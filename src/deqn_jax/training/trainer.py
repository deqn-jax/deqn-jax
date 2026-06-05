"""Main training loop for DEQN-JAX.

Key design: single JIT boundary around entire train_step for maximum performance.
Three step variants dispatched at construction time (before JIT):

- STANDARD: normal jax.grad + opt.update(grads, state, params)
- MAO: jax.jacrev(per_eq_loss_vector) -> per-equation Jacobian -> mao.update(eq_jac, state, params)
- LBFGS: optax.lbfgs (GradientTransformationExtraArgs) -- needs value + value_fn for line search
"""

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.metrics import create_logger
from deqn_jax.optimizers.registry import OptimizerKind, create_optimizer
from deqn_jax.training.checkpointing import (
    best_checkpoint_path as _best_checkpoint_path,
)
from deqn_jax.training.checkpointing import (
    final_save_best_fallback as _final_save_best_fallback,
)
from deqn_jax.training.checkpointing import (
    maybe_checkpoint as _maybe_checkpoint,
)
from deqn_jax.training.checkpointing import (
    maybe_save_best as _maybe_save_best,
)
from deqn_jax.training.history import get_history_len
from deqn_jax.training.loss import compute_loss, gauss_hermite_nd
from deqn_jax.training.reporting import (
    count_params as _count_params,
)
from deqn_jax.training.reporting import (
    log_episode as _log_episode,
)
from deqn_jax.training.reporting import (
    print_episode_progress as _print_episode_progress,
)
from deqn_jax.training.reporting import (
    print_final as _print_final,
)
from deqn_jax.training.reporting import (
    print_header as _print_header,
)

# Console banners and residual formatting now live in
# training/reporting.py; periodic / best-snapshot checkpointing lives
# in training/checkpointing.py. Both are imported above under their
# previous private names so call sites read unchanged.
# ---------------------------------------------------------------------------
# State + optimizer construction
# ---------------------------------------------------------------------------
# create_train_state / make_train_step / _validate_train_config /
# _resolve_model_for_training / _build_initial_state now live in
# training/state_init.py (pre-loop construction). Re-imported here under
# the same names so call sites + external imports read unchanged.
from deqn_jax.training.state_init import (  # noqa: E402  (re-export)
    _build_initial_state,
    _resolve_model_for_training,
    _validate_train_config,
    make_train_step,
)
from deqn_jax.training.state_init import (
    create_train_state as create_train_state,  # re-export (not used internally)
)
from deqn_jax.types import ModelSpec, TrainState


def _build_custom_loss_fn(config, model: ModelSpec, history_len: int):
    """Build the wrapped loss function for non-default loss configurations.

    Returns the custom loss callable (or None if the default MSE
    `compute_loss` should be used as-is). Handles three layered cases:
    composite loss, state-barrier penalty, and Huber loss for the bare
    path.
    """
    from functools import partial

    custom_loss_fn = None
    if config.loss_type == "composite":
        from deqn_jax.training.composite_loss import (
            make_composite_loss,
            prepare_composite_data,
        )
        from deqn_jax.training.linearize import linearize_model

        if config.verbose:
            print("  Building composite loss (linearize + ergodic cov)...")
        P, Q = linearize_model(model, verbose=config.verbose)

        comp_cfg = config.composite_loss
        comp_data = prepare_composite_data(
            model,
            P,
            Q,
            n_anchor_points=comp_cfg.n_anchor_points,
            anchor_sigma=comp_cfg.anchor_sigma,
            seed=config.seed,
            verbose=config.verbose,
        )
        custom_loss_fn = make_composite_loss(
            model,
            comp_data,
            anchor_weight=comp_cfg.anchor_weight,
            jac_weight=comp_cfg.jac_weight,
            jac_anchor_weight=comp_cfg.jac_anchor_weight,
            barrier_weight=comp_cfg.barrier_weight,
            newton_weight=comp_cfg.newton_weight,
            leverage_mult=comp_cfg.leverage_mult,
            aux_decay_floor=comp_cfg.aux_decay_floor,
            history_len=history_len,
            loss_choice=config.loss_choice,
            huber_delta=config.huber_delta,
        )
        if config.verbose:
            extras = []
            if config.loss_choice != "mse":
                extras.append(
                    f"loss_choice={config.loss_choice} (δ={config.huber_delta})"
                )
            if comp_cfg.jac_anchor_weight > 0:
                extras.append(f"sobolev-anchor w={comp_cfg.jac_anchor_weight}")
            extras_str = " · ".join(extras)
            print(
                f"  Composite loss ready.{(' · ' + extras_str) if extras_str else ''}"
            )

    barrier_weight = config.barrier_weight
    if (
        barrier_weight > 0
        and custom_loss_fn is None
        and model.state_barrier_fn is not None
    ):
        custom_loss_fn = partial(
            compute_loss,
            barrier_weight=barrier_weight,
            loss_choice=config.loss_choice,
            huber_delta=config.huber_delta,
        )
        if config.verbose:
            print(f"  State barrier: weight={barrier_weight}")

    if custom_loss_fn is None and config.loss_choice != "mse":
        custom_loss_fn = partial(
            compute_loss,
            loss_choice=config.loss_choice,
            huber_delta=config.huber_delta,
        )
        if config.verbose:
            print(f"  Loss choice: {config.loss_choice} (δ={config.huber_delta})")

    # Moment-matching aux loss layered on top of whatever was chosen above.
    # Uses Dynare's reference moments as the target. See
    # training/moment_loss.py for the design rationale.
    if (
        getattr(config, "moment_matching", None) is not None
        and config.moment_matching.enabled
    ):
        from deqn_jax.dynare_io import deqn_policy_to_dynare, load_dynare_moments
        from deqn_jax.training.moment_loss import (
            _resolve_target_indices,
            make_moment_matching_wrapper,
        )

        mom_cfg = config.moment_matching
        target_moments = load_dynare_moments(mom_cfg.dynare_dir)
        # DEQN ↔ Dynare name aliases (currently just `i` -> `i_var`); reuse
        # the canonical mapping from dynare_io.
        aliases = {p: deqn_policy_to_dynare(p) for p in model.policy_names}
        target_idx = _resolve_target_indices(
            policy_names=list(model.policy_names),
            target_moments=target_moments,
            name_aliases=aliases,
        )
        if config.verbose:
            print(
                f"  Moment-matching aux loss: weight={mom_cfg.weight}, "
                f"matching {len(target_idx)} policies against {mom_cfg.dynare_dir}"
            )
        custom_loss_fn = make_moment_matching_wrapper(
            custom_loss_fn,
            target_idx_to_moments=target_idx,
            weight=mom_cfg.weight,
            mean_weight=mom_cfg.mean_weight,
            std_weight=mom_cfg.std_weight,
            scale_eps=mom_cfg.scale_eps,
        )

    return custom_loss_fn


# ---------------------------------------------------------------------------
# Per-episode phase helpers for _run_training_loop
# ---------------------------------------------------------------------------


@dataclass
class _OptimizerRuntime:
    """Optimizer state that can be swapped mid-training.

    Mutated in place by ``_maybe_switch_optimizer``; everything else
    reads from it. Lives separately from ``TrainState`` because the
    optimizer object itself, the train_step closure, and the schedule
    function are Python-side (not part of the JIT-traced state).
    """

    opt: Any
    kind: OptimizerKind
    train_step: Callable
    lr_schedule_fn: Optional[Callable]
    switched: bool = False


@dataclass
class _NanRollback:
    """NaN recovery counters + last-known-good snapshot.

    ``last_good_state`` is refreshed at every periodic checkpoint and
    consumed by ``_handle_nan`` (roll-back path) and the end-of-training
    fallback save-best.
    """

    enabled: bool
    lr_scale: float = 1.0
    count: int = 0
    max_rollbacks: int = 10
    lr_reduction: float = 0.75
    last_good_state: Optional[TrainState] = None
    last_good_episode: int = 0


@dataclass
class _SaveBestTracker:
    best_loss: float = float("inf")
    best_episode: int = 0
    grace: int = 0


def _maybe_switch_optimizer(
    config,
    model: ModelSpec,
    n_equations: int,
    history_len: int,
    gradient_surgery: str,
    quad_nodes_jax: Optional[Array],
    quad_weights_jax: Optional[Array],
    custom_loss_fn: Optional[Callable],
    use_target: bool,
    state: TrainState,
    runtime: _OptimizerRuntime,
    ep_num: int,
) -> Tuple[TrainState, bool]:
    """Swap optimizer + train_step at config.switch_episode.

    Returns ``(state, did_switch)``. Mutates ``runtime`` in place. The
    caller resets early-stop counters when did_switch is True.
    """
    from deqn_jax.config import OptimizerConfig

    if (
        runtime.switched
        or config.switch_optimizer is None
        or config.switch_episode is None
        or ep_num != config.switch_episode
    ):
        return state, False

    switch_lr = config.switch_lr or config.optimizer.learning_rate
    switch_cfg = OptimizerConfig(
        name=config.switch_optimizer,
        learning_rate=switch_lr,
        grad_clip=config.optimizer.grad_clip,
    )
    new_opt, new_kind = create_optimizer(switch_cfg)
    if new_kind == OptimizerKind.MAO and hasattr(new_opt, "with_num_tasks"):
        new_opt = new_opt.with_num_tasks(n_equations)
    new_opt_state = new_opt.init(eqx.filter(state.params, eqx.is_array))
    state = state._replace(opt_state=new_opt_state)

    runtime.opt = new_opt
    runtime.kind = new_kind
    # Disable schedule after mid-training switch (uses constant LR).
    runtime.lr_schedule_fn = None
    runtime.train_step = make_train_step(
        model,
        new_opt,
        config.episode_length,
        config.mc_samples,
        config.batch_size,
        loss_reweight=config.loss_reweight,
        reweight_alpha=config.reweight_alpha,
        kind=new_kind,
        gradient_surgery=gradient_surgery,
        grad_clip=config.optimizer.grad_clip,
        quad_nodes=quad_nodes_jax,
        quad_weights=quad_weights_jax,
        history_len=history_len,
        compute_loss_fn=custom_loss_fn,
        ss_reset_frac=config.ss_reset_frac,
        use_target_network=use_target,
        n_epochs_per_rollout=config.n_epochs_per_rollout,
        n_minibatches_per_epoch=config.n_minibatches_per_epoch,
        initialize_each_episode=config.initialize_each_episode,
        sorted_within_batch=config.sorted_within_batch,
        replay_cfg=config.replay_buffer,
    )
    runtime.switched = True
    if config.verbose:
        print(
            f"  >> Switched to {config.switch_optimizer} (lr={switch_lr:.0e}) at episode {ep_num}"
        )
    return state, True


def _episode_lr_scale(
    config,
    lr_schedule_fn: Optional[Callable],
    history: Dict[str, list],
    nan: _NanRollback,
    ep_num: int,
) -> Tuple[Array, float]:
    """Return ``(lr_scale jnp scalar, current_lr float for logging)``.

    Stateful schedules (ReduceLROnPlateau) consume the most recent loss;
    stateless schedules accept but ignore it. NaN-rollback LR reduction
    is folded in via ``nan.lr_scale``.
    """
    if lr_schedule_fn is not None:
        last_loss = history["loss"][-1] if history["loss"] else None
        try:
            current_lr = float(lr_schedule_fn(ep_num, last_loss)) * nan.lr_scale
        except TypeError:
            # optax schedules accept a single positional arg; fall back.
            current_lr = float(lr_schedule_fn(ep_num)) * nan.lr_scale
        return jnp.array(current_lr), current_lr
    current_lr = config.optimizer.learning_rate * nan.lr_scale
    return jnp.array(nan.lr_scale), current_lr


def _episode_shock_scale(config, ep_num: int) -> Array:
    """Curriculum ramp + per-shock mask, fused into one shock_scale array.

    Returns a scalar when shock_mask is None, otherwise a vector
    [n_shocks]. Broadcasting in loss.py handles either shape.
    """
    if config.curriculum_episodes > 0 and ep_num < config.curriculum_episodes:
        t = ep_num / config.curriculum_episodes
        scale = config.curriculum_start + (1.0 - config.curriculum_start) * t
    else:
        scale = 1.0
    if config.shock_mask is not None:
        return jnp.array(scale) * jnp.array(config.shock_mask)
    return jnp.array(scale)


def _maybe_update_target(
    config, state: TrainState, ep_num: int, use_target: bool
) -> TrainState:
    if not use_target or ep_num % config.target_update_every != 0:
        return state
    if config.target_tau >= 1.0:
        return state._replace(target_params=state.params)
    tau = config.target_tau
    new_target = jax.tree.map(
        lambda p, t: tau * p + (1 - tau) * t,
        eqx.filter(state.params, eqx.is_array),
        eqx.filter(state.target_params, eqx.is_array),
    )
    return state._replace(target_params=eqx.combine(new_target, state.params))


def _handle_nan(
    config,
    state: TrainState,
    nan: _NanRollback,
    ep_num: int,
    loss_val: float,
) -> Tuple[TrainState, str]:
    """Detect NaN/Inf loss and decide rollback vs stop vs proceed.

    Returns ``(state, action)`` where action is one of
    ``"rollback" | "stop" | "proceed"``. State is the rolled-back snapshot
    on rollback, otherwise unchanged. ``nan`` is mutated in place.
    """
    if not (math.isnan(loss_val) or math.isinf(loss_val)):
        return state, "proceed"
    if (
        nan.enabled
        and nan.last_good_state is not None
        and nan.count < nan.max_rollbacks
    ):
        nan.count += 1
        nan.lr_scale *= nan.lr_reduction
        if config.verbose:
            effective_lr = config.optimizer.learning_rate * nan.lr_scale
            print(
                f"  >> NaN at episode {ep_num}! "
                f"Rolling back to ep {nan.last_good_episode}, "
                f"reducing LR to {effective_lr:.1e} "
                f"(rollback {nan.count}/{nan.max_rollbacks})"
            )
        return nan.last_good_state, "rollback"
    if nan.count >= nan.max_rollbacks:
        if config.verbose:
            print(
                f"  >> NaN at episode {ep_num} after {nan.max_rollbacks} rollbacks. Stopping."
            )
        return state, "stop"
    # No checkpoint to roll back to — just proceed (NaN will propagate).
    return state, "proceed"


def _check_early_stop(
    config,
    switched: bool,
    best_loss: float,
    patience: int,
    loss_val: float,
    ep_num: int,
) -> Tuple[float, int, bool]:
    """Update early-stop counters; return ``(best_loss, patience, should_break)``.

    Only active after a configured optimizer switch has fired, or when
    no switch is configured.
    """
    active = config.early_stop_patience is not None and (
        switched or config.switch_optimizer is None
    )
    if not active or math.isnan(loss_val):
        return best_loss, patience, False
    if loss_val < best_loss - config.early_stop_min_delta:
        return loss_val, 0, False
    patience += 1
    if patience >= config.early_stop_patience:
        if config.verbose:
            print(
                f"  >> Early stopping at episode {ep_num}: "
                f"no improvement for {config.early_stop_patience} episodes "
                f"(best={best_loss:.2e})"
            )
        return best_loss, patience, True
    return best_loss, patience, False


# _log_episode / _print_episode_progress now live in training/reporting.py
# (pure logging + console output); imported above under their previous
# private names so the call sites in _run_training_loop read unchanged.


# Checkpoint orchestration helpers (_maybe_checkpoint / _maybe_save_best /
# _final_save_best_fallback) moved to training/checkpointing.py and imported
# above. trainer.py keeps only the loop; checkpointing.py owns save policy.


def _run_training_loop(
    config,
    model: ModelSpec,
    state: TrainState,
    opt: Any,
    kind: OptimizerKind,
    gradient_surgery: str,
    train_step: Callable,
    lr_schedule_fn: Optional[Callable],
    quad_nodes_jax: Optional[Array],
    quad_weights_jax: Optional[Array],
    history_len: int,
    custom_loss_fn: Optional[Callable],
    use_target: bool,
    n_equations: int,
    start_episode: int,
    logger,
) -> Optional[Tuple[Any, Dict[str, list]]]:
    """Run the per-episode train loop.

    Orchestrates the per-episode phase helpers (mid-training switch,
    LR/curriculum scaling, train_step, target-net update, NaN rollback,
    early stop, logging, periodic + best-checkpoint writes). Each
    concern lives in its own helper above; this body is the algorithm
    in order.
    """
    if config.switch_optimizer and config.switch_episode is None:
        raise ValueError("--switch-optimizer requires --switch-episode")

    total_episodes = config.episodes
    if start_episode >= total_episodes:
        print(
            f"WARNING: checkpoint episode {start_episode} >= config.episodes {total_episodes}. Nothing to do."
        )
        return None

    runtime = _OptimizerRuntime(
        opt=opt,
        kind=kind,
        train_step=train_step,
        lr_schedule_fn=lr_schedule_fn,
    )
    nan = _NanRollback(
        enabled=(
            config.checkpoint_dir is not None and config.checkpoint_every is not None
        ),
        last_good_episode=start_episode,
    )
    save_best = _SaveBestTracker(
        # Don't save as "best" during curriculum ramp (shocks are reduced
        # → loss is artificially low). Falls back to log_every when no
        # curriculum is configured.
        grace=max(config.curriculum_episodes, config.log_every),
        best_episode=start_episode,
    )
    # Early-stop state is reset by mid-training switch, so kept as locals
    # rather than a third dataclass.
    es_best = float("inf")
    es_patience = 0

    history: Dict[str, list] = {"loss": [], "grad_norm": []}
    t_start = time.perf_counter()
    last_metrics = None
    ep_width = len(str(total_episodes))
    current_lr = config.optimizer.learning_rate

    for ep_num in range(start_episode + 1, total_episodes + 1):
        state, did_switch = _maybe_switch_optimizer(
            config,
            model,
            n_equations,
            history_len,
            gradient_surgery,
            quad_nodes_jax,
            quad_weights_jax,
            custom_loss_fn,
            use_target,
            state,
            runtime,
            ep_num,
        )
        if did_switch:
            es_best, es_patience = float("inf"), 0

        lr_scale, current_lr = _episode_lr_scale(
            config, runtime.lr_schedule_fn, history, nan, ep_num
        )
        shock_scale = _episode_shock_scale(config, ep_num)

        state, metrics = runtime.train_step(state, lr_scale, shock_scale)
        last_metrics = metrics
        state = _maybe_update_target(config, state, ep_num, use_target)

        loss_val = float(metrics.loss)
        grad_val = float(metrics.grad_norm)

        state, nan_action = _handle_nan(config, state, nan, ep_num, loss_val)
        if nan_action == "rollback":
            continue
        if nan_action == "stop":
            break

        es_best, es_patience, stop = _check_early_stop(
            config, runtime.switched, es_best, es_patience, loss_val, ep_num
        )
        if stop:
            break

        history["loss"].append(loss_val)
        history["grad_norm"].append(grad_val)

        if ep_num % config.log_every == 0 or ep_num == total_episodes:
            _log_episode(
                config,
                model,
                state,
                metrics,
                ep_num,
                total_episodes,
                start_episode,
                t_start,
                current_lr,
                history_len,
                logger,
            )
        if config.verbose and ep_num % config.log_every == 0:
            _print_episode_progress(
                metrics, ep_num, total_episodes, ep_width, start_episode, t_start
            )

        _maybe_checkpoint(config, state, nan, ep_num)
        _maybe_save_best(config, state, save_best, loss_val, ep_num)

    _final_save_best_fallback(config, state, nan, save_best, history)

    if config.verbose and last_metrics is not None:
        elapsed = time.perf_counter() - t_start
        _print_final(
            elapsed=elapsed,
            episodes=config.episodes,
            final_loss=float(last_metrics.loss),
            final_residuals=last_metrics.residuals,
        )
        if save_best.best_loss < float("inf"):
            print(
                f"Best checkpoint: {save_best.best_loss:.2e} at episode "
                f"{save_best.best_episode} → {_best_checkpoint_path(config.checkpoint_dir)}"
            )

    logger.close()
    return state.params, history


# ---------------------------------------------------------------------------
# Training entry points
# ---------------------------------------------------------------------------


def train_from_config(config) -> Tuple[Any, Dict[str, list]]:
    """Train from a TrainConfig object.

    This is the primary entry point for config-driven training.
    Supports checkpoint resume, mid-training optimizer switching,
    and grouped TensorBoard logging.

    Args:
        config: TrainConfig instance

    Returns:
        Tuple of (trained_params, history_dict)
    """
    _validate_train_config(config)
    model, n_equations = _resolve_model_for_training(config)

    fp64 = jnp.zeros(1).dtype == jnp.float64
    hidden_sizes = config.network.hidden_sizes
    key = jax.random.PRNGKey(config.seed)

    # ---- Build LR schedule helper for logging ----
    from deqn_jax.optimizers.registry import _build_lr_schedule

    # When a schedule is active, the optimizer is created with lr=1.0.
    # The actual LR is passed as a dynamic scalar to train_step each episode.
    has_schedule = config.optimizer.lr_schedule != "constant"
    if has_schedule:
        effective_opt_cfg = config.optimizer.model_copy(
            update={"learning_rate": 1.0, "lr_schedule": "constant"}
        )
    else:
        effective_opt_cfg = config.optimizer

    state, opt, kind, start_episode, total_for_schedule = _build_initial_state(
        config,
        model,
        key,
        n_equations,
        effective_opt_cfg,
    )

    # ---- Metric logger ----
    wandb_config = config.to_dict() if config.wandb_project else None
    logger = create_logger(
        tensorboard_dir=config.tensorboard_dir,
        wandb_project=config.wandb_project,
        wandb_config=wandb_config,
    )

    # ---- Print header ----
    if config.verbose:
        _print_header(
            model_spec=model,
            optimizer=config.optimizer.name,
            learning_rate=config.optimizer.learning_rate,
            hidden_sizes=hidden_sizes,
            n_params=_count_params(state.params),
            batch_size=config.batch_size,
            mc_samples=config.mc_samples,
            warm_start=config.warm_start,
            grad_clip=config.optimizer.grad_clip,
            loss_reweight=config.loss_reweight,
            fp64=fp64,
            lr_schedule=config.optimizer.lr_schedule,
            lr_warmup=config.optimizer.lr_warmup,
            lr_min_factor=config.optimizer.lr_min_factor,
            net_type=getattr(config.network, "type", "mlp")
            if config.network
            else "mlp",
            history_len=get_history_len(state.params),
        )

    # Build LR schedule function for computing per-episode LR (None if constant)
    lr_schedule_fn = None
    if has_schedule:
        lr_schedule_fn = _build_lr_schedule(config.optimizer, total_for_schedule)

    # ---- Pre-compute quadrature nodes (if using Gauss-Hermite) ----
    quad_nodes_jax = None
    quad_weights_jax = None
    exp_type = config.expectation_type
    if exp_type in ("quadrature", "gh", "gauss_hermite"):
        n_qp = config.n_quadrature_points
        quad = gauss_hermite_nd(n_qp, model.n_shocks)
        if quad is not None:
            quad_nodes_jax = jnp.array(quad[0])
            quad_weights_jax = jnp.array(quad[1])
            if config.verbose:
                print(
                    f"  Quadrature: {n_qp}^{model.n_shocks} = {quad[0].shape[0]} nodes (Gauss-Hermite)"
                )
        else:
            n_total = n_qp**model.n_shocks
            if config.verbose:
                print(
                    f"  Quadrature: {n_total} nodes exceeds limit, falling back to MC"
                )

    # ---- Determine history length from network (Python-level, before JIT) ----
    history_len = get_history_len(state.params)

    # ---- Shock mask ----
    if config.shock_mask is not None and config.verbose:
        shock_names = (
            model.shock_names
            if model.shock_names
            else tuple(f"shock_{i}" for i in range(model.n_shocks))
        )
        active = [n for n, m in zip(shock_names, config.shock_mask) if m > 0]
        zeroed = [n for n, m in zip(shock_names, config.shock_mask) if m == 0]
        print(f"  Shock mask: active={active}, zeroed={zeroed}")

    custom_loss_fn = _build_custom_loss_fn(config, model, history_len)

    # ---- Target network setup ----
    use_target = config.target_update_every > 0
    if use_target:
        state = state._replace(target_params=state.params)
        if config.verbose:
            print(
                f"  Target network: update every {config.target_update_every} episodes"
                f" (tau={config.target_tau})"
            )

    # ---- Create JIT-compiled train step ----
    gradient_surgery = config.gradient_surgery
    train_step = make_train_step(
        model,
        opt,
        config.episode_length,
        config.mc_samples,
        config.batch_size,
        loss_reweight=config.loss_reweight,
        reweight_alpha=config.reweight_alpha,
        kind=kind,
        gradient_surgery=gradient_surgery,
        grad_clip=config.optimizer.grad_clip,
        quad_nodes=quad_nodes_jax,
        quad_weights=quad_weights_jax,
        history_len=history_len,
        compute_loss_fn=custom_loss_fn,
        ss_reset_frac=config.ss_reset_frac,
        use_target_network=use_target,
        n_epochs_per_rollout=config.n_epochs_per_rollout,
        n_minibatches_per_epoch=config.n_minibatches_per_epoch,
        initialize_each_episode=config.initialize_each_episode,
        sorted_within_batch=config.sorted_within_batch,
        replay_cfg=config.replay_buffer,
    )

    if (
        config.verbose
        and kind == OptimizerKind.STANDARD
        and gradient_surgery != "pcgrad"
    ):
        # Compute and report the effective schedule so users can see what
        # the trainer is actually doing per outer iteration.
        ep_samples = config.episode_length * config.batch_size
        mbs_avail = max(1, ep_samples // config.batch_size)
        mbs_this_epoch = (
            min(config.n_minibatches_per_epoch, mbs_avail)
            if config.n_minibatches_per_epoch is not None
            else mbs_avail
        )
        updates_per_cycle = config.n_epochs_per_rollout * mbs_this_epoch
        print(
            f"  Schedule: 1 rollout ({config.episode_length}×{config.batch_size}="
            f"{ep_samples} states) → {config.n_epochs_per_rollout} epoch(s) × "
            f"{mbs_this_epoch} minibatch(es) of {config.batch_size} "
            f"= {updates_per_cycle} grad updates/cycle "
            f"({updates_per_cycle * config.episodes} total over "
            f"{config.episodes} cycles)"
        )

    return _run_training_loop(
        config=config,
        model=model,
        state=state,
        opt=opt,
        kind=kind,
        gradient_surgery=gradient_surgery,
        train_step=train_step,
        lr_schedule_fn=lr_schedule_fn,
        quad_nodes_jax=quad_nodes_jax,
        quad_weights_jax=quad_weights_jax,
        history_len=history_len,
        custom_loss_fn=custom_loss_fn,
        use_target=use_target,
        n_equations=n_equations,
        start_episode=start_episode,
        logger=logger,
    )


def train(
    model_name: str,
    episodes: int = 1000,
    hidden_sizes: Tuple[int, ...] = (64, 64),
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    episode_length: int = 100,
    mc_samples: int = 5,
    optimizer: str = "adam",
    warm_start: bool = False,
    seed: int = 42,
    log_every: int = 100,
    verbose: bool = True,
    grad_clip: Optional[float] = None,
    loss_weights: Optional[List[float]] = None,
    loss_reweight: str = "none",
    reweight_alpha: float = 0.9,
    tensorboard_dir: Optional[str] = None,
    wandb_project: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: Optional[int] = None,
) -> Tuple[Any, Dict[str, list]]:
    """Train DEQN model (backward-compatible wrapper).

    Builds a TrainConfig and delegates to train_from_config().

    Preserves the pre-sweep per-cycle training budget (one gradient step
    per cycle) to avoid surprising legacy callers. New code should prefer
    ``train_from_config(TrainConfig(...))`` which defaults to the full
    rollout+minibatch-sweep schedule matching reference DEQN.
    """
    from deqn_jax.config import NetworkConfig, OptimizerConfig, TrainConfig

    config = TrainConfig(
        model=model_name,
        episodes=episodes,
        batch_size=batch_size,
        episode_length=episode_length,
        mc_samples=mc_samples,
        seed=seed,
        network=NetworkConfig(hidden_sizes=hidden_sizes),
        optimizer=OptimizerConfig(
            name=optimizer,
            learning_rate=learning_rate,
            grad_clip=grad_clip,
        ),
        warm_start=warm_start,
        loss_weights=list(loss_weights) if loss_weights is not None else None,
        loss_reweight=loss_reweight,
        reweight_alpha=reweight_alpha,
        log_every=log_every,
        verbose=verbose,
        tensorboard_dir=tensorboard_dir,
        wandb_project=wandb_project,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        n_minibatches_per_epoch=1,
    )

    return train_from_config(config)
