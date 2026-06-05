"""Main training loop for DEQN-JAX.

Key design: single JIT boundary around entire train_step for maximum performance.
Three step variants dispatched at construction time (before JIT):

- STANDARD: normal jax.grad + opt.update(grads, state, params)
- MAO: jax.jacrev(per_eq_loss_vector) -> per-equation Jacobian -> mao.update(eq_jac, state, params)
- LBFGS: optax.lbfgs (GradientTransformationExtraArgs) -- needs value + value_fn for line search
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.metrics import create_logger
from deqn_jax.optimizers.registry import OptimizerKind
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

# _build_custom_loss_fn (builds the composite / barrier / huber / moment loss
# object) now lives in training/composite_loss.py, beside make_composite_loss.
from deqn_jax.training.composite_loss import (  # noqa: E402  (re-export)
    _build_custom_loss_fn,
)
from deqn_jax.training.history import get_history_len

# ---------------------------------------------------------------------------
# Per-episode phase helpers for _run_training_loop
# ---------------------------------------------------------------------------
# Per-episode runtime controllers now live in training/loop_control.py.
# Re-imported under the same names so _run_training_loop reads unchanged.
from deqn_jax.training.loop_control import (  # noqa: E402  (re-export)
    _check_early_stop,
    _episode_lr_scale,
    _episode_shock_scale,
    _handle_nan,
    _maybe_switch_optimizer,
    _maybe_update_target,
    _NanRollback,
    _OptimizerRuntime,
    _SaveBestTracker,
)
from deqn_jax.training.loss import gauss_hermite_nd
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
