"""Per-episode runtime controllers for the DEQN-JAX training loop.

The small, Python-side (non-JIT) controllers that ``_run_training_loop``
orchestrates each episode -- mid-training optimizer switch, LR / curriculum
scaling, target-network update, NaN rollback, and early stop -- plus the three
mutable runtime-state dataclasses they read/write. Extracted from trainer.py so
the orchestrator body reads as the algorithm in order. Pure move (no logic
changes); trainer.py re-imports these under the same names.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.optimizers.registry import OptimizerKind, create_optimizer
from deqn_jax.training.state_init import make_train_step
from deqn_jax.types import ModelSpec, TrainState


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
