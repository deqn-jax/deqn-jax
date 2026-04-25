"""DEQN-style train cycle: 1 rollout → many minibatch gradient steps.

The cycle wrapper is kind-agnostic: it knows how to roll an episode
forward, slice the resulting trajectory into minibatches, and call a
caller-supplied ``grad_step`` over them. Per-optimizer gradient
mechanics live in ``deqn_jax.optimizers.<name>.make_grad_step_<name>``
and are composed into a cycle by ``make_train_step`` in ``trainer.py``.

Two builders:
- ``make_rollout_fn`` — JIT'd ``state -> (trajectory, final_state, history, key)``.
- ``make_cycle_step`` — wraps a rollout + a grad step into the
  ``(state, lr_scale, shock_scale) -> (state, Metrics)`` shape that
  ``train_from_config`` consumes.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.training.episode import run_episode, run_episode_with_history
from deqn_jax.training.history import build_history_windows
from deqn_jax.types import Metrics, ModelSpec, TrainState


def make_rollout_fn(
    model: ModelSpec,
    episode_length: int,
    history_len: int,
    ss_reset_frac: float,
    initialize_each_episode: bool = False,
):
    """JIT'd: run one rollout → return (trajectory, final_state, history, new_key).

    The trajectory has shape [episode_length, batch, n_states] (MLP) or
    needs to be wrapped into windows for sequence nets. The outer loop
    then draws minibatches from it and calls the grad step many times.

    initialize_each_episode: if True, replace the entire batch with a fresh
    init_state_fn draw every rollout (deterministic-model "fresh sample
    per episode" semantics; see TrainConfig docstring).
    """

    @jax.jit
    def rollout_fn(
        state: TrainState,
        shock_scale: Array = jnp.array(1.0),
    ) -> Tuple[Array, Array, Array, Array]:
        key, episode_key, reset_key = jax.random.split(state.key, 3)
        ep_states = state.episode_state
        if initialize_each_episode and model.init_state_fn is not None:
            ep_states = model.init_state_fn(
                reset_key, ep_states.shape[0], model.constants
            )
        elif ss_reset_frac > 0.0:
            batch_n = ep_states.shape[0]
            n_reset = int(ss_reset_frac * batch_n)
            if n_reset > 0:
                assert model.steady_state_fn is not None, (
                    "ss_reset_frac>0 requires a model with steady_state_fn"
                )
                ss_state, _ = model.steady_state_fn(model.constants)
                noise = jax.random.uniform(
                    reset_key,
                    (n_reset, model.n_states),
                    minval=-0.05,
                    maxval=0.05,
                )
                fresh = ss_state * (1 + noise)
                ep_states = ep_states.at[:n_reset].set(fresh)
        if history_len > 1:
            trajectory, final_state, final_history = run_episode_with_history(
                model,
                state.params,
                ep_states,
                episode_key,
                episode_length,
                history_len,
                shock_scale=shock_scale,
                init_history=state.history_state,
            )
        else:
            trajectory, final_state = run_episode(
                model,
                state.params,
                ep_states,
                episode_key,
                episode_length,
                shock_scale=shock_scale,
            )
            final_history = state.history_state  # None for MLP; pass through unchanged
        return trajectory, final_state, final_history, key

    return rollout_fn


def make_cycle_step(
    rollout_fn: Callable,
    grad_step: Callable,
    model: ModelSpec,
    batch_size: int,
    n_epochs_per_rollout: int,
    n_minibatches_per_epoch: Optional[int],
    history_len: int = 1,
    sorted_within_batch: bool = False,
    replay_cfg: Optional[Any] = None,
):
    """Generic DEQN-style cycle: 1 rollout + n_epochs × minibatch sweep.

    Works for any ``grad_step`` with signature
    ``(state, batch, lr_scale, shock_scale) → (new_state, metrics)``.
    The rollout is kind-agnostic; only the per-batch grad step differs.

    ``sorted_within_batch`` (default False): when True and history_len==1,
    minibatches are contiguous slices of single trajectories (RL-style)
    rather than IID-shuffled samples. Batch order is shuffled; intra-batch
    order is preserved. See TrainConfig docstring.

    ``replay_cfg`` (default None): when provided AND ``replay_cfg.enabled``,
    a prioritized state-replay buffer (held on ``state.replay_state``) is
    written to after each rollout and sampled into the minibatch dataset.
    Incompatible with ``sorted_within_batch`` (buffer rows break trajectory
    contiguity) — raises at builder time.
    """
    use_replay = replay_cfg is not None and getattr(replay_cfg, "enabled", False)
    if use_replay and sorted_within_batch:
        raise ValueError(
            "replay_buffer.enabled and sorted_within_batch are incompatible: "
            "buffer rows break the trajectory-contiguous-chunk semantics that "
            "sorted_within_batch relies on. Disable one."
        )
    if use_replay and history_len > 1:
        raise NotImplementedError(
            "Replay buffer for sequence networks (history_len > 1) is a v2 "
            "follow-up. v1 supports MLP only."
        )

    def cycle_step(
        state: TrainState,
        lr_scale: Array,
        shock_scale: Array = jnp.array(1.0),
    ) -> Tuple[TrainState, Metrics]:
        # 1. Rollout — one JIT'd call.
        # Note: `rollout_fn` returns trajectory = [s_0, ..., s_{T-1}] and
        # final_carry_state = s_T (T transitions total). The *reference*
        # DEQN_MAO stores [s_0, ..., s_{T-1}] in state_episode and seeds
        # the next cycle from state_episode[T-1] = s_{T-1} (so cycles
        # overlap by one state, and each cycle advances T-1 transitions).
        # We use trajectory[-1] = s_{T-1} for seeding to match that
        # convention; final_carry_state is discarded.
        trajectory, _final_after_T, final_history, new_key = rollout_fn(
            state, shock_scale
        )
        next_seed = trajectory[-1]
        # Persist the history window alongside episode_state so recurrent
        # policies see continuous trajectories across cycles instead of
        # a constant-prefix rebuild at every rollout. For MLP (history_len=1)
        # final_history passes through as None.
        state = state._replace(
            episode_state=next_seed,
            key=new_key,
            history_state=final_history,
        )

        # 1b. Replay buffer write — once per cycle, after rollout.
        # Computes residual-based priorities at the trajectory states under
        # the *current* policy and stores both into a fixed-shape ring on
        # state.replay_state. Enabled only when replay_cfg.enabled is True.
        if use_replay:
            from deqn_jax.training import replay as _replay

            traj_flat = trajectory.reshape(-1, model.n_states)
            prio_key, state_key = jax.random.split(state.key)
            state = state._replace(key=state_key)
            prios = _replay.compute_priorities(
                model, state.params, traj_flat, prio_key, shock_scale=shock_scale
            )
            new_replay = _replay.write(state.replay_state, traj_flat, prios)
            state = state._replace(replay_state=new_replay)

        # 2. Build the minibatch dataset from the full trajectory.
        # Two layouts (chosen at trace time via sorted_within_batch):
        # - time-major [T*B, D]: reshape trajectory [T, B, D] directly.
        #   IID shuffling of all samples, used when sorted_within_batch=False.
        # - trajectory-major [B*T, D]: transpose to [B, T, D] first.
        #   Contiguous slices of length `batch_size` are single-trajectory
        #   temporal segments; used when sorted_within_batch=True.
        if history_len > 1:
            # Sequence path: windows already carry temporal coherence.
            # sorted_within_batch is a no-op here.
            dataset = build_history_windows(trajectory, history_len)
        elif sorted_within_batch:
            # [T, B, D] -> [B, T, D] -> [B*T, D]
            dataset = jnp.transpose(trajectory, (1, 0, 2)).reshape(-1, model.n_states)
        else:
            dataset = trajectory.reshape(-1, model.n_states)

        # 2b. Replay buffer sample — concat priority-weighted past states into
        # the dataset. Python-side warmup gate: until n_filled crosses the
        # min_fill_frac threshold the dataset is current-trajectory only.
        if use_replay:
            from deqn_jax.training import replay as _replay

            assert replay_cfg is not None  # narrowing for type-checkers
            if _replay.is_warm(
                state.replay_state, replay_cfg.capacity, replay_cfg.min_fill_frac
            ):
                n_buffered = int(replay_cfg.mix_ratio * dataset.shape[0])
                if n_buffered > 0:
                    sample_key, state_key = jax.random.split(state.key)
                    state = state._replace(key=state_key)
                    buffered, _ = _replay.sample(
                        state.replay_state,
                        sample_key,
                        n_buffered,
                        alpha=replay_cfg.priority_alpha,
                        eps=replay_cfg.priority_eps,
                    )
                    dataset = jnp.concatenate([dataset, buffered], axis=0)

        n_samples = dataset.shape[0]
        n_mbs_available = max(1, n_samples // batch_size)
        n_mbs = (
            min(n_minibatches_per_epoch, n_mbs_available)
            if n_minibatches_per_epoch is not None
            else n_mbs_available
        )

        # 3. Sweep: n_epochs × n_minibatches gradient updates.
        # Cycle-level metrics are aggregated across ALL minibatches in the
        # sweep, matching DEQN_MAO's run_cycle which reports
        # MSE_epoch_loss = epoch_loss / (N_episode_length * N_sim_batch).
        # Per-minibatch loss from grad_step is already a mean over its
        # batch, so averaging over minibatches gives an overall mean.
        # grad_norm is tracked as a max across the sweep (the spike is
        # what matters for stability diagnostics). Residuals are averaged
        # per equation.
        last_metrics = None
        loss_acc = jnp.array(0.0)
        grad_max = jnp.array(0.0)
        eq_acc: Optional[Dict[str, Array]] = None
        n_steps = 0
        for _ in range(n_epochs_per_rollout):
            perm_key, state_key = jax.random.split(state.key)
            state = state._replace(key=state_key)
            if sorted_within_batch and history_len == 1:
                # Shuffle the *order of contiguous chunks*, not samples.
                # Each chunk is dataset[b*batch_size : (b+1)*batch_size],
                # a trajectory segment of length batch_size.
                batch_perm = jax.random.permutation(perm_key, n_mbs_available)
            else:
                # Shuffle individual sample indices (IID minibatch).
                perm = jax.random.permutation(perm_key, n_samples)
            for mb_idx in range(n_mbs):
                if sorted_within_batch and history_len == 1:
                    b = batch_perm[mb_idx]
                    start = b * batch_size
                    minibatch = jax.lax.dynamic_slice_in_dim(dataset, start, batch_size)
                else:
                    idx = perm[mb_idx * batch_size : (mb_idx + 1) * batch_size]
                    minibatch = dataset[idx]
                state, last_metrics = grad_step(state, minibatch, lr_scale, shock_scale)
                loss_acc = loss_acc + last_metrics.loss
                grad_max = jnp.maximum(grad_max, last_metrics.grad_norm)
                if eq_acc is None:
                    eq_acc = {k: v * 0.0 for k, v in last_metrics.residuals.items()}
                for k, v in last_metrics.residuals.items():
                    eq_acc[k] = eq_acc[k] + v
                n_steps += 1

        # 4. Bump episode counter once per cycle.
        state = state._replace(episode=state.episode + 1)

        # 5. Build aggregated metrics.
        if n_steps == 0 or last_metrics is None:
            return state, last_metrics
        n_steps_f = float(n_steps)
        avg_loss = loss_acc / n_steps_f
        avg_eq = {k: v / n_steps_f for k, v in (eq_acc or {}).items()}
        aggregated = Metrics(loss=avg_loss, residuals=avg_eq, grad_norm=grad_max)
        return state, aggregated

    return cycle_step
