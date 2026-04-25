"""Prioritized state-replay buffer for DEQN training.

Pure JIT-friendly helpers used by the cycle orchestration to mix past states
into each minibatch. Lives here, not in ``trainer.py``, so every optimizer
kind (STANDARD / PCGRAD / MAO / LBFGS / GN) shares the same buffer mechanics.

Why: DEQN samples states from the ergodic distribution under the *current*
policy. As the policy improves it stops visiting regions it earlier visited,
and the network forgets how to satisfy equilibrium there. Symptom on real
research models (especially disaster, ZLB, occasionally-binding constraints):
the network "solves" the equilibrium on the often-visited branch and loses
signal on the rare-event branch where the moments researchers care about live.
A ring buffer of past states + priority-weighted sampling addresses both
the anti-forgetting and spectral-bias failure modes.

API:
    make_replay_state(capacity, n_states) -> ReplayState
    write(state, samples, priorities) -> ReplayState
    sample(state, key, n, alpha, eps) -> (samples, new_key)
    is_warm(state, capacity, min_fill_frac) -> bool   (Python-side gate)
    compute_priorities(model, params, states, key, shock_scale) -> Array

Priority semantics (PER, Schaul et al. 2015):
    sampling_prob(i) ∝ (priority[i] + eps) ** alpha

Priorities are computed at WRITE time as the per-element sum of squared
equilibrium residuals. v1 does NOT update priorities lazily after gradient
steps see those buffered states; the FIFO ring eventually evicts stale-high
entries and α=0.6 (sub-linear weighting) caps the over-sampling. Lazy
in-place priority updates are a v2 follow-up that requires plumbing
per-element residuals through every train-step variant's ``Metrics``.

Important behavioral note: the residual is evaluated as
``equations_fn(s, π_now(s), s', π_now(s'))`` for buffered states ``s``.
That's the *anti-forgetting* signal — we want the *current* policy to
satisfy equilibrium at *past* states. This is NOT off-policy importance-
sampling correction; if the equations are mis-specified the buffer
won't help, only if the sampling distribution drifts.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.types import ReplayState, make_replay_state

__all__ = [
    "ReplayState",
    "make_replay_state",
    "write",
    "sample",
    "is_warm",
    "compute_priorities",
]


def write(state: ReplayState, samples: Array, priorities: Array) -> ReplayState:
    """Append ``samples`` + ``priorities`` to the ring buffer.

    Args:
        state: Current ``ReplayState``.
        samples: ``[N, n_states]`` new states to store.
        priorities: ``[N]`` per-state priorities (sum-of-squared residuals).

    Returns:
        Updated ``ReplayState`` with the new rows written at positions
        ``[write_idx, write_idx+1, ..., write_idx+N-1] mod capacity``.
        ``write_idx`` advances modulo capacity; ``n_filled`` is clamped at
        capacity. If ``N > capacity``, only the last ``capacity`` rows are
        retained (the older rows in the input batch get overwritten by the
        newer ones during the same write — same as if you'd called write
        twice with halves).
    """
    capacity = state.buffer.shape[0]
    n = samples.shape[0]

    # If a single write batch exceeds capacity, drop the oldest rows of the
    # batch to keep only the last `capacity` rows. This is a compile-time
    # decision (n is static from the input shape).
    if n > capacity:
        samples = samples[-capacity:]
        priorities = priorities[-capacity:]
        n = capacity

    indices = (state.write_idx + jnp.arange(n, dtype=jnp.int32)) % capacity

    new_buffer = state.buffer.at[indices].set(samples.astype(state.buffer.dtype))
    new_priorities = state.priorities.at[indices].set(
        priorities.astype(state.priorities.dtype)
    )
    new_write_idx = (state.write_idx + jnp.int32(n)) % jnp.int32(capacity)
    new_n_filled = jnp.minimum(state.n_filled + jnp.int32(n), jnp.int32(capacity))

    return state._replace(
        buffer=new_buffer,
        priorities=new_priorities,
        write_idx=new_write_idx,
        n_filled=new_n_filled,
    )


def sample(
    state: ReplayState,
    key: Array,
    n: int,
    alpha: float,
    eps: float,
) -> Tuple[Array, Array]:
    """Draw ``n`` priority-weighted samples (with replacement) from the buffer.

    Sampling probability over rows ``[0, n_filled)`` is proportional to
    ``(priority + eps) ** alpha``; rows in the unfilled tail are masked out
    via a zero weight, so they're never selected even though the underlying
    array is shape ``[capacity]``.

    Args:
        state: Current ``ReplayState``. Caller is expected to have gated on
            ``is_warm`` so ``n_filled > 0``.
        key: PRNG key.
        n: Number of samples to draw.
        alpha: PER exponent. ``alpha=0`` recovers uniform sampling.
        eps: Floor added to priorities before exponentiation. Prevents
            zero-priority rows from being completely starved.

    Returns:
        ``(samples [n, n_states], new_key)``.
    """
    capacity = state.buffer.shape[0]
    mask = jnp.arange(capacity, dtype=jnp.int32) < state.n_filled
    weights = jnp.where(mask, (state.priorities + eps) ** alpha, jnp.float32(0.0))
    # Normalize. If buffer is empty (sum=0), this would NaN; the caller is
    # supposed to gate via is_warm. Add a tiny stabilizer so we degrade to a
    # uniform draw over the (still-empty) prefix instead of crashing.
    weights_sum = jnp.sum(weights)
    weights = weights / jnp.maximum(weights_sum, jnp.float32(1e-30))

    sub_key, new_key = jax.random.split(key)
    indices = jax.random.choice(sub_key, capacity, shape=(n,), replace=True, p=weights)
    samples = state.buffer[indices]
    return samples, new_key


def is_warm(state: ReplayState, capacity: int, min_fill_frac: float) -> bool:
    """Python-side gate: has the buffer reached the warmup threshold?

    Resolves to a concrete bool when called outside JIT (which is where the
    cycle orchestration lives). Use this to decide whether to mix buffered
    samples into the current minibatch dataset.
    """
    threshold = int(capacity * min_fill_frac)
    # n_filled is a 0-d JAX array; int() forces concretization. Safe outside JIT.
    return int(state.n_filled) >= threshold


def compute_priorities(
    model: Any,
    policy_fn: Callable[[Array], Array],
    states: Array,
    key: Array,
    shock_scale: float | Array = 1.0,
) -> Array:
    """One forward+residual pass over ``states`` → per-row priority scalar.

    Priority := sum across equations of squared residuals at one shock
    realization. Used to seed buffer entries at write time.

    Args:
        model: ``ModelSpec`` (only ``n_shocks`` and ``equations_fn`` /
            ``step_fn`` are accessed).
        policy_fn: Callable mapping ``[n_states]`` (or ``[H, n_states]``) to
            policy vector. Pass the Equinox module directly, same as the
            optimizer-side ``compute_loss`` does.
        states: ``[N, n_states]`` (or ``[N, H, n_states]`` for sequence nets,
            but those are blocked at the config-validation layer in v1).
        key: PRNG key for shock sampling.
        shock_scale: Curriculum-style scaling. Default 1.0 (full shock magnitude).

    Returns:
        ``[N]`` non-negative priorities (NaN/Inf scrubbed to 0.0 for stability).
    """
    # Local import to avoid circulars (loss.py imports from types/spec, fine).
    from deqn_jax.training.loss import compute_residuals

    n = states.shape[0]
    shock = jax.random.normal(key, (n, model.n_shocks)) * jnp.asarray(shock_scale)
    residuals_dict = compute_residuals(model, policy_fn, states, shock)
    res_arr = jnp.stack(list(residuals_dict.values()), axis=0)  # [n_eq, N]
    prio = jnp.sum(res_arr**2, axis=0)  # [N]
    prio = jnp.where(jnp.isfinite(prio), prio, jnp.float32(0.0))
    return prio
