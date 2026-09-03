"""Shared eval-rollout helpers: discrete-chain detection, shock draw, rollout loop.

Every evaluation-side simulation (Euler residuals, simulated moments,
stability, the active-subspace sampler, the deterministic IRF path) is the
same loop: draw a shock, step, record, clip, repeat. It used to be written
out six times, each copy carrying a different subset of the branches
(disaster Bernoulli, discrete Markov chain, two-stage quadrature). The loop
lives here once; callers supply the per-step function (still JIT-compiled on
their side) and a ``record`` callback.

The shock draw itself routes through ``deqn_jax.training.shocks`` so the
verifier samples from the same primitives the trainer does.
"""

from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from deqn_jax.training.shocks import (
    draw_discrete_shocks,
    draw_training_shocks,
    maybe_draw_disaster,
)

# ---------------------------------------------------------------------------
# Eval-time shock dispatcher (shared by all eval primitives)
# ---------------------------------------------------------------------------


def _model_uses_discrete_chain(model) -> bool:
    """True iff the model declares a discrete Markov-chain shock.

    Mirrors the trainer-side check in ``training/loss.py`` and
    ``training/shocks.py`` so the verifier visits the same support the
    trainer trained on.
    """
    return (
        getattr(model, "transition_matrix", None) is not None
        and getattr(model, "z_state_idx", None) is not None
    )


def _draw_eval_shock(model, key, state):
    """Draw one shock for a single-batch eval step.

    Continuous case: ``[1, n_shocks]`` Gaussian (legacy behavior).
    Discrete case:   ``[1]`` int32 sampled from ``Π[z_t]``, where
    ``z_t = state[:, z_state_idx]``. The shock IS the next-period
    categorical index — ``step_fn`` is responsible for embedding it
    into next-state.

    Both branches call the trainer's own helpers
    (``training.shocks.draw_discrete_shocks`` / ``draw_training_shocks``)
    at eval defaults (unit scale, no mask), so the drawn values are
    identical to the trainer's for the same key.
    """
    if _model_uses_discrete_chain(model):
        z_idx = int(model.z_state_idx)
        current_z = state[:, z_idx].astype(jnp.int32)
        return draw_discrete_shocks(
            key, current_z, jnp.asarray(model.transition_matrix)
        )
    return draw_training_shocks(key, 1, model.n_shocks)


# ---------------------------------------------------------------------------
# The one rollout loop
# ---------------------------------------------------------------------------

# ``step_fn(state, shock, d_disaster) -> outputs``; ``outputs[0]`` must be the
# next state, the rest is whatever the caller wants to record.
StepFn = Callable[[Any, Any, Optional[Any]], Tuple]
# ``record(t, outputs) -> bool``; return truthy to stop the rollout early.
RecordFn = Callable[[int, Tuple], Any]


def _clip(model, state):
    """Trajectory-propagation clip (identity when the model has none)."""
    return model.clip_state_fn(state) if model.clip_state_fn is not None else state


def _rollout(model, state, n_periods, step_fn, record, draw, after_step):
    for t in range(n_periods):
        shock, d_disaster = draw(state)
        outputs = step_fn(state, shock, d_disaster)
        if record(t, outputs):
            break
        state = _clip(model, outputs[0])
        if after_step is not None:
            after_step(t, state)
    return state


def eval_rollout(
    model,
    state,
    key,
    n_periods: int,
    step_fn: StepFn,
    record: RecordFn,
    p_disaster: float = 0.0,
    after_step: Optional[Callable[[int, Any], None]] = None,
):
    """Run a stochastic single-path rollout, recording each period.

    Args:
        model: ModelSpec (used for ``n_shocks``, the discrete chain, the
            disaster draw, and ``clip_state_fn``).
        state: ``[1, n_states]`` start state.
        key: PRNG key. Split per period — ``split(key, 3)`` when
            ``p_disaster > 0`` (key, shock, disaster), ``split(key)``
            otherwise. This order is the historical one and is what makes
            reported numbers reproducible across releases.
        n_periods: number of steps.
        step_fn: ``(state, shock, d_disaster) -> outputs`` with
            ``outputs[0]`` the next state. Callers pass their own JIT'd
            step; ``d_disaster`` is ``None`` unless a disaster was drawn.
        record: ``(t, outputs) -> bool``; called after the step, before
            the clip. Return truthy to stop (the rollout then leaves
            ``state`` at the *pre*-step value, as the hand-rolled loops did).
        p_disaster: Bernoulli disaster probability; > 0 selects the
            disaster branch.
        after_step: optional ``(t, state)`` hook called after the clip.

    Returns:
        The final state.
    """

    def draw(state):
        nonlocal key
        if p_disaster > 0.0:
            key, shock_key, d_key = jax.random.split(key, 3)
            shock = draw_training_shocks(shock_key, 1, model.n_shocks)
            return shock, maybe_draw_disaster(d_key, 1, model)
        key, shock_key = jax.random.split(key)
        return _draw_eval_shock(model, shock_key, state), None

    return _rollout(model, state, n_periods, step_fn, record, draw, after_step)


def deterministic_rollout(
    model,
    state,
    n_periods: int,
    step_fn: StepFn,
    record: RecordFn,
):
    """Run a zero-shock rollout (the IRF path). No PRNG is consumed."""
    zero_shock = jnp.zeros((1, model.n_shocks))
    return _rollout(
        model,
        state,
        n_periods,
        step_fn,
        record,
        lambda _state: (zero_shock, None),
        None,
    )
