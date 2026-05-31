"""Shock-drawing helpers for rollout, evaluation, and IRF paths.

Training-time and diagnostic simulation all share a single contract
for sampling shocks that actually drive the state forward. Previously
this logic was duplicated inconsistently across:

- ``episode.simulate_step`` -- training rollouts
- ``evaluate.euler_equation_errors`` -- ergodic residual diagnostics
- ``irf.run_irf`` -- impulse response paths

The duplication was exposed during a code review: the disaster model's
``d_disaster`` Bernoulli indicator was threaded only through the loss
expectation, never the rollout -- so training data never visited
disaster states even when ``p_disaster > 0``. Similarly ``shock_mask``
(masking individual shock dimensions to zero for curriculum / ablation)
applied only to the loss draws, and ``shock_scale`` (curriculum ramp)
did the same.

This module provides one place where all three concerns are handled:

- ``draw_training_shocks``: Gaussian draws with curriculum scale + mask.
- ``maybe_draw_disaster``: Bernoulli(p_disaster) draw when the model's
  step_fn takes a ``d_disaster`` kwarg and ``constants['p_disaster']``
  is positive; else ``None``.
- ``step_accepts_disaster``: signature introspection helper. Called
  at setup time (outside JIT).
- ``simulation_step``: one-step roll-forward that composes the above
  and dispatches to ``step_fn`` with or without the disaster kwarg.
"""

import inspect
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array


def step_accepts_disaster(step_fn: Callable) -> bool:
    """Does ``step_fn`` accept a ``d_disaster`` kwarg?

    Inspected from the function signature. Safe to call at setup time
    (outside JIT); returns a plain Python bool that becomes a
    compile-time constant when used inside a JIT-compiled function.
    """
    try:
        return "d_disaster" in inspect.signature(step_fn).parameters
    except (ValueError, TypeError):
        return False


def draw_training_shocks(
    key: Array,
    batch_size: int,
    n_shocks: int,
    shock_scale: Any = 1.0,
    shock_mask: Optional[Array] = None,
) -> Array:
    """Draw ``[batch, n_shocks]`` Gaussian shocks with curriculum scale + mask.

    ``shock_scale`` can be a scalar (ramps all shock dimensions together)
    or a length-``n_shocks`` vector (per-dimension ramp). ``shock_mask``,
    when provided, is a length-``n_shocks`` vector of 0/1 entries that
    zeros specific shock dimensions (used for shock ablations).

    Both are multiplicative, so ``shock_scale=0`` freezes all rollouts
    to deterministic dynamics.
    """
    shock = jax.random.normal(key, (batch_size, n_shocks))
    shock = shock * jnp.asarray(shock_scale)
    if shock_mask is not None:
        shock = shock * jnp.asarray(shock_mask)
    return shock


def draw_discrete_shocks(
    key: Array,
    current_z: Array,
    transition_matrix: Array,
) -> Array:
    """Sample next-z categorical indices from a finite Markov chain.

    For each batch element b, draws ``z_{t+1}[b] ~ Multinomial(Π[z_t[b]])``
    using ``jax.random.categorical`` on log-probabilities of the row of Π
    indexed by the current z-state.

    Args:
        key: PRNG key.
        current_z: Current z-state indices, shape ``[batch]`` int32.
        transition_matrix: Π of shape ``[K, K]``, rows must sum to 1.

    Returns:
        ``[batch]`` int32 of next-z indices in ``[0, K)``.

    The shock returned is the *next-period* z (what step_fn embeds into
    state, conventionally at ``state[z_state_idx]``). Curriculum scale
    and shock_mask are not applied — discrete chains have no continuous
    magnitude to ramp; curriculum on a discrete chain would mean
    something model-specific (e.g. concentrating mass on the central
    state) and is left to the caller.
    """
    log_probs = jnp.log(transition_matrix + 1e-30)  # [K, K]
    rows = log_probs[current_z]  # [batch, K]
    return jax.random.categorical(key, rows, axis=-1).astype(jnp.int32)


def maybe_draw_disaster(
    key: Array,
    batch_size: int,
    model,
) -> Optional[Array]:
    """Draw per-sample Bernoulli(p_disaster) if the model supports disasters.

    Returns a ``[batch]`` array of 0/1 floats, or ``None`` if:
    - the step_fn doesn't have a ``d_disaster`` kwarg, or
    - ``constants['p_disaster']`` is missing or zero.

    ``None`` is the caller's signal to call step_fn without the kwarg.

    Shape note: the residual-side ``compute_residuals`` calls step_fn
    with a *scalar* d_disaster (one value per branch of the disaster
    mixture). The rollout side wants a per-sample indicator. Both must
    broadcast cleanly against ``defs["k"]`` which is ``[batch]``; a
    trailing singleton dim here would broadcast to ``[batch, batch]``
    inside step_fn (silent until ``jnp.stack`` later mismatches), so
    we return a flat ``[batch]`` vector.
    """
    if not step_accepts_disaster(model.step_fn):
        return None
    p = model.constants.get("p_disaster", 0.0)
    if p <= 0.0:
        return None
    u = jax.random.uniform(key, (batch_size,))
    return (u < p).astype(jnp.float32)


def simulation_step(
    model,
    policy_fn: Callable[[Array], Array],
    state: Array,
    key: Array,
    shock_scale: Any = 1.0,
    shock_mask: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """One rollout step under training-time shock conventions.

    Returns ``(next_state, shock)``. Dispatches on the model:

    - If ``model.transition_matrix is not None``: sample next-z categorical
      from ``Π[z_t]`` and pass that integer index as the shock. The model's
      ``step_fn`` is responsible for embedding the index into next-state
      (e.g. ``state[z_state_idx] = shock`` and looking up any continuous
      realisations from a model-side table).
    - Else: sample ``[batch, n_shocks]`` Gaussian shocks (legacy path).

    Disaster indicator is drawn and passed to step_fn only if the model
    supports it; otherwise step_fn is called positionally. The signature
    inspection is done at trace time (a compile-time constant), so there
    is no per-step cost.
    """
    batch_size = state.shape[0]
    shock_key, disaster_key = jax.random.split(key)

    if getattr(model, "transition_matrix", None) is not None:
        z_idx = int(model.z_state_idx)
        current_z = state[:, z_idx].astype(jnp.int32)
        shock = draw_discrete_shocks(
            shock_key, current_z, jnp.asarray(model.transition_matrix)
        )
    else:
        shock = draw_training_shocks(
            shock_key,
            batch_size,
            model.n_shocks,
            shock_scale,
            shock_mask,
        )
    policy = policy_fn(state)

    d_disaster = maybe_draw_disaster(disaster_key, batch_size, model)
    if d_disaster is None:
        next_state = model.step_fn(state, policy, shock, model.constants)
    else:
        next_state = model.step_fn(
            state,
            policy,
            shock,
            model.constants,
            d_disaster=d_disaster,
        )
    return next_state, shock
