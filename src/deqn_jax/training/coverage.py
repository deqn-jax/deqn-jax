"""EWM coverage sampling: rollout helpers + a compute_loss wrapper.

Coverage imposes the existing residual loss on a mixture measure
(base + stress + local) instead of the base batch alone. State
generation is never differentiated (every pool is stop_gradient'd);
gradient flows only through the policy re-evaluated at the fixed pool
states inside compute_loss. Generated states are repaired (clipped)
into a feasible box before the residual is evaluated -- the repair
bounds the simulation only, never penalizes the residual.

Reference: Scheidegger & Schaab (2026), "Equilibrium World Models",
arXiv:2606.23463 -- the surrogate-free coverage arm.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.training.episode import run_episode
from deqn_jax.types import ModelSpec


def sample_stress_seeds(
    key: Array,
    n: int,
    n_states: int,
    ss_state: Array,
    stress_idx: Array,
    lows: Array,
    highs: Array,
) -> Array:
    """Draw n stress seeds: SS-filled, with stress dims uniform in their box."""
    seeds = jnp.broadcast_to(ss_state, (n, n_states))
    u = jax.random.uniform(key, (n, stress_idx.shape[0]), minval=lows, maxval=highs)
    return seeds.at[:, stress_idx].set(u)


def roll_states(
    model: ModelSpec,
    policy_fn: Callable[[Array], Array],
    seeds: Array,
    key: Array,
    horizon: int,
    shock_scale=1.0,
    lo: Optional[Array] = None,
    hi: Optional[Array] = None,
) -> Array:
    """Roll seeds horizon steps through the EXACT transition; return the
    projected landings s_1..s_H (raw seed excluded), repaired (clipped to
    [lo, hi] when given), stop-gradient'd.

    Reuses run_episode (MC shocks even under quadrature: state generation
    is a different object from the loss expectation). trajectory holds
    pre-transition states, so trajectory[0] is the raw seed and final_state
    is s_H; landings = trajectory[1:] ++ final_state.
    """
    trajectory, final_state = run_episode(
        model, policy_fn, seeds, key, episode_length=horizon, shock_scale=shock_scale
    )
    landings = jnp.concatenate([trajectory[1:], final_state[None]], axis=0)
    landings = landings.reshape(-1, model.n_states)
    if lo is not None:
        landings = jnp.clip(landings, lo, hi)
    return jax.lax.stop_gradient(landings)


def make_local_pool(
    states: Array,
    key: Array,
    n: int,
    sigma: float,
    lo: Optional[Array] = None,
    hi: Optional[Array] = None,
) -> Array:
    """n locally-perturbed copies of base states (sampled with replacement),
    plus Gaussian noise; repaired (clipped) when a box is given; stop-gradient'd."""
    k_idx, k_noise = jax.random.split(key)
    idx = jax.random.randint(k_idx, (n,), 0, states.shape[0])
    base = states[idx]
    noise = jax.random.normal(k_noise, base.shape) * sigma
    out = base + noise
    if lo is not None:
        out = jnp.clip(out, lo, hi)
    return jax.lax.stop_gradient(out)
