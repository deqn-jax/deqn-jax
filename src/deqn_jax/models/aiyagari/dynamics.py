"""State transitions for Aiyagari model."""

from typing import Dict

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.aiyagari.equations import definitions
from deqn_jax.models.aiyagari.variables import SPEC


def step(
    state: Array,
    policy: Array,
    shock: Array,
    constants: Dict,
) -> Array:
    """Transition to next state.

    Capital: k' = (1 - c_share) * coh, clipped to [k_min, inf)
    Productivity: a' = rho * a + sigma * sqrt(1 - rho^2) * eps
    """
    s = SPEC.unpack_state(state)
    defs = definitions(state, policy, constants)

    rho_a = constants["rho_a"]
    sigma_a = constants["sigma_a"]

    # Capital from savings (clipped to prevent overflow in trajectories)
    k_next = jnp.clip(defs["k_next"], constants["k_min"], constants["k_max"])

    # Idiosyncratic productivity shock (AR(1), unconditional std = sigma_a)
    eps = shock[:, 0] if shock.ndim > 1 else shock
    a_next = rho_a * s.a + jnp.sqrt(1 - rho_a**2) * sigma_a * eps

    return jnp.stack([k_next, a_next], axis=1)
