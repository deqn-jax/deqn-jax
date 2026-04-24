"""State transitions for Brock-Mirman model."""

from typing import Dict

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.brock_mirman.equations import definitions
from deqn_jax.models.brock_mirman.variables import SPEC


def step(
    state: Array,
    policy: Array,
    shock: Array,
    constants: Dict,
) -> Array:
    """Transition to next state.

    Capital: k' = (1 - delta) * k + s
    TFP:     z' = rho * z + sigma * eps
    """
    s = SPEC.unpack_state(state)
    defs = definitions(state, policy, constants)

    delta = constants["delta"]
    rho_z = constants["rho_z"]
    sigma_z = constants["sigma_z"]

    # Capital accumulation
    k_next = (1 - delta) * s.k + defs["s"]

    # TFP shock
    eps = shock[:, 0] if shock.ndim > 1 else shock
    z_next = rho_z * s.z + sigma_z * eps

    return jnp.stack([k_next, z_next], axis=1)
