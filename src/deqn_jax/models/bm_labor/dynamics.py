"""State transition for Brock-Mirman with endogenous labor."""

from typing import Dict

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.bm_labor.equations import definitions
from deqn_jax.models.bm_labor.variables import SPEC


def step(state: Array, policy: Array, shock: Array, constants: Dict) -> Array:
    """Next state.

        k' = (1 - delta) k + s * Y
        z' = rho_z * z + sigma_z * eps
    """
    s = SPEC.unpack_state(state)
    defs = definitions(state, policy, constants)

    delta = constants["delta"]
    rho_z = constants["rho_z"]
    sigma_z = constants["sigma_z"]

    k_next = (1.0 - delta) * s.k + defs["s"]
    eps = shock[:, 0] if shock.ndim > 1 else shock
    z_next = rho_z * s.z + sigma_z * eps

    return jnp.stack([k_next, z_next], axis=1)
