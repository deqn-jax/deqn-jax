"""State transition for the deterministic Brock-Mirman model.

    K_{t+1} = (1 - delta) * K_t + sav_rate_t * Y_t

With the default delta = 1 this reduces to K_{t+1} = sav_rate_t * Y_t.

The framework passes a ``shock`` argument for interface compatibility
with stochastic models; since N_SHOCKS=0 the shock has shape (batch, 0)
and is ignored.
"""

from typing import Dict

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.bm_deterministic.equations import definitions
from deqn_jax.models.bm_deterministic.variables import SPEC


def step(
    state: Array,
    policy: Array,
    shock: Array,
    constants: Dict,
) -> Array:
    s = SPEC.unpack_state(state)
    defs = definitions(state, policy, constants)

    delta = constants["delta"]
    k_next = (1.0 - delta) * s.k + defs["s"]

    # Preserve leading batch dimension when present.
    if state.ndim == 1:
        return jnp.stack([k_next], axis=-1)
    return jnp.stack([k_next], axis=1)
