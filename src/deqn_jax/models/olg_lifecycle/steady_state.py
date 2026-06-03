"""Init-state sampler for the 6-generation life-cycle OLG (Geneva Day 2 Ex 4).

There is no closed-form steady state: the cross-sectional capital distribution
and TFP are solved jointly by training over the ergodic distribution (exactly as
in the notebook, which seeds random states and simulates). ``steady_state_fn``
is therefore ``None``; the contract test skips its SS fixed-point checks in that
case, and episodes are seeded by ``init_state`` below.

Init matches the notebook: ``Z`` and each cohort's capital ~ ``exp(U(0,1))``
(strictly positive seeds; ``k^0`` is reset to 0 by the first dynamics step).
"""

import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.models.olg_lifecycle.variables import H


def init_state(key: Array, batch_size: int, constants) -> Array:
    """Sample ``[batch_size, 1 + H]`` initial states (Z, k0..k5) ~ exp(U(0,1))."""
    k_z, k_k = jax.random.split(key)
    Z = jnp.exp(jax.random.uniform(k_z, (batch_size, 1)))
    k = jnp.exp(jax.random.uniform(k_k, (batch_size, H)))
    return jnp.concatenate([Z, k], axis=1)
