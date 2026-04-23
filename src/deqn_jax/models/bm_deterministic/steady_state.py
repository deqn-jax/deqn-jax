"""Steady state and initial-state sampler for deterministic Brock-Mirman.

The (unique, interior) deterministic steady state solves
    1 = beta * (1 - delta + alpha * K^{alpha - 1})
    => K_ss = ((1/beta - 1 + delta) / alpha) ** (1 / (alpha - 1))

With delta = 1 this simplifies to K_ss = (alpha * beta) ** (1/(1-alpha)).

``init_state_fn`` samples K uniformly on [K_LB, K_UB] (declared in
variables.py). Combined with ``initialize_each_episode=True`` in the
trainer config, this gives fresh uniform coverage of the domain every
gradient cycle, matching the reference TF/Keras training recipe.
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.models.bm_deterministic.variables import K_LB, K_UB


def steady_state(constants: Dict) -> Tuple[Array, Array]:
    alpha = constants["alpha"]
    beta = constants["beta"]
    delta = constants["delta"]

    k_ss = ((1.0 / beta - 1.0 + delta) / alpha) ** (1.0 / (alpha - 1.0))

    y_ss = k_ss ** alpha
    sav_rate_ss = delta * k_ss / y_ss  # equal to alpha*beta when delta=1

    return jnp.array([k_ss]), jnp.array([sav_rate_ss])


def init_state(key: Array, batch_size: int, constants: Dict) -> Array:
    k_init = jax.random.uniform(key, (batch_size,), minval=K_LB, maxval=K_UB)
    return jnp.stack([k_init], axis=1)
