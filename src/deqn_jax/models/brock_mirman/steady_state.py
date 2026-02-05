"""Steady state computation for Brock-Mirman model."""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.models.brock_mirman.variables import SPEC


def steady_state(constants: Dict) -> Tuple[Array, Array]:
    """Compute deterministic steady state.

    At steady state with z=0:
        1 = beta * (1 + alpha * k^(alpha-1) - delta)
    """
    alpha = constants["alpha"]
    beta = constants["beta"]
    delta = constants["delta"]

    # Steady state capital
    k_ss = ((1 / beta - 1 + delta) / alpha) ** (1 / (alpha - 1))
    z_ss = 0.0

    # Steady state output and savings rate
    y_ss = k_ss ** alpha
    s_ss = delta * k_ss
    sav_rate_ss = s_ss / y_ss

    return jnp.array([k_ss, z_ss]), jnp.array([sav_rate_ss])


def init_state(key: Array, batch_size: int, constants: Dict) -> Array:
    """Sample initial states around steady state."""
    ss_state, _ = steady_state(constants)

    k_key, z_key = jax.random.split(key)
    k_init = ss_state[0] * (1 + 0.2 * jax.random.uniform(k_key, (batch_size,), minval=-1, maxval=1))
    z_init = jax.random.normal(z_key, (batch_size,)) * constants["sigma_z"] * 2

    return jnp.stack([k_init, z_init], axis=1)
