"""Steady state computation for Disaster (NK-DSGE) model."""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.models.disaster.variables import SPEC, STEADY_STATE


def steady_state(constants: Dict) -> Tuple[Array, Array]:
    ss_state = jnp.array([STEADY_STATE[n] for n in SPEC.state_names])
    ss_policy = jnp.array([STEADY_STATE[n] for n in SPEC.policy_names])
    return ss_state, ss_policy


def init_state(key: Array, batch_size: int, constants: Dict) -> Array:
    ss_state, _ = steady_state(constants)
    noise = jax.random.uniform(key, (batch_size, 13), minval=-0.05, maxval=0.05)
    return ss_state * (1 + noise)
