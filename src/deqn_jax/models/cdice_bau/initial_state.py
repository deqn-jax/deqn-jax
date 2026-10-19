"""Initial state of the CDICE economy (2015) and the training-set sampler.

The model is non-stationary, so it has no steady state to anchor on. The
reference trains on simulated paths that all start at the 2015 state (its run
config redraws the starting batch every episode with zero spread); the
sampler below reproduces that: every row is the 2015 state at tau = 0.
"""

from typing import Dict

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.cdice_bau.variables import SPEC

_INITIAL_KEYS = ("k0", "MAT0", "MUO0", "MLO0", "TAT0", "TOC0")


def initial_state(constants: Dict) -> Array:
    """The 2015 state vector [n_states] (tau = 0)."""
    return jnp.array([constants[k] for k in _INITIAL_KEYS] + [0.0])


def init_state(key: Array, batch_size: int, constants: Dict) -> Array:
    """``init_state_fn``: ``batch_size`` copies of the 2015 state."""
    del key  # deterministic start
    s0 = initial_state(constants)
    return jnp.broadcast_to(s0, (batch_size, SPEC.n_states))
