"""Steady state and init-state sampler for the stochastic Brock-Mirman model.

Deterministic steady state (z=0):

    1 = beta * (1 - delta + alpha * k^(alpha - 1))
    => k_ss = ((1/beta - 1 + delta) / alpha) ** (1 / (alpha - 1))

Init sampling mirrors the reference notebook's exogenous rect:

    k ~ Uniform[K_LB, K_UB]
    z ~ Uniform[ln(Z_LB), ln(Z_UB)]

where z is log TFP (the reference stores level TFP Z and samples on a
rect in Z; our dynamics carry the log directly so we sample on the log
bounds). Combined with ``initialize_each_episode=True`` in the trainer
config, this produces fresh uniform coverage of the training domain
every gradient cycle. Drop ``initialize_each_episode`` to switch to
trajectory-ergodic training.

The sampler is built declaratively via ``make_init_state_fn``; if the
model needs more complex initial-state logic than one distribution per
variable, a hand-written ``init_state_fn`` is still supported.
"""

import math
from typing import Dict, Tuple

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.variable_spec import make_init_state_fn

K_LB = 0.9
K_UB = 12.0
Z_LEVEL_LB = 0.7
Z_LEVEL_UB = 1.3
Z_LB = math.log(Z_LEVEL_LB)
Z_UB = math.log(Z_LEVEL_UB)


def steady_state(constants: Dict) -> Tuple[Array, Array]:
    alpha = constants["alpha"]
    beta = constants["beta"]
    delta = constants["delta"]

    k_ss = ((1.0 / beta - 1.0 + delta) / alpha) ** (1.0 / (alpha - 1.0))
    z_ss = 0.0

    y_ss = k_ss ** alpha
    s_ss = delta * k_ss
    sav_rate_ss = s_ss / y_ss

    return jnp.array([k_ss, z_ss]), jnp.array([sav_rate_ss])


INIT_SPECS = {
    "k": {"distribution": "uniform", "kwargs": {"minval": K_LB, "maxval": K_UB}},
    "z": {"distribution": "uniform", "kwargs": {"minval": Z_LB, "maxval": Z_UB}},
}

init_state = make_init_state_fn(("k", "z"), INIT_SPECS)
