"""Steady state and init-state sampler for Brock-Mirman + Epstein-Zin.

For the Euler FOC, the deterministic SS condition

    1 = β · (1 - δ + α · k^{α-1})

is identical to the standard Brock-Mirman model — Euler at SS does not
involve ψ or γ_ez (the EZ adjustment factor M̃ collapses to 1 under
deterministic dynamics). So (k_ss, sav_rate_ss) match brock_mirman.

V_ss is *not* part of the policy vector (it lives in the critic), so
the steady_state_fn returns the same shape as the standard brock_mirman
model. The critic learns V on its own.

Init-state sampling mirrors brock_mirman:

    k ~ Uniform[K_LB, K_UB]
    z ~ Uniform[ln(Z_LB), ln(Z_UB)]
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

    y_ss = k_ss**alpha
    s_ss = delta * k_ss
    sav_rate_ss = s_ss / y_ss

    return jnp.array([k_ss, z_ss]), jnp.array([sav_rate_ss])


INIT_SPECS = {
    "k": {"distribution": "uniform", "kwargs": {"minval": K_LB, "maxval": K_UB}},
    "z": {"distribution": "uniform", "kwargs": {"minval": Z_LB, "maxval": Z_UB}},
}

init_state = make_init_state_fn(("k", "z"), INIT_SPECS)
