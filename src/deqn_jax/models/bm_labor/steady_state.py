"""Steady state and init-state sampler for Brock-Mirman with endogenous labor.

Analytical derivation at z=0 (Z=1). From:

    Euler:    1 = beta * (1 - delta + alpha * K^{alpha-1} * L^{1-alpha})
    Resource: C = Y - delta K = K^alpha L^{1-alpha} - delta K
    Labor:    psi * L^theta = w / C = (1 - alpha) (K/L)^alpha / C

Let kappa = K/L. Euler gives

    kappa = ((1/beta - 1 + delta) / alpha)^(1/(alpha - 1))

(independent of L -- standard Cobb-Douglas result). Resource constraint
gives C = L (kappa^alpha - delta kappa). Substituting into the labor FOC
and collecting,

    L^(theta + 1) = (1 - alpha) / [psi * (1 - delta * kappa^{1-alpha})]

which pins down L_ss. Then K_ss = kappa * L_ss. This is the general case;
for log utility (gamma=1) it matches; for other gamma the derivation
generalizes by carrying the u'(C) factor through.

Init sampling: uniform rect covering the brock_mirman-matching training
domain (easier side-by-side comparison).
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
    psi = constants["psi"]
    theta = constants["theta"]

    # kappa = K/L from the capital Euler
    kappa = ((1.0 / beta - 1.0 + delta) / alpha) ** (1.0 / (alpha - 1.0))

    # Labor FOC gives L_ss
    L_ss = ((1.0 - alpha) / (psi * (1.0 - delta * kappa ** (1.0 - alpha)))) ** (
        1.0 / (theta + 1.0)
    )
    K_ss = kappa * L_ss

    # Output and savings rate at SS
    y_ss = (K_ss ** alpha) * (L_ss ** (1.0 - alpha))
    sav_rate_ss = delta * K_ss / y_ss
    z_ss = 0.0

    return jnp.array([K_ss, z_ss]), jnp.array([sav_rate_ss, L_ss])


INIT_SPECS = {
    "k": {"distribution": "uniform", "kwargs": {"minval": K_LB, "maxval": K_UB}},
    "z": {"distribution": "uniform", "kwargs": {"minval": Z_LB, "maxval": Z_UB}},
}

init_state = make_init_state_fn(("k", "z"), INIT_SPECS)
