"""Steady state and init-state sampler for the stochastic Brock-Mirman model.

Deterministic steady state (z=0):

    1 = beta * (1 - delta + alpha * k^(alpha - 1))
    => k_ss = ((1/beta - 1 + delta) / alpha) ** (1 / (alpha - 1))

Calibration note (resolves the historical "three-way SS mismatch").
For the canonical constants here (alpha=0.36, beta=0.99, delta=0.1):

    k_ss   = 6.36684          # the value this function returns
    sav_ss = delta*k/y = 0.32697

The formula is extremely sensitive to (beta, delta), which is the ENTIRE
source of the old "sim 4.0 / closed-form 0.18 / partial-delta 14"
discrepancy recorded in project memory. It was a calibration mixup, not a
solver bug:

    * delta=1 (full depreciation) collapses the formula to the log-utility
      closed form k_ss = (alpha*beta)**(1/(1-alpha)) = 0.1995 here -- a
      DIFFERENT model. That is the "~0.18" figure.
    * Evaluating the partial-delta formula with an off-canonical beta swings
      k_ss wildly: beta=0.96 -> 4.29, delta=0.025 -> 37.99. Legacy figures
      computed against off-calibration constants, or read off an unconverged
      simulation, are simply not comparable to the canonical 6.367.

See docs/dev/framework_audit_2026-05.md (bm-ss / models-06) and the value
regression test in tests/test_brock_mirman_ss.py.

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

    y_ss = k_ss**alpha
    s_ss = delta * k_ss
    sav_rate_ss = s_ss / y_ss

    return jnp.array([k_ss, z_ss]), jnp.array([sav_rate_ss])


INIT_SPECS = {
    "k": {"distribution": "uniform", "kwargs": {"minval": K_LB, "maxval": K_UB}},
    "z": {"distribution": "uniform", "kwargs": {"minval": Z_LB, "maxval": Z_UB}},
}

init_state = make_init_state_fn(("k", "z"), INIT_SPECS)
