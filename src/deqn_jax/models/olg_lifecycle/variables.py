"""Variables/constants for the 6-generation life-cycle OLG with borrowing constraints.

Geneva 2026 course, Day 2 Exercise 4. Households live ``H = 6`` deterministic
periods (ages 20-80, one model period ≈ 10 years), save in capital subject to a
borrowing constraint ``k >= 0``, and supply exogenous age-dependent efficient
labor. The firm side is the standard Cobb-Douglas block from the earlier
Brock-Mirman exercises.

State: TFP level ``Z`` plus the capital held by each of the ``H`` age groups.
Policy: the saving rate out of cash-at-hand for cohorts ``0..H-2``; the last
cohort consumes everything (``s^{H-1} ≡ 0``), so it is NOT a network output.
"""

import jax.numpy as jnp

from deqn_jax.models.variable_spec import VariableSpec

# Number of age groups (one period ≈ 10 years -> ages 20..80).
H = 6

# Named layout. State = (Z, k0..k5); policy = saving rates (s0..s4) for the
# H-1 = 5 cohorts that still choose how much to save.
SPEC = VariableSpec(
    state_names=("Z", "k0", "k1", "k2", "k3", "k4", "k5"),
    policy_names=("s0", "s1", "s2", "s3", "s4"),
)

# Age-dependent efficient labor supply (exogenous). The last two cohorts supply
# less, standing in for retirement without modeling a government. Matches the
# notebook's ``l_cycle``.
L_CYCLE = (1.0, 1.8, 2.3, 2.5, 1.6, 1.25)

# One model period ≈ 10 years, so beta/delta/rho are the per-decade values from
# the notebook ("we adjust the parameters a bit because one period is 10 years").
CONSTANTS = {
    "alpha": 0.36,
    "beta": 0.99**10,
    "delta": 0.8,
    "rho_z": 0.9**10,
    "sigma_z": 0.10,
    "l_cycle": L_CYCLE,
}

# Saving rates live in (0, 1). A finite upper bound makes the output layer use
# sigmoid bounding (see networks/common._apply_bounds), so 0 < s < 1 by
# construction => consumption c = cah*(1-s) > 0 and savings cah*s >= 0 (the
# borrowing constraint k' >= 0 holds automatically). FB encodes the complementary
# slackness between each Euler and its borrowing constraint.
POLICY_LOWER = jnp.array([1e-6] * (H - 1))
POLICY_UPPER = jnp.array([1.0 - 1e-6] * (H - 1))

N_SHOCKS = 1

DESCRIPTION = (
    "6-generation life-cycle OLG with borrowing constraints "
    "(Fischer-Burmeister Euler complementarity, two-stage loss)"
)
