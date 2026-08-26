"""Variables/constants for the 56-generation life-cycle OLG.

The large-scale companion of ``olg_lifecycle`` (the DEQN-paper-style
dimensionality demonstration): identical economics — households save in
capital under a borrowing constraint ``k >= 0`` with exogenous age-dependent
efficient labor — at annual frequency with ``H = 56`` cohorts (ages 20-75,
one model period = 1 year). State: TFP level ``Z`` plus 56 cohort capital
holdings (57-dim); policy: 55 saving rates (the last cohort consumes
everything). All equations, dynamics, and the init sampler are reused from
``olg_lifecycle`` — they derive H from ``len(constants["l_cycle"])``.

Calibration = the annual-frequency counterpart of the 6-generation decade
calibration (each derivation documented inline), so the two models describe
the same economy at different time aggregation.
"""

import jax.numpy as jnp
import numpy as np

from deqn_jax.models.variable_spec import VariableSpec

H = 56

SPEC = VariableSpec(
    state_names=("Z",) + tuple(f"k{h}" for h in range(H)),
    policy_names=tuple(f"s{h}" for h in range(H - 1)),
)

# Age-efficiency profile: the 6-generation anchors (1.0, 1.8, 2.3, 2.5, 1.6,
# 1.25) interpolated to 56 annual ages — same hump (peak in the 50s, lower
# tail standing in for retirement), finer grid.
_ANCHORS = np.array([1.0, 1.8, 2.3, 2.5, 1.6, 1.25])
L_CYCLE = tuple(
    float(v) for v in np.interp(np.linspace(0.0, 5.0, H), np.arange(6.0), _ANCHORS)
)

# Annual counterparts of the decade calibration:
#   beta:    0.99 per year            (decade value was 0.99^10)
#   delta:   (1-delta_a)^10 = 1-0.8   -> delta_a = 1 - 0.2^(1/10) ~ 0.1487
#   rho_z:   0.9 per year             (decade value was 0.9^10)
#   sigma_z: matched to the DECADE process's stationary std (0.1067):
#            sigma_a = sqrt(var_dec * (1 - 0.9^2)) ~ 0.0465, so the annual
#            AR(1) has the same unconditional TFP dispersion.
CONSTANTS = {
    "alpha": 0.36,
    "beta": 0.99,
    "delta": 1.0 - 0.2 ** (1.0 / 10.0),
    "rho_z": 0.9,
    "sigma_z": 0.0465,
    "l_cycle": L_CYCLE,
}

# Saving rates in (0, 1): sigmoid output bounding gives c > 0 and k' >= 0 by
# construction; FB encodes the Euler/borrowing complementary slackness.
POLICY_LOWER = jnp.array([1e-6] * (H - 1))
POLICY_UPPER = jnp.array([1.0 - 1e-6] * (H - 1))

N_SHOCKS = 1

DESCRIPTION = (
    "56-generation life-cycle OLG with borrowing constraints "
    "(annual frequency; reuses the olg_lifecycle machinery at H=56)"
)
