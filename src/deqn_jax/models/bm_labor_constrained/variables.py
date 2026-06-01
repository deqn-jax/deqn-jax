"""Variables/constants for Brock-Mirman with endogenous labor under an upper labor cap.

Geneva 2026 course, Day 2 Exercise 3. Same economics as ``bm_labor``, but labor is
capped at ``L <= L_max``. The cap is enforced as a Fischer-Burmeister complementarity
on the labor optimality condition (see ``equations.py`` and
``models/_complementarity.py``); the network's ``L`` output is additionally bounded to
``(0, L_max]`` so the cap holds by construction, and FB makes the binding case a valid
zero-residual solution.

Reuses ``bm_labor``'s SPEC and calibration; only adds ``L_max`` and caps the ``L``
output bound.
"""

import jax.numpy as jnp

from deqn_jax.models.bm_labor.variables import CONSTANTS as _BM_LABOR_CONSTANTS
from deqn_jax.models.variable_spec import VariableSpec

# Identical state/policy layout to bm_labor (k, z; sav_rate, L). Defined locally
# (not re-imported) so the module is self-contained and the import isn't stripped.
SPEC = VariableSpec(state_names=("k", "z"), policy_names=("sav_rate", "L"))

L_MAX = 1.01

# Same calibration as bm_labor, plus the labor cap. With the default constants the
# unconstrained labor SS is L_ss ~= 0.975 < L_max, so the cap is SLACK at the
# deterministic steady state (it binds only in high-TFP / high-effort regions).
CONSTANTS = {**_BM_LABOR_CONSTANTS, "L_max": L_MAX}

# sav_rate in (0, 1); L capped to (0, L_max]. A finite upper bound makes the output
# layer use sigmoid bounding for L (see networks/common._apply_bounds), so the cap is
# never violated by construction; the FB residual handles complementary slackness.
POLICY_LOWER = jnp.array([1e-6, 1e-6])
POLICY_UPPER = jnp.array([1 - 1e-6, L_MAX])

N_SHOCKS = 1

DESCRIPTION = (
    "Brock-Mirman with endogenous labor and an upper labor cap (FB-constrained)"
)
