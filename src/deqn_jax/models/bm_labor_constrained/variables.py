"""Variables/constants for Brock-Mirman with endogenous labor under an upper labor cap.

Geneva 2026 course, Day 2 Exercise 3. Same economics as ``bm_labor``, but labor is
capped at ``L <= L_max``, enforced as a Fischer-Burmeister complementarity on the labor
optimality condition (see ``equations.py`` and ``models/_complementarity.py``).

Two distinct labor bounds, matching Simon's notebook exactly:

  - ``L_OUT_BOUND = 1.01`` -- the network's ``L`` output ceiling
    (``lab_pol = 1.01 * sigmoid(.)`` in the reference), so ``L in (0, 1.01)``.
  - ``L_max = 1.02`` -- the Fischer-Burmeister cap (slack ``b = L_max - L``;
    reference uses ``b = 1.02 - L``).

Because the FB cap (1.02) is strictly ABOVE the output ceiling (1.01), the slack
``b >= 0.01 > 0`` always: the cap is SLACK by construction and the FB reduces to
enforcing the interior labor FOC (``psi*L^theta = w*u'(c)``). The unconstrained
labor SS is ``L_ss ~= 0.975``, well inside the ceiling. (An earlier port set both
bounds to 1.01, which made the cap bind in the high-TFP ergodic tail -- a deviation
from the reference; this matches Simon's 1.01/1.02 split so the cap is non-binding
as in his solution.)

Reuses ``bm_labor``'s SPEC and calibration; only adds the labor bounds.
"""

import jax.numpy as jnp

from deqn_jax.models.bm_labor.variables import CONSTANTS as _BM_LABOR_CONSTANTS
from deqn_jax.models.variable_spec import VariableSpec

# Identical state/policy layout to bm_labor (k, z; sav_rate, L). Defined locally
# (not re-imported) so the module is self-contained and the import isn't stripped.
SPEC = VariableSpec(state_names=("k", "z"), policy_names=("sav_rate", "L"))

# Network output ceiling for L (the sigmoid scale) vs the FB cap. Kept distinct,
# matching Simon's `1.01 * sigmoid` output and `1.02 - L` FB slack.
L_OUT_BOUND = 1.01
L_MAX = 1.02

# Same calibration as bm_labor, plus the FB labor cap. L_ss ~= 0.975 < L_OUT_BOUND,
# so the policy never reaches the output ceiling let alone the (higher) FB cap; the
# cap is slack throughout and the FB enforces the interior labor FOC.
CONSTANTS = {**_BM_LABOR_CONSTANTS, "L_max": L_MAX}

# sav_rate in (0, 1); L bounded to (0, L_OUT_BOUND]. The finite upper bound makes the
# output layer use sigmoid bounding for L (see networks/common._apply_bounds). The FB
# residual (with the higher 1.02 cap) handles complementary slackness.
POLICY_LOWER = jnp.array([1e-6, 1e-6])
POLICY_UPPER = jnp.array([1 - 1e-6, L_OUT_BOUND])

N_SHOCKS = 1

DESCRIPTION = (
    "Brock-Mirman with endogenous labor and an upper labor cap (FB-constrained)"
)
