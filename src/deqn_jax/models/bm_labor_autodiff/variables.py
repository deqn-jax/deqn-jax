"""Variables for the autodiff variant of Brock-Mirman-with-labor.

Re-exports the canonical `bm_labor` variable spec: calibration, policy
bounds, SPEC, N_SHOCKS. The only thing that differs between the two
variants is ``equations.py`` -- here the two FOCs (capital Euler +
labor FOC) are synthesized from a scalar Pi via ``jax.grad``, rather
than hand-derived.
"""

# Intentional re-exports for sibling modules in this package.
from deqn_jax.models.bm_labor.variables import (  # noqa: F401
    CONSTANTS,
    N_SHOCKS,
    POLICY_LOWER,
    POLICY_UPPER,
    SPEC,
)

DESCRIPTION = (
    "Brock-Mirman with endogenous labor, both FOCs synthesized from Pi via "
    "jax.grad (multi-policy autodiff POC)"
)
