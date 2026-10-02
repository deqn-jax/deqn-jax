"""Variables for the Lagrangian variant of the 6-agent OLG.

Re-exports the calibration, bounds and variable spec of ``olg_analytic_6``
so that the two variants differ only in how the equations are obtained.
"""

# Intentional re-exports for sibling modules in this package.
from deqn_jax.models.olg_analytic_6.variables import (  # noqa: F401
    CONSTANTS,
    N_SHOCKS,
    POLICY_LOWER,
    POLICY_UPPER,
    SPEC,
    A,
)

DESCRIPTION = "6-agent OLG with every Euler derived from the Lagrangian via jax.grad"
