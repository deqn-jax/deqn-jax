"""Variables for the autodiff variant of stochastic Brock-Mirman.

Same state, policy, bounds, and calibration as the canonical
brock_mirman. Lives in its own subpackage so the Euler equation can
be synthesized from the period-return function rather than
hand-derived.
"""

# Re-export the canonical model's variable spec. We don't want drift
# between the two versions' calibrations -- any accuracy difference
# should come from residual synthesis, nothing else. noqa: these are
# intentional re-exports consumed by sibling modules in this package.
from deqn_jax.models.brock_mirman.variables import (  # noqa: F401
    CONSTANTS,
    N_SHOCKS,
    POLICY_LOWER,
    POLICY_UPPER,
    SPEC,
)

DESCRIPTION = "Stochastic Brock-Mirman with Euler synthesized from period-return via jax.grad (autodiff POC)"
