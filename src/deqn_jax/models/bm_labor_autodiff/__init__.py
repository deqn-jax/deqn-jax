"""Autodiff POC for Brock-Mirman with endogenous labor.

Same economics as ``bm_labor`` but both FOCs (capital Euler +
intratemporal labor FOC) are synthesized from a single scalar Pi via
``jax.grad``, using the framework helper ``euler_from_period_return``
with ``intratemporal_policy_idx=(1,)``.

This is the minimum test case for Simon's "write a Lagrangian, not
hand-derived FOCs" vision extended to multi-policy models.
"""

from deqn_jax.models.bm_labor_autodiff.dynamics import step
from deqn_jax.models.bm_labor_autodiff.equations import (
    EQUATION_NAMES,
    definitions,
    equations,
    period_return,
)
from deqn_jax.models.bm_labor_autodiff.steady_state import init_state, steady_state
from deqn_jax.models.bm_labor_autodiff.variables import (
    CONSTANTS,
    N_SHOCKS,
    POLICY_LOWER,
    POLICY_UPPER,
    SPEC,
)
from deqn_jax.types import ModelSpec

MODEL = ModelSpec(
    name="bm_labor_autodiff",
    n_states=SPEC.n_states,
    n_policies=SPEC.n_policies,
    n_shocks=N_SHOCKS,
    state_names=SPEC.state_names,
    policy_names=SPEC.policy_names,
    equation_names=EQUATION_NAMES,
    shock_names=("eps_z",),
    constants=CONSTANTS,
    equations_fn=equations,
    step_fn=step,
    steady_state_fn=steady_state,
    init_state_fn=init_state,
    definitions_fn=definitions,
    policy_lower=POLICY_LOWER,
    policy_upper=POLICY_UPPER,
)

__all__ = ["MODEL", "period_return"]
