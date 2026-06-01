"""Brock-Mirman with endogenous labor under an upper labor cap.

Geneva 2026 course, Day 2 Exercise 3. Same economics as ``bm_labor`` with labor
capped at ``L_max``, enforced via a Fischer-Burmeister complementarity on the labor
optimality condition (see ``equations.py``).

Extend, don't fork: reuses ``bm_labor``'s dynamics, steady state, init sampler, and
``definitions`` verbatim. Only the labor equation (now FB) and the ``L`` upper bound
differ. The cap is slack at the deterministic SS (L_ss ~= 0.975 < L_max = 1.01), so
``bm_labor``'s analytical steady state is also the constrained model's SS.
"""

from deqn_jax.models.bm_labor.dynamics import step
from deqn_jax.models.bm_labor.equations import definitions
from deqn_jax.models.bm_labor.steady_state import init_state, steady_state
from deqn_jax.models.bm_labor_constrained.equations import EQUATION_NAMES, equations
from deqn_jax.models.bm_labor_constrained.variables import (
    CONSTANTS,
    N_SHOCKS,
    POLICY_LOWER,
    POLICY_UPPER,
    SPEC,
)
from deqn_jax.types import ModelSpec

MODEL = ModelSpec(
    name="bm_labor_constrained",
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
