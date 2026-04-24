"""2-country International Real Business Cycle model.

Heterogeneous risk aversion across countries (gamma_0=0.25, gamma_1=1.0),
symmetric Pareto weights, quadratic capital-adjustment cost, and an
irreversibility constraint i_j >= 0 enforced via Fischer-Burmeister
complementarity residuals.

5 equations (2 Eulers + 1 ARC + 2 FB), 5 policy outputs (2 k_next +
1 lambda + 2 mu), 4 states (2 k + 2 z), 3 shocks (2 country-specific +
1 aggregate). First model in the repo to use the Fischer-Burmeister
pattern; same function will later drive bm_labor_constrained.
"""

from deqn_jax.models.irbc.dynamics import step
from deqn_jax.models.irbc.equations import EQUATION_NAMES, definitions, equations
from deqn_jax.models.irbc.steady_state import init_state, steady_state
from deqn_jax.models.irbc.variables import (
    CONSTANTS,
    N_SHOCKS,
    POLICY_LOWER,
    POLICY_UPPER,
    SPEC,
)
from deqn_jax.types import ModelSpec

MODEL = ModelSpec(
    name="irbc",
    n_states=SPEC.n_states,
    n_policies=SPEC.n_policies,
    n_shocks=N_SHOCKS,
    state_names=SPEC.state_names,
    policy_names=SPEC.policy_names,
    equation_names=EQUATION_NAMES,
    shock_names=("eps_0", "eps_1", "eps_agg"),
    constants=CONSTANTS,
    equations_fn=equations,
    step_fn=step,
    steady_state_fn=steady_state,
    init_state_fn=init_state,
    definitions_fn=definitions,
    policy_lower=POLICY_LOWER,
    policy_upper=POLICY_UPPER,
)
