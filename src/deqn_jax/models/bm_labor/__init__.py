"""Brock-Mirman with endogenous labor supply.

Two states (k, z), two policies (sav_rate, L), two equations (Euler +
intratemporal labor FOC), one TFP shock. Extends the stochastic
brock_mirman by adding a labor-leisure choice with log(C) utility and
convex-in-effort disutility psi * L^(1+theta) / (1+theta).
"""

from deqn_jax.models.bm_labor.dynamics import step
from deqn_jax.models.bm_labor.equations import EQUATION_NAMES, definitions, equations
from deqn_jax.models.bm_labor.steady_state import init_state, steady_state
from deqn_jax.models.bm_labor.variables import (
    CONSTANTS,
    N_SHOCKS,
    POLICY_LOWER,
    POLICY_UPPER,
    SPEC,
)
from deqn_jax.types import ModelSpec

MODEL = ModelSpec(
    name="bm_labor",
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
