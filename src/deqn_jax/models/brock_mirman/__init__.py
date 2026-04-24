"""Brock-Mirman (1972) optimal growth model.

A simple RBC model with:
- State: (k, z) = (capital, log TFP)
- Policy: sav_rate (savings rate)
- One Euler equation

This is the canonical test case for DEQN methods.
"""

from deqn_jax.models.brock_mirman.dynamics import step
from deqn_jax.models.brock_mirman.equations import EQUATION_NAMES, definitions, equations
from deqn_jax.models.brock_mirman.steady_state import init_state, steady_state
from deqn_jax.models.brock_mirman.variables import (
    CONSTANTS,
    N_SHOCKS,
    POLICY_LOWER,
    POLICY_UPPER,
    SPEC,
)
from deqn_jax.types import ModelSpec

MODEL = ModelSpec(
    name="brock_mirman",
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
