"""Brock-Mirman (1972) optimal growth model.

A simple RBC model with:
- State: (k, z) = (capital, log TFP)
- Policy: sav_rate (savings rate)
- One Euler equation

This is the canonical test case for DEQN methods.
"""

from deqn_jax.types import ModelSpec
from deqn_jax.models.brock_mirman.variables import (
    SPEC, CONSTANTS, N_SHOCKS, POLICY_LOWER, POLICY_UPPER,
)
from deqn_jax.models.brock_mirman.equations import equations, definitions, EQUATION_NAMES
from deqn_jax.models.brock_mirman.dynamics import step
from deqn_jax.models.brock_mirman.steady_state import steady_state, init_state

MODEL = ModelSpec(
    name="brock_mirman",
    n_states=SPEC.n_states,
    n_policies=SPEC.n_policies,
    n_shocks=N_SHOCKS,
    state_names=SPEC.state_names,
    policy_names=SPEC.policy_names,
    equation_names=EQUATION_NAMES,
    constants=CONSTANTS,
    equations_fn=equations,
    step_fn=step,
    steady_state_fn=steady_state,
    init_state_fn=init_state,
    policy_lower=POLICY_LOWER,
    policy_upper=POLICY_UPPER,
)
