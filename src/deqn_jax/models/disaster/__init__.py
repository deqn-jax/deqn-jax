"""CMR-style NK-DSGE with Financial Frictions ("Disaster Model").

A medium-scale New Keynesian model with:
- 13 state variables (8 endogenous + 5 exogenous)
- 11 policy variables (s, L, omega_bar computed analytically)
- 11 equilibrium equations
- Financial frictions (costly state verification banking)

Analytical eliminations (12 original -> 9):
  s (cost min), L (balance sheet), omega_bar (bank participation)
"""

from deqn_jax.types import ModelSpec
from deqn_jax.models.disaster.variables import (
    SPEC, CONSTANTS, N_SHOCKS, POLICY_LOWER, POLICY_UPPER,
)
from deqn_jax.models.disaster.equations import equations, definitions, EQUATION_NAMES
from deqn_jax.models.disaster.dynamics import step, clip_state
from deqn_jax.models.disaster.steady_state import steady_state, init_state

MODEL = ModelSpec(
    name="disaster",
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
    definitions_fn=definitions,
    policy_lower=POLICY_LOWER,
    policy_upper=POLICY_UPPER,  # None → softplus bounding (no gradient death)
    clip_state_fn=clip_state,   # Simulation safety only (NOT in training loss path)
)
