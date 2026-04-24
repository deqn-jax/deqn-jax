"""CMR-style NK-DSGE with Financial Frictions ("Disaster Model").

A medium-scale New Keynesian model with:
- 13 state variables (8 endogenous + 5 exogenous)
- 11 policy variables (s, L, omega_bar computed analytically)
- 11 equilibrium equations
- Financial frictions (costly state verification banking)

Analytical eliminations (12 original -> 9):
  s (cost min), L (balance sheet), omega_bar (bank participation)
"""

from deqn_jax.models.disaster.dynamics import clip_state, compute_state_barrier, step
from deqn_jax.models.disaster.equations import EQUATION_NAMES, definitions, equations
from deqn_jax.models.disaster.steady_state import init_state, steady_state
from deqn_jax.models.disaster.variables import (
    CONSTANTS,
    N_SHOCKS,
    POLICY_LOWER,
    POLICY_UPPER,
    SPEC,
)
from deqn_jax.types import ModelSpec

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
    clip_state_fn=clip_state,          # Hard clip for eval/irf only
    state_barrier_fn=compute_state_barrier,  # Box penalty for loss
    # Order MUST match dynamics.step()'s shock[:, i] unpacking order.
    shock_names=("eps", "mu_ups", "mu_z", "g", "m_p"),
)
