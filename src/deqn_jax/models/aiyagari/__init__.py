"""Aiyagari (1994) incomplete-markets savings model.

INTERNAL / NOT PUBLICLY REGISTERED as of v0.1.0. This module exists in
source but is intentionally not exposed through the public model
registry (``deqn_jax.models.load_model``). It has no test coverage yet
and should not be relied upon for release-quality results.

Partial-equilibrium individual problem:
- State: (k, a) = (individual capital, log idiosyncratic productivity)
- Policy: c_share (consumption share of cash-on-hand)
- One Euler equation with borrowing constraint
- Prices fixed at representative-agent steady-state levels

Testbed for DEQN/GRPO on a model with:
- Borrowing constraint (k >= 0)
- Idiosyncratic risk (precautionary savings motive)
- Nonlinear optimal policy (savings rate depends on wealth)

To promote to a public model: add to ``deqn_jax/models/__init__.py``
registry dict and add smoke tests under ``tests/``.
"""

from deqn_jax.types import ModelSpec
from deqn_jax.models.aiyagari.variables import (
    SPEC, CONSTANTS, N_SHOCKS, POLICY_LOWER, POLICY_UPPER,
)
from deqn_jax.models.aiyagari.equations import equations, definitions, EQUATION_NAMES
from deqn_jax.models.aiyagari.dynamics import step
from deqn_jax.models.aiyagari.steady_state import steady_state, init_state

MODEL = ModelSpec(
    name="aiyagari",
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
    policy_upper=POLICY_UPPER,
)
