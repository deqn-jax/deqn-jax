"""Brock-Mirman with Epstein-Zin recursive utility.

Same one-good RBC dynamics as ``brock_mirman``, but the household has
Epstein-Zin recursive preferences with separate parameters ψ (IES) and
γ_ez (RRA). The Bellman recursion is enforced via a residual that
references the critic's V(s) — so this is the canonical demo of the
actor-critic framework on a small, well-understood model.

Use with ``actor_critic.mode = 'shared'`` (one ``ActorCriticMLP`` for
policy + value) or ``'separate'`` (policy MLP + standalone critic in
``aux_params``). See ``configs/brock_mirman_ez.yaml`` for a starting
config.
"""

from deqn_jax.models.brock_mirman_ez.dynamics import step
from deqn_jax.models.brock_mirman_ez.equations import (
    EQUATION_NAMES,
    definitions,
    equations,
)
from deqn_jax.models.brock_mirman_ez.steady_state import init_state, steady_state
from deqn_jax.models.brock_mirman_ez.variables import (
    CONSTANTS,
    N_SHOCKS,
    POLICY_LOWER,
    POLICY_UPPER,
    SPEC,
)
from deqn_jax.types import ModelSpec

MODEL = ModelSpec(
    name="brock_mirman_ez",
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
