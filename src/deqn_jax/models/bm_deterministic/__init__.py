"""Deterministic Brock-Mirman optimal growth model.

The simplest possible DEQN setup: 1 state (K), 1 policy (sav_rate), 1
Euler equation, no shocks. With the default calibration (alpha=0.36,
beta=0.99, delta=1, gamma=1) the optimal policy is the constant
sav_rate* = alpha * beta (Brock & Mirman 1972); used as oracle.

Training recipe: sample K uniformly on [0.10, 1.00] every cycle (set
``initialize_each_episode: True`` in the run config). No rollouts; the
deterministic attractor would otherwise collapse the training
distribution onto K_ss.

References: Brock & Mirman (1972), J. Econ. Theory 4(3), 479-513.
Azinovic, Gaegauf & Scheidegger (2022), IER 63(4), 1471-1525.
"""

from deqn_jax.types import ModelSpec
from deqn_jax.models.bm_deterministic.variables import (
    SPEC, CONSTANTS, N_SHOCKS, POLICY_LOWER, POLICY_UPPER,
)
from deqn_jax.models.bm_deterministic.equations import equations, definitions, EQUATION_NAMES
from deqn_jax.models.bm_deterministic.dynamics import step
from deqn_jax.models.bm_deterministic.steady_state import steady_state, init_state
from deqn_jax.models.bm_deterministic.hooks import make_cycle_hook

MODEL = ModelSpec(
    name="bm_deterministic",
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
    cycle_hook=make_cycle_hook(),  # default: writes to ./figures/bm_deterministic
)
