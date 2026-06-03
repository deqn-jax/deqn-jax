"""6-generation life-cycle OLG with borrowing constraints (Geneva Day 2 Ex 4).

Households live ``H = 6`` deterministic periods (ages 20-80, one period ≈ 10
years), save in capital subject to ``k >= 0``, and supply exogenous
age-dependent labor. Five intertemporal Euler conditions (one per cohort except
the last, which consumes everything) are encoded as Fischer-Burmeister
complementarities. Because each FB wraps an EXPECTATION (``E[fb] != fb(E)``), the
model uses the framework's two-stage loss hooks (``inside_fn`` + ``combine_fn``).
There is no closed-form steady state: it is trained over the ergodic
distribution from a random init (``steady_state_fn=None``).
"""

from deqn_jax.models.olg_lifecycle.dynamics import step
from deqn_jax.models.olg_lifecycle.equations import (
    EQUATION_NAMES,
    combine_fn,
    definitions,
    equations,
    inside_fn,
)
from deqn_jax.models.olg_lifecycle.steady_state import init_state
from deqn_jax.models.olg_lifecycle.variables import (
    CONSTANTS,
    N_SHOCKS,
    POLICY_LOWER,
    POLICY_UPPER,
    SPEC,
)
from deqn_jax.types import ModelSpec

MODEL = ModelSpec(
    name="olg_lifecycle",
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
    steady_state_fn=None,
    init_state_fn=init_state,
    definitions_fn=definitions,
    inside_fn=inside_fn,
    combine_fn=combine_fn,
    policy_lower=POLICY_LOWER,
    policy_upper=POLICY_UPPER,
)
