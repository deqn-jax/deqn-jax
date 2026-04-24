"""Autodiff POC: stochastic Brock-Mirman with Euler synthesized from primitives.

Same economics as ``brock_mirman``, but equations.py *does not hand-derive*
the Euler FOC. Instead it defines the per-period return

    Pi(K_t, K_{t+1}, z_t, constants) = u(C_t)

and uses ``jax.grad`` to produce the residual

    -(dPi/dK' + beta * dPi/dK_at_next).

This is the smallest possible demonstration of Simon's "researcher supplies
a Lagrangian, framework does the DEQN mechanics" vision. At steady state
the synthesized residual matches the hand-derived one to floating-point
noise (see ``tests/test_autodiff_equations.py``).

The longer-term framing is a framework-level helper ``equations_from_Pi``
that any ``ModelSpec`` could opt into by supplying a period-return function
instead of (or alongside) ``equations_fn``. See ``docs/site/autodiff.md``
for the design note.
"""

from deqn_jax.models.brock_mirman_autodiff.dynamics import step
from deqn_jax.models.brock_mirman_autodiff.equations import (
    EQUATION_NAMES,
    definitions,
    equations,
    period_return,
)
from deqn_jax.models.brock_mirman_autodiff.steady_state import init_state, steady_state
from deqn_jax.models.brock_mirman_autodiff.variables import (
    CONSTANTS,
    N_SHOCKS,
    POLICY_LOWER,
    POLICY_UPPER,
    SPEC,
)
from deqn_jax.types import ModelSpec

MODEL = ModelSpec(
    name="brock_mirman_autodiff",
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

__all__ = ["MODEL", "period_return"]
