"""Framework helper: synthesize equations_fn from a period-return function.

The researcher writes one scalar function

    Pi(K_t, K_{t+1}, z_t, policy_t, constants) = per-period return

(or a multi-agent variant ``Pi(..., agent_index=i)`` where each agent has
its own capital column). This is the special case of
``deqn_jax.training.lagrangian.residuals_from_lagrangian`` with a scalar
endogenous state per agent, no prices, no constraints and the raw Euler
form; the helper adapts ``Pi`` to that interface and delegates.

**Euler** (one per capital column), from the envelope theorem:

    euler = -(dPi/dK_{t+1} at (K_t, K_{t+1}, z_t, policy_t)
            + beta * dPi/dK_t at (K_{t+1}, K_{t+2}, z_{t+1}, policy_{t+1})),

with K_{t+2} rebuilt by the model's ``step_fn`` at zero shock and
``policy_{t+1}`` frozen (``stop_gradient``). The loss module averages the
per-shock residuals, which is the expectation.

**Intratemporal FOCs** (optional): ``-dPi/d(policy[j])`` for each listed
index, holding ``K_{t+1}`` fixed. In multi-agent mode the derivative is
taken of the sum of the agents' returns, so each agent's own static
choice gets its own condition.

For constraints with multipliers, prices taken as given, or the ratio
Euler form, use ``residuals_from_lagrangian`` directly.
"""

from typing import Callable, Iterable, Optional, Sequence

import jax.numpy as jnp

from deqn_jax.training.lagrangian import residuals_from_lagrangian


def euler_from_period_return(
    period_return_fn: Callable,
    step_fn: Callable,
    capital_idx: Optional[int] = None,
    exog_idx: Iterable[int] = (1,),
    n_shocks: int = 1,
    equation_name: Optional[str] = None,
    intratemporal_policy_idx: Iterable[int] = (),
    intratemporal_equation_names: Iterable[str] = (),
    *,
    capital_indices: Optional[Sequence[int]] = None,
    equation_names: Optional[Sequence[str]] = None,
) -> Callable:
    """Build an ``equations_fn`` that synthesizes residuals via ``jax.grad``.

    Two modes, dispatched on whether ``capital_indices`` is given:

    **Single-agent (legacy):** pass ``capital_idx=int`` and optionally
    ``equation_name=str``. ``period_return_fn`` has signature
    ``Pi(K_scalar, K_next_scalar, z_vec, policy_vec, constants) -> scalar``.

    **Multi-agent OLG:** pass ``capital_indices=Sequence[int]`` and
    ``equation_names=Sequence[str]`` (same length). ``period_return_fn``
    has signature
    ``Pi(K_scalar, K_next_scalar, z_vec, policy_vec, constants, *, agent_index: int) -> scalar``.
    The factory builds N per-agent gradient functions and returns N
    Euler residuals, one per ``capital_indices[i]``.

    Args:
        period_return_fn: per-period return; see modes above.
        step_fn: ``step(state, policy, shock, constants) -> next_state``,
            used at zero shock to reconstruct ``K_{t+2}``. For multi-agent,
            step_fn must update *all* capital states from the policy vector;
            the factory simply calls it once and indexes into the resulting
            next_next_state.
        capital_idx: legacy single-agent capital column. Mutually exclusive
            with ``capital_indices``.
        exog_idx: columns of ``state`` that are exogenous. Passed into
            ``period_return_fn`` as a 1-D vector in that order.
        n_shocks: number of shocks on the model; used to build a zero
            shock for the deterministic step.
        equation_name: legacy single-agent equation key (default ``"euler"``).
        intratemporal_policy_idx: policy indices whose intratemporal FOC
            ``dPi/d(policy[j]) = 0`` is synthesized as an additional
            equation. Default empty.
        intratemporal_equation_names: optional custom names for each
            intratemporal equation.
        capital_indices: keyword-only; when provided, switches to
            multi-agent mode. One entry per savings-choosing agent.
        equation_names: keyword-only; required when ``capital_indices`` is
            given; one entry per agent.

    Returns:
        ``equations_fn(state, policy, next_state, next_policy, constants)``
        returning a dict of residuals.
    """
    exog_idx = tuple(exog_idx)
    intratemporal_policy_idx = tuple(intratemporal_policy_idx)
    intratemporal_equation_names = tuple(intratemporal_equation_names)

    # Resolve single- vs multi-agent mode.
    if capital_indices is not None:
        if capital_idx is not None:
            raise ValueError(
                "Pass either 'capital_idx' (single-agent) or 'capital_indices' "
                "(multi-agent), not both."
            )
        capital_indices = tuple(int(i) for i in capital_indices)
        if equation_names is None:
            raise ValueError(
                "'equation_names' is required when 'capital_indices' is given "
                "(one equation key per agent)."
            )
        equation_names = tuple(equation_names)
        if len(equation_names) != len(capital_indices):
            raise ValueError(
                f"equation_names length ({len(equation_names)}) must equal "
                f"capital_indices length ({len(capital_indices)})"
            )
        is_multi_agent = True
    else:
        if capital_idx is None:
            capital_idx = 0
        if equation_name is None:
            equation_name = "euler"
        capital_indices = (int(capital_idx),)
        equation_names = (equation_name,)
        is_multi_agent = False

    if intratemporal_equation_names and len(intratemporal_equation_names) != len(
        intratemporal_policy_idx
    ):
        raise ValueError(
            f"intratemporal_equation_names length ({len(intratemporal_equation_names)}) "
            f"must equal intratemporal_policy_idx length "
            f"({len(intratemporal_policy_idx)})"
        )
    if not intratemporal_equation_names:
        intratemporal_equation_names = tuple(
            f"intratemporal_j{j}" for j in intratemporal_policy_idx
        )

    exog = jnp.asarray(exog_idx)

    def objective(state, x_next, policy, prices, constants):
        """Sum of the agents' returns, each on its own capital column."""
        del prices
        z = jnp.take(state, exog)
        if not is_multi_agent:
            return period_return_fn(
                state[capital_indices[0]], x_next[0], z, policy, constants
            )
        return sum(
            period_return_fn(state[cap], x_next[i], z, policy, constants, agent_index=i)
            for i, cap in enumerate(capital_indices)
        )

    return residuals_from_lagrangian(
        objective,
        step_fn,
        endogenous=capital_indices,
        euler_names=equation_names,
        n_shocks=n_shocks,
        static_controls=intratemporal_policy_idx,
        static_names=intratemporal_equation_names,
        euler_form="raw",
        stop_next_policy_gradient=True,
    )
