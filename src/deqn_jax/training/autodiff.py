"""Framework helper: synthesize equations_fn from a period-return function.

The researcher writes one scalar function

    Pi(K_t, K_{t+1}, z_t, policy_t, constants) = per-period return

(or a multi-agent variant ``Pi(..., agent_index=i)`` where each agent
has its own savings dimension; see below). The helper autodiffs Pi to
produce equilibrium residuals:

**Single-agent capital Euler** — one equation named ``equation_name``
(default ``"euler"``). Formed via the envelope theorem:

    euler = -(dPi/dK_{t+1} at (K_t, K_{t+1}, z_t, policy_t)
            + beta * dPi/dK_t at (K_{t+1}, K_{t+2}, z_{t+1}, policy_{t+1})).

K_{t+2} is reconstructed via the model's own ``step_fn`` at zero shock.
Expectation over shocks is handled as always by the loss module
(per-shock residual averaging over MC or quadrature nodes).

**Multi-agent Euler (OLG)** — one equation per savings-choosing agent.
Each agent ``i`` has its own intertemporal capital state column
``capital_indices[i]``; ``Pi`` is called with an additional
``agent_index=i`` keyword to select its agent-specific consumption /
effort / payoff. Returns one Euler residual per agent, named via
``equation_names``.

**Intratemporal FOCs (optional)** — one equation per index listed in
``intratemporal_policy_idx``. For index ``j`` the residual is

    foc_j = dPi/d(policy[j]) at (K_t, K_{t+1}, z_t, policy_t)

i.e. zero at the intratemporal optimum (labor FOC, effort FOC, etc.).
These are completely local — no expectation, no K_{t+2} reconstruction.
For multi-agent, intratemporal FOCs are computed against the
``agent_index=0`` form of Pi by default.

Current scope:

- Single intertemporal state OR multiple agent-specific intertemporal states.
- Any number of exogenous state dimensions (``exog_idx``).
- Any number of intratemporal policy dimensions (``intratemporal_policy_idx``).
- Shock expectation handled by the loss module.

Not yet supported:

- Lagrangian with explicit multipliers (KKT for borrowing constraints,
  Fischer-Burmeister).
"""

from typing import Callable, Dict, Iterable, Optional, Sequence

import jax
import jax.numpy as jnp
from jax import Array


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

    # Build per-agent gradient functions. Each agent's Pi closure pins its
    # agent_index; jax.grad operates on the closed-over scalar function.
    n_agents = len(capital_indices)

    def _make_per_agent_grads(agent_idx: int):
        if is_multi_agent:

            def pi_i(K, K_next, z, policy, c):
                return period_return_fn(K, K_next, z, policy, c, agent_index=agent_idx)
        else:
            pi_i = period_return_fn
        dK_next = jax.vmap(jax.grad(pi_i, argnums=1), in_axes=(0, 0, 0, 0, None))
        dK = jax.vmap(jax.grad(pi_i, argnums=0), in_axes=(0, 0, 0, 0, None))
        return dK_next, dK

    per_agent_dK_next = []
    per_agent_dK = []
    for i in range(n_agents):
        dn, d0 = _make_per_agent_grads(i)
        per_agent_dK_next.append(dn)
        per_agent_dK.append(d0)

    # Intratemporal FOC gradient. For multi-agent, take it against the
    # agent-0 form of Pi by default; researchers wanting per-agent
    # intratemporal FOCs should hand-write or extend this helper later.
    if is_multi_agent:

        def pi_for_intratemporal(K, K_next, z, policy, c):
            return period_return_fn(K, K_next, z, policy, c, agent_index=0)
    else:
        pi_for_intratemporal = period_return_fn
    _dPi_dpolicy_v = jax.vmap(
        jax.grad(pi_for_intratemporal, argnums=3), in_axes=(0, 0, 0, 0, None)
    )

    def _extract_exog(s: Array) -> Array:
        return jnp.take(s, jnp.asarray(exog_idx), axis=1)

    def equations_fn(
        state: Array,
        policy: Array,
        next_state: Array,
        next_policy: Array,
        constants: Dict,
    ) -> Dict[str, Array]:
        z_t = _extract_exog(state)
        z_tp1 = _extract_exog(next_state)

        # ENVELOPE CONTRACT: freeze next_policy via stop_gradient before
        # using it in K_{t+2} reconstruction or per-agent dPi2 gradients.
        # See the original single-agent docstring above for derivation.
        next_policy_frozen = jax.lax.stop_gradient(next_policy)

        # K_{t+2} — single deterministic step shared across all agents;
        # each agent indexes into the resulting next_next_state at its own
        # capital column.
        zero_shock = jnp.zeros((state.shape[0], n_shocks))
        next_next_state = step_fn(next_state, next_policy_frozen, zero_shock, constants)

        out: Dict[str, Array] = {}
        for i, (cap_idx, eq_name) in enumerate(zip(capital_indices, equation_names)):
            K_t = state[:, cap_idx]
            K_tp1 = next_state[:, cap_idx]
            K_tp2 = next_next_state[:, cap_idx]
            dPi1 = per_agent_dK_next[i](K_t, K_tp1, z_t, policy, constants)
            dPi2 = per_agent_dK[i](K_tp1, K_tp2, z_tp1, next_policy_frozen, constants)
            out[eq_name] = -(dPi1 + constants["beta"] * dPi2)

        # Intratemporal FOCs: -dPi/d(policy[j]) at t, per listed index.
        if intratemporal_policy_idx:
            # Use agent-0's K_t / K_tp1 (consistent with single-agent legacy).
            K_t_for_intra = state[:, capital_indices[0]]
            K_tp1_for_intra = next_state[:, capital_indices[0]]
            dPi_dp = _dPi_dpolicy_v(
                K_t_for_intra, K_tp1_for_intra, z_t, policy, constants
            )  # [batch, n_policies]
            for j, name in zip(intratemporal_policy_idx, intratemporal_equation_names):
                out[name] = -dPi_dp[:, j]

        return out

    return equations_fn
