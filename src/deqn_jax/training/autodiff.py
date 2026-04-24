"""Framework helper: synthesize equations_fn from a period-return function.

The researcher writes one scalar function

    Pi(K_t, K_{t+1}, z_t, policy_t, constants) = per-period return

where ``policy_t`` is the full policy vector the network outputs and
consumption / effort / whatever is implied by the budget constraint
inside Pi. The helper autodiffs Pi to produce equilibrium residuals:

**Capital Euler (always)** — one equation named ``euler_name`` (default
``"euler"``). Formed via the envelope theorem:

    euler = -(dPi/dK_{t+1} at (K_t, K_{t+1}, z_t, policy_t)
            + beta * dPi/dK_t at (K_{t+1}, K_{t+2}, z_{t+1}, policy_{t+1})).

K_{t+2} is reconstructed via the model's own ``step_fn`` at zero shock.
Expectation over shocks is handled as always by the loss module
(per-shock residual averaging over MC or quadrature nodes).

**Intratemporal FOCs (optional)** — one equation per index listed in
``intratemporal_policy_idx``. For index ``j`` the residual is

    foc_j = dPi/d(policy[j]) at (K_t, K_{t+1}, z_t, policy_t)

i.e. zero at the intratemporal optimum (labor FOC, effort FOC, etc.).
These are completely local — no expectation, no K_{t+2} reconstruction.

Current scope:

- Single intertemporal state dimension (``capital_idx``).
- Any number of exogenous state dimensions (``exog_idx``).
- Any number of intratemporal policy dimensions (``intratemporal_policy_idx``).
- Shock expectation handled by the loss module.

Not yet supported (tracked in ``docs/site/autodiff.md``):

- Multi-agent Euler (one equation per savings-choosing agent, OLG-style)
  — needs a vmap over an agent dimension plus a way to identify which
  policy element saves into which next-state element.
- Lagrangian with explicit multipliers (KKT for borrowing constraints,
  Fischer-Burmeister).
"""

from typing import Callable, Dict, Iterable

import jax
import jax.numpy as jnp
from jax import Array


def euler_from_period_return(
    period_return_fn: Callable,
    step_fn: Callable,
    capital_idx: int = 0,
    exog_idx: Iterable[int] = (1,),
    n_shocks: int = 1,
    equation_name: str = "euler",
    intratemporal_policy_idx: Iterable[int] = (),
    intratemporal_equation_names: Iterable[str] = (),
) -> Callable:
    """Build an ``equations_fn`` that synthesizes residuals via ``jax.grad``.

    Args:
        period_return_fn: ``Pi(K_scalar, K_next_scalar, z_vec, policy_vec,
            constants) -> scalar``. ``policy_vec`` is the per-sample policy
            array (not batched); the author reads whichever components their
            Pi actually depends on. For brock_mirman style models that
            absorb the savings rate into K_next via the budget, Pi can
            ignore ``policy_vec`` entirely -- the argument is still passed.
        step_fn: ``step(state, policy, shock, constants) -> next_state``,
            used at zero shock to reconstruct ``K_{t+2}``.
        capital_idx: which column of ``state`` is the intertemporal capital.
        exog_idx: columns of ``state`` that are exogenous (AR(1), shocks,
            etc.). Passed into ``period_return_fn`` as a 1-D vector in
            that order.
        n_shocks: number of shocks on the model; used to build a zero
            shock for the deterministic step.
        equation_name: key under which the Euler residual is returned
            (default ``"euler"``).
        intratemporal_policy_idx: policy indices whose intratemporal FOC
            ``dPi/d(policy[j]) = 0`` is synthesized as an additional
            equation. Default empty (= no intratemporal equations, just
            the capital Euler).
        intratemporal_equation_names: optional custom names for each
            intratemporal equation (same length as
            ``intratemporal_policy_idx``). Defaults to
            ``("intratemporal_j0", "intratemporal_j1", ...)`` using the
            policy index.

    Returns:
        ``equations_fn(state, policy, next_state, next_policy, constants)``
        returning ``{equation_name: [batch], ...}`` with one entry for
        the capital Euler plus one for each intratemporal FOC.
    """
    exog_idx = tuple(exog_idx)
    intratemporal_policy_idx = tuple(intratemporal_policy_idx)
    intratemporal_equation_names = tuple(intratemporal_equation_names)
    if intratemporal_equation_names and len(intratemporal_equation_names) != len(intratemporal_policy_idx):
        raise ValueError(
            f"intratemporal_equation_names length ({len(intratemporal_equation_names)}) "
            f"must equal intratemporal_policy_idx length "
            f"({len(intratemporal_policy_idx)})"
        )
    if not intratemporal_equation_names:
        intratemporal_equation_names = tuple(
            f"intratemporal_j{j}" for j in intratemporal_policy_idx
        )

    # Autodiffs against specific arg positions of Pi(K, K_next, z, policy, constants):
    #   argnum 0 -> K_t            (for Euler's dPi/dK at t+1)
    #   argnum 1 -> K_{t+1}        (for Euler's dPi/dK_{t+1} at t)
    #   argnum 3 -> policy vector  (for intratemporal FOCs)
    dPi_dK_next = jax.grad(period_return_fn, argnums=1)
    dPi_dK = jax.grad(period_return_fn, argnums=0)
    dPi_dpolicy = jax.grad(period_return_fn, argnums=3)

    _dPi_dK_next_v = jax.vmap(dPi_dK_next, in_axes=(0, 0, 0, 0, None))
    _dPi_dK_v = jax.vmap(dPi_dK, in_axes=(0, 0, 0, 0, None))
    _dPi_dpolicy_v = jax.vmap(dPi_dpolicy, in_axes=(0, 0, 0, 0, None))

    def _extract_exog(s: Array) -> Array:
        return jnp.take(s, jnp.asarray(exog_idx), axis=1)

    def equations_fn(
        state: Array,
        policy: Array,
        next_state: Array,
        next_policy: Array,
        constants: Dict,
    ) -> Dict[str, Array]:
        K_t = state[:, capital_idx]
        K_tp1 = next_state[:, capital_idx]
        z_t = _extract_exog(state)
        z_tp1 = _extract_exog(next_state)

        # K_{t+2} via zero-shock deterministic step from (K_{t+1}, policy_{t+1}).
        zero_shock = jnp.zeros((state.shape[0], n_shocks))
        next_next_state = step_fn(next_state, next_policy, zero_shock, constants)
        K_tp2 = next_next_state[:, capital_idx]

        # Capital Euler (sign convention matches the hand-derived
        # `u_c - beta E[u_c'(1 + r' - delta)]` form used elsewhere).
        dPi1 = _dPi_dK_next_v(K_t, K_tp1, z_t, policy, constants)
        dPi2 = _dPi_dK_v(K_tp1, K_tp2, z_tp1, next_policy, constants)
        out = {equation_name: -(dPi1 + constants["beta"] * dPi2)}

        # Intratemporal FOCs: -dPi/d(policy[j]) at t, per listed index.
        # Sign flip mirrors the Euler convention (helper returns -(dPi1 + beta dPi2))
        # so the generated residual matches hand-derived forms like
        # `psi * L^theta - w * u_c` rather than their negation.
        if intratemporal_policy_idx:
            dPi_dp = _dPi_dpolicy_v(K_t, K_tp1, z_t, policy, constants)  # [batch, n_policies]
            for j, name in zip(intratemporal_policy_idx, intratemporal_equation_names):
                out[name] = -dPi_dp[:, j]

        return out

    return equations_fn
