"""Framework helper: synthesize equations_fn from a period-return function.

For a representative-agent problem with a single intertemporal state
(capital, wealth, debt stock, ...), the researcher writes one scalar
function

    Pi(K_t, K_{t+1}, z_t, constants) = per-period return (utility, payoff)

where consumption / effort / whatever is implied by the budget constraint
inside Pi. The helper here autodiffs Pi twice to produce the Euler
residual

    euler = -(dPi/dK_{t+1}  at (K_t, K_{t+1}, z_t)
            + beta * dPi/dK_t  at (K_{t+1}, K_{t+2}, z_{t+1})).

K_{t+2} is reconstructed via the model's own ``step_fn`` evaluated at
zero shock, so the researcher does not separately supply a deterministic
law of motion.

Current scope (POC-level, matches ``brock_mirman_autodiff``):

- Single intertemporal state dimension (``capital_idx``).
- Any number of exogenous state dimensions (``exog_idx``).
- Single policy (though multi-policy models still work as long as only
  one policy dimension affects the intertemporal tradeoff; the others
  enter Pi via the budget and their intratemporal FOCs are out of scope
  for this helper).
- Shock expectation is handled as always by the loss module (per-shock
  residual averaging over MC or quadrature nodes).

Not yet supported (tracked in ``docs/site/autodiff.md``):

- Multi-agent Euler (one equation per savings-choosing agent, OLG-style).
- Intratemporal FOCs (e.g. labor FOC from ``dPi/dL = 0``).
- Lagrangian with explicit multipliers (KKT for borrowing constraints,
  Fischer-Burmeister).
"""

from typing import Callable, Dict, Iterable, Tuple

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
) -> Callable:
    """Build an ``equations_fn`` that synthesizes the Euler residual via ``jax.grad``.

    Args:
        period_return_fn: ``Pi(K_scalar, K_next_scalar, z_vec, constants) -> scalar``.
            ``z_vec`` is a 1-D array containing the exogenous state
            components (empty if the model has no exogenous state).
        step_fn: the model's standard ``step(state, policy, shock, constants)``;
            used with zero shocks to reconstruct ``K_{t+2}``.
        capital_idx: which column of ``state`` is the intertemporal capital.
        exog_idx: columns of ``state`` that are exogenous shocks. Passed
            into ``period_return_fn`` as a vector in that order.
        n_shocks: number of shocks the model has; used to build a zero
            shock for the deterministic step.
        equation_name: key under which the residual is returned
            (default ``"euler"``).

    Returns:
        An ``equations_fn(state, policy, next_state, next_policy, constants)``
        that returns ``{equation_name: [batch]}`` -- shape and contract
        matching the canonical hand-derived model.
    """
    exog_idx = tuple(exog_idx)

    dPi_dK_next = jax.grad(period_return_fn, argnums=1)
    dPi_dK = jax.grad(period_return_fn, argnums=0)

    # vmap the per-element autodiff calls over the batch axis; constants
    # passed through as a static pytree (in_axes=None).
    _dPi_dK_next_v = jax.vmap(dPi_dK_next, in_axes=(0, 0, 0, None))
    _dPi_dK_v = jax.vmap(dPi_dK, in_axes=(0, 0, 0, None))

    def _extract_exog(s: Array) -> Array:
        # jnp.take with axis=1 keeps the result 2-D; shape [batch, n_exog].
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

        # K_{t+2}: apply the model's step with zero shock. That gives the
        # deterministic continuation; expectation over shocks is handled
        # by the loss module averaging per-shock residuals.
        zero_shock = jnp.zeros((state.shape[0], n_shocks))
        next_next_state = step_fn(next_state, next_policy, zero_shock, constants)
        K_tp2 = next_next_state[:, capital_idx]

        dPi1 = _dPi_dK_next_v(K_t, K_tp1, z_t, constants)
        dPi2 = _dPi_dK_v(K_tp1, K_tp2, z_tp1, constants)

        beta = constants["beta"]
        # Sign: match the "u_c - beta E[u_c' (1 + r' - delta)]" convention
        # already used in the hand-derived brock_mirman model.
        euler = -(dPi1 + beta * dPi2)

        return {equation_name: euler}

    return equations_fn
