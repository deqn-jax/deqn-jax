"""Equilibrium residuals derived from a period Lagrangian by ``jax.grad``.

The model author writes, per sample (unbatched),

    F(state, x_next, policy, prices, constants)          period objective
    h_k(state, x_next, policy, prices, constants) = 0    equality constraints
    g_k(state, x_next, policy, prices, constants) >= 0   inequality constraints

where ``x_next`` is the vector of endogenous next-period states (the state
columns listed in ``endogenous``, as chosen at t) and the multipliers
``lambda_k`` / ``mu_k`` are policy outputs. The framework forms

    L = F + sum_k lambda_k h_k + sum_k mu_k g_k

and derives

- one Euler residual per endogenous state ``x_j``:
  ``dL_t/dx'_j + beta * E_t[dL_{t+1}/dx_j] = 0``;
- one first-order condition per static control ``p_c``: ``dL_t/dp_c = 0``;
- each equality constraint ``h_k`` as its own residual;
- each inequality constraint as ``FB(mu_k, g_k)`` (Fischer-Burmeister).

Prices and aggregates the agents take as given come from ``prices_fn`` and
enter ``L`` as an argument that is never differentiated, so a competitive
equilibrium (OLG, heterogeneous agents) is not mistaken for a planner's
problem. Summing the period objectives of all agents into one ``F`` gives
each agent's own conditions as long as every endogenous state and static
control belongs to one agent.

The t+1 term is evaluated at the next state the trainer passed in, with
``x_{t+2}`` rebuilt by ``step_fn`` at zero shock; the trainer averages the
per-shock residuals, so the expectation sits where the Euler needs it. The
residual is linear in every t+1 quantity (the ratio form divides by a
period-t quantity only), which keeps that average unbiased.
"""

from typing import Callable, Dict, NamedTuple, Optional, Sequence, Union

import jax
import jax.numpy as jnp
from jax import Array

EULER_FORMS = ("ratio", "raw")


class Constraint(NamedTuple):
    """A constraint ``fn(...) = 0`` or ``fn(...) >= 0`` with its multiplier.

    ``fn`` has the objective's signature; ``multiplier`` is the policy
    column carrying the constraint's Lagrange multiplier.
    """

    name: str
    fn: Callable
    multiplier: int


def _check_names(names: Sequence[str], n: int, what: str) -> tuple:
    names = tuple(names)
    if len(names) != n:
        raise ValueError(f"{what}: {len(names)} names for {n} entries")
    return names


def _discount_fn(discount: Union[str, float, Callable]) -> Callable:
    """Return ``beta(state, constants)`` from a constants key, a number or a callable."""
    if callable(discount):
        return discount
    if isinstance(discount, str):
        return lambda state, constants: constants[discount]
    return lambda state, constants: discount


def residuals_from_lagrangian(
    objective: Callable,
    step_fn: Callable,
    endogenous: Sequence[int],
    euler_names: Sequence[str],
    *,
    n_shocks: int,
    prices_fn: Optional[Callable] = None,
    static_controls: Sequence[int] = (),
    static_names: Sequence[str] = (),
    equalities: Sequence[Constraint] = (),
    inequalities: Sequence[Constraint] = (),
    discount: Union[str, float, Callable] = "beta",
    euler_form: str = "ratio",
    stop_next_policy_gradient: bool = True,
) -> Callable:
    """Build ``equations_fn(state, policy, next_state, next_policy, constants)``.

    Args:
        objective: ``F(state, x_next, policy, prices, constants) -> scalar``,
            one sample. ``x_next[i]`` is next-period state column
            ``endogenous[i]``.
        step_fn: the model's law of motion (batched), used at zero shock to
            rebuild ``x_{t+2}`` from ``next_state`` and ``next_policy``.
        endogenous: state columns chosen one period ahead; one Euler each.
            Their next-period values must not depend on the shock.
        euler_names: one equation name per endogenous state.
        n_shocks: shock dimension, for the zero shock.
        prices_fn: ``prices(state, policy, constants)`` for one sample: the
            prices and aggregates agents take as given. Any pytree; ``None``
            passes ``None``.
        static_controls: policy columns chosen within the period; one FOC
            ``-dL/dp_c`` each, holding ``x_next`` fixed.
        static_names: names for the static FOCs (default ``foc_p{c}``).
        equalities: ``h = 0`` constraints; residual ``h``.
        inequalities: ``g >= 0`` constraints; residual ``FB(mu, g)``.
        discount: constants key, number, or ``beta(state, constants)`` for
            one sample at t (time-varying discounting).
        euler_form: ``"ratio"`` divides the raw residual
            ``-(dL_t/dx'_j + beta dL_{t+1}/dx_j)`` by the period-t marginal
            cost ``A_j = -d(F + lambda.h)/dx'_j``, giving
            ``1 - (M_j + beta E[dL_{t+1}/dx_j]) / A_j`` with ``M_j`` the
            inequality-multiplier terms. ``A_j`` must be positive on the
            training support. ``"raw"`` returns the residual in objective
            units.
        stop_next_policy_gradient: freeze ``next_policy`` (semi-gradient),
            the convention of the period-return helper. The residual value
            is unaffected; only the training gradient changes.

    Returns:
        ``equations_fn`` returning residuals in the order Euler, static
        FOCs, equalities, inequalities.
    """
    endogenous = tuple(int(j) for j in endogenous)
    if len(set(endogenous)) != len(endogenous):
        raise ValueError(f"endogenous state columns repeat: {endogenous}")
    euler_names = _check_names(euler_names, len(endogenous), "euler_names")
    static_controls = tuple(int(c) for c in static_controls)
    if not static_names:
        static_names = tuple(f"foc_p{c}" for c in static_controls)
    static_names = _check_names(static_names, len(static_controls), "static_names")
    equalities = tuple(equalities)
    inequalities = tuple(inequalities)
    multipliers = {c.multiplier for c in equalities + inequalities}
    if multipliers & set(static_controls):
        raise ValueError("a multiplier column is also listed as a static control")
    if euler_form not in EULER_FORMS:
        raise ValueError(f"euler_form must be one of {EULER_FORMS}, got {euler_form!r}")
    # Imported here: the models package imports this module while it loads.
    from deqn_jax.models._complementarity import fischer_burmeister

    beta_of = _discount_fn(discount)
    endo_idx = jnp.asarray(endogenous)

    def equations_fn(
        state: Array,
        policy: Array,
        next_state: Array,
        next_policy: Array,
        constants: Dict,
    ) -> Dict[str, Array]:
        def terms(constraints, s, xn, p, pr):
            return sum(
                p[k.multiplier] * k.fn(s, xn, p, pr, constants) for k in constraints
            )

        def cost_part(s, xn, p, pr):  # F + lambda.h, which sets the ratio scale
            f = objective(s, xn, p, pr, constants)
            return f + terms(equalities, s, xn, p, pr) if equalities else f

        def lagrangian(s, xn, p, pr):
            f = cost_part(s, xn, p, pr)
            return f + terms(inequalities, s, xn, p, pr) if inequalities else f

        def prices(s, p):
            if prices_fn is None:
                return None
            return jax.vmap(lambda si, pi: prices_fn(si, pi, constants))(s, p)

        def grad(fn, argnum, s, xn, p):
            return jax.vmap(jax.grad(fn, argnums=argnum))(s, xn, p, prices(s, p))

        if stop_next_policy_gradient:
            next_policy = jax.lax.stop_gradient(next_policy)
        zero_shock = jnp.zeros((state.shape[0], n_shocks))
        next_next_state = step_fn(next_state, next_policy, zero_shock, constants)
        x_tp1 = jnp.take(next_state, endo_idx, axis=1)
        x_tp2 = jnp.take(next_next_state, endo_idx, axis=1)

        # dL_t/dx'_j at t; dL_{t+1}/dx_j at t+1 (taken on the state vector).
        dL_now = grad(lagrangian, 1, state, x_tp1, policy)
        dL_next = grad(lagrangian, 0, next_state, x_tp2, next_policy)
        dL_next = jnp.take(dL_next, endo_idx, axis=1)
        if callable(discount):
            beta = jax.vmap(lambda s: beta_of(s, constants))(state)[:, None]
        else:
            beta = beta_of(state, constants)

        raw = -(dL_now + beta * dL_next)
        if euler_form == "ratio":
            d_cost = (
                grad(cost_part, 1, state, x_tp1, policy) if inequalities else dL_now
            )
            raw = raw / -d_cost
        out: Dict[str, Array] = {name: raw[:, i] for i, name in enumerate(euler_names)}

        if static_controls:
            dL_dp = grad(lagrangian, 2, state, x_tp1, policy)
            for c, name in zip(static_controls, static_names):
                out[name] = -dL_dp[:, c]

        pr = prices(state, policy)
        for is_ineq, constraints in ((False, equalities), (True, inequalities)):
            for k in constraints:
                val = jax.vmap(lambda *a, k=k: k.fn(*a, constants))(
                    state, x_tp1, policy, pr
                )
                if is_ineq:
                    val = fischer_burmeister(policy[:, k.multiplier], val)
                out[k.name] = val
        return out

    return equations_fn
