"""The 6-agent OLG written as a Lagrangian; equations derived by jax.grad.

State ``(k^2..k^6, eta, delta)``, policy ``(s^1..s^5)`` with ``s^h`` the
capital cohort h carries into the next period, ``k'^{h+1} = s^h``. The
period objective is the sum of the six cohorts' utilities,

    F = sum_h u(c^h),   c^h = r k^h + w l^h - k'^{h+1}   (k^1 = k'^7 = 0),

with the budget substituted in, so there are no multipliers. The interest
factor ``r`` and the wage ``w`` come from ``prices`` and are taken as given
by every cohort. Each saving choice ``k'^{h+1}`` appears in cohort h's
utility today and in cohort h+1's utility tomorrow only, so the summed
objective yields each cohort's own Euler equation,

    1 - beta r' u'(c'^{h+1}) / u'(c^h) = 0,

in the ratio form the hand-written ``olg_analytic_6`` uses.
"""

from typing import Dict

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.olg_analytic_6.dynamics import step
from deqn_jax.models.olg_analytic_6_autodiff.variables import N_SHOCKS, A
from deqn_jax.training.lagrangian import residuals_from_lagrangian

EQUATION_NAMES = tuple(f"euler_h{h + 1}" for h in range(A - 1))

# Below this consumption level u is continued linearly, so u'(c) is capped
# at u'(C_FLOOR), the same cap the hand-written model applies to u'(c).
C_FLOOR = 1e-3


def _utility(c: Array, gamma: float) -> Array:
    """CRRA utility, continued linearly below ``C_FLOOR``."""
    safe = jnp.maximum(c, C_FLOOR)
    if gamma == 1.0:
        u, u_floor = jnp.log(safe), jnp.log(C_FLOOR)
    else:
        u = safe ** (1.0 - gamma) / (1.0 - gamma)
        u_floor = C_FLOOR ** (1.0 - gamma) / (1.0 - gamma)
    slope = C_FLOOR**-gamma
    return jnp.where(c > C_FLOOR, u, u_floor + slope * (c - C_FLOOR))


def prices(state: Array, policy: Array, constants: Dict) -> Dict[str, Array]:
    """Interest factor and wage from aggregate capital, for one sample."""
    del policy
    alpha, labor = constants["alpha"], constants["labor_1"]
    eta, delta = state[A - 1], state[A]
    K = jnp.sum(state[: A - 1])
    r = alpha * eta * K ** (alpha - 1.0) * labor ** (1.0 - alpha) + 1.0 - delta
    w = (1.0 - alpha) * eta * K**alpha * labor ** (-alpha)
    return {"r": r, "w": w}


def objective(
    state: Array, k_next: Array, policy: Array, prices: Dict, constants: Dict
) -> Array:
    """Sum over cohorts of u(c^h) at given prices; ``k_next = (k'^2..k'^6)``."""
    del policy
    zero = jnp.zeros((1,))
    k = jnp.concatenate([zero, state[: A - 1]])
    labor = jnp.zeros(A).at[0].set(constants["labor_1"])
    savings = jnp.concatenate([k_next, zero])
    c = prices["r"] * k + prices["w"] * labor - savings
    return jnp.sum(_utility(c, constants["gamma"]))


equations = residuals_from_lagrangian(
    objective,
    step,
    endogenous=range(A - 1),  # k^2..k^6 are state columns 0..4
    euler_names=EQUATION_NAMES,
    n_shocks=N_SHOCKS,
    prices_fn=prices,
    euler_form="ratio",
)
