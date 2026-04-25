"""Autodiff-synthesized equations for Brock-Mirman with endogenous labor.

Two FOCs, both drop out of differentiating a single scalar Pi:

    Pi(K_t, K_{t+1}, z_t, policy_t) = u(C_t) - psi * L_t^(1+theta) / (1+theta)

where C_t is implied by the budget constraint
``C_t = Y(K_t, L_t, z_t) - (K_{t+1} - (1 - delta) K_t)``, Y is Cobb-
Douglas in (K, L), and policy_t = (sav_rate, L). Only L enters Pi's
intratemporal argument list -- sav_rate is absorbed into K_{t+1} via
the law of motion, exactly as in the no-labor autodiff variant.

The helper produces:
- ``euler`` = capital Euler via ``dPi/dK_{t+1} + beta * dPi/dK at t+1``.
- ``labor_foc`` = intratemporal labor FOC ``dPi/d(policy[1])``, evaluated
  pointwise (no expectation).

Parity against the hand-derived ``bm_labor`` is verified in
``tests/test_autodiff_equations.py``.
"""

from typing import Dict

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.bm_labor.dynamics import step
from deqn_jax.models.bm_labor_autodiff.variables import SPEC
from deqn_jax.training.autodiff import euler_from_period_return

EQUATION_NAMES = ("euler", "labor_foc")


def period_return(
    k: Array,
    k_next: Array,
    z: Array,
    policy: Array,
    constants: Dict,
) -> Array:
    """Pi(K, K_next, z, policy) = ln C(K, K_next, L, z) - psi L^(1+theta)/(1+theta).

    Budget: C = Z * L^(1-alpha) * K^alpha - (K_next - (1-delta) K).
    Savings rate is absorbed into K_next; only L enters Pi explicitly.
    """
    alpha = constants["alpha"]
    delta = constants["delta"]
    gamma = constants["gamma"]
    psi = constants["psi"]
    theta = constants["theta"]

    L = policy[1]  # labor is policy dim 1
    Z = jnp.exp(z[0])

    y = Z * jnp.power(L, 1.0 - alpha) * jnp.power(k, alpha)
    invest = k_next - (1.0 - delta) * k
    c = jnp.maximum(y - invest, 1e-6)  # small floor for stability

    if gamma == 1.0:
        u_c = jnp.log(c)
    else:
        u_c = (jnp.power(c, 1.0 - gamma) - 1.0) / (1.0 - gamma)

    # Convex effort disutility
    v_L = psi * jnp.power(L, 1.0 + theta) / (1.0 + theta)
    return u_c - v_L


def definitions(state: Array, policy: Array, constants: Dict) -> Dict[str, Array]:
    """Shared with canonical bm_labor for diagnostic / plotting parity."""
    s = SPEC.unpack_state(state)
    p = SPEC.unpack_policy(policy)

    alpha = constants["alpha"]
    gamma = constants["gamma"]

    Z = jnp.exp(s.z)
    y = Z * jnp.power(p.L, 1.0 - alpha) * jnp.power(s.k, alpha)
    mpk = alpha * Z * jnp.power(p.L, 1.0 - alpha) * jnp.power(s.k, alpha - 1.0)
    w = (1.0 - alpha) * Z * jnp.power(p.L, -alpha) * jnp.power(s.k, alpha)

    c = (1.0 - p.sav_rate) * y
    sav = p.sav_rate * y

    u_c = jnp.power(jnp.maximum(c, 1e-12), -gamma)

    return {"Z": Z, "y": y, "mpk": mpk, "w": w, "c": c, "s": sav, "u_c": u_c}


# State is (k, z); policy is (sav_rate, L). Labor FOC = dPi/d(policy[1]).
equations = euler_from_period_return(
    period_return_fn=period_return,
    step_fn=step,
    capital_idx=0,
    exog_idx=(1,),
    n_shocks=1,
    equation_name="euler",
    intratemporal_policy_idx=(1,),
    intratemporal_equation_names=("labor_foc",),
)
