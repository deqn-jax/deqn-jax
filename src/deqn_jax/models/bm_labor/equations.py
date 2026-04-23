"""Equilibrium equations for Brock-Mirman with endogenous labor.

Two residuals:

    euler     = u'(c) - beta * u'(c') * (1 + mpk' - delta)
    labor_foc = psi * L^theta - w * u'(c)

Both in raw residual form. MC-safe (residuals are linear in shock-
dependent quantities), and free of the u_c-division trap that the
LHS-normalized dimensionless form introduces for Euler. The labor
FOC is intratemporal, so shock structure is irrelevant -- any
consistent form is fine; raw keeps magnitudes commensurable with the
Euler residual for shared optimizer dynamics.
"""

from typing import Dict

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.bm_labor.variables import SPEC

EQUATION_NAMES = ("euler", "labor_foc")


def definitions(state: Array, policy: Array, constants: Dict) -> Dict[str, Array]:
    """Derived quantities from state and policy."""
    s = SPEC.unpack_state(state)
    p = SPEC.unpack_policy(policy)

    alpha = constants["alpha"]
    gamma = constants["gamma"]

    Z = jnp.exp(s.z)
    # Cobb-Douglas: Y = Z * L^(1-alpha) * K^alpha
    y = Z * jnp.power(p.L, 1.0 - alpha) * jnp.power(s.k, alpha)

    # Factor prices
    mpk = alpha * Z * jnp.power(p.L, 1.0 - alpha) * jnp.power(s.k, alpha - 1.0)
    w = (1.0 - alpha) * Z * jnp.power(p.L, -alpha) * jnp.power(s.k, alpha)

    # Consumption via savings rate out of output
    c = (1.0 - p.sav_rate) * y
    sav = p.sav_rate * y

    u_c = jnp.power(c, -gamma)

    return {"Z": Z, "y": y, "mpk": mpk, "w": w, "c": c, "s": sav, "u_c": u_c}


def equations(
    state: Array,
    policy: Array,
    next_state: Array,
    next_policy: Array,
    constants: Dict,
) -> Dict[str, Array]:
    beta = constants["beta"]
    delta = constants["delta"]
    psi = constants["psi"]
    theta = constants["theta"]

    p = SPEC.unpack_policy(policy)
    defs = definitions(state, policy, constants)
    next_defs = definitions(next_state, next_policy, constants)

    u_c = defs["u_c"]
    w = defs["w"]
    u_c_next = next_defs["u_c"]
    mpk_next = next_defs["mpk"]

    euler = u_c - beta * u_c_next * (1.0 + mpk_next - delta)
    labor_foc = psi * jnp.power(p.L, theta) - w * u_c

    return {"euler": euler, "labor_foc": labor_foc}
