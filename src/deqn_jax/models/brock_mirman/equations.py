"""Equilibrium equations for Brock-Mirman model."""

from typing import Dict

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.brock_mirman.variables import SPEC

EQUATION_NAMES = ("euler",)


def definitions(state: Array, policy: Array, constants: Dict) -> Dict[str, Array]:
    """Compute derived quantities from state and policy."""
    s = SPEC.unpack_state(state)
    p = SPEC.unpack_policy(policy)

    alpha = constants["alpha"]
    gamma = constants["gamma"]

    # TFP level
    Z = jnp.exp(s.z)

    # Production: y = Z * k^alpha
    y = Z * jnp.power(s.k, alpha)

    # Marginal product of capital
    mpk = alpha * Z * jnp.power(s.k, alpha - 1)

    # Consumption and savings
    c = (1 - p.sav_rate) * y
    sav = p.sav_rate * y

    # Marginal utility (CRRA)
    u_c = jnp.power(c, -gamma)

    return {"Z": Z, "y": y, "mpk": mpk, "c": c, "s": sav, "u_c": u_c}


def equations(
    state: Array,
    policy: Array,
    next_state: Array,
    next_policy: Array,
    constants: Dict,
) -> Dict[str, Array]:
    """Compute equilibrium equation residuals.

    Euler equation:
        u'(c) = beta * E[u'(c') * (1 + mpk' - delta)]
    """
    beta = constants["beta"]
    delta = constants["delta"]

    defs = definitions(state, policy, constants)
    next_defs = definitions(next_state, next_policy, constants)

    # Euler equation residual
    euler = defs["u_c"] - beta * next_defs["u_c"] * (1 + next_defs["mpk"] - delta)

    return {"euler": euler}
