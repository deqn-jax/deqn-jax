"""Equilibrium equations for Aiyagari model."""

from typing import Dict

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.aiyagari.variables import SPEC

EQUATION_NAMES = ("euler",)


def definitions(state: Array, policy: Array, constants: Dict) -> Dict[str, Array]:
    """Compute derived quantities from state and policy."""
    s = SPEC.unpack_state(state)
    p = SPEC.unpack_policy(policy)

    r = constants["r_ss"]
    w = constants["w_ss"]
    gamma = constants["gamma"]

    # Cash-on-hand
    coh = (1 + r) * s.k + jnp.exp(s.a) * w

    # Consumption and savings
    c = jnp.maximum(p.c_share * coh, 1e-10)
    k_next = jnp.maximum((1 - p.c_share) * coh, constants["k_min"])

    # Marginal utility (CRRA)
    u_c = jnp.power(c, -gamma)

    return {"coh": coh, "c": c, "k_next": k_next, "u_c": u_c}


def equations(
    state: Array,
    policy: Array,
    next_state: Array,
    next_policy: Array,
    constants: Dict,
) -> Dict[str, Array]:
    """Compute equilibrium equation residuals.

    Euler equation:
        u'(c) = beta * (1 + r) * E[u'(c')]

    Note: this is the per-realization residual. The expectation over
    shocks is handled externally (by averaging over shock samples in
    the training loop or GRPO reward computation).

    At the borrowing constraint, the Euler inequality u'(c) >= beta*(1+r)*E[u'(c')]
    holds. The bounded policy (c_share in [0.01, 0.99]) enforces feasibility.
    """
    beta = constants["beta"]
    r = constants["r_ss"]

    defs = definitions(state, policy, constants)
    next_defs = definitions(next_state, next_policy, constants)

    # Normalized Euler equation residual:
    #   (u'(c) - β*(1+r)*u'(c')) / u'(c) = 1 - β*(1+r) * (c/c')^γ
    #
    # Division by u'(c) prevents scale blow-up when c → 0 (near the
    # borrowing constraint). E[residual] = 0 at the Euler optimum.
    euler = 1.0 - beta * (1 + r) * next_defs["u_c"] / defs["u_c"]

    return {"euler": euler}
