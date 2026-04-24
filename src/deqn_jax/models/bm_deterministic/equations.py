"""Euler residual for the deterministic Brock-Mirman model.

We use the dimensionless relative Euler error from the reference:

    rel_ee = 1 - (C_{t+1} / C_t)^gamma / (beta * (R_{t+1} + 1 - delta))

At equilibrium the Euler FOC is (C_{t+1}/C_t)^gamma = beta (R' + 1 - delta),
so rel_ee = 0. Using this form (rather than raw u'(c) units) keeps the
loss magnitude consistent across the sampling range -- the MSE then
weights states near the edges of the domain similarly to states near
the steady state, which is what we want for a pointwise solve.

For gamma = 1 the consumption-growth factor is just C_{t+1}/C_t.
"""

from typing import Dict

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.bm_deterministic.variables import SPEC

EQUATION_NAMES = ("euler",)


def definitions(state: Array, policy: Array, constants: Dict) -> Dict[str, Array]:
    s = SPEC.unpack_state(state)
    p = SPEC.unpack_policy(policy)

    alpha = constants["alpha"]
    delta = constants["delta"]

    y = jnp.power(s.k, alpha)               # gross output (delta=1 so no undepreciated stock in the resource constraint)
    mpk = alpha * jnp.power(s.k, alpha - 1)
    c = (1.0 - p.sav_rate) * y
    sav = p.sav_rate * y

    return {"y": y, "mpk": mpk, "c": c, "s": sav}


def equations(
    state: Array,
    policy: Array,
    next_state: Array,
    next_policy: Array,
    constants: Dict,
) -> Dict[str, Array]:
    beta = constants["beta"]
    gamma = constants["gamma"]
    delta = constants["delta"]

    defs = definitions(state, policy, constants)
    next_defs = definitions(next_state, next_policy, constants)

    c = defs["c"]
    c_next = next_defs["c"]
    r_next = next_defs["mpk"]

    consumption_growth = jnp.power(c_next / c, gamma)
    discount = beta * (r_next + 1.0 - delta)
    rel_ee = 1.0 - consumption_growth / discount

    return {"euler": rel_ee}
