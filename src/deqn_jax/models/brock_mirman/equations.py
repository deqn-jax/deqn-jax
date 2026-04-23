"""Equilibrium equations for the stochastic Brock-Mirman model.

Residual form: LHS-normalized dimensionless Euler error,

    resid = 1 - beta * u'(c') * (1 + mpk' - delta) / u'(c)

which equals ``(raw_residual) / u'(c)``. This is MC-compatible:
``E_eps[resid]`` is linear in the shock-dependent quantity
``u'(c'(eps)) * (1 + mpk'(eps) - delta)`` and so the framework's
"average residuals over shocks, then square" aggregation gives the
correct stochastic Euler loss. The alternative ``1 - (c'/c)^gamma /
(beta (1 + r' - delta))`` form is also dimensionless but introduces
the shock-dependent ``(1 + r' - delta)`` *in a denominator*, which
breaks under MC averaging via Jensen's inequality. That form is fine
for purely deterministic models but not here.

For gamma=1 (log utility) this reduces to
``1 - beta * (c / c') * (1 + mpk' - delta)``, matching the reference
notebook's ``errREE`` after algebraically inverting its inner
expectation.
"""

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

    Z = jnp.exp(s.z)
    y = Z * jnp.power(s.k, alpha)
    mpk = alpha * Z * jnp.power(s.k, alpha - 1)

    c = (1 - p.sav_rate) * y
    sav = p.sav_rate * y

    u_c = jnp.power(c, -gamma)

    return {"Z": Z, "y": y, "mpk": mpk, "c": c, "s": sav, "u_c": u_c}


def equations(
    state: Array,
    policy: Array,
    next_state: Array,
    next_policy: Array,
    constants: Dict,
) -> Dict[str, Array]:
    beta = constants["beta"]
    delta = constants["delta"]

    defs = definitions(state, policy, constants)
    next_defs = definitions(next_state, next_policy, constants)

    u_c = defs["u_c"]
    u_c_next = next_defs["u_c"]
    mpk_next = next_defs["mpk"]

    euler = 1.0 - beta * u_c_next * (1.0 + mpk_next - delta) / u_c

    return {"euler": euler}
