"""Equilibrium equations for the stochastic Brock-Mirman model.

Residual form: raw Euler FOC,

    resid = u'(c) - beta * u'(c') * (1 + mpk' - delta)

which is MC-safe (linear in the shock-dependent quantity
``u'(c') (1 + mpk' - delta)``) and at equilibrium has
``E_eps[resid] = 0``. The framework's "average residuals over shocks,
then square" aggregation applies directly.

Why not a dimensionless ``resid / u'(c)``? It is MC-safe too, but at
bad policies that drive consumption to zero it *shrinks* the residual
magnitude (divides by ``u'(c)`` which blows up for small c), removing
the gradient pressure that would otherwise push the policy away from
the low-consumption region. Raw form keeps that pressure. For
accuracy reporting we convert to a dimensionless log10 magnitude
post-training in the evaluation module, which is the standard DEQN
diagnostic (Azinovic et al. 2022).

Why not the RHS-normalized ``1 - u'(c) / (beta E[u'(c')(1+r'-delta)])``?
That form requires computing the expectation *inside* the residual
before taking MC samples (Simon's Gauss-Hermite notebook does
this). With per-shock residual averaging the per-shock form becomes
``1 - u'(c) / (beta u'(c'(eps))(1+r'(eps)-delta))``, which is biased
under ``E[1/X] != 1/E[X]`` (Jensen). Good for deterministic or
analytic-quadrature solvers, unsafe under MC.

Parameterization note -- capital accumulation law.
   Ours:       k' = (1 - delta) * k + s * Y(k, z)       [dynamics.py]
   DEQN-MAO:   k' = sav_rate * (Y + (1 - delta) * k)    [their Variables.py]

   Expanding DEQN-MAO: k' = s*Z*k^alpha + s*(1-delta)*k, whereas ours
   has an unscaled (1 - delta)*k retention term. The two agree only at
   delta = 1 (full depreciation) or s = 1, not in general. Our form is
   the economically standard "savings rate out of net output plus
   retained undepreciated capital"; DEQN-MAO's absorbs retained capital
   into the savings decision. For the calibration used here (delta=0.1)
   the two implementations visit different ergodic distributions even
   with identical network + optimizer + shock setup, and cross-comparing
   ergodic Euler error magnitudes against DEQN-MAO's brock_mirman
   requires accounting for this difference. This port mirrors Simon's
   Day-2 notebook, which uses the standard form.
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

    euler = u_c - beta * u_c_next * (1.0 + mpk_next - delta)

    return {"euler": euler}
