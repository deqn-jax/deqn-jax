"""Equilibrium equations for Brock-Mirman with endogenous labor under an upper labor cap.

Geneva 2026 course, Day 2 Exercise 3.

The capital Euler is unchanged from ``bm_labor``. The labor optimality condition
becomes a KKT/complementarity problem for the cap ``L <= L_max``, written as a single
smooth Fischer-Burmeister residual:

    fb(slack, wedge) = 0,   slack = L_max - L,   wedge = w*u'(c) - psi*L^theta

  - ``slack >= 0``  : the cap is respected (also guaranteed by the L output bound).
  - ``wedge >= 0``  : marginal benefit of labor >= marginal cost.
  - ``slack * wedge = 0`` : either interior (L < L_max and the FOC ``psi*L^theta =
    w*u'(c)`` holds, wedge = 0) or binding (L = L_max with a positive wedge).

Why FB rather than the plain FOC ``psi*L^theta - w*u'(c)``: when the cap binds, the
plain FOC residual is strictly negative and can never reach zero, so MSE training
fights an unsatisfiable target. ``fb`` makes the binding corner a genuine zero of the
residual, so the loss can converge there.
"""

from typing import Dict

import jax.numpy as jnp
from jax import Array

from deqn_jax.models._complementarity import fischer_burmeister
from deqn_jax.models.bm_labor.equations import definitions
from deqn_jax.models.bm_labor_constrained.variables import SPEC

EQUATION_NAMES = ("euler", "labor_foc")


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
    L_max = constants["L_max"]

    p = SPEC.unpack_policy(policy)
    defs = definitions(state, policy, constants)
    next_defs = definitions(next_state, next_policy, constants)

    u_c = defs["u_c"]
    w = defs["w"]
    u_c_next = next_defs["u_c"]
    mpk_next = next_defs["mpk"]

    euler = u_c - beta * u_c_next * (1.0 + mpk_next - delta)

    # KKT for L <= L_max as a Fischer-Burmeister residual.
    #   slack a = L_max - L  >= 0
    #   wedge b = w*u_c/(psi*L^theta) - 1  >= 0   (ratio form, matching the Geneva
    #     Day 2 Ex 3 notebook: interior => labor FOC holds => b = 0; binding =>
    #     marginal benefit exceeds cost => b > 0). L is bounded >= 1e-6 so the
    #     denominator is safe.
    slack = L_max - p.L
    wedge = w * u_c / (psi * jnp.power(p.L, theta)) - 1.0
    labor_foc = fischer_burmeister(slack, wedge)

    return {"euler": euler, "labor_foc": labor_foc}
