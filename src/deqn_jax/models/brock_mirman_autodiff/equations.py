"""Autodiff-synthesized Euler for stochastic Brock-Mirman.

Proof of concept for the "no pen-and-paper FOCs" path. The user supplies
a per-period return function

    Pi(K_t, K_{t+1}, z_t, constants) = u(C_t(K_t, K_{t+1}, z_t))

where consumption is implied by the budget constraint
``C_t = Y(K_t, z_t) - (K_{t+1} - (1 - delta) K_t)``. The framework then
synthesizes the capital Euler residual via two ``jax.grad`` calls:

    euler = dPi/dK_{t+1} |_{(K_t, K_{t+1}, z_t)}
          + beta * dPi/dK_t |_{(K_{t+1}, K_{t+2}, z_{t+1})}

The first term is how today's consumption responds to the choice of
tomorrow's capital (negative -- investing costs consumption). The second
is how tomorrow's consumption responds to tomorrow's incoming capital
(positive -- more capital gives more income). At the optimum they
balance in expectation: ``0 = -u'(c) + beta E[u'(c') r']``.

K_{t+2} is needed only to *evaluate* Pi at t+1; we reconstruct it from
next_state + next_policy using the model's own capital-accumulation law.
No free parameters: if the user provides Pi, Y, and the law of motion,
the Euler residual is mechanical.

Match check: the closed form of Pi with log utility + Cobb-Douglas gives
    dPi/dK' = -u'(C) = -1/C
    dPi/dK  = u'(C) * (1 + r - delta)
so the synthesized residual is
    -1/C_t + beta * (1 + r_{t+1} - delta) / C_{t+1}
which is exactly the hand-derived raw Euler from the canonical
brock_mirman model (sign-flipped, so equivalent up to sign). We expose
the negated form here to match the "u'(c) - beta E[u'(c')(1+r-delta)]"
sign convention already used elsewhere in the framework.
"""

from typing import Dict

import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.models.brock_mirman_autodiff.variables import SPEC


EQUATION_NAMES = ("euler",)


def period_return(k: Array, k_next: Array, z: Array, constants: Dict) -> Array:
    """Scalar per-period return Pi(K_t, K_{t+1}, z_t) = u(C_t).

    Pure function of primitives (production, budget, utility) -- no FOCs,
    no hand-derivation. This is the only object the user has to write.
    """
    alpha = constants["alpha"]
    delta = constants["delta"]
    gamma = constants["gamma"]

    Z = jnp.exp(z)
    y = Z * jnp.power(k, alpha)
    invest = k_next - (1.0 - delta) * k
    c = y - invest

    # Small floor for numerical safety; identical to the hand-derived model.
    c = jnp.maximum(c, 1e-6)

    if gamma == 1.0:
        return jnp.log(c)
    return (jnp.power(c, 1.0 - gamma) - 1.0) / (1.0 - gamma)


# Autodiff once, up front. These are pure functions; JAX traces them as
# needed when called inside vmap.
_dPi_dK_next = jax.grad(period_return, argnums=1)   # derivative wrt K_{t+1}
_dPi_dK      = jax.grad(period_return, argnums=0)   # derivative wrt K_t


def definitions(state: Array, policy: Array, constants: Dict) -> Dict[str, Array]:
    """Shared with canonical brock_mirman for diagnostic / plotting parity."""
    s = SPEC.unpack_state(state)
    p = SPEC.unpack_policy(policy)

    alpha = constants["alpha"]
    gamma = constants["gamma"]

    Z = jnp.exp(s.z)
    y = Z * jnp.power(s.k, alpha)
    mpk = alpha * Z * jnp.power(s.k, alpha - 1)

    c = (1 - p.sav_rate) * y
    sav = p.sav_rate * y

    u_c = jnp.power(jnp.maximum(c, 1e-12), -gamma)

    return {"Z": Z, "y": y, "mpk": mpk, "c": c, "s": sav, "u_c": u_c}


def equations(
    state: Array,
    policy: Array,
    next_state: Array,
    next_policy: Array,
    constants: Dict,
) -> Dict[str, Array]:
    """Euler residual, synthesized by autodiffing period_return.

    Reconstructs K_{t+2} from next_state + next_policy (using the
    same capital-accumulation law the model's step_fn uses), so Pi
    at t+1 can be evaluated as Pi(K_{t+1}, K_{t+2}, z_{t+1}).
    """
    beta = constants["beta"]
    alpha = constants["alpha"]
    delta = constants["delta"]

    s = SPEC.unpack_state(state)
    s_next = SPEC.unpack_state(next_state)
    p_next = SPEC.unpack_policy(next_policy)

    # K_{t+2} from (K_{t+1}, z_{t+1}, sav_rate_{t+1}) using the same
    # law of motion as dynamics.step. Keeps the autodiff version
    # truly pen-and-paper-free -- the investment formula is the
    # only place the model's dynamics show up in the Euler.
    Z_next = jnp.exp(s_next.z)
    y_next = Z_next * jnp.power(s_next.k, alpha)
    k_nextnext = (1.0 - delta) * s_next.k + p_next.sav_rate * y_next

    # Evaluate the two partials per batch element.
    dPi_dK_next_now = jax.vmap(_dPi_dK_next, in_axes=(0, 0, 0, None))(
        s.k, s_next.k, s.z, constants,
    )
    dPi_dK_at_next = jax.vmap(_dPi_dK, in_axes=(0, 0, 0, None))(
        s_next.k, k_nextnext, s_next.z, constants,
    )

    # Envelope-theorem Euler: at the optimum, today's marginal cost of
    # saving + tomorrow's expected marginal benefit of having that
    # capital = 0. We negate so the residual sign matches the
    # u_c - beta E[u_c' (1 + r - delta)] convention used in the canonical
    # brock_mirman model.
    euler = -(dPi_dK_next_now + beta * dPi_dK_at_next)

    return {"euler": euler}
