"""Autodiff-synthesized equations for stochastic Brock-Mirman.

Model author writes one scalar function -- the per-period return -- and
hands it to ``euler_from_period_return``. That helper autodiffs the Euler
residual. Nothing else here.

No hand-derived FOC. No explicit ``u'(c) - beta E[u'(c')(1 + r' - delta)]``
anywhere in this file. The Euler drops out of differentiating Pi.
"""

from typing import Dict

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.brock_mirman.dynamics import step
from deqn_jax.models.brock_mirman_autodiff.variables import SPEC
from deqn_jax.training.autodiff import euler_from_period_return

EQUATION_NAMES = ("euler",)


def period_return(
    k: Array, k_next: Array, z: Array, policy: Array, constants: Dict
) -> Array:
    """Pi(K_t, K_{t+1}, z_t, policy_t) = u(C_t). Budget constraint baked in.

    Brock-Mirman absorbs the savings rate into K_next via the budget,
    so ``policy`` is unused here -- the argument is still part of the
    helper's Pi contract for multi-policy models (see bm_labor_autodiff).
    """
    del policy  # brock_mirman has no intratemporal policy dependence
    alpha = constants["alpha"]
    delta = constants["delta"]
    gamma = constants["gamma"]

    Z = jnp.exp(z[0])
    y = Z * jnp.power(k, alpha)
    invest = k_next - (1.0 - delta) * k
    c = jnp.maximum(y - invest, 1e-6)  # small floor for numerical safety

    if gamma == 1.0:
        return jnp.log(c)
    return (jnp.power(c, 1.0 - gamma) - 1.0) / (1.0 - gamma)


# One line: hand the per-period return + the model's own step_fn to the
# framework helper; get back the Euler residual equations_fn.
equations = euler_from_period_return(
    period_return_fn=period_return,
    step_fn=step,
    capital_idx=0,  # state = (k, z); capital is dim 0
    exog_idx=(1,),  # z is dim 1
    n_shocks=1,
)


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
