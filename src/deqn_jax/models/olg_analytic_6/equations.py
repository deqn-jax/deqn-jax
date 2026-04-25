"""Equilibrium equations for 6-agent OLG.

Five Euler equations, one per savings-choosing agent (h = 1..A-1):

    u'(c_t^h) = beta * E[r_{t+1} * u'(c_{t+1}^{h+1})]

Raw residual form: LHS - RHS. Under our per-shock averaging aggregation
this gives E_shock[raw] = LHS - beta * E[r' u_c(c'^{h+1})] which is zero
at equilibrium and free of the Jensen-under-reciprocal trap that the
"LHS/RHS - 1" form introduces.

The 5x per-agent dimension of the residual comes out as 5 separate
named equations (euler_h1..euler_h5) so the reweighting machinery can
balance them independently.

Timing detail: agent h at time t, who saves k^{h+1}, gets consumption
c'^{h+1} at time t+1 when they age into cohort h+1. So the Euler for
savings-choosing agent h involves c_{t+1}^{h+1}, NOT c_{t+1}^{h}.
"""

from typing import Dict

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.olg_analytic_6.variables import SPEC, A

EQUATION_NAMES = tuple(f"euler_h{h + 1}" for h in range(A - 1))


def definitions(state: Array, policy: Array, constants: Dict) -> Dict[str, Array]:
    """Shape-agnostic: works on either 1-D unbatched state (shape [n_states])
    or 2-D batched state (shape [batch, n_states]), because unpack_state
    returns named fields with matching leading dims and we never reference
    a batch axis explicitly.
    """
    s = SPEC.unpack_state(state)
    p = SPEC.unpack_policy(policy)
    alpha = constants["alpha"]
    gamma = constants["gamma"]
    labor_1 = constants["labor_1"]

    eta = s.eta
    delta = s.delta
    zero = jnp.zeros_like(s.k2)  # same shape as leaf fields

    # Per-agent arrays, stacked on the LAST axis. axis=-1 works for
    # both 0-D fields (gives shape [A]) and 1-D fields (gives [batch, A]).
    k = jnp.stack([zero, s.k2, s.k3, s.k4, s.k5, s.k6], axis=-1)

    K = jnp.sum(k, axis=-1)
    L = jnp.full_like(K, labor_1)

    r = alpha * eta * jnp.power(K, alpha - 1.0) * jnp.power(L, 1.0 - alpha) + (
        1.0 - delta
    )
    w = (1.0 - alpha) * eta * jnp.power(K, alpha) * jnp.power(L, -alpha)
    Y = eta * jnp.power(K, alpha) * jnp.power(L, 1.0 - alpha) + (1.0 - delta) * K

    # Per-agent income. Broadcasting: r has shape matching the leading
    # dims of k; k has one extra trailing axis.
    fin = k * r[..., None]
    lab = jnp.zeros_like(k).at[..., 0].set(labor_1 * w)
    inc = fin + lab

    s_full = jnp.stack([p.s1, p.s2, p.s3, p.s4, p.s5, zero], axis=-1)
    c = inc - s_full
    # Clamp c before computing u'(c). Floor at 1e-3 caps u_c at 1000 for
    # log utility -- enough to leave meaningful gradient space to the
    # Euler, while preventing the explosion that a 1e-12 floor would give
    # on untrained states. Infeasibility is handled by the definition_bounds
    # penalty on c (in __init__.py), which drives the policy out of the
    # negative-c region; the clamp here just keeps the Euler from turning
    # infeasible states into gradient noise. The true c (pre-clamp) is
    # what goes into the penalty and is reported in diagnostics.
    u_c = jnp.power(jnp.maximum(c, 1e-3), -gamma)

    # Per-agent capital (k^h) is NOT added to defs to avoid name collisions
    # with state_names = ("k2", ..., "k6") in IRF recording. Read k^h from
    # the state directly where needed; k^1 is always 0 (newborn).
    defs = {"K": K, "L": L, "r": r, "w": w, "Y": Y}
    for h in range(A):
        defs[f"inc{h + 1}"] = inc[..., h]
        defs[f"c{h + 1}"] = c[..., h]
        defs[f"u_c{h + 1}"] = u_c[..., h]

    return defs


def equations(
    state: Array,
    policy: Array,
    next_state: Array,
    next_policy: Array,
    constants: Dict,
) -> Dict[str, Array]:
    """Five Euler equations, each normalized by its own (shock-independent) LHS.

    Per-agent u'(c^h) varies by ~40x between young and old agents because
    consumption varies by ~40x at SS. Raw residuals u'(c^h) - beta E[r' u'(c'^{h+1})]
    inherit that per-agent scale, so the loss is dominated by old-agent
    equations and the optimizer starves the young-agent ones. Dividing
    each residual by its own LHS (u'(c^h)) gives a dimensionless equation:

        resid_h = 1 - beta * r_{t+1} * u'(c_{t+1}^{h+1}) / u'(c_t^h)

    MC-safe: u'(c^h) is shock-independent (current-state quantity), so
    it's a pass-through scalar -- no Jensen issue from the normalization.
    """
    beta = constants["beta"]

    defs = definitions(state, policy, constants)
    next_defs = definitions(next_state, next_policy, constants)

    r_next = next_defs["r"]

    residuals = {}
    for h in range(A - 1):
        u_c_now = defs[f"u_c{h + 1}"]
        u_c_next = next_defs[f"u_c{h + 2}"]
        residuals[f"euler_h{h + 1}"] = 1.0 - beta * r_next * u_c_next / u_c_now

    return residuals
