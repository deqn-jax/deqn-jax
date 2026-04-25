"""State transition for 6-agent OLG.

Agents age up: agent h at time t holds capital k^h, chooses savings s^h
= k'^{h+1}. So at t+1:

    k^{h+1}_{t+1} = s^h_t  for h = 1..A-1

Agent A dies (consumes all), agent 1 is born with k^1 = 0 (implicit; not
part of the state vector).

Shocks: eta' = eta_mid + eta_half * eps1, delta' = delta_mid + delta_half
* eps2, where eps1, eps2 are the per-sample shock values passed from the
expectation machinery. With Gauss-Hermite n=2 quadrature the eps's are
+/-1 exactly, so eta' ∈ {0.95, 1.05} and delta' ∈ {0.5, 0.9} with
uniform weights -- matching the reference's 4-state i.i.d. Markov chain.
"""

from typing import Dict

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.olg_analytic_6.variables import SPEC


def step(state: Array, policy: Array, shock: Array, constants: Dict) -> Array:
    p = SPEC.unpack_policy(policy)
    eta_mid = constants["eta_mid"]
    eta_half = constants["eta_half"]
    delta_mid = constants["delta_mid"]
    delta_half = constants["delta_half"]

    # Age-up: agent h's savings become agent (h+1)'s capital tomorrow.
    # state order: k2, k3, k4, k5, k6, eta, delta  -> next: k'^2..k'^6, eta', delta'
    k2_next = p.s1
    k3_next = p.s2
    k4_next = p.s3
    k5_next = p.s4
    k6_next = p.s5

    # Shocks: eps_eta, eps_delta; pull out robustly for 1-D or 2-D shock
    if shock.ndim > 1:
        eps_eta = shock[:, 0]
        eps_delta = shock[:, 1]
    else:
        eps_eta = shock[0] * jnp.ones_like(k2_next)
        eps_delta = shock[1] * jnp.ones_like(k2_next)

    eta_next = eta_mid + eta_half * eps_eta
    delta_next = delta_mid + delta_half * eps_delta

    return jnp.stack(
        [k2_next, k3_next, k4_next, k5_next, k6_next, eta_next, delta_next], axis=1
    )
