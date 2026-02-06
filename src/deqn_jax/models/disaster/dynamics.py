"""State transitions for Disaster (NK-DSGE) model."""

from typing import Dict

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.disaster.variables import SPEC
from deqn_jax.models.disaster.equations import definitions


def step(state: Array, policy: Array, shock: Array, constants: Dict) -> Array:
    """State transition."""
    st = SPEC.unpack_state(state)
    p = SPEC.unpack_policy(policy)
    defs = definitions(state, policy, constants)
    c = constants

    # Shocks
    eps_shock, mu_ups_shock, mu_z_shock, g_shock, mp_shock = shock[:, 0], shock[:, 1], shock[:, 2], shock[:, 3], shock[:, 4]

    safe_log = lambda x: jnp.log(jnp.maximum(x, 1e-8))

    # Evolve exogenous states
    eps_next = jnp.exp(c["rho_eps"] * safe_log(st.eps) + c["sigma_eps"] * eps_shock)
    mu_ups_next = jnp.exp(c["rho_mu_ups"] * safe_log(st.mu_ups) + c["sigma_mu_ups"] * mu_ups_shock)
    mu_z_next = jnp.exp(c["rho_mu_z"] * safe_log(st.mu_z) + (1 - c["rho_mu_z"]) * safe_log(c["mu_z_ss"]) + c["sigma_mu_z"] * mu_z_shock)
    g_next = jnp.exp(c["rho_g"] * safe_log(st.g) + (1 - c["rho_g"]) * safe_log(c["g_ss"]) + c["sigma_g"] * g_shock)
    m_p_next = c["sigma_mp"] * mp_shock

    next_state = jnp.stack([
        p.pi, defs["k"], defs["c"], p.q, p.i, defs["R"], p.w_tilda, defs["L"],
        eps_next, mu_ups_next, g_next, mu_z_next, m_p_next
    ], axis=1)

    lower = jnp.array([0.9, 10.0, 0.5, 0.5, 0.3, 1.0, 1.0, 1.0, 0.8, 0.9, 0.4, 0.99, -2.0])
    upper = jnp.array([1.2, 50.0, 3.0, 2.0, 2.0, 1.1, 3.0, 4.0, 1.2, 1.1, 0.9, 1.02, 2.0])
    return jnp.clip(next_state, lower, upper)
