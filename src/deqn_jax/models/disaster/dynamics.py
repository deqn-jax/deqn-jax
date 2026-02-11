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

    return jnp.stack([
        p.pi, defs["k"], defs["c"], p.q, p.i, defs["R"], p.w_tilda, defs["L"],
        eps_next, mu_ups_next, g_next, mu_z_next, m_p_next
    ], axis=1)


# Safety bounds for long-horizon simulation only (evaluate, irf).
# NOT used in training loss path — hard clips kill gradients.
_SIM_LOWER = jnp.array([0.8, 5.0, 0.1, 0.3, 0.1, 0.99, 0.5, 0.5, 0.7, 0.8, 0.3, 0.97, -3.0])
_SIM_UPPER = jnp.array([1.3, 80.0, 5.0, 3.0, 3.0, 1.15, 5.0, 8.0, 1.4, 1.3, 1.2, 1.04, 3.0])


def clip_state(state: Array) -> Array:
    """Clip states for simulation safety (eval/irf only, NOT training)."""
    return jnp.clip(state, _SIM_LOWER, _SIM_UPPER)
