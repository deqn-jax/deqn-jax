"""State transitions for Disaster (NK-DSGE) model."""

from typing import Dict

import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.models.disaster.equations import definitions
from deqn_jax.models.disaster.variables import SPEC


def step(state: Array, policy: Array, shock: Array, constants: Dict,
         d_disaster: float = 0.0) -> Array:
    """State transition.

    Args:
        state: Current state [batch, n_states]
        policy: Current policy [batch, n_policies]
        shock: Gaussian shock realizations [batch, n_shocks]
        constants: Model parameters
        d_disaster: Disaster indicator in [0, 1]. 0 = no disaster (default),
            1 = disaster. When d_disaster=1, the capital k entering next period
            is multiplied by exp(-theta_disaster) to represent capital
            destruction.
    """
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

    # Capital destruction in disaster branch: k_effective = k_raw * exp(-theta)
    theta = c.get("theta_disaster", 0.0)
    k_factor = jnp.exp(-theta * d_disaster)
    k_next = defs["k"] * k_factor

    next_state = jnp.stack([
        p.pi, k_next, defs["c"], p.q, p.i, defs["R"], p.w_tilda, defs["L"],
        eps_next, mu_ups_next, g_next, mu_z_next, m_p_next
    ], axis=1)

    return soft_clip_state(next_state)


# ---------- State bounds ----------
# Hard clip bounds for eval/irf simulation safety (NOT training).
#          pi    k     c     q     i     R     w_t   L     eps   mu_u  g     mu_z  m_p
_SIM_LOWER = jnp.array([0.8, 5.0, 0.1, 0.3, 0.1, 0.99, 0.5, 0.5, 0.7, 0.8, 0.3, 0.97, -3.0])
_SIM_UPPER = jnp.array([1.3, 80.0, 5.0, 3.0, 3.0, 1.15, 5.0, 8.0, 1.4, 1.3, 1.2, 1.04, 3.0])

# Soft clip bounds: very wide, margins >= 2.0 from SS for all variables.
# With k=5 softplus, margin=2.0 gives distortion < 5e-5 at SS.
# Negative lower bounds are OK for positive quantities — the dynamics keep
# states positive via exp(), so the lower softplus never activates in practice.
#              pi     k     c     q     i     R     w_t   L     eps   mu_u  g     mu_z  m_p
# SS values: 1.014  27.35  1.59  1.00  0.79  1.02  1.92  1.97  1.00  1.00  0.62  1.00  0.00
_SOFT_LOWER = jnp.array([-1.0, 1.0, -1.0, -1.5, -1.5, -1.0, -0.5, -0.5, -1.5, -1.5, -2.0, -1.5, -5.0])
_SOFT_UPPER = jnp.array([ 3.5, 100.0, 8.0,  5.0,  5.0,  3.5,  6.0, 10.0,  3.5,  3.5,  3.0,  3.5,  5.0])


def clip_state(state: Array) -> Array:
    """Hard clip for simulation safety (eval/irf only, NOT training)."""
    return jnp.clip(state, _SIM_LOWER, _SIM_UPPER)


def soft_clip_state(state: Array) -> Array:
    """Differentiable clip for episode trajectories.

    Uses chained softplus with very wide bounds: gradient attenuates
    smoothly near bounds but never reaches exactly zero. Wide bounds
    ensure < 1% distortion at SS values.
    """
    k = 5.0
    x = _SOFT_LOWER + jax.nn.softplus(k * (state - _SOFT_LOWER)) / k
    x = _SOFT_UPPER - jax.nn.softplus(k * (_SOFT_UPPER - x)) / k
    return x


def compute_state_barrier(state: Array) -> Array:
    """Box barrier penalty on states outside plausible bounds.

    Returns per-sample penalty [batch]. Added to loss with barrier_weight.
    Uses relu² (exactly zero inside bounds, quadratic outside).
    """
    over = jnp.maximum(state - _SIM_UPPER, 0.0)
    under = jnp.maximum(_SIM_LOWER - state, 0.0)
    return jnp.mean(over**2 + under**2, axis=-1)
