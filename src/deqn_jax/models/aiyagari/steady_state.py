"""Steady state computation for Aiyagari model."""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import Array


def steady_state(constants: Dict) -> Tuple[Array, Array]:
    """Compute approximate deterministic steady state.

    With beta*(1+r) < 1, the deterministic agent converges to the
    borrowing constraint. But with idiosyncratic risk, precautionary
    savings keep the mean capital positive. We use the aggregate K
    (implied by prices) as the reference capital level.

    The steady-state c_share at k = K_agg with a = 0:
      coh = (1+r)*K + w
      At the asymptotic savings rate (high k, no risk):
        1 - c_share = (beta * (1+r)^(1-gamma))^(1/gamma)
    """
    beta = constants["beta"]
    r = constants["r_ss"]
    w = constants["w_ss"]
    gamma = constants["gamma"]
    K = constants["K_agg"]

    k_ss = K
    a_ss = 0.0

    # Capital-stabilizing c_share: k' = k when c_share = (r*k + w) / coh
    # Not the true optimum (Euler doesn't hold exactly here since
    # beta*(1+r) < 1), but a good reference for normalization.
    coh_ss = (1 + r) * K + w
    c_share_ss = (r * K + w) / coh_ss

    return jnp.array([k_ss, a_ss]), jnp.array([c_share_ss])


def init_state(key: Array, batch_size: int, constants: Dict) -> Array:
    """Sample initial states around the aggregate capital level.

    k sampled log-normally around K_agg, a from unconditional AR(1).
    """
    K = constants["K_agg"]
    sigma_a = constants["sigma_a"]
    k_min = constants["k_min"]
    k_max = constants["k_max"]

    k_key, a_key = jax.random.split(key)

    # Log-normal around K with moderate spread
    k_init = K * jnp.exp(0.5 * jax.random.normal(k_key, (batch_size,)))
    k_init = jnp.clip(k_init, k_min + 0.01, k_max)

    # Unconditional distribution of a
    a_init = jax.random.normal(a_key, (batch_size,)) * sigma_a

    return jnp.stack([k_init, a_init], axis=1)
