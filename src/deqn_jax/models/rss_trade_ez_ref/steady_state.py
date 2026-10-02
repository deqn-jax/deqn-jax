"""Initial-state sampler for the RSS trade DSGE replica.

No closed-form steady state exists for the asymmetric three-country economy
(``steady_state_fn = None``). The sampler starts every path near the
calibrated economy's deterministic rest point: capital at the reference
steady-state stocks jittered multiplicatively, zero net foreign assets, zero
tariffs, log-volatility at its unconditional mean, and the scaffolding columns
at their converged values (``homo = homo_1 = A_min = U_store = 1``,
``a_mask = 0``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.models.rss_trade_ez_ref.variables import K_SS_REFERENCE, Layout


def sample_capital(
    key: Array, batch_size: int, n: int, k_center=K_SS_REFERENCE, jitter: float = 0.1
) -> Array:
    """``[b, n]`` capital stocks: the reference steady-state stocks jittered
    multiplicatively by ``exp(U(-jitter, jitter))``."""
    center = jnp.asarray(k_center[:n], dtype=jnp.float32)
    u = jax.random.uniform(key, (batch_size, n), minval=-jitter, maxval=jitter)
    return center[None, :] * jnp.exp(u)


def make_init_state(layout: Layout, k_center=K_SS_REFERENCE, jitter: float = 0.1):
    n = layout.n

    def init_state(key: Array, batch_size: int, constants) -> Array:
        bar_sigma = jnp.asarray(constants["bar_sigma_tau"]).reshape(-1)
        s = jnp.zeros((batch_size, layout.n_states))
        s = s.at[:, layout.K].set(sample_capital(key, batch_size, n, k_center, jitter))
        s = s.at[:, layout.sigma_tau.reshape(-1)].set(bar_sigma[None, :])
        s = s.at[:, layout.homo].set(1.0)
        s = s.at[:, layout.homo_1].set(1.0)
        s = s.at[:, layout.A_min].set(1.0)
        s = s.at[:, layout.U_store].set(1.0)
        s = s.at[:, layout.a_mask].set(0.0)
        return s

    return init_state
