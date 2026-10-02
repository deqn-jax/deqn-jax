"""State transition and initial sampler of the Phase-1 variant ``rss_trade_ez``.

Exogenous block (writeup (34)–(35)), on the six off-diagonal pairs only: the
12 shocks are standard normals (log-volatility innovations, then tariff
innovations), and the tariff innovation is mapped through the replica's
truncated-normal transport, CDF matching of the standard normal onto
``N(mu, sigma^2)`` truncated to ``[0, inf)``. Simulation draws and
quadrature nodes go through the same ``step_fn``, so the expectation
operator integrates exactly the law the simulator samples.

Endogenous block, as in the replica: capital follows the ``K_prime`` output
(the ``law_of_motion_K`` residual ties it to the accumulation identity; see
``definitions.next_endogenous`` for why not the identity itself); the bond
follows the cleared position ``A``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.rss_trade_ez.definitions import next_endogenous
from deqn_jax.models.rss_trade_ez.variables import K_SS_REFERENCE, Layout
from deqn_jax.models.rss_trade_ez_ref.dynamics import tariff_step
from deqn_jax.models.rss_trade_ez_ref.steady_state import sample_capital


def make_step(layout: Layout):
    m = layout.n_pairs
    pairs = jnp.asarray(layout.pair_flat)

    def step(state: Array, policy: Array, shock: Array, constants) -> Array:
        tau_next, sig_next = tariff_step(
            state[:, layout.tau],
            state[:, layout.sig],
            shock[:, m : 2 * m],
            shock[:, :m],
            constants,
            pairs,
        )
        K_next, A_next = next_endogenous(state, policy, constants, layout)
        nxt = state
        nxt = nxt.at[:, layout.sig].set(sig_next)
        nxt = nxt.at[:, layout.tau].set(jnp.maximum(tau_next, 0.0))
        nxt = nxt.at[:, layout.K].set(K_next)
        nxt = nxt.at[:, layout.A].set(A_next)
        return nxt

    return step


def make_clip_state(layout: Layout):
    """Evaluation/IRF-only projection to the admissible domain (never enters
    the residual): capital floor, tariffs in ``[0, tau_cap]``."""

    def clip_state(state: Array, constants) -> Array:
        min_K = float(constants["min_K"])
        tau_cap = float(constants["tau_cap"])
        s = state.at[..., layout.K].set(jnp.maximum(state[..., layout.K], min_K))
        return s.at[..., layout.tau].set(jnp.clip(s[..., layout.tau], 0.0, tau_cap))

    return clip_state


def make_init_state(layout: Layout, k_center=K_SS_REFERENCE, jitter: float = 0.1):
    """Paths start near the calibrated economy's deterministic rest point:
    capital at the reference steady-state stocks jittered multiplicatively,
    zero net foreign assets, zero tariffs, log-volatility at its mean."""

    def init_state(key: Array, batch_size: int, constants) -> Array:
        bar_sigma = jnp.asarray(constants["bar_sigma_tau"]).reshape(-1)
        s = jnp.zeros((batch_size, layout.n_states))
        s = s.at[:, layout.K].set(
            sample_capital(key, batch_size, layout.n, k_center, jitter)
        )
        return s.at[:, layout.sig].set(bar_sigma[layout.pair_flat][None, :])

    return init_state
