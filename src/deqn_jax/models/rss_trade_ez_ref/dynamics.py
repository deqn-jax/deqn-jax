"""State transition of the RSS trade DSGE (writeup §3 plus the endogenous states).

One measure for simulation and quadrature: the shock vector is ``2 n²``
standard normals (log-volatility innovations first, then tariff innovations,
importer-major), and the tariff step maps its innovation through the
truncated-normal transport ``T(z; mu, sigma) = mu + sigma Phi^-1(Phi(a) +
(1 - Phi(a)) Phi(z))``, ``a = (0 - mu)/sigma`` — CDF matching of the standard
normal onto ``N(mu, sigma^2)`` truncated to ``[0, inf)`` — so tariffs never
go negative and the expectation operator integrates the same law the
simulator draws from.

Endogenous states advance from the policy (``K``, ``A``, ``U_store``), as
in the reference (the law-of-motion residual enforces the accumulation
identity). The scaffolding columns (``homo``, ``homo_1``, ``A_min``,
``a_mask``) are constants of the replica and pass through unchanged.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.scipy.special import erf, erfinv

from deqn_jax.models.rss_trade_ez_ref.variables import Layout

_SQRT2 = 1.4142135623730951


def normal_cdf(x: Array) -> Array:
    return 0.5 * (1.0 + erf(x / _SQRT2))


def normal_ppf(u: Array) -> Array:
    u = jnp.clip(u, 1e-7, 1.0 - 1e-7)
    return _SQRT2 * erfinv(2.0 * u - 1.0)


def transport_truncated_normal(
    z: Array, mu: Array, sigma: Array, low: float = 0.0
) -> Array:
    """Map a standard-normal draw/node ``z`` to ``N(mu, sigma^2)`` truncated to
    ``[low, inf)`` by CDF matching. Monotone in ``z``, differentiable, and
    exactly the truncated law when ``z ~ N(0, 1)``."""
    a = (low - mu) / (sigma + 1e-10)
    phi_a = normal_cdf(a)
    u = phi_a + (1.0 - phi_a) * normal_cdf(z)
    u = jnp.clip(u, phi_a + 1e-7, 1.0 - 1e-7)
    return mu + sigma * normal_ppf(u)


def make_step(layout: Layout):
    n = layout.n
    n_pairs = n * n
    tau_idx = layout.tau.reshape(-1)
    sig_idx = layout.sigma_tau.reshape(-1)

    def step(state: Array, policy: Array, shock: Array, constants) -> Array:
        rho_s = jnp.asarray(constants["rho_sigma_tau"]).reshape(-1)
        bar_s = jnp.asarray(constants["bar_sigma_tau"]).reshape(-1)
        sig_s = jnp.asarray(constants["sigma_sigma_tau"]).reshape(-1)
        rho_t = jnp.asarray(constants["rho_tau"]).reshape(-1)
        bar_t = jnp.asarray(constants["bar_tau"]).reshape(-1)
        offdiag = (~jnp.eye(n, dtype=bool)).reshape(-1).astype(state.dtype)

        eps_sigma = shock[:, :n_pairs]
        eps_tau = shock[:, n_pairs : 2 * n_pairs]
        sig = state[:, sig_idx]
        tau = state[:, tau_idx]

        # (35) log-volatility AR(1); diagonal parameters are zero -> stays 0
        sig_next = (1.0 - rho_s) * bar_s + rho_s * sig + sig_s * eps_sigma
        # (34) tariff level: AR(1) mean with the transported innovation
        mu_tau = (1.0 - rho_t) * bar_t + rho_t * tau
        tau_next = transport_truncated_normal(eps_tau, mu_tau, jnp.exp(sig_next))
        tau_next = jnp.maximum(tau_next * offdiag, 0.0)

        nxt = state
        nxt = nxt.at[:, sig_idx].set(sig_next)
        nxt = nxt.at[:, tau_idx].set(tau_next)
        nxt = nxt.at[:, layout.K].set(policy[:, layout.blocks["K"]])
        nxt = nxt.at[:, layout.A].set(policy[:, layout.blocks["A"]])
        nxt = nxt.at[:, layout.U_store].set(policy[:, layout.blocks["U_store"]])
        return nxt

    return step


def make_clip_state(layout: Layout):
    """Evaluation/IRF-only projection to the admissible domain (never enters
    the residual): capital floor, tariffs in [0, tau_cap], scaffolding in [0, 1],
    accumulators non-negative."""
    tau_idx = layout.tau.reshape(-1)
    scaff = jnp.array([layout.homo, layout.homo_1, layout.A_min, layout.a_mask])

    def clip_state(state: Array, constants=None) -> Array:
        min_K = 0.005 if constants is None else float(constants["min_K"])
        tau_cap = 2.5 if constants is None else float(constants["tau_cap"])
        s = state
        s = s.at[..., layout.K].set(jnp.maximum(s[..., layout.K], min_K))
        s = s.at[..., tau_idx].set(jnp.clip(s[..., tau_idx], 0.0, tau_cap))
        s = s.at[..., scaff].set(jnp.clip(s[..., scaff], 0.0, 1.0))
        s = s.at[..., layout.U_store].set(jnp.maximum(s[..., layout.U_store], 0.0))
        return s

    return clip_state
