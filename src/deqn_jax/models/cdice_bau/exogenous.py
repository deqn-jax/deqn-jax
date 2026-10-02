"""Deterministic exogenous paths of the CDICE economy, as functions of time.

Time enters the model as the state ``tau = 1 - exp(-vartheta * t)`` (t in
years since 2015), which maps the infinite horizon onto [0, 1) so the network
sees a bounded input. Every path below takes real time ``t``; ``real_time``
and ``next_tau`` convert between the two clocks. Annual time step throughout.

Growth rates (``gr_tfp``, ``gr_lab``) are the instantaneous rates d log X / dt
at t; the model uses them, not the discrete ratios X_{t+1}/X_t, in the growth
correction of the budget constraint and in the effective discount factor, as
the reference does.
"""

from typing import Dict

import jax.numpy as jnp
from jax import Array


def real_time(tau: Array, c: Dict) -> Array:
    """Years since 2015 from computational time."""
    return -jnp.log1p(-tau) / c["vartheta"]


def next_tau(tau: Array, c: Dict) -> Array:
    """Computational time one year later."""
    return -jnp.expm1(-c["vartheta"] * (real_time(tau, c) + 1.0))


def tfp(t: Array, c: Dict) -> Array:
    """Labour-augmenting productivity A_t (output per effective worker is k^alpha)."""
    g0, d = c["gA0hat"], c["deltaA"]
    return c["A0hat"] * jnp.exp(g0 * (1.0 - jnp.exp(-d * t)) / d)


def gr_tfp(t: Array, c: Dict) -> Array:
    """Instantaneous growth rate of A_t [1/year]."""
    return c["gA0hat"] * jnp.exp(-c["deltaA"] * t)


def lab(t: Array, c: Dict) -> Array:
    """World population L_t [millions]."""
    L0, Linf = c["L0"], c["Linfty"]
    return L0 + (Linf - L0) * (1.0 - jnp.exp(-c["deltaL"] * t))


def gr_lab(t: Array, c: Dict) -> Array:
    """Instantaneous growth rate of L_t [1/year]."""
    L0, Linf, d = c["L0"], c["Linfty"], c["deltaL"]
    return d / ((Linf / (Linf - L0)) * jnp.exp(d * t) - 1.0)


def sigma(t: Array, c: Dict) -> Array:
    """Carbon intensity of gross output [1000 GtC per trillion USD]."""
    ds = c["deltaSigma"]
    return c["sigma0"] * jnp.exp(c["gSigma0"] / jnp.log1p(ds) * ((1.0 + ds) ** t - 1.0))


def theta1(t: Array, c: Dict) -> Array:
    """Abatement-cost coefficient (zero abatement under BAU; reported only)."""
    return (
        c["pback"]
        * (1000.0 * c["c2co2"] * sigma(t, c))
        * jnp.exp(-c["gback"] * t)
        / c["theta2"]
    )


def eland(t: Array, c: Dict) -> Array:
    """Land-use emissions [1000 GtC / year]."""
    return c["ELand0"] * jnp.exp(-c["deltaLand"] * t)


def fex(t: Array, c: Dict) -> Array:
    """Non-CO2 radiative forcing [W m^-2]: linear until 2100, flat after.

    The reference truncates the ramp length to whole periods (``int``);
    with an annual step and Tyears = 85 that is 85 years exactly.
    """
    years = float(int(c["Tyears"]))
    return c["fex0"] + (c["fex1"] - c["fex0"]) * jnp.minimum(t, years) / years


def beta_hat(t: Array, c: Dict) -> Array:
    """Growth-adjusted discount factor exp(-rho + (1 - 1/psi) g_A + g_L)."""
    return jnp.exp(-c["rho"] + (1.0 - 1.0 / c["psi"]) * gr_tfp(t, c) + gr_lab(t, c))


def growth_factor(t: Array, c: Dict) -> Array:
    """exp(g_A + g_L): converts next-period capital to this period's effective units."""
    return jnp.exp(gr_tfp(t, c) + gr_lab(t, c))
