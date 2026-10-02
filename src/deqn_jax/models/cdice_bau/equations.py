"""Equilibrium conditions of the CDICE business-as-usual planner.

Business as usual: abatement is fixed at zero, the planner chooses savings
taking the climate externality of production into account (DICE-2016
"baseline"). In per-effective-worker units, with consumption
c = lam^(-psi) (lam is the normalized marginal utility), the conditions are

    budget:   (1 - Omega(T_AT)) k^a - c + (1 - delta) k - G_t k'          = 0
    foc_k:    G_t lam - b_t [ lam' ((1 - Omega(T_AT')) a k'^(a-1) + 1 - delta)
                              - nu_at' sigma' A' L' a k'^(a-1) ]               = 0
    foc_tat:  eta_at - b_t [ -lam' dOmega(T_AT') k'^a
                             + eta_at' (1 - c1 c3 - c1f) + eta_oc' c4 ]        = 0
    foc_mat:  -nu_at - b_t [ -nu_at' (1 - b12) + nu_uo' b12
                             + eta_at' c1 F2x / (ln 2 M_AT') ]                  = 0
    foc_muo:  nu_uo - b_t [ -nu_at' b21 + nu_uo' (1 - b21 - b23) + nu_lo' b23 ] = 0
    foc_mlo:  nu_lo - b_t [ nu_uo' b32 + nu_lo' (1 - b32) ]                     = 0
    foc_toc:  eta_oc - b_t [ eta_at' c1 c3 + eta_oc' (1 - c4) ]                 = 0

where G_t = exp(g_A + g_L), b_t = beta_hat_t, primes are next-period values,
dOmega is the derivative of the damage share Omega, and nu_at > 0 is the
sign-flipped shadow value of atmospheric carbon. This is the system the
reference code solves (``gdice_baseline/Equations.py``; its derivation is in
Online Appendix D of Folini et al. 2024) at an annual step. Residuals are in
the reference's units, so Euler error statistics compare directly.
"""

from typing import Dict

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.cdice_bau import climate, exogenous
from deqn_jax.models.cdice_bau.variables import SPEC

EQUATION_NAMES = (
    "foc_k",
    "budget",
    "foc_tat",
    "foc_mat",
    "foc_muo",
    "foc_mlo",
    "foc_toc",
)


def definitions(state: Array, policy: Array, constants: Dict) -> Dict[str, Array]:
    """Exogenous paths and economic quantities at (state, policy).

    Quantities per effective worker unless suffixed ``_level`` (trillions of
    2010 USD) or stated otherwise. ``emissions`` is total (industrial + land)
    CO2 emissions in 1000 GtC per year; ``scc`` is the social cost of carbon
    in USD per tC (divide by ``c2co2`` for USD per tCO2).
    """
    c = constants
    s = SPEC.unpack_state(state)
    p = SPEC.unpack_policy(policy)
    alpha = c["alpha"]

    t = exogenous.real_time(s.tau, c)
    a_t = exogenous.tfp(t, c)
    l_t = exogenous.lab(t, c)
    sig = exogenous.sigma(t, c)
    e_land = exogenous.eland(t, c)
    scale = a_t * l_t  # effective labour: per-effective-worker -> level

    omega = climate.damage(s.t_at, c)
    ygross = s.k**alpha
    ynet = (1.0 - omega) * ygross
    con = p.lam ** (-c["psi"])
    inv = ynet - con
    e_ind = sig * ygross * scale  # 1000 GtC / year
    emissions = e_ind + e_land

    # Social cost of carbon: -(dV/dM_AT) / (dV/dK), the reference's formula
    # at an annual step.
    q = climate.coeffs(c)
    mpk = alpha * s.k ** (alpha - 1.0)
    dv_dk = p.lam * ((1.0 - omega) * mpk + (1.0 - c["delta"])) - p.nu_at * sig * (
        scale * mpk
    )
    dv_dmat = (
        -p.nu_at * (1.0 - q.b12)
        + p.nu_uo * q.b12
        + p.eta_at * q.c1 * c["f2xco2"] / (jnp.log(2.0) * s.m_at)
    )
    scc = -dv_dmat / dv_dk * scale

    return {
        "t": t,
        "tfp": a_t,
        "gr_tfp": exogenous.gr_tfp(t, c),
        "lab": l_t,
        "gr_lab": exogenous.gr_lab(t, c),
        "sigma": sig,
        "theta1": exogenous.theta1(t, c),
        "eland": e_land,
        "fex": exogenous.fex(t, c),
        "beta_hat": exogenous.beta_hat(t, c),
        "omega": omega,
        "ygross": ygross,
        "ynet": ynet,
        "con": con,
        "inv": inv,
        "savings_rate": inv / ynet,
        "damages": omega * ygross,
        "e_ind": e_ind,
        "emissions": emissions,
        "scc": scc,
        "effective_labor": scale,
    }


def equations(
    state: Array,
    policy: Array,
    next_state: Array,
    next_policy: Array,
    constants: Dict,
) -> Dict[str, Array]:
    c = constants
    s = SPEC.unpack_state(state)
    p = SPEC.unpack_policy(policy)
    sn = SPEC.unpack_state(next_state)
    pn = SPEC.unpack_policy(next_policy)
    alpha, delta = c["alpha"], c["delta"]
    q = climate.coeffs(c)

    t = exogenous.real_time(s.tau, c)
    g = exogenous.growth_factor(t, c)
    b = exogenous.beta_hat(t, c)

    tn = exogenous.real_time(sn.tau, c)
    scale_n = exogenous.tfp(tn, c) * exogenous.lab(tn, c)
    sigma_n = exogenous.sigma(tn, c)
    omega_n = climate.damage(sn.t_at, c)
    # k' is the policy; sn.k equals it (no state clipping), the reference
    # writes the next-period terms with the policy.
    mpk_n = alpha * p.k_next ** (alpha - 1.0)

    omega = climate.damage(s.t_at, c)
    con = p.lam ** (-c["psi"])
    budget = (1.0 - omega) * s.k**alpha - con + (1.0 - delta) * s.k - g * p.k_next

    foc_k = g * p.lam - b * (
        pn.lam * ((1.0 - omega_n) * mpk_n + (1.0 - delta))
        - pn.nu_at * sigma_n * scale_n * mpk_n
    )
    foc_tat = p.eta_at - b * (
        -pn.lam * climate.damage_prime(sn.t_at, c) * p.k_next**alpha
        + pn.eta_at * (1.0 - q.c1c3 - q.c1f)
        + pn.eta_oc * q.c4
    )
    foc_mat = -p.nu_at - b * (
        -pn.nu_at * (1.0 - q.b12)
        + pn.nu_uo * q.b12
        + pn.eta_at * q.c1 * c["f2xco2"] / (jnp.log(2.0) * sn.m_at)
    )
    foc_muo = p.nu_uo - b * (
        -pn.nu_at * q.b21 + pn.nu_uo * (1.0 - q.b21 - q.b23) + pn.nu_lo * q.b23
    )
    foc_mlo = p.nu_lo - b * (pn.nu_uo * q.b32 + pn.nu_lo * (1.0 - q.b32))
    foc_toc = p.eta_oc - b * (pn.eta_at * q.c1c3 + pn.eta_oc * (1.0 - q.c4))

    return {
        "foc_k": foc_k,
        "budget": budget,
        "foc_tat": foc_tat,
        "foc_mat": foc_mat,
        "foc_muo": foc_muo,
        "foc_mlo": foc_mlo,
        "foc_toc": foc_toc,
    }
