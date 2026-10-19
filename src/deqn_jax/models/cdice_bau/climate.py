"""CDICE climate block: three-reservoir carbon cycle and two-layer temperature.

Annual transition (Folini et al. 2024, Section 4 and Online Appendix D):

    M_AT' = (1 - b12) M_AT + b21 M_UO + E_t
    M_UO' = b12 M_AT + (1 - b21 - b23) M_UO + b32 M_LO
    M_LO' = b23 M_UO + (1 - b32) M_LO
    T_AT' = (1 - c1 c3 - c1 F2x/T2x) T_AT + c1 c3 T_OC + c1 (F_CO2(M_AT) + F_ex,t)
    T_OC' = c4 T_AT + (1 - c4) T_OC

with mass balance b21 = b12 MATeq/MUOeq, b32 = b23 MUOeq/MLOeq and CO2 forcing
F_CO2(M) = F2x log2(M / MATbase). Forcing in T_AT' is evaluated at the current
period's carbon mass, the timing of DICE-2016 and of the reference code.
"""

from typing import Dict, NamedTuple

import jax.numpy as jnp
from jax import Array


class ClimateCoeffs(NamedTuple):
    b12: float
    b21: float
    b23: float
    b32: float
    c1: float
    c1c3: float
    c1f: float  # c1 * F2x / T2x: feedback (climate-sensitivity) term
    c4: float


def coeffs(c: Dict) -> ClimateCoeffs:
    return ClimateCoeffs(
        b12=c["b12"],
        b21=c["b12"] * c["MATeq"] / c["MUOeq"],
        b23=c["b23"],
        b32=c["b23"] * c["MUOeq"] / c["MLOeq"],
        c1=c["c1"],
        c1c3=c["c1"] * c["c3"],
        c1f=c["c1"] * c["f2xco2"] / c["t2xco2"],
        c4=c["c4"],
    )


def co2_forcing(m_at: Array, c: Dict) -> Array:
    """Radiative forcing of atmospheric CO2 [W m^-2]."""
    return c["f2xco2"] * jnp.log(m_at / c["MATbase"]) / jnp.log(2.0)


def damage(t_at: Array, c: Dict) -> Array:
    """Damage share of gross output, Omega(T_AT)."""
    return c["pi1"] * t_at ** c["pow1"] + c["pi2"] * t_at ** c["pow2"]


def damage_prime(t_at: Array, c: Dict) -> Array:
    """dOmega / dT_AT."""
    return c["pow1"] * c["pi1"] * t_at ** (c["pow1"] - 1.0) + c["pow2"] * c[
        "pi2"
    ] * t_at ** (c["pow2"] - 1.0)


def transition(
    m_at: Array,
    m_uo: Array,
    m_lo: Array,
    t_at: Array,
    t_oc: Array,
    emissions: Array,
    forcing_ex: Array,
    c: Dict,
):
    """One-year climate transition; returns (M_AT', M_UO', M_LO', T_AT', T_OC')."""
    q = coeffs(c)
    m_at_next = (1.0 - q.b12) * m_at + q.b21 * m_uo + emissions
    m_uo_next = q.b12 * m_at + (1.0 - q.b21 - q.b23) * m_uo + q.b32 * m_lo
    m_lo_next = q.b23 * m_uo + (1.0 - q.b32) * m_lo
    t_at_next = (
        (1.0 - q.c1c3 - q.c1f) * t_at
        + q.c1c3 * t_oc
        + q.c1 * (co2_forcing(m_at, c) + forcing_ex)
    )
    t_oc_next = q.c4 * t_at + (1.0 - q.c4) * t_oc
    return m_at_next, m_uo_next, m_lo_next, t_at_next, t_oc_next
