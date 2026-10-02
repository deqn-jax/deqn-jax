"""Variables and calibration for the CDICE business-as-usual model.

Calibration: the multi-model-mean ("mmm_mmm") CDICE parameterization of
Folini, Friedl, Kübler and Scheidegger (2024), annual time step. Values are
those of the public replication package's ``gdice_baseline_mmm_mmm`` constants
(https://github.com/ClimateChangeEcon/Climate_in_Climate_Economics).

Normalizations (kept from the reference so that solutions compare one to one):

- ``k`` is capital per effective worker, K / (A_t L_t); gross output per
  effective worker is ``k ** alpha``. Levels in trillions of 2010 USD are
  recovered by multiplying with ``A_t L_t`` (``exogenous.tfp * exogenous.lab``).
- Carbon masses are in 1000 GtC; temperatures are in degrees C above
  pre-industrial.
- ``tau = 1 - exp(-vartheta * t)`` is the computational time in [0, 1); t is
  real years since 2015.

The seven policy outputs are next-period capital and the six normalized
shadow prices of the planner's problem (consumption, the three carbon
reservoirs, the two temperatures). ``nu_at`` enters every equation with a
minus sign (the shadow value of atmospheric carbon is negative), so the
network carries it as a positive number, as in the reference.
"""

import math

import jax.numpy as jnp

from deqn_jax.models.variable_spec import VariableSpec

SPEC = VariableSpec(
    state_names=("k", "m_at", "m_uo", "m_lo", "t_at", "t_oc", "tau"),
    policy_names=("k_next", "lam", "nu_at", "nu_uo", "nu_lo", "eta_at", "eta_oc"),
)

CONSTANTS = {
    # --- time --------------------------------------------------------------
    "vartheta": 0.015,  # computational-time compression, tau = 1 - exp(-vartheta t)
    # --- population [millions] ---------------------------------------------
    "L0": 7403.0,  # 2015
    "Linfty": 11500.0,  # asymptote
    "deltaL": 0.0268,  # convergence rate
    # --- labour-augmenting productivity A_t ---------------------------------
    "A0hat": 0.010295,
    "gA0hat": 0.0217,  # initial growth rate
    "deltaA": 0.005,  # decline rate of the growth rate
    # --- carbon intensity [1000 GtC per trillion USD] ----------------------
    "sigma0": 0.0000955592,
    "gSigma0": -0.0152,  # initial growth rate
    "deltaSigma": 0.001,  # decline rate of decarbonization
    # --- abatement cost (reported only: abatement is zero under BAU) -------
    "theta2": 2.6,
    "pback": 0.55,  # backstop price, thousand 2010 USD per tCO2
    "gback": 0.005,
    "c2co2": 3.666,  # tC -> tCO2
    # --- land-use emissions [1000 GtC / year] ------------------------------
    "ELand0": 0.00070922,
    "deltaLand": 0.023,
    # --- non-CO2 forcing [W m^-2], linear 2015 -> 2100, then flat ------------
    "fex0": 0.5,
    "fex1": 1.0,
    "Tyears": 85.0,
    # --- preferences and technology ----------------------------------------
    "rho": 0.015,  # pure rate of time preference (continuous time)
    "psi": 0.68965517,  # intertemporal elasticity of substitution
    "alpha": 0.3,
    "delta": 0.1,  # annual depreciation
    # --- damages: Omega(T) = pi1 T^pow1 + pi2 T^pow2 -------------------------
    "pi1": 0.0,
    "pi2": 0.00236,
    "pow1": 1.0,
    "pow2": 2.0,
    # --- CDICE multi-model-mean carbon cycle (annual rates) ----------------
    "b12": 0.054,  # atmosphere -> upper ocean
    "b23": 0.0082,  # upper ocean -> lower ocean
    "MATeq": 0.607,
    "MUOeq": 0.489,
    "MLOeq": 1.281,
    # --- CDICE multi-model-mean temperature module (annual rates) ----------
    "c1": 0.137,
    "c3": 0.73,
    "c4": 0.00689,
    "f2xco2": 3.45,  # forcing of a CO2 doubling [W m^-2]
    "t2xco2": 3.25,  # equilibrium climate sensitivity [C]
    "MATbase": 0.607,  # pre-industrial atmospheric carbon [1000 GtC]
    # --- initial state, 2015 -------------------------------------------------
    "k0": 2.926,
    "MAT0": 0.851,
    "MUO0": 0.628,
    "MLO0": 1.323,
    "TAT0": 1.1,
    "TOC0": 0.27,
}

# The reference runs: softplus on next capital, consumption marginal utility
# and the (sign-flipped) atmospheric-carbon price; the other four shadow prices
# are unbounded. A lower bound of -inf marks an unbounded (linear) output
# (networks/common._apply_bounds).
_NEG_INF = -math.inf
POLICY_LOWER = jnp.array([0.0, 0.0, 0.0, _NEG_INF, _NEG_INF, _NEG_INF, _NEG_INF])
POLICY_UPPER = None

# Deterministic: no shocks. The framework's expectation collapses to the single
# zero-width draw (as for bm_deterministic).
N_SHOCKS = 0

DESCRIPTION = (
    "CDICE business-as-usual, multi-model-mean climate "
    "(Folini, Friedl, Kübler & Scheidegger, REStud 2024; "
    "github.com/ClimateChangeEcon/Climate_in_Climate_Economics)"
)
