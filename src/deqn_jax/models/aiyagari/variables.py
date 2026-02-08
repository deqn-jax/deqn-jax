"""Variable specification and constants for Aiyagari model.

Partial-equilibrium individual savings problem:
  - State: (k, a) = (individual capital, log idiosyncratic productivity)
  - Policy: c_share (consumption share of cash-on-hand)
  - Prices r, w set at GE-consistent levels with beta*(1+r) < 1
"""

import jax.numpy as jnp

from deqn_jax.models.variable_spec import VariableSpec

SPEC = VariableSpec(
    state_names=("k", "a"),
    policy_names=("c_share",),
)

CONSTANTS = {
    "alpha": 0.34,       # Capital share
    "beta": 0.98,        # Discount factor
    "gamma": 2.0,        # Risk aversion (CRRA)
    "delta": 0.05,       # Depreciation rate
    "rho_a": 0.75,       # Productivity persistence
    "sigma_a": 0.05,     # Productivity shock std (unconditional)
    "k_min": 0.0,        # Borrowing constraint
    "k_max": 50.0,       # Capital cap (prevents overflow)
}

# Set r below 1/beta - 1 so that beta*(1+r) < 1.
# This gives a well-defined ergodic distribution in partial equilibrium.
# r = 0.01 gives beta*(1+r) = 0.9898.
_r = 0.01
_alpha = CONSTANTS["alpha"]
_delta = CONSTANTS["delta"]
_K_agg = ((_r + _delta) / _alpha) ** (1 / (_alpha - 1))
_w = (1 - _alpha) * _K_agg ** _alpha

CONSTANTS["r_ss"] = _r
CONSTANTS["w_ss"] = float(_w)
CONSTANTS["K_agg"] = float(_K_agg)

POLICY_LOWER = jnp.array([0.01])
POLICY_UPPER = jnp.array([0.99])

N_SHOCKS = 1
