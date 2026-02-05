"""Variable specification and constants for Brock-Mirman model."""

import jax.numpy as jnp

from deqn_jax.models.variable_spec import VariableSpec

SPEC = VariableSpec(
    state_names=("k", "z"),
    policy_names=("sav_rate",),
)

CONSTANTS = {
    "alpha": 1 / 3,      # Capital share
    "beta": 0.95,        # Discount factor
    "gamma": 2.0,        # Risk aversion (CRRA)
    "delta": 0.1,        # Depreciation rate
    "rho_z": 0.8,        # TFP persistence
    "sigma_z": 0.03,     # TFP shock std
}

POLICY_LOWER = jnp.array([1e-6])
POLICY_UPPER = jnp.array([1 - 1e-6])

N_SHOCKS = 1

DESCRIPTION = "Brock-Mirman (1972) optimal growth model"
