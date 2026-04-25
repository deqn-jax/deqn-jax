"""Variable specification and constants for Brock-Mirman model."""

import jax.numpy as jnp

from deqn_jax.models.variable_spec import VariableSpec

SPEC = VariableSpec(
    state_names=("k", "z"),
    policy_names=("sav_rate",),
)

CONSTANTS = {
    "alpha": 0.36,  # Capital share
    "beta": 0.99,  # Discount factor
    "gamma": 1.0,  # Risk aversion (CRRA; gamma=1 is log utility)
    "delta": 0.1,  # Depreciation rate
    "rho_z": 0.9,  # TFP persistence (AR(1) on log TFP)
    "sigma_z": 0.04,  # TFP shock std
}

POLICY_LOWER = jnp.array([1e-6])
POLICY_UPPER = jnp.array([1 - 1e-6])

N_SHOCKS = 1

DESCRIPTION = "Brock-Mirman (1972) optimal growth model"
