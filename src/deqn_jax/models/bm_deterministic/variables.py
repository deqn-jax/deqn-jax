"""Variable specification for the deterministic Brock-Mirman model.

Single state K, single policy sav_rate, one Euler equation, no shocks.
Calibration matches the TF/Keras reference notebook:

    alpha = 0.36,  beta = 0.99,  delta = 1.0,  gamma = 1.0

With delta = 1 (full depreciation) and log utility the optimal policy
is the constant sav_rate* = alpha * beta (Brock & Mirman 1972), which
serves as the oracle for validating the trained network across the
whole compact domain we sample from.
"""

import jax.numpy as jnp

from deqn_jax.models.variable_spec import VariableSpec

SPEC = VariableSpec(
    state_names=("k",),
    policy_names=("sav_rate",),
)

CONSTANTS = {
    "alpha": 0.36,
    "beta": 0.99,
    "gamma": 1.0,
    "delta": 1.0,
}

# sav_rate in (0, 1) enforced by sigmoid output via the MLP's bounding.
POLICY_LOWER = jnp.array([1e-6])
POLICY_UPPER = jnp.array([1.0 - 1e-6])

# Truly deterministic. The framework's MC path short-circuits to a
# single zero-width shock sample, so N_SHOCKS=0 is the right value.
N_SHOCKS = 0

# Training distribution: uniform on [K_LB, K_UB]. Same as the reference
# notebook. ``init_state_fn`` samples from this interval, and when the
# trainer is configured with ``initialize_each_episode=True`` the full
# batch is redrawn every cycle — no rollouts, no attractor collapse.
K_LB = 0.10
K_UB = 1.00

DESCRIPTION = "Deterministic Brock-Mirman (s* = alpha*beta closed form)"
