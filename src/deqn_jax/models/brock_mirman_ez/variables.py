"""Variable specification and constants for Brock-Mirman + Epstein-Zin.

Same state/policy shape as the canonical Brock-Mirman model. The value
function V(s) is *not* a policy — it comes from the actor-critic value
head (shared trunk's ``value_head`` or a standalone critic in
``aux_params``). The model author writes equations that reference
``value_now``, ``value_next`` — the framework supplies them.

Calibration defaults: ψ=1.5 (IES > 1), γ_ez=5.0 (RRA > 1). With ψ ≠
γ_ez we are in the genuinely-recursive Epstein-Zin regime; the
discounted-utility / CRRA recovery happens at ψ = γ_ez.
"""

import jax.numpy as jnp

from deqn_jax.models.variable_spec import VariableSpec

SPEC = VariableSpec(
    state_names=("k", "z"),
    policy_names=("sav_rate",),
)

CONSTANTS = {
    "alpha": 0.36,  # Capital share
    "beta": 0.99,  # Discount factor
    "delta": 0.1,  # Depreciation rate
    "rho_z": 0.9,  # TFP persistence
    "sigma_z": 0.04,  # TFP shock std
    "psi": 1.5,  # IES (intertemporal elasticity of substitution)
    "gamma_ez": 5.0,  # Risk aversion (Epstein-Zin)
}

POLICY_LOWER = jnp.array([1e-6])
POLICY_UPPER = jnp.array([1 - 1e-6])

N_SHOCKS = 1

DESCRIPTION = "Brock-Mirman with Epstein-Zin recursive utility (actor-critic demo)"
