"""Variable specification and constants for Disaster (NK-DSGE) model."""

import jax.numpy as jnp

from deqn_jax.models.variable_spec import VariableSpec

SPEC = VariableSpec(
    state_names=(
        "pi_lag", "k_lag", "c_lag", "q_lag", "i_lag", "R_lag",
        "w_tilda_lag", "L_lag", "eps", "mu_ups", "g", "mu_z", "m_p"
    ),
    policy_names=(
        "lambda_z", "i", "pi", "w_tilda",
        "h", "F_w", "F_p", "q"
    ),
)

CONSTANTS = {
    # Preferences
    "beta": 0.9985, "b": 0.74, "sigma_L": 1.0, "psi_L": 0.7705,
    # Production
    "alpha": 0.4, "delta": 0.025, "kappa": 2.0, "Phi": 0.606,
    # Price/wage setting
    "lambda_f": 1.2, "lambda_w": 1.2, "xi_p": 0.6, "xi_w": 0.6,
    "iota": 0.9, "iota_w": 0.49, "iota_mu": 0.94,
    # Monetary policy
    "rho_p": 0.85, "alpha_pi": 1.5, "alpha_y": 0.36,
    # Taxes
    "tau_c": 0.047, "tau_k": 0.32, "tau_l": 0.24,
    # Financial frictions
    "Theta": 0.005, "gamma_e": 0.985, "w_e": 0.005,
    "sigma_omega": 0.26822, "mu_mon": 0.22,
    # Steady states
    "pi_ss": 1.006, "mu_z_ss": 1.0041, "R_ss": 1.011678,
    "y_ss": 3.0308, "g_ss": 0.616,
    # Shock parameters
    "rho_eps": 0.809, "sigma_eps": 0.0046,
    "rho_mu_ups": 0.987, "sigma_mu_ups": 0.004,
    "rho_mu_z": 0.146, "sigma_mu_z": 0.00715,
    "rho_g": 0.94, "sigma_g": 0.023,
    "sigma_mp": 0.0049,
}

STEADY_STATE = {
    # Numerically solved (max |residual| < 1e-7)
    "pi_lag": 1.012286, "k_lag": 27.421619, "c_lag": 1.593684, "q_lag": 1.0,
    "i_lag": 0.794711, "R_lag": 1.017964, "w_tilda_lag": 1.920219, "L_lag": 1.965713,
    "eps": 1.0, "mu_ups": 1.0, "g": 0.616, "mu_z": 1.0041, "m_p": 0.0,
    "lambda_z": 0.601828, "i": 0.794711, "pi": 1.012286,
    "w_tilda": 1.920219, "h": 0.944029,
    "F_w": 0.885310, "F_p": 4.735931, "q": 1.0,
}

# omega_bar SS value (computed analytically from bank participation constraint)
OMEGA_BAR_SS = 0.488466

# Per-variable bounding — tight to prevent pathological local minima.
# Lower bounds raised close to SS; upper bounds on pi to prevent blowup.
# omega_bar eliminated analytically (Newton solver on bank participation constraint).
#
# Policy:     lambda_z  i     pi    w_tilda  h     F_w   F_p   q
# SS:         0.602     0.795 1.012 1.920    0.944 0.885 4.736 1.000
# Bounding:   softplus  soft  sigm  softplus soft  soft  soft  softplus
POLICY_LOWER = jnp.array([0.2, 0.4,  0.95, 1.0, 0.6, 0.3, 2.0, 0.5])
_inf = float("inf")
POLICY_UPPER = jnp.array([_inf, _inf, 1.1,  _inf, _inf, _inf, _inf, _inf])

N_SHOCKS = 5

DESCRIPTION = "NK-DSGE with financial frictions"
