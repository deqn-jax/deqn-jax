"""Variable specification and constants for Disaster (NK-DSGE) model."""

import jax.numpy as jnp

from deqn_jax.models.variable_spec import VariableSpec

SPEC = VariableSpec(
    state_names=(
        "pi_lag", "k_lag", "c_lag", "q_lag", "i_lag", "R_lag",
        "w_tilda_lag", "L_lag", "eps", "mu_ups", "g", "mu_z", "m_p"
    ),
    policy_names=(
        "lambda_z", "i", "pi", "c", "w_tilda",
        "h", "F_w", "F_p", "q", "K_p", "K_w",
    ),
)

CONSTANTS = {
    # Preferences
    "beta": 0.9985, "b": 0.74, "sigma_L": 1.0, "psi_L": 0.7705,
    # Production
    "alpha": 0.4, "delta": 0.025, "kappa": 2.0, "Phi": 0.606,
    # Price/wage setting
    # NOTE: xi=0.6 is load-bearing for Blanchard-Kahn determinacy under this
    # calibration. Attempted xi=0.5 to widen the Calvo validity edge; broke
    # determinacy (14 stable eigenvalues vs expected 13). ξ cannot be lowered
    # without recalibrating other constants simultaneously.
    "lambda_f": 1.2, "lambda_w": 1.2, "xi_p": 0.6, "xi_w": 0.6,
    "iota": 0.9, "iota_w": 0.49, "iota_mu": 0.94,
    # Monetary policy
    "rho_p": 0.85, "alpha_pi": 1.5, "alpha_y": 0.36,
    # Effective lower bound on gross nominal rate (1.0 = 0% nominal floor).
    # Taylor-rule output R is soft-floored to R_lb in definitions() via a
    # high-sharpness softplus so SS distortion is negligible (~0.1% shift).
    # To disable, set R_lb very low (e.g. 0.5). Raising sharpness tightens
    # the floor at the cost of gradient transition.
    "R_lb": 1.0, "R_lb_sharpness": 500.0,
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
    # Disaster parameters
    # p_disaster: per-period disaster probability. 0 = baseline CMR (no disasters).
    # theta_disaster: capital destruction magnitude (fraction destroyed = 1 - exp(-theta)).
    # Default 0 → acts as identity, no disaster code path activates.
    "p_disaster": 0.0,
    "theta_disaster": 0.05,
}

STEADY_STATE = {
    # Numerically solved (max |residual| < 1e-7)
    "pi_lag": 1.012286, "k_lag": 27.421619, "c_lag": 1.593684, "q_lag": 1.0,
    "i_lag": 0.794711, "R_lag": 1.017964, "w_tilda_lag": 1.920219, "L_lag": 1.965713,
    "eps": 1.0, "mu_ups": 1.0, "g": 0.616, "mu_z": 1.0041, "m_p": 0.0,
    "lambda_z": 0.601828, "i": 0.794711, "pi": 1.012286, "c": 1.593684,
    "w_tilda": 1.920219, "h": 0.944029,
    "F_w": 0.885310, "F_p": 4.735931, "q": 1.0,
    "K_p": 4.831129, "K_w": 2.207212,
}

# omega_bar SS value (computed analytically from bank participation constraint)
OMEGA_BAR_SS = 0.488466

# Per-variable bounding.
#
# CRITICAL: pi upper bound is PINNED AT CALVO VALIDITY EDGE. With xi_p=0.6
# and lambda_f=1.2, the Calvo price dispersion formula
#   K_p_inner = (1 - xi_p * (pi_tilda/pi)^-5) / (1 - xi_p)
# requires pi < ~1.1*pi_tilda for K_p_inner > 0 (i.e., for the equations
# to admit a nonlinear solution). Widening pi upper is UNSAFE — network
# will enter regions where eq2a/eq2b have no valid solution, triggering
# gradient explosion through the soft_floor at 0.01.
#
# Policy:     lambda_z  i     pi    c     w_tilda  h     F_w   F_p   q     K_p   K_w
# SS:         0.602     0.795 1.012 1.594 1.920    0.944 0.885 4.736 1.000 4.78  2.18
# Bounding:   softplus  soft  sigm  soft  softplus soft  soft  soft  soft  soft  soft
POLICY_LOWER = jnp.array([0.2, 0.4,  0.95, 0.3, 1.0, 0.6, 0.3, 2.0, 0.5, 1.0, 0.5])
_inf = float("inf")
POLICY_UPPER = jnp.array([_inf, _inf, 1.1,  _inf, _inf, _inf, _inf, _inf, _inf, _inf, _inf])

N_SHOCKS = 5

DESCRIPTION = "NK-DSGE with financial frictions"
