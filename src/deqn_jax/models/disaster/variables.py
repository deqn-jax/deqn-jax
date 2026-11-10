"""Variable specification and constants for Disaster (NK-DSGE) model."""

import numpy as np

from deqn_jax.models.variable_spec import VariableSpec

SPEC = VariableSpec(
    state_names=(
        "pi_lag",
        "k_lag",
        "c_lag",
        "q_lag",
        "i_lag",
        "R_lag",
        "w_tilda_lag",
        "L_lag",
        "eps",
        "mu_ups",
        "g",
        "mu_z",
        "m_p",
    ),
    policy_names=(
        "lambda_z",
        "i",
        "pi",
        "c",
        "w_tilda",
        "h",
        "F_w",
        "F_p",
        "q",
        "K_p",
        "K_w",
    ),
)

CONSTANTS = {
    # Preferences
    "beta": 0.9985,
    "b": 0.74,
    "sigma_L": 1.0,
    "psi_L": 0.7705,
    # Production
    "alpha": 0.4,
    "delta": 0.025,
    "kappa": 2.0,
    "Phi": 0.606,
    # Price/wage setting
    # NOTE: xi=0.6 is load-bearing for Blanchard-Kahn determinacy under this
    # calibration. Attempted xi=0.5 to widen the Calvo validity edge; broke
    # determinacy (14 stable eigenvalues vs expected 13). ξ cannot be lowered
    # without recalibrating other constants simultaneously.
    "lambda_f": 1.2,
    "lambda_w": 1.2,
    "xi_p": 0.6,
    "xi_w": 0.6,
    "iota": 0.9,
    "iota_w": 0.49,
    "iota_mu": 0.94,
    # Monetary policy
    "rho_p": 0.85,
    "alpha_pi": 1.5,
    "alpha_y": 0.36,
    # Effective lower bound on gross nominal rate. Default 1.0 = 0% nominal,
    # the textbook ZLB. Realistic ELBs are often slightly below 1: the SNB
    # held a ~-0.75% annual policy rate for seven years (Jan 2015 - Sep 2022),
    # equivalent to ~R_lb = 0.998 at quarterly frequency.
    # Taylor-rule output R is soft-floored to R_lb in definitions() via a
    # high-sharpness softplus so SS distortion is negligible (~1e-7 at
    # sharpness=500). To disable the floor, set R_lb very low (e.g. 0.5).
    # Raising sharpness tightens the floor at the cost of sharper gradient
    # transition around R_lb.
    "R_lb": 1.0,
    "R_lb_sharpness": 500.0,
    # Taxes
    "tau_c": 0.047,
    "tau_k": 0.32,
    "tau_l": 0.24,
    # Financial frictions
    "Theta": 0.005,
    "gamma_e": 0.985,
    "w_e": 0.005,
    "sigma_omega": 0.26822,
    "mu_mon": 0.22,
    # Steady states. y_ss and R_ss are the model-consistent pair: y_ss is the
    # model's own steady-state GDP and R_ss is Fisher-consistent with pi_ss,
    # so the solved steady state lands on pi_ss. The constants of the source
    # calibration (y_ss 3.0308, R_ss 1.011678) put y_ss 0.87% above the
    # model's GDP; the Taylor output-gap term then holds solved inflation at
    # 1.0123. They stay runnable as configs/disaster_cmr_constants.yaml.
    "pi_ss": 1.006,
    "mu_z_ss": 1.0041,
    "R_ss": 1.0116421,
    "y_ss": 3.006808,
    "g_ss": 0.616,
    # Shock parameters
    "rho_eps": 0.809,
    "sigma_eps": 0.0046,
    "rho_mu_ups": 0.987,
    "sigma_mu_ups": 0.004,
    "rho_mu_z": 0.146,
    "sigma_mu_z": 0.00715,
    "rho_g": 0.94,
    "sigma_g": 0.023,
    "sigma_mp": 0.0049,
    # Disaster parameters
    # p_disaster: per-period disaster probability. 0 = baseline CMR (no disasters).
    # theta_disaster: capital destruction magnitude (fraction destroyed = 1 - exp(-theta)).
    # Default 0 → acts as identity, no disaster code path activates.
    "p_disaster": 0.0,
    "theta_disaster": 0.05,
}

STEADY_STATE = {
    # Numerically solved (max |residual| < 2e-15) at CONSTANTS, with the
    # eq5 habit soft floor at sharpness 100 (see the eq5 block in
    # equations.py). Used as the initial guess of the steady-state solver.
    # pi sits 8e-5 under pi_ss: the wedge of the ELB softplus.
    "pi_lag": 1.005918,
    "k_lag": 27.495364,
    "c_lag": 1.593960,
    "q_lag": 1.0,
    "i_lag": 0.796848,
    "R_lag": 1.011560,
    "w_tilda_lag": 1.923460,
    "L_lag": 1.965715,
    "eps": 1.0,
    "mu_ups": 1.0,
    "g": 0.616,
    "mu_z": 1.0041,
    "m_p": 0.0,
    "lambda_z": 0.601724,
    "i": 0.796848,
    "pi": 1.005918,
    "c": 1.593960,
    "w_tilda": 1.923460,
    "h": 0.943405,
    "F_w": 0.865186,
    "F_p": 4.535149,
    "q": 1.0,
    "K_p": 4.534645,
    "K_w": 2.070384,
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
#
# numpy float64, not jnp: a jnp array built here takes the precision in force
# at import, so an fp64 run imported before x64 would bound at float32 values.
POLICY_LOWER = np.array([0.2, 0.4, 0.95, 0.3, 1.0, 0.6, 0.3, 2.0, 0.5, 1.0, 0.5])
_inf = float("inf")
POLICY_UPPER = np.array(
    [_inf, _inf, 1.1, _inf, _inf, _inf, _inf, _inf, _inf, _inf, _inf]
)

N_SHOCKS = 5

DESCRIPTION = "NK-DSGE with financial frictions"
