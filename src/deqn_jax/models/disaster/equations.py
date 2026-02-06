"""Equilibrium equations for Disaster (NK-DSGE) model."""

from typing import Dict

import jax.numpy as jnp
from jax import Array
from jax.scipy.special import erf

from deqn_jax.models.disaster.variables import SPEC

EQUATION_NAMES = (
    "eq1_price_phillips_F", "eq2_price_phillips_K", "eq3_wage_phillips_F",
    "eq4_wage_phillips_K", "eq5_consumption_euler", "eq6_bond_euler",
    "eq7_investment_euler", "eq8_bank_participation", "eq9_entrepreneur_contract",
)


# Financial friction helpers
def normal_cdf(x: Array) -> Array:
    return 0.5 * (1.0 + erf(x / jnp.sqrt(2.0)))

def normal_pdf(x: Array) -> Array:
    return jnp.exp(-0.5 * x ** 2) / jnp.sqrt(2.0 * jnp.pi)

def F_omega(omega_bar: Array, sigma: float) -> Array:
    """Default probability."""
    z = (jnp.log(omega_bar) + 0.5 * sigma ** 2) / sigma
    return normal_cdf(z)

def G_omega(omega_bar: Array, sigma: float) -> Array:
    """Expected value conditional on default."""
    z = (jnp.log(omega_bar) - 0.5 * sigma ** 2) / sigma
    return normal_cdf(z)

def G_omega_prime(omega_bar: Array, sigma: float) -> Array:
    z = (jnp.log(omega_bar) - 0.5 * sigma ** 2) / sigma
    return normal_pdf(z) / (sigma * omega_bar)

def Gamma(omega_bar: Array, sigma: float) -> Array:
    return omega_bar * (1.0 - F_omega(omega_bar, sigma)) + G_omega(omega_bar, sigma)

def Gamma_prime(omega_bar: Array, sigma: float) -> Array:
    return 1.0 - F_omega(omega_bar, sigma)

def S_adj(ratio: Array, mu_z_ss: float, kappa: float) -> Array:
    return 0.5 * kappa * (ratio - mu_z_ss) ** 2

def S_adj_prime(ratio: Array, mu_z_ss: float, kappa: float) -> Array:
    return kappa * (ratio - mu_z_ss)


def definitions(state: Array, policy: Array, constants: Dict) -> Dict[str, Array]:
    """Compute derived quantities."""
    st = SPEC.unpack_state(state)
    p = SPEC.unpack_policy(policy)
    c = constants  # shorthand

    # Inflation indexation
    pi_tilda = c["pi_ss"] ** c["iota"] * st.pi_lag ** (1 - c["iota"])
    pi_w_tilda = c["pi_ss"] ** c["iota_w"] * st.pi_lag ** (1 - c["iota_w"])
    pi_w = p.pi * p.w_tilda / st.w_tilda_lag

    # Investment adjustment
    i_ratio = st.mu_z * p.i / st.i_lag
    S_val = S_adj(i_ratio, c["mu_z_ss"], c["kappa"])
    S_prime_val = S_adj_prime(i_ratio, c["mu_z_ss"], c["kappa"])

    # Financial frictions
    F_val = F_omega(p.omega_bar, c["sigma_omega"])
    G_val = G_omega(p.omega_bar, c["sigma_omega"])
    Gamma_val = Gamma(p.omega_bar, c["sigma_omega"])

    # Capital and returns
    k = (1 - c["delta"]) * st.k_lag / st.mu_z + (1 - S_val) * p.i
    # Marginal cost — solved analytically from cost minimization (eliminates eq10)
    s = (1.0 / st.eps) * (st.mu_z * p.h / st.k_lag) ** c["alpha"] * p.w_tilda / (1 - c["alpha"])
    r_k = st.eps * c["alpha"] * (st.mu_z * p.h / st.k_lag) ** (1 - c["alpha"]) * s
    R_k = ((1 - c["tau_k"]) * r_k + (1 - c["delta"]) * p.q) / st.q_lag * p.pi + c["tau_k"] * c["delta"]

    # Net worth: entrepreneur keeps (1-Gamma) share of gross return on capital
    n = (c["gamma_e"] / (p.pi * st.mu_z)) * (1.0 - Gamma_val) * R_k * st.q_lag * st.k_lag + c["w_e"]

    # Leverage — balance sheet identity (eliminates eq12)
    L = p.q * k / (n + 1e-8)

    # Output
    y_z = st.eps * (st.k_lag / st.mu_z) ** c["alpha"] * p.h ** (1 - c["alpha"]) - c["Phi"]

    # Consumption — solved analytically from resource constraint (eliminates eq11)
    monitoring_cost = c["mu_mon"] * G_val * R_k * st.q_lag * st.k_lag / (st.mu_z * p.pi)
    entrepreneur_cons = c["Theta"] * (1 - c["gamma_e"]) / c["gamma_e"] * (n - c["w_e"])
    cc = jnp.maximum(y_z - st.g - p.i / st.mu_ups - entrepreneur_cons - monitoring_cost, 1e-4)

    y_gdp = st.g + cc + p.i / st.mu_ups

    # Interest rate (Taylor rule)
    R = c["R_ss"] * (st.R_lag / c["R_ss"]) ** c["rho_p"] * (
        (p.pi / c["pi_ss"]) ** c["alpha_pi"] * (y_gdp / c["y_ss"]) ** c["alpha_y"]
    ) ** (1 - c["rho_p"]) * jnp.exp(st.m_p)

    # Phillips curve auxiliaries (with numerical floor)
    K_p_inner = (1 - c["xi_p"] * (pi_tilda / p.pi) ** (1 / (1 - c["lambda_f"]))) / (1 - c["xi_p"])
    K_p = p.F_p * jnp.maximum(K_p_inner, 0.01) ** (1 - c["lambda_f"])

    K_w_inner = (1 - c["xi_w"] * (pi_w_tilda / pi_w * c["mu_z_ss"]) ** (1 / (1 - c["lambda_w"]))) / (1 - c["xi_w"])
    K_w = (1 / c["psi_L"]) * jnp.maximum(K_w_inner, 0.01) ** (1 - c["lambda_w"] * (1 + c["sigma_L"])) * p.w_tilda * p.F_w

    return {
        "pi_tilda": pi_tilda, "pi_w_tilda": pi_w_tilda, "pi_w": pi_w,
        "S_val": S_val, "S_prime_val": S_prime_val,
        "F_val": F_val, "G_val": G_val, "Gamma_val": Gamma_val,
        "s": s, "L": L, "c": cc,
        "k": k, "r_k": r_k, "R_k": R_k, "n": n, "y_z": y_z, "y_gdp": y_gdp,
        "R": R, "K_p": K_p, "K_w": K_w, "i_ratio": i_ratio,
    }


def equations(
    state: Array, policy: Array,
    next_state: Array, next_policy: Array,
    constants: Dict,
) -> Dict[str, Array]:
    """Compute equilibrium equation residuals."""
    st = SPEC.unpack_state(state)
    p = SPEC.unpack_policy(policy)
    st_n = SPEC.unpack_state(next_state)
    p_n = SPEC.unpack_policy(next_policy)
    c = constants

    defs = definitions(state, policy, c)
    defs_n = definitions(next_state, next_policy, c)

    residuals = {}

    # Eq 1: Price Phillips (F_p)
    eq1_expect = (defs_n["pi_tilda"] / p_n.pi) ** (1 / (1 - c["lambda_f"])) * p_n.F_p
    residuals["eq1_price_phillips_F"] = p.lambda_z * defs["y_z"] + c["beta"] * c["xi_p"] * eq1_expect - p.F_p

    # Eq 2: Price Phillips (K_p)
    eq2_expect = (defs_n["pi_tilda"] / p_n.pi) ** (c["lambda_f"] / (1 - c["lambda_f"])) * defs_n["K_p"]
    residuals["eq2_price_phillips_K"] = p.lambda_z * c["lambda_f"] * defs["y_z"] * defs["s"] + c["beta"] * c["xi_p"] * eq2_expect - defs["K_p"]

    # Eq 3: Wage Phillips (F_w)
    eq3_coef = st_n.mu_z ** (c["iota_mu"] / (1 - c["lambda_w"]) - 1) * c["mu_z_ss"] ** ((1 - c["iota_mu"]) / (1 - c["lambda_w"]))
    eq3_expect = eq3_coef * defs_n["pi_w_tilda"] ** (1 / (1 - c["lambda_w"])) * \
                 (1 / defs_n["pi_w"]) ** (c["lambda_w"] / (1 - c["lambda_w"])) * (1 / p_n.pi) * p_n.F_w
    residuals["eq3_wage_phillips_F"] = p.h * (1 - c["tau_l"]) / c["lambda_w"] * p.lambda_z + c["beta"] * c["xi_w"] * eq3_expect - p.F_w

    # Eq 4: Wage Phillips (K_w)
    eq4_ratio = (defs_n["pi_w_tilda"] * c["mu_z_ss"] / defs_n["pi_w"]) ** (c["lambda_w"] / (1 - c["lambda_w"]) * (1 + c["sigma_L"]))
    residuals["eq4_wage_phillips_K"] = p.h ** (1 + c["sigma_L"]) + c["beta"] * c["xi_w"] * eq4_ratio * defs_n["K_w"] - defs["K_w"]

    # Eq 5: Consumption Euler (c from resource constraint via defs)
    habit_now = defs["c"] * st.mu_z - c["b"] * st.c_lag
    habit_next = defs_n["c"] * st_n.mu_z - c["b"] * defs["c"]
    residuals["eq5_consumption_euler"] = (1 + c["tau_c"]) * p.lambda_z - st.mu_z / habit_now + c["beta"] * c["b"] / habit_next

    # Eq 6: Bond Euler
    residuals["eq6_bond_euler"] = -p.lambda_z + defs["R"] * c["beta"] * p_n.lambda_z / (p_n.pi * st_n.mu_z)

    # Eq 7: Investment Euler
    eq7_term1 = p.lambda_z * p.q * (1 - defs["S_val"] - defs["i_ratio"] * defs["S_prime_val"])
    i_ratio_next = st_n.mu_z * p_n.i / p.i
    S_prime_next = S_adj_prime(i_ratio_next, c["mu_z_ss"], c["kappa"])
    eq7_expect = p_n.lambda_z * p_n.q * st_n.mu_z * (p_n.i / p.i) ** 2 * S_prime_next
    residuals["eq7_investment_euler"] = eq7_term1 - p.lambda_z / st.mu_ups + c["beta"] * eq7_expect

    # Eq 8: Bank participation
    survival_prob = 1.0 - defs["F_val"]
    residuals["eq8_bank_participation"] = st.L_lag * (defs["R_k"] / st.R_lag) * (
        p.omega_bar * survival_prob + (1 - c["mu_mon"]) * defs["G_val"]
    ) - st.L_lag + 1

    # Eq 9: Entrepreneur contract
    Gamma_next = Gamma(p_n.omega_bar, c["sigma_omega"])
    Gamma_prime_next = Gamma_prime(p_n.omega_bar, c["sigma_omega"])
    G_prime_next = G_omega_prime(p_n.omega_bar, c["sigma_omega"])
    G_next = G_omega(p_n.omega_bar, c["sigma_omega"])
    Rk_over_R = defs_n["R_k"] / defs["R"]
    ratio_term = Gamma_prime_next / (Gamma_prime_next - c["mu_mon"] * G_prime_next + 1e-8)
    bracket_term = 1 - Rk_over_R * (Gamma_next - c["mu_mon"] * G_next)
    residuals["eq9_entrepreneur_contract"] = Rk_over_R * (1 - Gamma_next) - ratio_term * bracket_term

    return residuals
