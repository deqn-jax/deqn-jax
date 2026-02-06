"""Rescaled Euler error equations for Disaster (NK-DSGE) model.

Each equation is written as LHS/RHS - 1 = 0 (unit-free percentage deviations),
following the Euler error formulation in the paper. This makes all residuals
comparable in magnitude and improves gradient balance during training.
"""

from typing import Dict

import jax.numpy as jnp
from jax import Array
from jax.scipy.special import erf

from deqn_jax.models.disaster.variables import SPEC
from deqn_jax.models.disaster.equations import (
    definitions,
    S_adj_prime,
    Gamma, Gamma_prime,
    G_omega, G_omega_prime,
    EQUATION_NAMES,
)


def equations(
    state: Array, policy: Array,
    next_state: Array, next_policy: Array,
    constants: Dict,
) -> Dict[str, Array]:
    """Compute rescaled equilibrium equation residuals (unit-free)."""
    st = SPEC.unpack_state(state)
    p = SPEC.unpack_policy(policy)
    st_n = SPEC.unpack_state(next_state)
    p_n = SPEC.unpack_policy(next_policy)
    c = constants

    defs = definitions(state, policy, c)
    defs_n = definitions(next_state, next_policy, c)

    residuals = {}

    # Eq 1: Price Phillips (F_p) — divide by F_p
    eq1_expect = (defs_n["pi_tilda"] / p_n.pi) ** (1 / (1 - c["lambda_f"])) * p_n.F_p
    residuals["eq1_price_phillips_F"] = (
        p.lambda_z * defs["y_z"] + c["beta"] * c["xi_p"] * eq1_expect
    ) / (p.F_p + 1e-8) - 1.0

    # Eq 2: Price Phillips (K_p) — divide by K_p
    eq2_expect = (defs_n["pi_tilda"] / p_n.pi) ** (c["lambda_f"] / (1 - c["lambda_f"])) * defs_n["K_p"]
    residuals["eq2_price_phillips_K"] = (
        p.lambda_z * c["lambda_f"] * defs["y_z"] * p.s + c["beta"] * c["xi_p"] * eq2_expect
    ) / (defs["K_p"] + 1e-8) - 1.0

    # Eq 3: Wage Phillips (F_w) — divide by F_w
    eq3_coef = st_n.mu_z ** (c["iota_mu"] / (1 - c["lambda_w"]) - 1) * c["mu_z_ss"] ** ((1 - c["iota_mu"]) / (1 - c["lambda_w"]))
    eq3_expect = eq3_coef * defs_n["pi_w_tilda"] ** (1 / (1 - c["lambda_w"])) * \
                 (1 / defs_n["pi_w"]) ** (c["lambda_w"] / (1 - c["lambda_w"])) * (1 / p_n.pi) * p_n.F_w
    residuals["eq3_wage_phillips_F"] = (
        p.h * (1 - c["tau_l"]) / c["lambda_w"] * p.lambda_z + c["beta"] * c["xi_w"] * eq3_expect
    ) / (p.F_w + 1e-8) - 1.0

    # Eq 4: Wage Phillips (K_w) — divide by K_w
    eq4_ratio = (defs_n["pi_w_tilda"] * c["mu_z_ss"] / defs_n["pi_w"]) ** (c["lambda_w"] / (1 - c["lambda_w"]) * (1 + c["sigma_L"]))
    residuals["eq4_wage_phillips_K"] = (
        p.h ** (1 + c["sigma_L"]) + c["beta"] * c["xi_w"] * eq4_ratio * defs_n["K_w"]
    ) / (defs["K_w"] + 1e-8) - 1.0

    # Eq 5: Consumption Euler — multiply by habit_now, divide by mu_z
    habit_now = p.c * st.mu_z - c["b"] * st.c_lag
    habit_next = p_n.c * st_n.mu_z - c["b"] * p.c
    residuals["eq5_consumption_euler"] = (
        (1 + c["tau_c"]) * p.lambda_z * habit_now + c["beta"] * c["b"] * habit_now / (habit_next + 1e-8)
    ) / (st.mu_z + 1e-8) - 1.0

    # Eq 6: Bond Euler — divide by lambda_z
    residuals["eq6_bond_euler"] = (
        defs["R"] * c["beta"] * p_n.lambda_z / (p_n.pi * st_n.mu_z)
    ) / (p.lambda_z + 1e-8) - 1.0

    # Eq 7: Investment Euler — divide by lambda_z/mu_ups (multiply by mu_ups/lambda_z)
    eq7_term1 = p.q * (1 - defs["S_val"] - defs["i_ratio"] * defs["S_prime_val"])
    i_ratio_next = st_n.mu_z * p_n.i / p.i
    S_prime_next = S_adj_prime(i_ratio_next, c["mu_z_ss"], c["kappa"])
    eq7_expect = p_n.lambda_z * p_n.q * st_n.mu_z * (p_n.i / p.i) ** 2 * S_prime_next
    residuals["eq7_investment_euler"] = (
        st.mu_ups * eq7_term1 - 1.0
        + c["beta"] * st.mu_ups * eq7_expect / (p.lambda_z + 1e-8)
    )

    # Eq 8: Bank participation — unchanged (already O(1))
    survival_prob = 1.0 - defs["F_val"]
    residuals["eq8_bank_participation"] = st.L_lag * (defs["R_k"] / st.R_lag) * (
        p.omega_bar * survival_prob + (1 - c["mu_mon"]) * defs["G_val"]
    ) - st.L_lag + 1

    # Eq 9: Entrepreneur contract — unchanged (already O(1))
    Gamma_next = Gamma(p_n.omega_bar, c["sigma_omega"])
    Gamma_prime_next = Gamma_prime(p_n.omega_bar, c["sigma_omega"])
    G_prime_next = G_omega_prime(p_n.omega_bar, c["sigma_omega"])
    G_next = G_omega(p_n.omega_bar, c["sigma_omega"])
    Rk_over_R = defs_n["R_k"] / defs["R"]
    ratio_term = Gamma_prime_next / (Gamma_prime_next - c["mu_mon"] * G_prime_next + 1e-8)
    bracket_term = 1 - Rk_over_R * (Gamma_next - c["mu_mon"] * G_next)
    residuals["eq9_entrepreneur_contract"] = Rk_over_R * (1 - Gamma_next) - ratio_term * bracket_term

    # Eq 10: Marginal cost — divide by s*eps
    rhs_10 = (defs["r_k"] / c["alpha"]) ** c["alpha"] * (p.w_tilda / (1 - c["alpha"])) ** (1 - c["alpha"])
    residuals["eq10_marginal_cost"] = rhs_10 / (p.s * st.eps + 1e-8) - 1.0

    # Eq 11: Resource constraint — divide by y_z
    monitoring_cost = c["mu_mon"] * defs["G_val"] * defs["R_k"] * st.q_lag * st.k_lag / (st.mu_z * p.pi)
    entrepreneur_cons = c["Theta"] * (1 - c["gamma_e"]) / c["gamma_e"] * (defs["n"] - c["w_e"])
    residuals["eq11_resource_constraint"] = (
        st.g + p.c + p.i / st.mu_ups + entrepreneur_cons + monitoring_cost
    ) / (defs["y_z"] + 1e-8) - 1.0

    # Eq 12: Leverage definition — already unit-free (ratio - 1)
    residuals["eq12_leverage_definition"] = p.L * defs["n"] / (p.q * defs["k"] + 1e-8) - 1.0

    return residuals
