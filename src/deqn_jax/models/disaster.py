"""CMR-style NK-DSGE with Financial Frictions ("Disaster Model").

A medium-scale New Keynesian model with:
- 13 state variables (8 endogenous + 5 exogenous)
- 12 policy variables
- 12 equilibrium equations
- Financial frictions (costly state verification banking)
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import erf

from deqn_jax.types import ModelSpec
from deqn_jax.models.variables import DISASTER_SPEC as SPEC


# Model constants
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
    "sigma_mp": 0.49,
}

EQUATION_NAMES = (
    "eq1_price_phillips_F", "eq2_price_phillips_K", "eq3_wage_phillips_F",
    "eq4_wage_phillips_K", "eq5_consumption_euler", "eq6_bond_euler",
    "eq7_investment_euler", "eq8_bank_participation", "eq9_entrepreneur_contract",
    "eq10_marginal_cost", "eq11_resource_constraint", "eq12_capital_accumulation"
)

# Steady state values
STEADY_STATE = {
    "pi_lag": 1.006, "k_lag": 27.531, "c_lag": 1.6, "q_lag": 1.0,
    "i_lag": 0.79853, "R_lag": 1.011678, "w_tilda_lag": 1.9224, "L_lag": 1.9658,
    "eps": 1.0, "mu_ups": 1.0, "g": 0.616, "mu_z": 1.0041, "m_p": 0.0,
    "lambda_z": 0.59945, "i": 0.79853, "pi": 1.006, "c": 1.6,
    "w_tilda": 1.9224, "s": 0.83333, "omega_bar": 0.48848, "h": 0.94596,
    "F_w": 0.89465, "F_p": 4.5318, "q": 1.0, "L": 1.9658,
}


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

    # Capital and returns
    k = (1 - c["delta"]) * st.k_lag / st.mu_z + (1 - S_val) * p.i
    r_k = st.eps * c["alpha"] * (st.mu_z * p.h / st.k_lag) ** (1 - c["alpha"]) * p.s
    R_k = ((1 - c["tau_k"]) * r_k + (1 - c["delta"]) * p.q) / st.q_lag * p.pi + c["tau_k"] * c["delta"]

    # Net worth
    n = (c["gamma_e"] / (p.pi * st.mu_z)) * (
        R_k * st.q_lag * st.k_lag - st.R_lag * st.q_lag * st.k_lag / st.L_lag
        - c["mu_mon"] * G_val * R_k * st.q_lag * st.k_lag
    ) + c["w_e"] + c["gamma_e"] * st.R_lag / (p.pi * st.mu_z) * st.q_lag * st.k_lag / st.L_lag

    # Output
    y_z = st.eps * (st.k_lag / st.mu_z) ** c["alpha"] * p.h ** (1 - c["alpha"]) - c["Phi"]
    y_gdp = st.g + p.c + p.i / st.mu_ups

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
        "S_val": S_val, "S_prime_val": S_prime_val, "F_val": F_val, "G_val": G_val,
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
    residuals["eq2_price_phillips_K"] = p.lambda_z * c["lambda_f"] * defs["y_z"] * p.s + c["beta"] * c["xi_p"] * eq2_expect - defs["K_p"]

    # Eq 3: Wage Phillips (F_w)
    eq3_coef = st_n.mu_z ** (c["iota_mu"] / (1 - c["lambda_w"]) - 1) * c["mu_z_ss"] ** ((1 - c["iota_mu"]) / (1 - c["lambda_w"]))
    eq3_expect = eq3_coef * defs_n["pi_w_tilda"] ** (1 / (1 - c["lambda_w"])) * \
                 (1 / defs_n["pi_w"]) ** (c["lambda_w"] / (1 - c["lambda_w"])) * (1 / p_n.pi) * p_n.F_w
    residuals["eq3_wage_phillips_F"] = p.h * (1 - c["tau_l"]) / c["lambda_w"] * p.lambda_z + c["beta"] * c["xi_w"] * eq3_expect - p.F_w

    # Eq 4: Wage Phillips (K_w)
    eq4_ratio = (defs_n["pi_w_tilda"] * c["mu_z_ss"] / defs_n["pi_w"]) ** (c["lambda_w"] / (1 - c["lambda_w"]) * (1 + c["sigma_L"]))
    residuals["eq4_wage_phillips_K"] = p.h ** (1 + c["sigma_L"]) + c["beta"] * c["xi_w"] * eq4_ratio * defs_n["K_w"] - defs["K_w"]

    # Eq 5: Consumption Euler
    habit_now = p.c * st.mu_z - c["b"] * st.c_lag
    habit_next = p_n.c * st_n.mu_z - c["b"] * p.c
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

    # Eq 10: Marginal cost
    residuals["eq10_marginal_cost"] = p.s - (1 / st.eps) * (defs["r_k"] / c["alpha"]) ** c["alpha"] * (p.w_tilda / (1 - c["alpha"])) ** (1 - c["alpha"])

    # Eq 11: Resource constraint
    monitoring_cost = c["mu_mon"] * defs["G_val"] * defs["R_k"] * st.q_lag * st.k_lag / (st.mu_z * p.pi)
    entrepreneur_cons = c["Theta"] * (1 - c["gamma_e"]) / c["gamma_e"] * (defs["n"] - c["w_e"])
    residuals["eq11_resource_constraint"] = defs["y_z"] - st.g - p.c - p.i / st.mu_ups - entrepreneur_cons - monitoring_cost

    # Eq 12: Capital accumulation
    residuals["eq12_capital_accumulation"] = defs["k"] - (1 - c["delta"]) * st.k_lag / st.mu_z - (1 - defs["S_val"]) * p.i

    return residuals


def step(state: Array, policy: Array, shock: Array, constants: Dict) -> Array:
    """State transition."""
    st = SPEC.unpack_state(state)
    p = SPEC.unpack_policy(policy)
    defs = definitions(state, policy, constants)
    c = constants

    # Shocks
    eps_shock, mu_ups_shock, mu_z_shock, g_shock, mp_shock = shock[:, 0], shock[:, 1], shock[:, 2], shock[:, 3], shock[:, 4]

    safe_log = lambda x: jnp.log(jnp.maximum(x, 1e-8))

    # Evolve exogenous states
    eps_next = jnp.exp(c["rho_eps"] * safe_log(st.eps) + c["sigma_eps"] * eps_shock)
    mu_ups_next = jnp.exp(c["rho_mu_ups"] * safe_log(st.mu_ups) + c["sigma_mu_ups"] * mu_ups_shock)
    mu_z_next = jnp.exp(c["rho_mu_z"] * safe_log(st.mu_z) + (1 - c["rho_mu_z"]) * safe_log(c["mu_z_ss"]) + c["sigma_mu_z"] * mu_z_shock)
    g_next = jnp.exp(c["rho_g"] * safe_log(st.g) + (1 - c["rho_g"]) * safe_log(c["g_ss"]) + c["sigma_g"] * g_shock)
    m_p_next = c["sigma_mp"] * mp_shock

    next_state = jnp.stack([
        p.pi, defs["k"], p.c, p.q, p.i, defs["R"], p.w_tilda, p.L,
        eps_next, mu_ups_next, g_next, mu_z_next, m_p_next
    ], axis=1)

    lower = jnp.array([0.9, 10.0, 0.5, 0.5, 0.3, 1.0, 1.0, 1.0, 0.8, 0.9, 0.4, 0.99, -2.0])
    upper = jnp.array([1.2, 50.0, 3.0, 2.0, 2.0, 1.1, 3.0, 4.0, 1.2, 1.1, 0.9, 1.02, 2.0])
    return jnp.clip(next_state, lower, upper)


def steady_state(constants: Dict) -> Tuple[Array, Array]:
    ss_state = jnp.array([STEADY_STATE[n] for n in SPEC.state_names])
    ss_policy = jnp.array([STEADY_STATE[n] for n in SPEC.policy_names])
    return ss_state, ss_policy


def init_state(key: Array, batch_size: int, constants: Dict) -> Array:
    ss_state, _ = steady_state(constants)
    noise = jax.random.uniform(key, (batch_size, 13), minval=-0.05, maxval=0.05)
    return ss_state * (1 + noise)


POLICY_LOWER = jnp.array([0.1, 0.3, 0.9, 0.5, 1.0, 0.5, 0.1, 0.5, 0.1, 1.0, 0.5, 1.0])
POLICY_UPPER = jnp.array([2.0, 2.0, 1.2, 3.0, 3.0, 1.5, 1.5, 1.5, 3.0, 10.0, 2.0, 4.0])

MODEL = ModelSpec(
    name="disaster",
    n_states=SPEC.n_states,
    n_policies=SPEC.n_policies,
    n_shocks=5,
    state_names=SPEC.state_names,
    policy_names=SPEC.policy_names,
    equation_names=EQUATION_NAMES,
    constants=CONSTANTS,
    equations_fn=equations,
    step_fn=step,
    steady_state_fn=steady_state,
    init_state_fn=init_state,
    policy_lower=POLICY_LOWER,
    policy_upper=POLICY_UPPER,
)
