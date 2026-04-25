"""Disaster-model diagnostic decompositions, exposed via ModelSpec.scalar_diagnostics_fn.

These two helpers decompose the Phillips-curve recursion equations into
their constituent terms so we can watch the ratio bases, log-residuals,
and soft-floor saturation fractions on TensorBoard / console during a
disaster-model training run. They live here, not in trainer.py, because
they read disaster-specific definition keys (``K_p``, ``K_w``,
``pi_tilda``, ``pi_w_tilda``, ``pi_w``, ``y_z``, ``s``) and disaster-
specific calibration constants (``xi_p``, ``xi_w``, ``lambda_f``,
``lambda_w``, ``sigma_L``, ``mu_z_ss``).

Wired into the model via ``ModelSpec.scalar_diagnostics_fn`` in
``disaster/__init__.py``.
"""

from typing import Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from deqn_jax.models.disaster.equations import S_adj_prime, _soft_floor
from deqn_jax.types import ModelSpec


def _eq1_diagnostics(
    model: ModelSpec,
    policy_fn: Callable,
    states: Array,
    policy_out: Array,
    defs: Dict[str, Array],
) -> Dict[str, float]:
    """Compute eq1 (price_phillips_F) decomposition at training states.

    Mirrors ``_eq2_diagnostics`` for the F_p side of the price-Phillips
    recursion. Residual structure:
        F_p = lambda_z * y_z + beta * xi_p * (pi_tilda_n/pi_n)^(1/(1-lambda_f)) * F_p_n
    """
    c = model.constants
    batch_size = states.shape[0]

    zero_shock = jnp.zeros((batch_size, model.n_shocks))
    next_states = model.step_fn(states, policy_out, zero_shock, c)
    next_policies = jax.vmap(policy_fn)(next_states)
    assert model.definitions_fn is not None, "disaster diagnostics needs definitions_fn"
    defs_fn = model.definitions_fn
    defs_n = jax.vmap(lambda s, p: defs_fn(s, p, c))(next_states, next_policies)

    pi_idx = list(model.policy_names).index("pi")
    F_p_idx = list(model.policy_names).index("F_p")
    lambda_z_idx = list(model.policy_names).index("lambda_z")

    F_p = policy_out[:, F_p_idx]
    F_p_n = next_policies[:, F_p_idx]
    pi_n = next_policies[:, pi_idx]
    lambda_z = policy_out[:, lambda_z_idx]

    ratio_base = defs_n["pi_tilda"] / pi_n
    exponent = 1.0 / (1.0 - c["lambda_f"])
    eq1_ratio = ratio_base**exponent

    eq1_expect = eq1_ratio * F_p_n
    lhs_term = lambda_z * defs["y_z"]
    expect_term = c["beta"] * c["xi_p"] * eq1_expect
    rhs = lhs_term + expect_term
    log_residual = jnp.log(jnp.maximum(rhs, 1e-8)) - jnp.log(jnp.maximum(F_p, 1e-8))

    rb = np.asarray(ratio_base)
    lr = np.asarray(log_residual)
    return {
        "ratio_base_mean": float(np.mean(rb)),
        "ratio_base_std": float(np.std(rb)),
        "ratio_base_min": float(np.min(rb)),
        "ratio_base_max": float(np.max(rb)),
        "eq1_ratio_mean": float(np.mean(np.asarray(eq1_ratio))),
        "F_p_mean": float(np.mean(np.asarray(F_p))),
        "F_p_n_mean": float(np.mean(np.asarray(F_p_n))),
        "lhs_term_mean": float(np.mean(np.asarray(lhs_term))),
        "expect_term_mean": float(np.mean(np.asarray(expect_term))),
        "log_residual_mean": float(np.mean(lr)),
        "log_residual_std": float(np.std(lr)),
        "exponent": float(exponent),
    }


def _eq4a_diagnostics(
    model: ModelSpec,
    states: Array,
    policy_out: Array,
    defs: Dict[str, Array],
) -> Dict[str, float]:
    """Compute eq4a (K_w definition) decomposition at training states.

    Algebraic identity in log-space:
        log(K_w) - log((1/psi_L) * K_w_inner^(1-lambda_w*(1+sigma_L))
                       * w_tilda * F_w)
    Uses only current-period defs — no rollout needed.
    """
    c = model.constants
    F_w_idx = list(model.policy_names).index("F_w")
    w_tilda_idx = list(model.policy_names).index("w_tilda")
    F_w = policy_out[:, F_w_idx]
    w_tilda = policy_out[:, w_tilda_idx]
    K_w = defs["K_w"]
    K_w_inner = defs["K_w_inner"]

    inner_exponent = 1.0 - c["lambda_w"] * (1.0 + c["sigma_L"])
    K_w_analytical = (1.0 / c["psi_L"]) * K_w_inner**inner_exponent * w_tilda * F_w
    log_residual = jnp.log(jnp.maximum(K_w, 1e-8)) - jnp.log(
        jnp.maximum(K_w_analytical, 1e-8)
    )

    floor_frac = float(np.mean(np.asarray(K_w_inner) < 0.02))
    lr = np.asarray(log_residual)
    return {
        "K_w_mean": float(np.mean(np.asarray(K_w))),
        "K_w_analytical_mean": float(np.mean(np.asarray(K_w_analytical))),
        "log_residual_mean": float(np.mean(lr)),
        "log_residual_std": float(np.std(lr)),
        "K_w_inner_floor_frac": floor_frac,
        "inner_exponent": float(inner_exponent),
    }


def _eq4_diagnostics(
    model: ModelSpec,
    policy_fn: Callable,
    states: Array,
    policy_out: Array,
    defs: Dict[str, Array],
) -> Dict[str, float]:
    """Compute eq4 (wage_phillips_K) decomposition at training states.

    Runs a zero-shock forward step to get next-period definitions, then
    decomposes the eq4 residual into its constituent terms. Returns
    scalar statistics (mean/std/min/max/saturation-fraction) ready for
    TensorBoard / console logging.
    """
    c = model.constants
    batch_size = states.shape[0]

    zero_shock = jnp.zeros((batch_size, model.n_shocks))
    next_states = model.step_fn(states, policy_out, zero_shock, c)
    next_policies = jax.vmap(policy_fn)(next_states)
    assert model.definitions_fn is not None, "disaster diagnostics needs definitions_fn"
    defs_fn = model.definitions_fn
    defs_n = jax.vmap(lambda s, p: defs_fn(s, p, c))(next_states, next_policies)

    ratio_base = defs_n["pi_w_tilda"] * c["mu_z_ss"] / defs_n["pi_w"]
    exponent = c["lambda_w"] / (1 - c["lambda_w"]) * (1 + c["sigma_L"])
    eq4_ratio = ratio_base**exponent

    sigma_L = c["sigma_L"]
    h_idx = list(model.policy_names).index("h") if "h" in model.policy_names else None
    if h_idx is not None:
        h = policy_out[:, h_idx]
        h_term = h ** (1 + sigma_L)
    else:
        h_term = jnp.zeros(batch_size)

    expect_term = c["beta"] * c["xi_w"] * eq4_ratio * defs_n["K_w"]
    K_w = defs["K_w"]
    eq4_rhs = h_term + expect_term
    log_residual = jnp.log(jnp.maximum(eq4_rhs, 1e-8)) - jnp.log(jnp.maximum(K_w, 1e-8))

    xi_w = c["xi_w"]
    lambda_w = c["lambda_w"]
    K_w_inner_ratio = (defs["pi_w_tilda"] / defs["pi_w"] * c["mu_z_ss"]) ** (
        1 / (1 - lambda_w)
    )
    K_w_inner = (1 - xi_w * K_w_inner_ratio) / (1 - xi_w)
    floor_frac = float(np.mean(np.asarray(K_w_inner) < 0.02))

    rb = np.asarray(ratio_base)
    lr = np.asarray(log_residual)
    return {
        "ratio_base_mean": float(np.mean(rb)),
        "ratio_base_std": float(np.std(rb)),
        "ratio_base_min": float(np.min(rb)),
        "ratio_base_max": float(np.max(rb)),
        "eq4_ratio_mean": float(np.mean(np.asarray(eq4_ratio))),
        "K_w_mean": float(np.mean(np.asarray(K_w))),
        "K_w_n_mean": float(np.mean(np.asarray(defs_n["K_w"]))),
        "h_term_mean": float(np.mean(np.asarray(h_term))),
        "expect_term_mean": float(np.mean(np.asarray(expect_term))),
        "log_residual_mean": float(np.mean(lr)),
        "log_residual_std": float(np.std(lr)),
        "K_w_inner_floor_frac": floor_frac,
        "exponent": float(exponent),
    }


def _eq3_diagnostics(
    model: ModelSpec,
    policy_fn: Callable,
    states: Array,
    policy_out: Array,
    defs: Dict[str, Array],
) -> Dict[str, float]:
    """Compute eq3 (wage_phillips_F) decomposition at training states.

    Mirrors ``_eq1_diagnostics`` for the F_w side of the wage-Phillips
    recursion. The expectation has more multiplicative factors than the
    price side, so each factor is logged separately:
        eq3_expect = mu_z_factor * pi_w_tilda_factor * pi_w_inv_factor
                     * (1/pi_n) * F_w_n
        F_w = h*(1-tau_l)/lambda_w * lambda_z + beta*xi_w * eq3_expect
    """
    c = model.constants
    batch_size = states.shape[0]

    zero_shock = jnp.zeros((batch_size, model.n_shocks))
    next_states = model.step_fn(states, policy_out, zero_shock, c)
    next_policies = jax.vmap(policy_fn)(next_states)
    assert model.definitions_fn is not None, "disaster diagnostics needs definitions_fn"
    defs_fn = model.definitions_fn
    defs_n = jax.vmap(lambda s, p: defs_fn(s, p, c))(next_states, next_policies)

    pi_idx = list(model.policy_names).index("pi")
    F_w_idx = list(model.policy_names).index("F_w")
    h_idx = list(model.policy_names).index("h")
    lambda_z_idx = list(model.policy_names).index("lambda_z")
    mu_z_state_idx = list(model.state_names).index("mu_z")

    F_w = policy_out[:, F_w_idx]
    F_w_n = next_policies[:, F_w_idx]
    pi_n = next_policies[:, pi_idx]
    h = policy_out[:, h_idx]
    lambda_z = policy_out[:, lambda_z_idx]
    mu_z_n = next_states[:, mu_z_state_idx]

    lambda_w = c["lambda_w"]
    iota_mu = c["iota_mu"]
    exponent_pi_w_tilda = 1.0 / (1.0 - lambda_w)
    exponent_pi_w_inv = lambda_w / (1.0 - lambda_w)
    mu_z_exponent = iota_mu / (1.0 - lambda_w) - 1.0
    mu_z_ss_exponent = (1.0 - iota_mu) / (1.0 - lambda_w)

    mu_z_factor = mu_z_n**mu_z_exponent * c["mu_z_ss"] ** mu_z_ss_exponent
    pi_w_tilda_factor = defs_n["pi_w_tilda"] ** exponent_pi_w_tilda
    pi_w_inv_factor = (1.0 / defs_n["pi_w"]) ** exponent_pi_w_inv
    pi_inv_factor = 1.0 / pi_n

    eq3_expect = (
        mu_z_factor * pi_w_tilda_factor * pi_w_inv_factor * pi_inv_factor * F_w_n
    )
    lhs_term = h * (1.0 - c["tau_l"]) / lambda_w * lambda_z
    expect_term = c["beta"] * c["xi_w"] * eq3_expect
    rhs = lhs_term + expect_term
    log_residual = jnp.log(jnp.maximum(rhs, 1e-8)) - jnp.log(jnp.maximum(F_w, 1e-8))

    lr = np.asarray(log_residual)
    return {
        "mu_z_factor_mean": float(np.mean(np.asarray(mu_z_factor))),
        "pi_w_tilda_factor_mean": float(np.mean(np.asarray(pi_w_tilda_factor))),
        "pi_w_inv_factor_mean": float(np.mean(np.asarray(pi_w_inv_factor))),
        "pi_inv_factor_mean": float(np.mean(np.asarray(pi_inv_factor))),
        "F_w_mean": float(np.mean(np.asarray(F_w))),
        "F_w_n_mean": float(np.mean(np.asarray(F_w_n))),
        "lhs_term_mean": float(np.mean(np.asarray(lhs_term))),
        "expect_term_mean": float(np.mean(np.asarray(expect_term))),
        "log_residual_mean": float(np.mean(lr)),
        "log_residual_std": float(np.std(lr)),
        "exponent_pi_w_tilda": float(exponent_pi_w_tilda),
        "exponent_pi_w_inv": float(exponent_pi_w_inv),
    }


def _eq2a_diagnostics(
    model: ModelSpec,
    states: Array,
    policy_out: Array,
    defs: Dict[str, Array],
) -> Dict[str, float]:
    """Compute eq2a (K_p definition) decomposition at training states.

    Algebraic identity in log-space: ``log(K_p) - log(F_p * K_p_inner^(1-lambda_f))``.
    Uses only current-period defs — no rollout needed (cheaper than the
    recursion helpers).
    """
    c = model.constants
    F_p_idx = list(model.policy_names).index("F_p")
    F_p = policy_out[:, F_p_idx]
    K_p = defs["K_p"]
    K_p_inner = defs["K_p_inner"]

    K_p_analytical = F_p * K_p_inner ** (1.0 - c["lambda_f"])
    log_residual = jnp.log(jnp.maximum(K_p, 1e-8)) - jnp.log(
        jnp.maximum(K_p_analytical, 1e-8)
    )

    floor_frac = float(np.mean(np.asarray(K_p_inner) < 0.02))
    lr = np.asarray(log_residual)
    return {
        "K_p_mean": float(np.mean(np.asarray(K_p))),
        "K_p_analytical_mean": float(np.mean(np.asarray(K_p_analytical))),
        "log_residual_mean": float(np.mean(lr)),
        "log_residual_std": float(np.std(lr)),
        "K_p_inner_floor_frac": floor_frac,
    }


def _eq2_diagnostics(
    model: ModelSpec,
    policy_fn: Callable,
    states: Array,
    policy_out: Array,
    defs: Dict[str, Array],
) -> Dict[str, float]:
    """Compute eq2 (price_phillips_K) decomposition at training states.

    Mirrors ``_eq4_diagnostics`` for the price-Phillips side.
    """
    c = model.constants
    batch_size = states.shape[0]

    zero_shock = jnp.zeros((batch_size, model.n_shocks))
    next_states = model.step_fn(states, policy_out, zero_shock, c)
    next_policies = jax.vmap(policy_fn)(next_states)
    assert model.definitions_fn is not None, "disaster diagnostics needs definitions_fn"
    defs_fn = model.definitions_fn
    defs_n = jax.vmap(lambda s, p: defs_fn(s, p, c))(next_states, next_policies)

    pi_idx = list(model.policy_names).index("pi")
    ratio_base = defs_n["pi_tilda"] / next_policies[:, pi_idx]
    exponent = c["lambda_f"] / (1 - c["lambda_f"])
    eq2_ratio = ratio_base**exponent

    lambda_z_idx = list(model.policy_names).index("lambda_z")
    lambda_z = policy_out[:, lambda_z_idx]
    lhs_term = lambda_z * c["lambda_f"] * defs["y_z"] * defs["s"]
    expect_term = c["beta"] * c["xi_p"] * eq2_ratio * defs_n["K_p"]
    K_p = defs["K_p"]
    eq2_rhs = lhs_term + expect_term
    log_residual = jnp.log(jnp.maximum(eq2_rhs, 1e-8)) - jnp.log(jnp.maximum(K_p, 1e-8))

    xi_p = c["xi_p"]
    lambda_f = c["lambda_f"]
    K_p_inner_ratio = (defs["pi_tilda"] / policy_out[:, pi_idx]) ** (1 / (1 - lambda_f))
    K_p_inner = (1 - xi_p * K_p_inner_ratio) / (1 - xi_p)
    floor_frac = float(np.mean(np.asarray(K_p_inner) < 0.02))

    rb = np.asarray(ratio_base)
    lr = np.asarray(log_residual)
    return {
        "ratio_base_mean": float(np.mean(rb)),
        "ratio_base_std": float(np.std(rb)),
        "ratio_base_min": float(np.min(rb)),
        "ratio_base_max": float(np.max(rb)),
        "eq2_ratio_mean": float(np.mean(np.asarray(eq2_ratio))),
        "K_p_mean": float(np.mean(np.asarray(K_p))),
        "K_p_n_mean": float(np.mean(np.asarray(defs_n["K_p"]))),
        "lhs_term_mean": float(np.mean(np.asarray(lhs_term))),
        "expect_term_mean": float(np.mean(np.asarray(expect_term))),
        "log_residual_mean": float(np.mean(lr)),
        "log_residual_std": float(np.std(lr)),
        "K_p_inner_floor_frac": floor_frac,
        "exponent": float(exponent),
    }


def _eq5_diagnostics(
    model: ModelSpec,
    policy_fn: Callable,
    states: Array,
    policy_out: Array,
    defs: Dict[str, Array],
) -> Dict[str, float]:
    """Compute eq5 (consumption_euler) decomposition at training states.

    Residual structure:
        habit_now  = soft_floor(c * mu_z - b * c_lag, 1e-2)
        habit_next = soft_floor(c_n * mu_z_n - b * c, 1e-2)
        eq5 = ((1 + tau_c) * lambda_z * habit_now
               + beta * b * habit_now / habit_next) / mu_z - 1
    Logs the two summands, habit means (raw + floored), the
    habit_now/habit_next ratio, and saturation fractions for each
    soft-floored habit term.
    """
    c_const = model.constants
    batch_size = states.shape[0]

    zero_shock = jnp.zeros((batch_size, model.n_shocks))
    next_states = model.step_fn(states, policy_out, zero_shock, c_const)
    next_policies = jax.vmap(policy_fn)(next_states)

    c_idx = list(model.policy_names).index("c")
    lambda_z_idx = list(model.policy_names).index("lambda_z")
    mu_z_state_idx = list(model.state_names).index("mu_z")
    c_lag_state_idx = list(model.state_names).index("c_lag")

    c_now = policy_out[:, c_idx]
    c_n = next_policies[:, c_idx]
    lambda_z = policy_out[:, lambda_z_idx]
    mu_z = states[:, mu_z_state_idx]
    mu_z_n = next_states[:, mu_z_state_idx]
    c_lag = states[:, c_lag_state_idx]

    habit_now_raw = c_now * mu_z - c_const["b"] * c_lag
    habit_next_raw = c_n * mu_z_n - c_const["b"] * c_now
    habit_now = _soft_floor(habit_now_raw, 1e-2)
    habit_next = _soft_floor(habit_next_raw, 1e-2)

    tax_term = (1.0 + c_const["tau_c"]) * lambda_z * habit_now
    habit_ratio_term = c_const["beta"] * c_const["b"] * habit_now / (habit_next + 1e-8)
    rhs = tax_term + habit_ratio_term
    log_residual = jnp.log(jnp.maximum(rhs, 1e-8)) - jnp.log(jnp.maximum(mu_z, 1e-8))
    habit_ratio = habit_now / (habit_next + 1e-8)

    floor_thresh = 0.012  # 1e-2 + 2e-3 buffer for saturation detection
    floor_frac_now = float(np.mean(np.asarray(habit_now_raw) < floor_thresh))
    floor_frac_next = float(np.mean(np.asarray(habit_next_raw) < floor_thresh))

    lr = np.asarray(log_residual)
    hr = np.asarray(habit_ratio)
    return {
        "tax_term_mean": float(np.mean(np.asarray(tax_term))),
        "habit_ratio_term_mean": float(np.mean(np.asarray(habit_ratio_term))),
        "habit_now_raw_mean": float(np.mean(np.asarray(habit_now_raw))),
        "habit_next_raw_mean": float(np.mean(np.asarray(habit_next_raw))),
        "habit_now_mean": float(np.mean(np.asarray(habit_now))),
        "habit_next_mean": float(np.mean(np.asarray(habit_next))),
        "habit_now_floor_frac": floor_frac_now,
        "habit_next_floor_frac": floor_frac_next,
        "habit_ratio_mean": float(np.mean(hr)),
        "habit_ratio_std": float(np.std(hr)),
        "habit_ratio_min": float(np.min(hr)),
        "habit_ratio_max": float(np.max(hr)),
        "log_residual_mean": float(np.mean(lr)),
        "log_residual_std": float(np.std(lr)),
    }


def _eq7_diagnostics(
    model: ModelSpec,
    policy_fn: Callable,
    states: Array,
    policy_out: Array,
    defs: Dict[str, Array],
) -> Dict[str, float]:
    """Compute eq7 (investment_euler) decomposition at training states.

    Residual structure:
        now_term  = mu_ups * q * (1 - S_val - i_ratio*S_prime_val)
        i_ratio_n = mu_z_n * i_n / i
        S_prime_n = S_adj_prime(i_ratio_n, mu_z_ss, kappa)
        expect    = lambda_z_n * q_n * mu_z_n * (i_n/i)^2 * S_prime_n
        eq7       = (now_term + beta*mu_ups*expect/lambda_z) - 1

    The (i_n/i)^2 factor is the dominant nonlinearity. Logs both
    summands, the i_ratio_next mean/std/min/max + its square, and the
    S_val / S_prime_val / S_prime_next means.
    """
    c = model.constants
    batch_size = states.shape[0]

    zero_shock = jnp.zeros((batch_size, model.n_shocks))
    next_states = model.step_fn(states, policy_out, zero_shock, c)
    next_policies = jax.vmap(policy_fn)(next_states)

    i_idx = list(model.policy_names).index("i")
    q_idx = list(model.policy_names).index("q")
    lambda_z_idx = list(model.policy_names).index("lambda_z")
    mu_z_state_idx = list(model.state_names).index("mu_z")
    mu_ups_state_idx = list(model.state_names).index("mu_ups")

    i_now = policy_out[:, i_idx]
    i_n = next_policies[:, i_idx]
    q = policy_out[:, q_idx]
    q_n = next_policies[:, q_idx]
    lambda_z = policy_out[:, lambda_z_idx]
    lambda_z_n = next_policies[:, lambda_z_idx]
    mu_z_n = next_states[:, mu_z_state_idx]
    mu_ups = states[:, mu_ups_state_idx]

    S_val = defs["S_val"]
    S_prime_val = defs["S_prime_val"]
    i_ratio = defs["i_ratio"]

    i_ratio_next = mu_z_n * i_n / (i_now + 1e-8)
    S_prime_next = S_adj_prime(i_ratio_next, c["mu_z_ss"], c["kappa"])

    now_term = mu_ups * q * (1.0 - S_val - i_ratio * S_prime_val)
    expect = lambda_z_n * q_n * mu_z_n * (i_n / (i_now + 1e-8)) ** 2 * S_prime_next
    expect_term = c["beta"] * mu_ups * expect / (lambda_z + 1e-8)
    rhs = now_term + expect_term
    residual = rhs - 1.0
    log_rhs_residual = jnp.log(jnp.maximum(jnp.abs(rhs), 1e-8))

    irn = np.asarray(i_ratio_next)
    res = np.asarray(residual)
    return {
        "now_term_mean": float(np.mean(np.asarray(now_term))),
        "expect_term_mean": float(np.mean(np.asarray(expect_term))),
        "i_ratio_next_mean": float(np.mean(irn)),
        "i_ratio_next_std": float(np.std(irn)),
        "i_ratio_next_min": float(np.min(irn)),
        "i_ratio_next_max": float(np.max(irn)),
        "i_ratio_next_sq_mean": float(np.mean(irn**2)),
        "S_val_mean": float(np.mean(np.asarray(S_val))),
        "S_prime_val_mean": float(np.mean(np.asarray(S_prime_val))),
        "S_prime_next_mean": float(np.mean(np.asarray(S_prime_next))),
        "q_mean": float(np.mean(np.asarray(q))),
        "q_n_mean": float(np.mean(np.asarray(q_n))),
        "rhs_mean": float(np.mean(np.asarray(rhs))),
        "residual_mean": float(np.mean(res)),
        "residual_std": float(np.std(res)),
        "log_abs_rhs_mean": float(np.mean(np.asarray(log_rhs_residual))),
    }


def scalar_diagnostics(
    model: ModelSpec,
    policy_fn: Callable,
    states: Array,
    policy_out: Array,
    defs: Dict[str, Array],
) -> Dict[str, float]:
    """Top-level disaster diagnostics dispatcher.

    Returns a dict of scalars to log, namespaced as ``eq2_diag/<name>``
    and ``eq4_diag/<name>``. Both decompositions are duck-type-guarded
    against ``defs`` so they're skipped for ablations that drop the
    relevant Phillips-curve definitions.
    """
    out: Dict[str, float] = {}

    if "pi_tilda" in defs and "y_z" in defs:
        for k, v in _eq1_diagnostics(
            model, policy_fn, states, policy_out, defs
        ).items():
            out[f"eq1_diag/{k}"] = v

    if "K_p" in defs and "K_p_inner" in defs:
        for k, v in _eq2a_diagnostics(model, states, policy_out, defs).items():
            out[f"eq2a_diag/{k}"] = v

    if "K_p" in defs and "pi_tilda" in defs and "s" in defs:
        for k, v in _eq2_diagnostics(
            model, policy_fn, states, policy_out, defs
        ).items():
            out[f"eq2_diag/{k}"] = v

    if "pi_w_tilda" in defs and "pi_w" in defs:
        for k, v in _eq3_diagnostics(
            model, policy_fn, states, policy_out, defs
        ).items():
            out[f"eq3_diag/{k}"] = v

    if "K_w" in defs and "K_w_inner" in defs:
        for k, v in _eq4a_diagnostics(model, states, policy_out, defs).items():
            out[f"eq4a_diag/{k}"] = v

    if "K_w" in defs and "pi_w_tilda" in defs and "pi_w" in defs:
        for k, v in _eq4_diagnostics(
            model, policy_fn, states, policy_out, defs
        ).items():
            out[f"eq4_diag/{k}"] = v

    if "c_lag" in model.state_names and "c" in model.policy_names:
        for k, v in _eq5_diagnostics(
            model, policy_fn, states, policy_out, defs
        ).items():
            out[f"eq5_diag/{k}"] = v

    if "S_val" in defs and "S_prime_val" in defs and "i_ratio" in defs:
        for k, v in _eq7_diagnostics(
            model, policy_fn, states, policy_out, defs
        ).items():
            out[f"eq7_diag/{k}"] = v

    return out
