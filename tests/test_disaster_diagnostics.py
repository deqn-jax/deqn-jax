"""Tests for disaster scalar_diagnostics decompositions.

Each test loads the disaster model, builds a small synthetic state batch
around the deterministic steady state, runs ``scalar_diagnostics``, and
asserts that every expected ``eq{N}_diag/*`` key is present, is a Python
``float``, and is finite.

These tests pin the dispatcher contract — if a helper accidentally returns
JAX arrays, NaNs, or drops a key, the test catches it before training
silently logs garbage.
"""

import math

import jax.numpy as jnp
import jax.random as random

from deqn_jax.models.disaster import MODEL
from deqn_jax.models.disaster.diagnostics import scalar_diagnostics


def _setup_batch(batch_size: int = 16, jitter: float = 0.01):
    """Build a state batch and matching policy_out around the deterministic SS.

    Returns: (states, policy_out, ss_policy_fn, defs_dict)
    """
    assert MODEL.steady_state_fn is not None
    ss_state, ss_policy = MODEL.steady_state_fn(MODEL.constants)

    key = random.PRNGKey(0)
    state_noise = jitter * random.normal(key, (batch_size, MODEL.n_states))
    states = ss_state[None, :] + state_noise
    policy_out = jnp.broadcast_to(ss_policy[None, :], (batch_size, MODEL.n_policies))

    def policy_fn(s):
        return ss_policy

    import jax

    assert MODEL.definitions_fn is not None
    defs_fn = MODEL.definitions_fn
    defs = jax.vmap(lambda s, p: defs_fn(s, p, MODEL.constants))(states, policy_out)

    return states, policy_out, policy_fn, defs


def _assert_all_float_finite(out: dict, prefix: str):
    """Every key starting with ``prefix`` must be float and finite."""
    matched = {k: v for k, v in out.items() if k.startswith(prefix)}
    assert matched, f"no keys with prefix {prefix!r} in {sorted(out.keys())}"
    for k, v in matched.items():
        assert isinstance(v, float), f"{k} = {v!r} is {type(v).__name__}, not float"
        assert math.isfinite(v), f"{k} = {v!r} is not finite"


def test_eq1_price_phillips_F_diagnostics():
    """eq1 helper emits a complete, finite scalar dict."""
    states, policy_out, policy_fn, defs = _setup_batch()
    out = scalar_diagnostics(MODEL, policy_fn, states, policy_out, defs)

    expected = {
        "eq1_diag/ratio_base_mean",
        "eq1_diag/ratio_base_std",
        "eq1_diag/ratio_base_min",
        "eq1_diag/ratio_base_max",
        "eq1_diag/eq1_ratio_mean",
        "eq1_diag/F_p_mean",
        "eq1_diag/F_p_n_mean",
        "eq1_diag/lhs_term_mean",
        "eq1_diag/expect_term_mean",
        "eq1_diag/log_residual_mean",
        "eq1_diag/log_residual_std",
        "eq1_diag/exponent",
    }
    missing = expected - set(out.keys())
    assert not missing, f"missing eq1_diag keys: {sorted(missing)}"
    _assert_all_float_finite(out, "eq1_diag/")


def test_eq3_wage_phillips_F_diagnostics():
    """eq3 helper emits a complete, finite scalar dict."""
    states, policy_out, policy_fn, defs = _setup_batch()
    out = scalar_diagnostics(MODEL, policy_fn, states, policy_out, defs)

    expected = {
        "eq3_diag/mu_z_factor_mean",
        "eq3_diag/pi_w_tilda_factor_mean",
        "eq3_diag/pi_w_inv_factor_mean",
        "eq3_diag/pi_inv_factor_mean",
        "eq3_diag/F_w_mean",
        "eq3_diag/F_w_n_mean",
        "eq3_diag/lhs_term_mean",
        "eq3_diag/expect_term_mean",
        "eq3_diag/log_residual_mean",
        "eq3_diag/log_residual_std",
        "eq3_diag/exponent_pi_w_tilda",
        "eq3_diag/exponent_pi_w_inv",
    }
    missing = expected - set(out.keys())
    assert not missing, f"missing eq3_diag keys: {sorted(missing)}"
    _assert_all_float_finite(out, "eq3_diag/")


def test_eq2a_Kp_definition_diagnostics():
    """eq2a (K_p definition) helper emits a complete, finite scalar dict."""
    states, policy_out, policy_fn, defs = _setup_batch()
    out = scalar_diagnostics(MODEL, policy_fn, states, policy_out, defs)

    expected = {
        "eq2a_diag/K_p_mean",
        "eq2a_diag/K_p_analytical_mean",
        "eq2a_diag/log_residual_mean",
        "eq2a_diag/log_residual_std",
        "eq2a_diag/K_p_inner_floor_frac",
    }
    missing = expected - set(out.keys())
    assert not missing, f"missing eq2a_diag keys: {sorted(missing)}"
    _assert_all_float_finite(out, "eq2a_diag/")


def test_eq4a_Kw_definition_diagnostics():
    """eq4a (K_w definition) helper emits a complete, finite scalar dict."""
    states, policy_out, policy_fn, defs = _setup_batch()
    out = scalar_diagnostics(MODEL, policy_fn, states, policy_out, defs)

    expected = {
        "eq4a_diag/K_w_mean",
        "eq4a_diag/K_w_analytical_mean",
        "eq4a_diag/log_residual_mean",
        "eq4a_diag/log_residual_std",
        "eq4a_diag/K_w_inner_floor_frac",
        "eq4a_diag/inner_exponent",
    }
    missing = expected - set(out.keys())
    assert not missing, f"missing eq4a_diag keys: {sorted(missing)}"
    _assert_all_float_finite(out, "eq4a_diag/")


def test_eq5_consumption_euler_diagnostics():
    """eq5 (consumption Euler) helper emits a complete, finite scalar dict."""
    states, policy_out, policy_fn, defs = _setup_batch()
    out = scalar_diagnostics(MODEL, policy_fn, states, policy_out, defs)

    expected = {
        "eq5_diag/tax_term_mean",
        "eq5_diag/habit_ratio_term_mean",
        "eq5_diag/habit_now_raw_mean",
        "eq5_diag/habit_next_raw_mean",
        "eq5_diag/habit_now_mean",
        "eq5_diag/habit_next_mean",
        "eq5_diag/habit_now_floor_frac",
        "eq5_diag/habit_next_floor_frac",
        "eq5_diag/habit_ratio_mean",
        "eq5_diag/habit_ratio_std",
        "eq5_diag/habit_ratio_min",
        "eq5_diag/habit_ratio_max",
        "eq5_diag/log_residual_mean",
        "eq5_diag/log_residual_std",
    }
    missing = expected - set(out.keys())
    assert not missing, f"missing eq5_diag keys: {sorted(missing)}"
    _assert_all_float_finite(out, "eq5_diag/")


def test_eq7_investment_euler_diagnostics():
    """eq7 (investment Euler) helper emits a complete, finite scalar dict."""
    states, policy_out, policy_fn, defs = _setup_batch()
    out = scalar_diagnostics(MODEL, policy_fn, states, policy_out, defs)

    expected = {
        "eq7_diag/now_term_mean",
        "eq7_diag/expect_term_mean",
        "eq7_diag/i_ratio_next_mean",
        "eq7_diag/i_ratio_next_std",
        "eq7_diag/i_ratio_next_min",
        "eq7_diag/i_ratio_next_max",
        "eq7_diag/i_ratio_next_sq_mean",
        "eq7_diag/S_val_mean",
        "eq7_diag/S_prime_val_mean",
        "eq7_diag/S_prime_next_mean",
        "eq7_diag/q_mean",
        "eq7_diag/q_n_mean",
        "eq7_diag/rhs_mean",
        "eq7_diag/residual_mean",
        "eq7_diag/residual_std",
        "eq7_diag/log_abs_rhs_mean",
    }
    missing = expected - set(out.keys())
    assert not missing, f"missing eq7_diag keys: {sorted(missing)}"
    _assert_all_float_finite(out, "eq7_diag/")


def test_eq8_entrepreneur_contract_diagnostics():
    """eq8 (entrepreneur contract) helper emits a complete, finite scalar dict."""
    states, policy_out, policy_fn, defs = _setup_batch()
    out = scalar_diagnostics(MODEL, policy_fn, states, policy_out, defs)

    expected = {
        "eq8_diag/Rk_over_R_mean",
        "eq8_diag/Rk_over_R_std",
        "eq8_diag/Rk_over_R_min",
        "eq8_diag/Rk_over_R_max",
        "eq8_diag/Gamma_next_mean",
        "eq8_diag/G_next_mean",
        "eq8_diag/Gamma_prime_next_mean",
        "eq8_diag/G_prime_next_mean",
        "eq8_diag/omega_bar_next_mean",
        "eq8_diag/omega_bar_next_std",
        "eq8_diag/omega_bar_next_min",
        "eq8_diag/omega_bar_next_max",
        "eq8_diag/newton_residual_n_mean",
        "eq8_diag/ratio_term_mean",
        "eq8_diag/bracket_term_mean",
        "eq8_diag/term_a_mean",
        "eq8_diag/term_b_mean",
        "eq8_diag/signed_residual_mean",
        "eq8_diag/signed_residual_std",
    }
    missing = expected - set(out.keys())
    assert not missing, f"missing eq8_diag keys: {sorted(missing)}"
    _assert_all_float_finite(out, "eq8_diag/")


def test_eq9_resource_constraint_diagnostics():
    """eq9 (resource constraint) helper emits a complete, finite scalar dict."""
    states, policy_out, policy_fn, defs = _setup_batch()
    out = scalar_diagnostics(MODEL, policy_fn, states, policy_out, defs)

    expected = {
        "eq9_diag/g_share_mean",
        "eq9_diag/c_share_mean",
        "eq9_diag/i_share_mean",
        "eq9_diag/entrepreneur_share_mean",
        "eq9_diag/monitoring_share_mean",
        "eq9_diag/total_share_mean",
        "eq9_diag/log_residual_mean",
        "eq9_diag/log_residual_std",
    }
    missing = expected - set(out.keys())
    assert not missing, f"missing eq9_diag keys: {sorted(missing)}"
    _assert_all_float_finite(out, "eq9_diag/")
