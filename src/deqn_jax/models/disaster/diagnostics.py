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

    if "K_p" in defs and "pi_tilda" in defs and "s" in defs:
        for k, v in _eq2_diagnostics(
            model, policy_fn, states, policy_out, defs
        ).items():
            out[f"eq2_diag/{k}"] = v

    if "K_w" in defs and "pi_w_tilda" in defs and "pi_w" in defs:
        for k, v in _eq4_diagnostics(
            model, policy_fn, states, policy_out, defs
        ).items():
            out[f"eq4_diag/{k}"] = v

    return out
