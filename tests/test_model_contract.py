"""Contract-conformance tests across every registered model.

Audit findings tests-01 / models-01 / models-02: nothing validated that a
registered model actually satisfies the ModelSpec contract, so a broken new
model (wrong dims, SS that isn't a fixed point, mislabeled equations) would
pass CI silently. This is the safety net that makes the disaster-decoupling
and monolith-refactor work safe to do.

Parametrized over ``list_models()`` (the registry), so a newly-registered
model is automatically held to the same bar. The contract checked here:

  1. Dimension consistency — len(state_names)==n_states, etc. (when populated).
  2. SS is a fixed point of the equilibrium equations: residuals ≈ 0 at SS.
  3. SS is a fixed point of the dynamics: step(ss, ss, shock=0) ≈ ss.
  4. equations() dict key order == equation_names (the silent per-equation
     mislabel guard from models-02; eq_losses_to_array relies on this).

Tolerances: shipped models solve SS to ≤3e-5 (per the audit); we allow a
~30x margin so the test pins the contract without being brittle to solver
roundoff.
"""

import jax.numpy as jnp
import pytest

from deqn_jax.models import list_models, load_model

MODEL_NAMES = [name for name, _ in list_models()]

SS_RESID_TOL = 1e-3
FIXED_POINT_ATOL = 1e-3
FIXED_POINT_RTOL = 1e-3


@pytest.fixture(params=MODEL_NAMES)
def model(request):
    return load_model(request.param)


def _b(arr):
    """Add a leading batch dim: [n] -> [1, n]."""
    return arr[None, :]


def test_registry_nonempty():
    assert len(MODEL_NAMES) >= 1, "model registry is empty"


def test_dimension_consistency(model):
    if model.state_names:
        assert len(model.state_names) == model.n_states, (
            f"{model.name}: {len(model.state_names)} state_names != "
            f"n_states={model.n_states}"
        )
    if model.policy_names:
        assert len(model.policy_names) == model.n_policies, (
            f"{model.name}: {len(model.policy_names)} policy_names != "
            f"n_policies={model.n_policies}"
        )
    if model.policy_lower is not None:
        assert model.policy_lower.shape == (model.n_policies,)
    if model.policy_upper is not None:
        assert model.policy_upper.shape == (model.n_policies,)


def test_steady_state_is_equation_fixed_point(model):
    """Residuals of the equilibrium equations vanish at the steady state."""
    if model.steady_state_fn is None:
        pytest.skip(f"{model.name} has no steady_state_fn")
    ss_state, ss_policy = model.steady_state_fn(model.constants)
    s, p = _b(ss_state), _b(ss_policy)
    resid = model.equations_fn(s, p, s, p, model.constants)
    worst = {}
    for name, val in resid.items():
        if name.startswith("aux_"):
            continue
        worst[name] = float(jnp.max(jnp.abs(val)))
    bad = {k: v for k, v in worst.items() if v >= SS_RESID_TOL}
    assert not bad, (
        f"{model.name}: equation residuals at SS exceed {SS_RESID_TOL}: {bad}"
    )


def test_steady_state_is_dynamics_fixed_point(model):
    """step(ss, ss, shock=0) returns the steady state."""
    if model.steady_state_fn is None:
        pytest.skip(f"{model.name} has no steady_state_fn")
    ss_state, ss_policy = model.steady_state_fn(model.constants)
    s, p = _b(ss_state), _b(ss_policy)
    zero_shock = jnp.zeros((1, model.n_shocks))
    next_state = model.step_fn(s, p, zero_shock, model.constants)
    assert next_state.shape == s.shape, (
        f"{model.name}: step returned shape {next_state.shape}, expected {s.shape}"
    )
    ok = jnp.allclose(next_state, s, atol=FIXED_POINT_ATOL, rtol=FIXED_POINT_RTOL)
    assert ok, (
        f"{model.name}: step(ss) is not a fixed point. "
        f"max abs dev = {float(jnp.max(jnp.abs(next_state - s))):.2e}; "
        f"next={jnp.ravel(next_state)}, ss={jnp.ravel(s)}"
    )


def test_equation_names_match_equation_keys(model):
    """equations() dict key order must equal equation_names (models-02).

    eq_losses_to_array stacks the per-equation loss vector from the dict's
    insertion order and labels it with equation_names; a mismatch silently
    mislabels every per-equation residual in reweighting / gradient surgery.
    """
    if not model.equation_names:
        pytest.skip(f"{model.name} has no equation_names")
    if model.steady_state_fn is None:
        pytest.skip(f"{model.name} has no steady_state_fn to evaluate equations")
    ss_state, ss_policy = model.steady_state_fn(model.constants)
    s, p = _b(ss_state), _b(ss_policy)
    resid = model.equations_fn(s, p, s, p, model.constants)
    keys = tuple(k for k in resid.keys() if not k.startswith("aux_"))
    assert keys == tuple(model.equation_names), (
        f"{model.name}: equations() keys {keys} != equation_names "
        f"{tuple(model.equation_names)} — per-equation residuals would be "
        f"mislabeled in reweighting/gradient surgery."
    )
