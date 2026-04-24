"""Sanity tests for the IRBC port.

Steady-state residuals are the strongest single assertion: the 5
equations (2 Euler + ARC + 2 FB) should all be at float32 noise when
evaluated at the analytically-solved SS (k_j=1, z_j=0, mu_j=0, with
lambda_ss from the Pareto-weighted aggregate consumption FOC).
"""

import jax.numpy as jnp
import pytest


@pytest.fixture
def model_and_ss():
    from deqn_jax.models.irbc import MODEL
    ss_state, ss_policy = MODEL.steady_state_fn(MODEL.constants)
    return MODEL, ss_state, ss_policy


def test_irbc_registers_and_loads():
    from deqn_jax.models import list_models, load_model
    model = load_model("irbc")
    assert model.name == "irbc"
    assert "irbc" in dict(list_models())
    assert model.equation_names == ("euler_0", "euler_1", "arc", "fb_0", "fb_1")
    assert model.n_states == 4
    assert model.n_policies == 5
    assert model.n_shocks == 3


def test_irbc_ss_residuals_below_noise(model_and_ss):
    """All 5 residuals at SS should be ≤ 1e-5 (float32 noise floor)."""
    MODEL, ss_state, ss_policy = model_and_ss
    state = ss_state[None, :]
    policy = ss_policy[None, :]
    resid = MODEL.equations_fn(state, policy, state, policy, MODEL.constants)
    for name, val in resid.items():
        v = float(val[0])
        assert abs(v) < 1e-5, f"Residual {name} at SS = {v:+.3e}, expected ~0"


def test_irbc_ss_step_is_fixed_point(model_and_ss):
    """step_fn at SS policy with zero shocks returns SS state."""
    MODEL, ss_state, ss_policy = model_and_ss
    state = ss_state[None, :]
    policy = ss_policy[None, :]
    next_state = MODEL.step_fn(state, policy, jnp.zeros((1, 3)), MODEL.constants)
    assert float(jnp.max(jnp.abs(next_state - state))) < 1e-6


def test_irbc_ss_consumption_positive(model_and_ss):
    """Per-country consumption c_j derived from lambda_ss must be > 0."""
    MODEL, ss_state, ss_policy = model_and_ss
    defs = MODEL.definitions_fn(ss_state[None, :], ss_policy[None, :], MODEL.constants)
    assert float(defs["c_0"][0]) > 0
    assert float(defs["c_1"][0]) > 0


def test_irbc_ss_investment_equals_delta_k(model_and_ss):
    """At SS k_j=1, investment i_j should equal delta (covers depreciation
    exactly). Adjustment cost vanishes when k == k_next."""
    MODEL, ss_state, ss_policy = model_and_ss
    defs = MODEL.definitions_fn(ss_state[None, :], ss_policy[None, :], MODEL.constants)
    delta = MODEL.constants["delta"]
    assert abs(float(defs["i_0"][0]) - delta) < 1e-6
    assert abs(float(defs["i_1"][0]) - delta) < 1e-6
    assert abs(float(defs["adj_cost_0"][0])) < 1e-8
    assert abs(float(defs["adj_cost_1"][0])) < 1e-8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
