"""Certificate-in-the-loop drift loss (composite_loss.drift_weight): unit guards."""

import jax
import jax.numpy as jnp
import numpy as np

from deqn_jax.training.composite_loss import CompositeData, _drift_loss
from deqn_jax.types import ModelSpec


def _toy_model(factor):
    """1-state toy: closed loop s' = factor * s + policy-contribution 0."""

    def step_fn(state, policy, shock, constants):
        return factor * state + 0.0 * policy

    return ModelSpec(
        name="toy",
        n_states=1,
        n_policies=1,
        n_shocks=1,
        equations_fn=lambda *a: {},
        step_fn=step_fn,
        constants={},
    )


def _toy_data():
    return CompositeData(
        P=jnp.zeros((1, 1)),
        ss_state=jnp.zeros(1),
        ss_policy=jnp.zeros(1),
        ergodic_cov_chol=jnp.eye(1),
        anchor_points=jnp.zeros((1, 1)),
        anchor_deviations=jnp.zeros((1, 1)),
        anchor_lin_policy=jnp.zeros((1, 1)),
        aux_constants={},
    )


def _policy(x):
    return jnp.zeros(1)


PROBES = jnp.array([[1e-3], [-1e-3]])
LOG_TARGET = float(jnp.log(0.99))


def test_contraction_is_free():
    val = _drift_loss(_toy_model(0.8), _policy, _toy_data(), PROBES, 20, LOG_TARGET)
    assert float(val) < 1e-6, f"contraction should not be penalized, got {val}"


def test_expansion_is_penalized_linearly():
    val = _drift_loss(_toy_model(1.2), _policy, _toy_data(), PROBES, 20, LOG_TARGET)
    expected = np.log(1.2) - LOG_TARGET  # hinge is linear above threshold
    np.testing.assert_allclose(float(val), expected, rtol=0.05)


def test_gradient_flows_through_policy():
    """Policy-dependent toy: s' = s + p(s); grad wrt policy scale must be nonzero."""

    def step_fn(state, policy, shock, constants):
        return state + policy

    model = ModelSpec(
        name="toy2",
        n_states=1,
        n_policies=1,
        n_shocks=1,
        equations_fn=lambda *a: {},
        step_fn=step_fn,
        constants={},
    )

    def loss_of_scale(a):
        return _drift_loss(model, lambda s: a * s, _toy_data(), PROBES, 10, LOG_TARGET)

    g = jax.grad(loss_of_scale)(0.5)  # s' = 1.5 s -> expanding -> grad real
    assert np.isfinite(float(g)) and abs(float(g)) > 1e-6


def test_composite_wiring_off_and_on():
    from deqn_jax.models import load_model
    from deqn_jax.networks.mlp import MLP
    from deqn_jax.training.composite_loss import (
        make_composite_loss,
        prepare_composite_data,
    )
    from deqn_jax.training.linearize import linearize_model

    model = load_model("brock_mirman")
    P, Q = linearize_model(model, verbose=False)
    data = prepare_composite_data(model, P, Q, n_anchor_points=8, verbose=False)
    net = MLP(
        in_features=model.n_states,
        out_features=model.n_policies,
        hidden_sizes=(8,),
        activations=(jax.nn.tanh,),
        key=jax.random.PRNGKey(0),
    )
    states = jnp.tile(jnp.asarray(data.ss_state)[None, :], (4, 1))
    key = jax.random.PRNGKey(1)

    fn_off = make_composite_loss(model, data)
    total_off, eq_off = fn_off(model, net, states, key, mc_samples=2)
    assert "aux_drift" not in eq_off  # build-time skip

    fn_on = make_composite_loss(model, data, drift_weight=1.0, drift_horizon=10)
    total_on, eq_on = fn_on(model, net, states, key, mc_samples=2)
    assert "aux_drift" in eq_on
    assert np.isfinite(float(eq_on["aux_drift"]))
    # totals must agree up to exactly the weighted drift term
    np.testing.assert_allclose(
        float(total_on) - float(eq_on["aux_drift"]), float(total_off), rtol=1e-5
    )
