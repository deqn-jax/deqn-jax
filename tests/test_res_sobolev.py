"""Residual-Sobolev loss (composite_loss.res_sobolev_weight): unit guards."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deqn_jax.models import load_model
from deqn_jax.training.composite_loss import _residual_sobolev_loss
from deqn_jax.training.loss import gauss_hermite_nd


def _bm_setup():
    model = load_model("brock_mirman")
    qn, qw = gauss_hermite_nd(7, model.n_shocks)
    ss_state, _ = model.steady_state_fn(model.constants)
    # states around the SS, spread in k
    k = jnp.linspace(0.8, 1.2, 6)[:, None] * ss_state[0]
    z = jnp.zeros((6, 1))
    states = jnp.concatenate([k, z], axis=1)
    dirs = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    return model, states, jnp.array(qn), jnp.array(qw), dirs


def _toy_model():
    """1-state toy with exact semantics: r = policy - state.

    True policy p(s) = s zeroes the residual EVERYWHERE, so its residual
    gradient is exactly 0. Impostor p(s) = c·s has d/ds E[r] = c - 1
    exactly, so the Sobolev value is (c-1)² — assertable to the digit.
    Deterministic step; single dummy quadrature node.
    """
    from deqn_jax.types import ModelSpec

    return ModelSpec(
        name="toy_rsob",
        n_states=1,
        n_policies=1,
        n_shocks=1,
        equations_fn=lambda s, p, sn, pn, c: {"eq": (p - s).reshape(-1)},
        step_fn=lambda s, p, shock, c: 0.5 * s,
        constants={},
        equation_names=("eq",),
    )


TOY_STATES = jnp.linspace(0.5, 1.5, 5)[:, None]
TOY_QN = jnp.zeros((1, 1))
TOY_QW = jnp.ones(1)
TOY_DIRS = jnp.array([[1.0]])


def test_true_policy_beats_impostor_exactly():
    model = _toy_model()

    v_true = float(
        _residual_sobolev_loss(
            model, lambda s: s, TOY_STATES, TOY_QN, TOY_QW, 1.0, TOY_DIRS
        )
    )
    v_imp = float(
        _residual_sobolev_loss(
            model, lambda s: 0.9 * s, TOY_STATES, TOY_QN, TOY_QW, 1.0, TOY_DIRS
        )
    )
    assert v_true < 1e-12, f"true policy residual-gradient must be 0, got {v_true}"
    np.testing.assert_allclose(v_imp, 0.01, rtol=1e-6)  # (0.9 - 1)² exactly


def test_gradient_flows():
    model = _toy_model()

    def loss_of_scale(a):
        return _residual_sobolev_loss(
            model, lambda s: a * s, TOY_STATES, TOY_QN, TOY_QW, 1.0, TOY_DIRS
        )

    g = jax.grad(loss_of_scale)(0.9)  # d/da (a-1)² = 2(a-1) = -0.2
    np.testing.assert_allclose(float(g), -0.2, rtol=1e-6)


def test_composite_wiring_off_and_on():
    from deqn_jax.networks.mlp import MLP
    from deqn_jax.training.composite_loss import (
        make_composite_loss,
        prepare_composite_data,
    )
    from deqn_jax.training.linearize import linearize_model

    model, states, qn, qw, _dirs = _bm_setup()
    P, Q = linearize_model(model, verbose=False)
    data = prepare_composite_data(model, P, Q, n_anchor_points=8, verbose=False)
    net = MLP(
        in_features=model.n_states,
        out_features=model.n_policies,
        hidden_sizes=(8,),
        activations=(jax.nn.tanh,),
        key=jax.random.PRNGKey(0),
    )
    key = jax.random.PRNGKey(1)

    fn_off = make_composite_loss(model, data)
    total_off, eq_off = fn_off(model, net, states, key, quad_nodes=qn, quad_weights=qw)
    assert "aux_res_sobolev" not in eq_off

    fn_on = make_composite_loss(
        model, data, res_sobolev_weight=2.0, res_sobolev_n_states=4
    )
    total_on, eq_on = fn_on(model, net, states, key, quad_nodes=qn, quad_weights=qw)
    assert "aux_res_sobolev" in eq_on
    np.testing.assert_allclose(
        float(total_on) - 2.0 * float(eq_on["aux_res_sobolev"]),
        float(total_off),
        rtol=1e-5,
    )


def test_validator_requires_quadrature():
    from deqn_jax.config import TrainConfig
    from deqn_jax.training.state_init import _validate_train_config

    cfg = TrainConfig.from_dict(
        {
            "model": "brock_mirman",
            "episodes": 2,
            "batch_size": 16,
            "verbose": False,
            "loss_type": "composite",
            "expectation_type": "mc",
            "composite_loss": {"res_sobolev_weight": 1.0},
            "network": {"type": "mlp", "hidden_sizes": [8]},
        }
    )
    with pytest.raises(ValueError, match="quadrature"):
        _validate_train_config(cfg)
