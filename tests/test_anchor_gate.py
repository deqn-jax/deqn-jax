"""Kink-aware anchor gate (composite_loss.anchor_gate): unit + wiring guards."""

import jax.numpy as jnp
import numpy as np
import pytest

from deqn_jax.training.composite_loss import CompositeData, _anchor_loss


def _toy_data(weights):
    n, n_states, n_pol = 4, 2, 3
    pts = jnp.arange(n * n_states, dtype=jnp.float32).reshape(n, n_states)
    lin = jnp.zeros((n, n_pol))
    return CompositeData(
        P=jnp.zeros((n_pol, n_states)),
        ss_state=jnp.zeros(n_states),
        ss_policy=jnp.zeros(n_pol),
        ergodic_cov_chol=jnp.eye(n_states),
        anchor_points=pts,
        anchor_deviations=pts,
        anchor_lin_policy=lin,
        aux_constants={},
        anchor_weights=None if weights is None else jnp.asarray(weights),
    )


def _policy(x):
    # deterministic nonzero policy: per-point error grows with the state sum
    return jnp.stack([x.sum(), 2.0 * x.sum(), 0.0])


def test_weights_none_is_plain_mean():
    a = _anchor_loss(_policy, _toy_data(None))
    b = _anchor_loss(_policy, _toy_data([1.0, 1.0, 1.0, 1.0]))
    np.testing.assert_allclose(float(a), float(b), rtol=1e-6)


def test_zero_weight_removes_point():
    # weight pattern keeps only point 0; loss must equal the plain anchor
    # loss computed on point 0 alone
    only_first = _anchor_loss(_policy, _toy_data([1.0, 0.0, 0.0, 0.0]))
    d = _toy_data(None)
    d0 = d._replace(
        anchor_points=d.anchor_points[:1], anchor_lin_policy=d.anchor_lin_policy[:1]
    )
    np.testing.assert_allclose(
        float(only_first), float(_anchor_loss(_policy, d0)), rtol=1e-6
    )


def test_disaster_gate_open_at_ss_closed_at_floor():
    from deqn_jax.models import load_model
    from deqn_jax.models.disaster.equations import anchor_gate
    from deqn_jax.models.disaster.variables import SPEC

    model = load_model("disaster")
    ss_state, ss_policy = model.steady_state_fn(model.constants)
    pts = jnp.tile(jnp.asarray(ss_state)[None, :], (3, 1))
    pol = jnp.tile(jnp.asarray(ss_policy)[None, :], (3, 1))

    # row 1: SS as-is (floor slack, R_taylor ≈ 1.018) -> weight ~1
    # row 2: monetary shock m_p pushed deep negative -> Taylor rate through
    #        the floor -> weight ~0
    m_p_idx = SPEC.state_names.index("m_p")
    pts = pts.at[1, m_p_idx].set(-0.05)
    # row 3: moderately negative m_p, near the floor -> intermediate/low
    pts = pts.at[2, m_p_idx].set(-0.012)

    w = np.asarray(anchor_gate(pts, pol, model.constants))
    assert w.shape == (3,)
    assert w[0] > 0.95, f"gate should be open at SS, got {w[0]}"
    assert w[1] < 0.05, f"gate should be closed past the floor, got {w[1]}"
    assert np.all((w >= 0) & (w <= 1))


def test_flag_without_model_hook_raises():
    from deqn_jax.config import TrainConfig
    from deqn_jax.models import load_model
    from deqn_jax.training.composite_loss import _build_custom_loss_fn

    cfg = TrainConfig.from_dict(
        {
            "model": "brock_mirman",
            "episodes": 2,
            "batch_size": 16,
            "verbose": False,
            "loss_type": "composite",
            "composite_loss": {"anchor_gate": True},
            "network": {"type": "mlp", "hidden_sizes": [8]},
        }
    )
    model = load_model("brock_mirman")
    with pytest.raises(ValueError, match="anchor_gate_fn"):
        _build_custom_loss_fn(cfg, model, history_len=1)


def test_prepare_data_gate_wiring_clips_and_stores():
    from deqn_jax.models import load_model
    from deqn_jax.training.composite_loss import prepare_composite_data
    from deqn_jax.training.linearize import linearize_model

    model = load_model("brock_mirman")
    P, Q = linearize_model(model, verbose=False)

    def fake_gate(points, lin_policy, constants):
        n = points.shape[0]
        return jnp.linspace(-0.5, 1.5, n)  # out-of-range on purpose

    data = prepare_composite_data(
        model, P, Q, n_anchor_points=8, verbose=False, anchor_gate_fn=fake_gate
    )
    w = np.asarray(data.anchor_weights)
    assert w.shape == (8,)
    assert w.min() == 0.0 and w.max() == 1.0  # clipped into [0, 1]

    data_off = prepare_composite_data(model, P, Q, n_anchor_points=8, verbose=False)
    assert data_off.anchor_weights is None
