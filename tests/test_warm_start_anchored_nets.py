"""The constant-SS warm start must never run on a BK-anchored network.

Anchored nets (``linear_plus_mlp``, ``disaster_policy_net``, ``kf_anchored_mlp``)
start at the Blanchard-Kahn linear policy by construction (``init_scale: 0``).
Fitting them to a CONSTANT steady-state policy teaches the MLP delta to cancel
the linear slope: measured on the shipped disaster recipe (2026-09-02), the
warm start moved the closed-loop spectral radius at the SS from the 0.98699
exogenous floor to 1.14 before the first training step. The dispatch used to
skip the warm start for ``linear_plus_mlp`` only.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deqn_jax.config import NetworkConfig, OptimizerConfig, TrainConfig
from deqn_jax.training.linearize import linearize_model
from deqn_jax.training.state_init import (
    _build_initial_state,
    _resolve_model_for_training,
)


def _cfg(net_type: str, **net_kw) -> TrainConfig:
    return TrainConfig(
        model="disaster",
        episodes=2,
        batch_size=16,
        episode_length=3,
        mc_samples=2,
        expectation_type="quadrature",
        n_quadrature_points=3,
        loss_type="composite",
        warm_start=True,
        network=NetworkConfig(
            type=net_type, hidden_sizes=(16,), init_scale=0.0, **net_kw
        ),
        optimizer=OptimizerConfig(name="adam", learning_rate=1e-3, lr_warmup=1),
        verbose=False,
        seed=0,
    )


@pytest.mark.parametrize("net_type", ["disaster_policy_net", "linear_plus_mlp"])
def test_warm_start_leaves_anchored_net_at_linear_policy(net_type):
    cfg = _cfg(net_type)
    model, n_eq = _resolve_model_for_training(cfg)
    state, *_ = _build_initial_state(
        cfg, model, jax.random.PRNGKey(0), n_eq, cfg.optimizer
    )
    net = state.params

    ss_state, ss_policy = model.steady_state_fn(model.constants)
    ss_state = jnp.asarray(ss_state)
    P, _ = linearize_model(model, verbose=False)

    # Value and slope at the SS are the linearization's.
    np.testing.assert_allclose(
        np.asarray(net(ss_state)), np.asarray(ss_policy), rtol=1e-6
    )
    J = jax.jacobian(net)(ss_state)
    np.testing.assert_allclose(np.asarray(J), np.asarray(P), atol=1e-6)

    # And so is the policy on a small cloud (inside the clip box).
    cloud = ss_state * (
        1
        + jax.random.uniform(
            jax.random.PRNGKey(1), (64, model.n_states), minval=-0.01, maxval=0.01
        )
    )
    lin = jnp.asarray(ss_policy) + (cloud - ss_state) @ P.T
    np.testing.assert_allclose(
        np.asarray(net(cloud)), np.asarray(lin), rtol=1e-5, atol=1e-8
    )
