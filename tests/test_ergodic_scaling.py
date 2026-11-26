"""Tests for ergodic_std input scaling and ergodic_std SS resets."""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from deqn_jax.config import NetworkConfig, TrainConfig
from deqn_jax.models import load_model
from deqn_jax.networks.factory import build_policy_net
from deqn_jax.training.checkpointing import load_policy_from_checkpoint
from deqn_jax.training.cycle import apply_ss_reset
from deqn_jax.training.linearize import compute_ergodic_covariance, linearize_model
from deqn_jax.training.trainer import train_from_config


def test_input_scale_mode_ergodic_std_matches_lyapunov_std():
    model = load_model("disaster")
    _, Q = linearize_model(model, verbose=False)
    sigma = compute_ergodic_covariance(Q, model, verbose=False)
    expected = jnp.maximum(jnp.sqrt(jnp.maximum(jnp.diag(sigma), 0.0)), 1e-6)

    net_cfg = NetworkConfig(
        type="disaster_policy_net",
        hidden_sizes=(16,),
        input_scale_mode="ergodic_std",
    )
    net = build_policy_net(model, jax.random.PRNGKey(0), (16,), net_cfg)
    assert jnp.allclose(jnp.asarray(net.mlp.input_scale), expected, rtol=1e-10)


def test_input_scale_mode_validation():
    with pytest.raises(ValueError, match="input_scale_mode"):
        NetworkConfig(type="mlp", input_scale_mode="bogus")
    with pytest.raises(ValueError, match="input_scale_mode"):
        NetworkConfig(type="lstm", input_scale_mode="ergodic_std")


def test_ss_reset_mode_ergodic_std_perturbs_zero_ss_and_bounds_tight_states():
    model = load_model("disaster")
    ss_state, _ = model.steady_state_fn(model.constants)
    _, Q = linearize_model(model, verbose=False)
    erg_std = jnp.sqrt(
        jnp.maximum(jnp.diag(compute_ergodic_covariance(Q, model, verbose=False)), 0.0)
    )
    ep_states = jnp.broadcast_to(ss_state[None, :], (128, model.n_states))
    reset = apply_ss_reset(
        jax.random.PRNGKey(0), ep_states, 1.0, ss_state, reset_std=erg_std
    )
    mp_idx = model.state_names.index("m_p")
    muz_idx = model.state_names.index("mu_z")
    # Zero-SS monetary shock state is actually perturbed (relative mode leaves it at 0)
    assert float(jnp.std(reset[:, mp_idx])) > 0.5 * float(erg_std[mp_idx])
    # Tight state mu_z stays strictly within +-1.5 ergodic std
    max_z = float(
        jnp.max(jnp.abs(reset[:, muz_idx] - ss_state[muz_idx])) / erg_std[muz_idx]
    )
    assert max_z <= 1.5 + 1e-12


def test_unblocked_config_loads_and_checkpoint_roundtrip(tmp_path: Path):
    cfg = TrainConfig.from_yaml("configs/disaster_unblocked_bkpin.yaml")
    assert cfg.ss_reset_mode == "ergodic_std"
    assert cfg.network.input_scale_mode == "ergodic_std"

    ckpt_dir = tmp_path / "bkpin_smoke"
    cfg = cfg.with_overrides(
        {
            "episodes": 2,
            "episode_length": 4,
            "batch_size": 8,
            "n_minibatches_per_epoch": 1,
            "expectation_type": "monomial",
            "network.hidden_sizes": [16],
            "optimizer.lr_warmup": 1,
            "checkpoint_dir": str(ckpt_dir),
            "checkpoint_every": 2,
            "tensorboard_dir": None,
            "verbose": False,
        }
    )
    net, _ = train_from_config(cfg)
    loaded, model = load_policy_from_checkpoint(str(ckpt_dir / "checkpoint_000002.eqx"))
    ss_state, ss_policy = model.steady_state_fn(model.constants)
    P, _ = linearize_model(model, verbose=False)
    assert jnp.allclose(loaded(ss_state), net(ss_state), atol=1e-12)
    assert jnp.allclose(loaded(ss_state), ss_policy, atol=1e-10)
    assert jnp.allclose(jax.jacfwd(loaded)(ss_state), P, atol=1e-10)
