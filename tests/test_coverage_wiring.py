"""Validators + bit-identical / divergence guards for coverage wiring."""

import numpy as np
import pytest

from deqn_jax.config import TrainConfig
from deqn_jax.training.state_init import _validate_train_config
from deqn_jax.training.trainer import train_from_config


def _cfg(**over):
    base = {
        "model": "brock_mirman",
        "episodes": 30,
        "episode_length": 20,
        "batch_size": 16,
        "sim_batch": 32,
        "mc_samples": 2,
        "verbose": False,
        "log_every": 1,
        "seed": 0,
        "network": {"type": "mlp", "hidden_sizes": [16, 16], "activation": "tanh"},
        "optimizer": {"name": "adam", "learning_rate": 1e-3},
    }
    base.update(over)
    return TrainConfig.from_dict(base)


def test_coverage_plus_composite_validates():
    # The v1 mutual exclusion was relaxed 2026-07-06: composite∘coverage
    # composes (base residual term = coverage mixture, anchor/jac on top).
    cfg = _cfg(
        loss_type="composite",
        coverage={"enabled": True, "stress_ranges": {"z": (-0.1, -0.05)}},
    )
    _validate_train_config(cfg)  # must not raise


def test_coverage_plus_moment_matching_rejected():
    cfg = _cfg(
        moment_matching={"enabled": True, "dynare_dir": "dynare/brock_mirman"},
        coverage={"enabled": True, "stress_ranges": {"z": (-0.1, -0.05)}},
    )
    with pytest.raises(ValueError):
        _validate_train_config(cfg)


def test_coverage_plus_mao_rejected():
    cfg = _cfg(
        optimizer={"name": "mao"},
        coverage={"enabled": True, "stress_ranges": {"z": (-0.1, -0.05)}},
    )
    with pytest.raises(ValueError):
        _validate_train_config(cfg)


def test_coverage_plus_barrier_rejected():
    cfg = _cfg(
        barrier_weight=1.0,
        coverage={"enabled": True, "stress_ranges": {"z": (-0.1, -0.05)}},
    )
    with pytest.raises(ValueError):
        _validate_train_config(cfg)


def test_coverage_plus_replay_rejected():
    cfg = _cfg(
        replay_buffer={"enabled": True},
        coverage={"enabled": True, "stress_ranges": {"z": (-0.1, -0.05)}},
    )
    with pytest.raises(ValueError):
        _validate_train_config(cfg)


def test_coverage_plus_sequence_net_rejected():
    cfg = _cfg(
        network={"type": "lstm", "hidden_sizes": [16], "history_len": 4},
        coverage={"enabled": True, "stress_ranges": {"z": (-0.1, -0.05)}},
    )
    with pytest.raises(NotImplementedError):
        _validate_train_config(cfg)


def test_coverage_unknown_state_name_rejected():
    # 'znope' is not an irbc state name -> error at model resolution
    cfg = TrainConfig.from_dict(
        {
            "model": "irbc",
            "episodes": 2,
            "batch_size": 16,
            "episode_length": 1,
            "initialize_each_episode": True,
            "expectation_type": "gauss_hermite",
            "n_quadrature_points": 3,
            "verbose": False,
            "network": {"type": "mlp", "hidden_sizes": [8]},
            "coverage": {"enabled": True, "stress_ranges": {"znope": (-0.5, -0.2)}},
        }
    )
    with pytest.raises(ValueError):
        train_from_config(cfg)


def test_bit_identical_when_disabled():
    # coverage block present but disabled == no coverage block at all
    _, h_off = train_from_config(_cfg())
    _, h_dis = train_from_config(
        _cfg(
            coverage={
                "enabled": False,
                "n_stress": 999,
                "stress_ranges": {"z": (-0.1, -0.05)},
            }
        )
    )
    a = np.asarray(h_off["loss"])
    b = np.asarray(h_dis["loss"])
    assert a.shape == b.shape
    np.testing.assert_array_equal(a, b)  # EXACT equality


def test_coverage_changes_trajectory_when_enabled():
    _, h_off = train_from_config(_cfg())
    _, h_on = train_from_config(
        _cfg(
            coverage={
                "enabled": True,
                "n_stress": 32,
                "n_local": 32,
                "rollout_horizon": 4,
                "stress_ranges": {"z": (-0.15, -0.05), "k": (7.0, 9.0)},
            }
        )
    )
    a = np.asarray(h_off["loss"])
    b = np.asarray(h_on["loss"])
    assert np.abs(a - b).max() > 1e-8  # coverage actually feeds gradients
