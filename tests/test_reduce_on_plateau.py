"""Tests for the ReduceLROnPlateau LR schedule."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Scheduler state-machine tests (unit-level)
# ---------------------------------------------------------------------------

def _make(factor=0.5, patience=3, cooldown=0, min_delta=1e-6, min_lr=1e-5):
    from deqn_jax.optimizers.registry import ReduceLROnPlateau
    return ReduceLROnPlateau(
        initial_lr=1e-3,
        factor=factor,
        patience=patience,
        cooldown=cooldown,
        min_delta=min_delta,
        min_lr=min_lr,
    )


def test_initial_lr_returned_before_first_loss():
    s = _make()
    assert s(0) == pytest.approx(1e-3)


def test_lr_unchanged_while_improving():
    s = _make(patience=3)
    for i, loss in enumerate([0.1, 0.09, 0.08, 0.07, 0.06]):
        lr = s(i, loss)
    assert lr == pytest.approx(1e-3)


def test_lr_drops_after_patience_without_improvement():
    s = _make(factor=0.5, patience=3, cooldown=0)
    s(0, 0.1)
    # Three non-improvements: counter=1, 2, 3 -> at third, drop.
    s(1, 0.1)
    s(2, 0.1)
    lr = s(3, 0.1)
    assert lr == pytest.approx(5e-4)


def test_lr_does_not_fall_below_min_lr():
    s = _make(factor=0.1, patience=1, cooldown=0, min_lr=1e-4)
    s(0, 0.1)
    # Each non-improvement (after patience=1) multiplies by 0.1.
    for i in range(1, 20):
        s(i, 0.1)
    assert s(100, 0.1) >= 1e-4 - 1e-12
    assert s(100, 0.1) == pytest.approx(1e-4)


def test_cooldown_delays_further_decay():
    s = _make(factor=0.5, patience=1, cooldown=5, min_lr=1e-10)
    s(0, 0.1)
    # First decay at step 1 (no improvement after patience=1 step).
    s(1, 0.1)
    lr_after_first_drop = s(2, 0.1)
    assert lr_after_first_drop == pytest.approx(5e-4)
    # During cooldown (5 episodes), LR should stay at 5e-4 even if no
    # improvement.
    for i in range(3, 7):
        assert s(i, 0.1) == pytest.approx(5e-4)
    # After cooldown elapsed, next stagnation should trigger another decay.
    s(7, 0.1)
    lr_after_second_drop = s(8, 0.1)
    assert lr_after_second_drop == pytest.approx(2.5e-4)


def test_improvement_resets_patience_counter():
    s = _make(factor=0.5, patience=3, cooldown=0)
    s(0, 0.1)
    s(1, 0.1)
    s(2, 0.1)          # 2 non-improvements; one more would trigger
    lr_after_improve = s(3, 0.05)   # improvement -> counter resets
    assert lr_after_improve == pytest.approx(1e-3)
    # After the reset, we need 3 more non-improvements to trigger decay.
    s(4, 0.06)         # non-improvement; wait=1
    s(5, 0.06)         # non-improvement; wait=2
    assert s(6, 0.06) == pytest.approx(5e-4)   # wait=3 -> decay


def test_min_delta_filters_noise():
    s = _make(factor=0.5, patience=2, cooldown=0, min_delta=0.01)
    s(0, 0.1)
    s(1, 0.099)        # 0.001 drop < min_delta -> counted as non-improvement
    s(2, 0.098)        # 0.001 drop < min_delta -> non-improvement
    lr = s(3, 0.097)   # third non-improvement -> decay
    assert lr == pytest.approx(5e-4)


# ---------------------------------------------------------------------------
# End-to-end tests (training loop integration)
# ---------------------------------------------------------------------------

def test_reduce_on_plateau_smoke_end_to_end():
    from deqn_jax.config import TrainConfig, NetworkConfig, OptimizerConfig
    from deqn_jax.training.trainer import train_from_config

    cfg = TrainConfig(
        model="brock_mirman",
        episodes=3, batch_size=16, episode_length=8, mc_samples=2,
        network=NetworkConfig(hidden_sizes=(8,)),
        optimizer=OptimizerConfig(
            name="adam", learning_rate=1e-3,
            lr_schedule="reduce_on_plateau",
            lr_reduce_patience=1, lr_reduce_cooldown=0, lr_reduce_factor=0.5,
            lr_min_factor=1e-3,
        ),
        verbose=False, log_every=1,
    )
    _, h = train_from_config(cfg)
    assert np.isfinite(h["loss"][-1])


def test_invalid_schedule_rejected():
    from deqn_jax.config import OptimizerConfig

    with pytest.raises(ValueError):
        OptimizerConfig(name="adam", learning_rate=1e-3, lr_schedule="no_such_schedule")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
