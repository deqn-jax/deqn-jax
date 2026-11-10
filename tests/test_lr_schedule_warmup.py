"""A cosine schedule whose warm-up is not shorter than the run is refused.

The cosine part decays over the episodes left after ``lr_warmup``; a run
of ``episodes <= lr_warmup`` leaves none. optax then raised from deep
inside ``warmup_cosine_decay_schedule`` ("requires positive decay_steps,
got -97"), naming neither the config field nor the remedy.
"""

import pytest

from deqn_jax.config import NetworkConfig, OptimizerConfig, TrainConfig
from deqn_jax.optimizers.registry import _build_lr_schedule
from deqn_jax.training.trainer import train_from_config


def _opt(warmup):
    return OptimizerConfig(
        name="adam", learning_rate=1e-3, lr_schedule="cosine", lr_warmup=warmup
    )


@pytest.mark.parametrize("warmup, episodes", [(100, 3), (3, 3)])
def test_warmup_not_shorter_than_run_is_refused(warmup, episodes):
    with pytest.raises(ValueError) as err:
        _build_lr_schedule(_opt(warmup), episodes)
    msg = str(err.value)
    assert "optimizer.lr_warmup" in msg
    assert str(warmup) in msg and str(episodes) in msg
    assert "--set optimizer.lr_warmup=" in msg


def test_warmup_shorter_than_run_builds_the_schedule():
    schedule = _build_lr_schedule(_opt(2), 3)
    assert float(schedule(0)) == 0.0
    assert float(schedule(2)) == pytest.approx(1e-3)


def test_short_run_is_refused_before_training():
    cfg = TrainConfig(
        model="brock_mirman",
        episodes=3,
        batch_size=16,
        network=NetworkConfig(hidden_sizes=(16,)),
        optimizer=_opt(100),
        verbose=False,
    )
    with pytest.raises(ValueError, match="optimizer.lr_warmup"):
        train_from_config(cfg)
