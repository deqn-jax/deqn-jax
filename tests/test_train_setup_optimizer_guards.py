"""Setup guards against options a train-step variant would silently drop.

Pins, for every registered optimizer, which configured options the setup
validators reject (the table below records the behaviour from before the
guards were derived from ``OptimizerKind``), and requires that a mid-run
``switch_optimizer`` is held to the same guards at setup time as the same
optimizer configured from the start.
"""

import pytest

from deqn_jax.config import TrainConfig
from deqn_jax.optimizers.registry import list_optimizers
from deqn_jax.training.state_init import (
    _resolve_model_for_training,
    _validate_train_config,
)

# feature -> config overrides layered on the base config
FEATURES = {
    "composite": {"loss_type": "composite"},
    "pcgrad": {"gradient_surgery": "pcgrad"},
    "grad_clip": {"optimizer": {"grad_clip": 1.0}},
    "reweight": {"loss_reweight": "lr_annealing"},
    "weights": {"model": "bm_labor", "loss_weights": [1.0, 2.0]},
    "huber": {"loss_choice": "huber"},
    "barrier": {"barrier_weight": 1.0},
    "coverage": {"coverage": {"enabled": True, "stress_ranges": {"z": [-0.1, -0.05]}}},
    # model-dependent: olg_lifecycle declares combine_fn (two-stage)
    "two_stage": {"model": "olg_lifecycle"},
}

# optimizer -> features rejected at setup
REJECTED = {
    "adam": set(),
    "adamw": set(),
    "lion": set(),
    "muon": set(),
    "ngd": set(),
    "sgd": set(),
    "shampoo": set(),
    "lbfgs": {"pcgrad", "grad_clip", "coverage"},
    "mao": {
        "composite",
        "pcgrad",
        "reweight",
        "weights",
        "huber",
        "barrier",
        "coverage",
    },
}
for _gn in ("gn", "ign", "lm"):
    REJECTED[_gn] = {
        "composite",
        "pcgrad",
        "grad_clip",
        "reweight",
        "weights",
        "huber",
        "barrier",
        "coverage",
        "two_stage",
    }


def _cfg(feature: str, optimizer: str, switch_to: str | None = None) -> TrainConfig:
    over = dict(FEATURES[feature])
    opt = {"name": optimizer, **over.pop("optimizer", {})}
    base = {
        "model": "brock_mirman",
        "verbose": False,
        "network": {"type": "mlp", "hidden_sizes": [8]},
        **over,
        "optimizer": opt,
    }
    if switch_to is not None:
        base.update(switch_optimizer=switch_to, switch_episode=2)
    return TrainConfig.from_dict(base)


def _rejected(cfg: TrainConfig) -> bool:
    try:
        _validate_train_config(cfg)
        _resolve_model_for_training(cfg)
    except (ValueError, NotImplementedError):
        return True
    return False


def test_table_covers_the_registry():
    assert set(REJECTED) == set(list_optimizers())


@pytest.mark.parametrize("name", list_optimizers())
@pytest.mark.parametrize("feature", list(FEATURES))
def test_guard_outcome_per_registered_optimizer(name, feature):
    assert _rejected(_cfg(feature, name)) == (feature in REJECTED[name])


@pytest.mark.parametrize("name", list_optimizers())
@pytest.mark.parametrize("feature", list(FEATURES))
def test_switched_optimizer_is_guarded_like_a_configured_one(name, feature):
    # adam accepts every feature, so any rejection comes from the switch.
    assert _rejected(_cfg(feature, "adam", switch_to=name)) == (
        feature in REJECTED[name]
    )
