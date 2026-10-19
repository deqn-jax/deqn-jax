"""SGD, AdamW and Lion are registered STANDARD optimizers again (restored 2026-10-02)."""

import pytest

from deqn_jax.config import OptimizerConfig
from deqn_jax.optimizers.registry import (
    OptimizerKind,
    create_optimizer,
    list_optimizers,
)


@pytest.mark.parametrize("name", ["sgd", "adamw", "lion"])
def test_restored_optimizer_is_registered_as_standard(name):
    assert name in list_optimizers()
    opt, kind = create_optimizer(OptimizerConfig(name=name))
    assert kind is OptimizerKind.STANDARD
    assert opt is not None


def test_weight_decay_is_a_config_field_again():
    assert OptimizerConfig(name="adamw", weight_decay=1e-4).weight_decay == 1e-4


def test_negative_weight_decay_is_rejected():
    with pytest.raises(ValueError, match="weight_decay"):
        OptimizerConfig(name="adamw", weight_decay=-1.0)
