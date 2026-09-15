"""Saved run configs from earlier releases still load.

Every run directory (including the DGX certification record) keeps the fully
resolved ``config.yaml`` of its run. Fields removed since then must be dropped
with a warning, not rejected as unknown, or the certified checkpoints stop
loading. Typing a removed key in a ``--set`` override is still an error.
"""

import pytest

from deqn_jax.config import TrainConfig
from deqn_jax.config.io import REMOVED_FIELDS, load_config


def test_saved_config_with_removed_fields_loads_with_a_warning():
    d = {
        "model": "brock_mirman",
        "network": {"type": "mlp", "hidden_sizes": [8], "multi_head": False},
        "optimizer": {
            "name": "adam",
            "learning_rate": 1e-3,
            "lr_reduce_factor": 0.5,
            "lr_reduce_patience": 500,
        },
    }
    with pytest.warns(UserWarning, match="ignoring removed field"):
        cfg = TrainConfig.from_dict(d)
    assert cfg.network.hidden_sizes == (8,)
    assert cfg.optimizer.learning_rate == 1e-3
    assert not hasattr(cfg.network, "multi_head")
    assert not hasattr(cfg.optimizer, "lr_reduce_factor")


def test_removed_fields_are_really_gone_from_the_models():
    from deqn_jax.config import NetworkConfig, OptimizerConfig

    for f in REMOVED_FIELDS["network"]:
        assert f not in NetworkConfig.model_fields
    for f in REMOVED_FIELDS["optimizer"]:
        assert f not in OptimizerConfig.model_fields


def test_removed_field_in_a_set_override_is_still_an_error():
    with pytest.raises(ValueError, match="Unknown keys"):
        load_config(overrides={"network.multi_head": True})
