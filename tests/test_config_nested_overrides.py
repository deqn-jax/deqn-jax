"""Every nested config block is reachable through dot-notation overrides.

The flatten/unflatten dispatch used to enumerate the nested blocks by hand in
three places; `coverage` was added to one of them only, so
``--set coverage.enabled=true`` raised "Unknown keys" (2026-09-02 review,
Track B). The dispatch is now derived from ``TrainConfig.model_fields``.
"""

import pytest

from deqn_jax.config import TrainConfig
from deqn_jax.config._base import _ConfigBase
from deqn_jax.config.io import _config_to_flat_dict, _flat_dict_to_config


def _nested_block_names():
    return [
        n
        for n, f in TrainConfig.model_fields.items()
        if isinstance(f.annotation, type) and issubclass(f.annotation, _ConfigBase)
    ]


def test_every_nested_block_is_flattened():
    cfg = TrainConfig(model="brock_mirman")
    flat = _config_to_flat_dict(cfg)
    for block in _nested_block_names():
        assert any(k.startswith(f"{block}.") for k in flat), block
        assert block not in flat


@pytest.mark.parametrize("block", _nested_block_names())
def test_override_reaches_every_nested_block(block):
    cfg = TrainConfig(model="brock_mirman")
    sub = next(iter(type(getattr(cfg, block)).model_fields))
    flat = _config_to_flat_dict(cfg)
    flat[f"{block}.{sub}"] = flat[f"{block}.{sub}"]  # identity override must round-trip
    assert _flat_dict_to_config(flat).model_dump() == cfg.model_dump()


def test_set_coverage_field_reaches_the_block():
    """The 2026-09-02 repro: this raised "Unknown keys ... 'coverage.enabled'"."""
    cfg = TrainConfig.from_yaml("configs/irbc_ewm.yaml")
    assert cfg.coverage.enabled is True
    off = cfg.with_overrides({"coverage.enabled": False, "coverage.n_stress": 7})
    assert off.coverage.enabled is False
    assert off.coverage.n_stress == 7


def test_unknown_nested_key_still_rejected():
    with pytest.raises(ValueError, match="Unknown keys"):
        TrainConfig(model="brock_mirman").with_overrides({"coverage.nope": 1})


def test_constants_override_folds_into_the_constants_dict(tmp_path):
    """``--set constants.<name>=<value>`` reaches one model constant without
    restating the others, and the run's config.yaml records it."""
    cfg = TrainConfig(model="brock_mirman", constants={"alpha": 0.3})
    cfg = cfg.with_overrides({"constants.beta": "0.9", "constants.alpha": "0.35"})
    assert cfg.constants == {"alpha": 0.35, "beta": 0.9}
    path = tmp_path / "config.yaml"
    cfg.to_yaml(str(path))
    assert TrainConfig.from_yaml(str(path)).constants == {"alpha": 0.35, "beta": 0.9}
    with pytest.raises(ValueError, match="Empty constant name"):
        cfg.with_overrides({"constants.": "1"})
