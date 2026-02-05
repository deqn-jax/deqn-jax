"""Tests for structured configuration system."""

import os
import tempfile

import pytest


class TestOptimizerConfig:
    """Test OptimizerConfig defaults."""

    def test_defaults(self):
        from deqn_jax.config import OptimizerConfig
        cfg = OptimizerConfig()
        assert cfg.name == "adam"
        assert cfg.learning_rate == 1e-3
        assert cfg.grad_clip is None


class TestNetworkConfig:
    """Test NetworkConfig defaults."""

    def test_defaults(self):
        from deqn_jax.config import NetworkConfig
        cfg = NetworkConfig()
        assert cfg.type == "mlp"
        assert cfg.hidden_sizes == (64, 64)
        assert cfg.activation == "tanh"


class TestTrainConfig:
    """Test TrainConfig construction and manipulation."""

    def test_defaults(self):
        from deqn_jax.config import TrainConfig
        cfg = TrainConfig()
        assert cfg.model == "brock_mirman"
        assert cfg.episodes == 1000
        assert cfg.optimizer.name == "adam"
        assert cfg.network.hidden_sizes == (64, 64)

    def test_from_dict_flat(self):
        from deqn_jax.config import TrainConfig
        cfg = TrainConfig.from_dict({
            "model": "disaster",
            "episodes": 500,
        })
        assert cfg.model == "disaster"
        assert cfg.episodes == 500
        assert cfg.optimizer.name == "adam"  # default

    def test_from_dict_nested(self):
        from deqn_jax.config import TrainConfig
        cfg = TrainConfig.from_dict({
            "model": "brock_mirman",
            "optimizer": {
                "name": "ngd",
                "learning_rate": 0.01,
            },
            "network": {
                "hidden_sizes": [128, 128],
            },
        })
        assert cfg.optimizer.name == "ngd"
        assert cfg.optimizer.learning_rate == 0.01
        assert cfg.network.hidden_sizes == (128, 128)

    def test_from_dict_string_optimizer(self):
        """Optimizer can be specified as a plain string."""
        from deqn_jax.config import TrainConfig
        cfg = TrainConfig.from_dict({"optimizer": "sgd"})
        assert cfg.optimizer.name == "sgd"

    def test_from_yaml(self):
        from deqn_jax.config import TrainConfig

        yaml_content = """
model: disaster
episodes: 2000
optimizer:
  name: mao
  learning_rate: 5.0e-4
network:
  hidden_sizes: [128, 64]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            cfg = TrainConfig.from_yaml(f.name)

        os.unlink(f.name)

        assert cfg.model == "disaster"
        assert cfg.episodes == 2000
        assert cfg.optimizer.name == "mao"
        assert cfg.optimizer.learning_rate == 5e-4
        assert cfg.network.hidden_sizes == (128, 64)

    def test_with_overrides(self):
        from deqn_jax.config import TrainConfig
        cfg = TrainConfig()
        new_cfg = cfg.with_overrides({
            "optimizer.learning_rate": "0.01",
            "episodes": "500",
            "optimizer.name": "ngd",
        })
        assert new_cfg.optimizer.learning_rate == 0.01
        assert new_cfg.episodes == 500
        assert new_cfg.optimizer.name == "ngd"

    def test_to_dict(self):
        from deqn_jax.config import TrainConfig
        cfg = TrainConfig(model="test", episodes=42)
        d = cfg.to_dict()
        assert d["model"] == "test"
        assert d["episodes"] == 42
        assert d["optimizer"]["name"] == "adam"


class TestLoadConfig:
    """Test config loading with priority merging."""

    def test_defaults_only(self):
        from deqn_jax.config import load_config
        cfg = load_config()
        assert cfg.model == "brock_mirman"
        assert cfg.optimizer.name == "adam"

    def test_cli_overrides_default(self):
        from deqn_jax.config import load_config
        cfg = load_config(episodes=500, model="disaster")
        assert cfg.episodes == 500
        assert cfg.model == "disaster"

    def test_set_overrides_cli(self):
        """--set overrides should take priority over CLI kwargs."""
        from deqn_jax.config import load_config
        cfg = load_config(
            overrides={"episodes": "100"},
            episodes=500,
        )
        assert cfg.episodes == 100  # --set wins

    def test_yaml_with_cli_override(self):
        from deqn_jax.config import load_config

        yaml_content = """
model: disaster
episodes: 2000
optimizer:
  name: mao
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            cfg = load_config(config_path=f.name, episodes=100)

        os.unlink(f.name)

        assert cfg.model == "disaster"  # from YAML
        assert cfg.episodes == 100  # CLI overrides YAML
        assert cfg.optimizer.name == "mao"  # from YAML


class TestTypeInference:
    """Test _infer_type for CLI --set values."""

    def test_bool(self):
        from deqn_jax.config import _infer_type
        assert _infer_type("true") is True
        assert _infer_type("false") is False
        assert _infer_type("none") is None

    def test_int(self):
        from deqn_jax.config import _infer_type
        assert _infer_type("42") == 42
        assert isinstance(_infer_type("42"), int)

    def test_float(self):
        from deqn_jax.config import _infer_type
        assert _infer_type("1e-3") == 1e-3
        assert isinstance(_infer_type("1e-3"), float)

    def test_tuple(self):
        from deqn_jax.config import _infer_type
        assert _infer_type("64,64,32") == (64, 64, 32)

    def test_string(self):
        from deqn_jax.config import _infer_type
        assert _infer_type("adam") == "adam"
