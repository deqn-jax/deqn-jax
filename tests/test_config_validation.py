"""Tests for config validation, type coercion, and edge cases."""

import os
import tempfile

import pytest

from deqn_jax.config import (
    CompositeLossConfig,
    NetworkConfig,
    OptimizerConfig,
    TrainConfig,
    _infer_type,
    load_config,
)


# ---------------------------------------------------------------------------
# OptimizerConfig validation
# ---------------------------------------------------------------------------
class TestOptimizerConfigValidation:
    def test_valid_config_passes(self):
        OptimizerConfig()  # defaults are valid

    def test_all_optimizer_names_valid(self):
        for name in OptimizerConfig.VALID_NAMES:
            OptimizerConfig(name=name)

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown optimizer"):
            OptimizerConfig(name="invalid_optimizer")

    def test_learning_rate_zero_raises(self):
        with pytest.raises(ValueError, match="learning_rate"):
            OptimizerConfig(learning_rate=0)

    def test_learning_rate_negative_raises(self):
        with pytest.raises(ValueError, match="learning_rate"):
            OptimizerConfig(learning_rate=-1e-3)

    def test_grad_clip_negative_raises(self):
        with pytest.raises(ValueError, match="grad_clip"):
            OptimizerConfig(grad_clip=-1.0)

    def test_grad_clip_zero_raises(self):
        with pytest.raises(ValueError, match="grad_clip"):
            OptimizerConfig(grad_clip=0.0)

    def test_grad_clip_none_ok(self):
        cfg = OptimizerConfig(grad_clip=None)
        assert cfg.grad_clip is None

    def test_weight_decay_negative_raises(self):
        with pytest.raises(ValueError, match="weight_decay"):
            OptimizerConfig(weight_decay=-0.01)

    def test_weight_decay_zero_ok(self):
        OptimizerConfig(weight_decay=0.0)

    def test_beta1_zero_raises(self):
        with pytest.raises(ValueError, match="beta1"):
            OptimizerConfig(beta1=0.0)

    def test_beta1_one_raises(self):
        with pytest.raises(ValueError, match="beta1"):
            OptimizerConfig(beta1=1.0)

    def test_beta2_out_of_range_raises(self):
        with pytest.raises(ValueError, match="beta2"):
            OptimizerConfig(beta2=1.5)

    def test_epsilon_zero_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            OptimizerConfig(epsilon=0.0)

    def test_epsilon_negative_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            OptimizerConfig(epsilon=-1e-8)

    def test_invalid_lr_schedule_raises(self):
        with pytest.raises(ValueError, match="lr_schedule"):
            OptimizerConfig(lr_schedule="linear")

    def test_lr_min_factor_negative_raises(self):
        with pytest.raises(ValueError, match="lr_min_factor"):
            OptimizerConfig(lr_min_factor=-0.1)

    def test_lr_min_factor_above_one_raises(self):
        with pytest.raises(ValueError, match="lr_min_factor"):
            OptimizerConfig(lr_min_factor=1.5)

    def test_lr_min_factor_boundary_ok(self):
        OptimizerConfig(lr_min_factor=0.0)
        OptimizerConfig(lr_min_factor=1.0)

    def test_lr_warmup_negative_raises(self):
        with pytest.raises(ValueError, match="lr_warmup"):
            OptimizerConfig(lr_warmup=-1)

    def test_damping_zero_raises(self):
        with pytest.raises(ValueError, match="damping"):
            OptimizerConfig(damping=0.0)

    def test_gn_and_lm_allow_zero_damping(self):
        OptimizerConfig(name="gn", damping=0.0)
        OptimizerConfig(name="lm", damping=0.0)

    def test_decay_out_of_range_raises(self):
        with pytest.raises(ValueError, match="decay"):
            OptimizerConfig(decay=1.0)

    def test_block_size_zero_raises(self):
        with pytest.raises(ValueError, match="block_size"):
            OptimizerConfig(block_size=0)

    def test_precond_update_freq_zero_raises(self):
        with pytest.raises(ValueError, match="precond_update_freq"):
            OptimizerConfig(precond_update_freq=0)

    def test_memory_size_zero_raises(self):
        with pytest.raises(ValueError, match="memory_size"):
            OptimizerConfig(memory_size=0)

    def test_ns_steps_zero_raises(self):
        with pytest.raises(ValueError, match="ns_steps"):
            OptimizerConfig(ns_steps=0)


# ---------------------------------------------------------------------------
# NetworkConfig validation
# ---------------------------------------------------------------------------
class TestNetworkConfigValidation:
    def test_valid_config_passes(self):
        NetworkConfig()

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown network type"):
            NetworkConfig(type="cnn")

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            NetworkConfig(activation="swish")

    def test_invalid_init_raises(self):
        with pytest.raises(ValueError, match="Unknown init"):
            NetworkConfig(init="glorot")

    def test_empty_hidden_sizes_raises(self):
        with pytest.raises(ValueError, match="hidden_sizes must be non-empty"):
            NetworkConfig(hidden_sizes=())

    def test_negative_hidden_size_raises(self):
        with pytest.raises(ValueError, match="hidden_sizes must be > 0"):
            NetworkConfig(hidden_sizes=(64, -1))

    def test_zero_hidden_size_raises(self):
        with pytest.raises(ValueError, match="hidden_sizes must be > 0"):
            NetworkConfig(hidden_sizes=(0,))

    def test_activations_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="activations length"):
            NetworkConfig(hidden_sizes=(64, 64), activations=("tanh",))

    def test_activations_matching_length_ok(self):
        NetworkConfig(hidden_sizes=(64, 32), activations=("tanh", "relu"))

    def test_history_len_zero_raises(self):
        with pytest.raises(ValueError, match="history_len"):
            NetworkConfig(history_len=0)

    def test_num_heads_zero_raises(self):
        with pytest.raises(ValueError, match="num_heads"):
            NetworkConfig(num_heads=0)

    def test_n_layers_zero_raises(self):
        with pytest.raises(ValueError, match="n_layers"):
            NetworkConfig(n_layers=0)

    def test_list_hidden_sizes_coerced_to_tuple(self):
        cfg = NetworkConfig(hidden_sizes=[128, 64])
        assert cfg.hidden_sizes == (128, 64)
        assert isinstance(cfg.hidden_sizes, tuple)

    def test_list_activations_coerced_to_tuple(self):
        cfg = NetworkConfig(
            hidden_sizes=(64, 32),
            activations=["tanh", "relu"],
        )
        assert isinstance(cfg.activations, tuple)


# ---------------------------------------------------------------------------
# CompositeLossConfig validation
# ---------------------------------------------------------------------------
class TestCompositeLossConfigValidation:
    def test_valid_config_passes(self):
        CompositeLossConfig()

    def test_negative_anchor_weight_raises(self):
        with pytest.raises(ValueError, match="anchor_weight"):
            CompositeLossConfig(anchor_weight=-0.1)

    def test_negative_jac_weight_raises(self):
        with pytest.raises(ValueError, match="jac_weight"):
            CompositeLossConfig(jac_weight=-0.01)

    def test_negative_barrier_weight_raises(self):
        with pytest.raises(ValueError, match="barrier_weight"):
            CompositeLossConfig(barrier_weight=-1.0)

    def test_negative_newton_weight_raises(self):
        with pytest.raises(ValueError, match="newton_weight"):
            CompositeLossConfig(newton_weight=-0.5)

    def test_zero_weights_ok(self):
        CompositeLossConfig(
            anchor_weight=0.0,
            jac_weight=0.0,
            barrier_weight=0.0,
            newton_weight=0.0,
        )

    def test_n_anchor_points_zero_raises(self):
        with pytest.raises(ValueError, match="n_anchor_points"):
            CompositeLossConfig(n_anchor_points=0)

    def test_anchor_sigma_zero_raises(self):
        with pytest.raises(ValueError, match="anchor_sigma"):
            CompositeLossConfig(anchor_sigma=0.0)

    def test_anchor_sigma_negative_raises(self):
        with pytest.raises(ValueError, match="anchor_sigma"):
            CompositeLossConfig(anchor_sigma=-1.0)

    def test_leverage_mult_zero_raises(self):
        with pytest.raises(ValueError, match="leverage_mult"):
            CompositeLossConfig(leverage_mult=0.0)


# ---------------------------------------------------------------------------
# TrainConfig validation
# ---------------------------------------------------------------------------
class TestTrainConfigValidation:
    def test_valid_config_passes(self):
        TrainConfig()

    def test_empty_model_raises(self):
        with pytest.raises(ValueError, match="model must be a non-empty"):
            TrainConfig(model="")

    def test_episodes_zero_raises(self):
        with pytest.raises(ValueError, match="episodes must be > 0"):
            TrainConfig(episodes=0)

    def test_episodes_negative_raises(self):
        with pytest.raises(ValueError, match="episodes must be > 0"):
            TrainConfig(episodes=-1)

    def test_batch_size_zero_raises(self):
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            TrainConfig(batch_size=0)

    def test_episode_length_zero_raises(self):
        with pytest.raises(ValueError, match="episode_length must be > 0"):
            TrainConfig(episode_length=0)

    def test_mc_samples_zero_raises(self):
        with pytest.raises(ValueError, match="mc_samples must be > 0"):
            TrainConfig(mc_samples=0)

    def test_seed_negative_raises(self):
        with pytest.raises(ValueError, match="seed must be >= 0"):
            TrainConfig(seed=-1)

    def test_seed_zero_ok(self):
        TrainConfig(seed=0)

    def test_invalid_loss_type_raises(self):
        with pytest.raises(ValueError, match="Unknown loss_type"):
            TrainConfig(loss_type="huber")

    def test_invalid_loss_reweight_raises(self):
        with pytest.raises(ValueError, match="Unknown loss_reweight"):
            TrainConfig(loss_reweight="softmax")

    def test_invalid_gradient_surgery_raises(self):
        with pytest.raises(ValueError, match="Unknown gradient_surgery"):
            TrainConfig(gradient_surgery="cagrad")

    def test_invalid_expectation_type_raises(self):
        with pytest.raises(ValueError, match="Unknown expectation_type"):
            TrainConfig(expectation_type="trapezoid")

    def test_n_quadrature_points_zero_raises(self):
        with pytest.raises(ValueError, match="n_quadrature_points"):
            TrainConfig(n_quadrature_points=0)

    def test_log_every_zero_raises(self):
        with pytest.raises(ValueError, match="log_every must be > 0"):
            TrainConfig(log_every=0)

    def test_curriculum_episodes_negative_raises(self):
        with pytest.raises(ValueError, match="curriculum_episodes"):
            TrainConfig(curriculum_episodes=-1)

    def test_curriculum_start_zero_when_active_raises(self):
        with pytest.raises(ValueError, match="curriculum_start"):
            TrainConfig(curriculum_episodes=100, curriculum_start=0.0)

    def test_curriculum_start_above_one_when_active_raises(self):
        with pytest.raises(ValueError, match="curriculum_start"):
            TrainConfig(curriculum_episodes=100, curriculum_start=1.5)

    def test_curriculum_start_ignored_when_inactive(self):
        # curriculum_start=0.0 is fine when curriculum_episodes=0
        TrainConfig(curriculum_episodes=0, curriculum_start=0.0)

    def test_early_stop_min_delta_negative_raises(self):
        with pytest.raises(ValueError, match="early_stop_min_delta"):
            TrainConfig(early_stop_min_delta=-1e-6)

    def test_switch_optimizer_without_episode_raises(self):
        with pytest.raises(ValueError, match="switch_episode must be set"):
            TrainConfig(switch_optimizer="lbfgs", switch_episode=None)

    def test_switch_optimizer_with_episode_ok(self):
        TrainConfig(switch_optimizer="lbfgs", switch_episode=500)

    def test_negative_loss_weights_raises(self):
        with pytest.raises(ValueError, match="loss_weights"):
            TrainConfig(loss_weights=[1.0, -0.5, 1.0])

    def test_zero_loss_weights_ok(self):
        TrainConfig(loss_weights=[1.0, 0.0, 1.0])

    def test_reweight_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="reweight_alpha"):
            TrainConfig(reweight_alpha=0.0)

    def test_reweight_alpha_one_raises(self):
        with pytest.raises(ValueError, match="reweight_alpha"):
            TrainConfig(reweight_alpha=1.0)

    def test_checkpoint_every_zero_raises(self):
        with pytest.raises(ValueError, match="checkpoint_every must be > 0"):
            TrainConfig(checkpoint_every=0)

    def test_checkpoint_every_negative_raises(self):
        with pytest.raises(ValueError, match="checkpoint_every must be > 0"):
            TrainConfig(checkpoint_every=-1)

    def test_checkpoint_every_none_ok(self):
        cfg = TrainConfig(checkpoint_every=None)
        assert cfg.checkpoint_every is None

    def test_checkpoint_every_positive_ok(self):
        TrainConfig(checkpoint_every=100)

    def test_history_len_exceeds_episode_length_raises(self):
        with pytest.raises(ValueError, match="history_len.*must be <= episode_length"):
            TrainConfig(
                network=NetworkConfig(history_len=50),
                episode_length=10,
            )

    def test_history_len_equals_episode_length_ok(self):
        TrainConfig(
            network=NetworkConfig(history_len=10),
            episode_length=10,
        )

    def test_history_len_less_than_episode_length_ok(self):
        TrainConfig(
            network=NetworkConfig(history_len=5),
            episode_length=100,
        )


class TestNetworkConfigTransformerValidation:
    def test_transformer_hidden_dim_not_divisible_by_num_heads_raises(self):
        with pytest.raises(
            ValueError, match="hidden_dim.*must be divisible by num_heads"
        ):
            NetworkConfig(type="transformer", hidden_sizes=(65,), num_heads=4)

    def test_transformer_hidden_dim_divisible_ok(self):
        NetworkConfig(type="transformer", hidden_sizes=(64,), num_heads=4)

    def test_non_transformer_hidden_dim_indivisible_ok(self):
        # Only enforced for transformer type
        NetworkConfig(type="mlp", hidden_sizes=(65,), num_heads=4)


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------
class TestTypeCoercion:
    def test_yaml_string_to_float(self):
        cfg = OptimizerConfig(learning_rate="0.001")
        assert cfg.learning_rate == 0.001
        assert isinstance(cfg.learning_rate, float)

    def test_yaml_string_to_int_via_float(self):
        # YAML often reads numbers as int; string coercion still works
        cfg = TrainConfig(episodes=100)
        assert cfg.episodes == 100

    def test_scientific_notation_coercion(self):
        cfg = OptimizerConfig(learning_rate="1e-3")
        assert cfg.learning_rate == 1e-3

    def test_composite_loss_coercion(self):
        cfg = CompositeLossConfig(
            anchor_weight="0.1",
            jac_weight="0.01",
            barrier_weight="0.01",
            newton_weight="0.01",
            n_anchor_points="64",
            anchor_sigma="1.0",
            leverage_mult="5.0",
        )
        assert cfg.anchor_weight == 0.1
        assert isinstance(cfg.n_anchor_points, int)
        assert isinstance(cfg.anchor_sigma, float)

    def test_switch_lr_string_coercion(self):
        cfg = TrainConfig(switch_lr="0.01", switch_optimizer="adam", switch_episode=500)
        assert cfg.switch_lr == 0.01
        assert isinstance(cfg.switch_lr, float)

    def test_switch_episode_string_coercion(self):
        cfg = TrainConfig(
            switch_episode="500",
            switch_optimizer="adam",
        )
        assert cfg.switch_episode == 500
        assert isinstance(cfg.switch_episode, int)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_infer_type_negative_int(self):
        assert _infer_type("-42") == -42
        assert isinstance(_infer_type("-42"), int)

    def test_infer_type_negative_float(self):
        assert _infer_type("-1.5") == -1.5

    def test_infer_type_scientific_notation(self):
        assert _infer_type("1e-4") == 1e-4

    def test_infer_type_empty_string(self):
        assert _infer_type("") == ""

    def test_infer_type_whitespace(self):
        assert _infer_type("   ") == "   "

    def test_infer_type_passthrough_non_string(self):
        assert _infer_type(42) == 42
        assert _infer_type(3.14) == 3.14
        assert _infer_type(None) is None

    def test_from_dict_unknown_top_level_key_raises(self):
        with pytest.raises(
            ValueError, match="(?s)Unknown keys in config.*totally_unknown_key"
        ):
            TrainConfig.from_dict(
                {
                    "model": "brock_mirman",
                    "totally_unknown_key": 999,
                }
            )

    def test_from_dict_unknown_optimizer_key_raises(self):
        with pytest.raises(ValueError, match="(?s)Unknown keys in optimizer.*momentum"):
            TrainConfig.from_dict(
                {
                    "optimizer": {"name": "adam", "momentum": 0.9},
                }
            )

    def test_from_dict_unknown_network_key_raises(self):
        with pytest.raises(ValueError, match="(?s)Unknown keys in network.*dropout"):
            TrainConfig.from_dict(
                {
                    "network": {"type": "mlp", "dropout": 0.1},
                }
            )

    def test_from_dict_unknown_composite_loss_key_raises(self):
        with pytest.raises(
            ValueError, match="(?s)Unknown keys in composite_loss.*temperature"
        ):
            TrainConfig.from_dict(
                {
                    "composite_loss": {"anchor_weight": 0.1, "temperature": 1.0},
                }
            )

    def test_from_dict_typo_suggests_correction(self):
        with pytest.raises(ValueError, match="did you mean.*'episodes'"):
            TrainConfig.from_dict({"episode": 500})

    def test_from_dict_optimizer_typo_suggests_correction(self):
        with pytest.raises(ValueError, match="did you mean.*'learning_rate'"):
            TrainConfig.from_dict(
                {
                    "optimizer": {"leaning_rate": 0.01},
                }
            )

    def test_from_dict_empty_sub_dicts(self):
        cfg = TrainConfig.from_dict(
            {
                "optimizer": {},
                "network": {},
                "composite_loss": {},
            }
        )
        assert cfg.optimizer.name == "adam"
        assert cfg.network.hidden_sizes == (64, 64)

    def test_from_yaml_empty_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        ) as f:
            f.write("")
            f.flush()
            cfg = TrainConfig.from_yaml(f.name)
        os.unlink(f.name)
        assert cfg.model == "brock_mirman"

    def test_to_dict_from_dict_roundtrip(self):
        original = TrainConfig(
            model="disaster",
            episodes=500,
            optimizer=OptimizerConfig(name="ngd", learning_rate=0.01),
            network=NetworkConfig(hidden_sizes=(128, 64)),
        )
        d = original.to_dict()
        restored = TrainConfig.from_dict(d)
        assert restored.model == original.model
        assert restored.episodes == original.episodes
        assert restored.optimizer.name == original.optimizer.name
        assert restored.optimizer.learning_rate == original.optimizer.learning_rate
        assert restored.network.hidden_sizes == original.network.hidden_sizes

    def test_to_yaml_from_yaml_roundtrip(self):
        original = TrainConfig(
            model="disaster",
            episodes=500,
            optimizer=OptimizerConfig(name="shampoo", learning_rate=0.005),
            network=NetworkConfig(hidden_sizes=(32, 32, 16)),
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        ) as f:
            path = f.name
        original.to_yaml(path)
        restored = TrainConfig.from_yaml(path)
        os.unlink(path)
        assert restored.model == original.model
        assert restored.episodes == original.episodes
        assert restored.optimizer.name == original.optimizer.name
        assert restored.optimizer.learning_rate == original.optimizer.learning_rate
        assert restored.network.hidden_sizes == original.network.hidden_sizes

    def test_partial_yaml_only_optimizer(self):
        yaml_content = """
optimizer:
  name: lion
  learning_rate: 0.002
"""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        ) as f:
            f.write(yaml_content)
            f.flush()
            cfg = TrainConfig.from_yaml(f.name)
        os.unlink(f.name)
        assert cfg.optimizer.name == "lion"
        assert cfg.optimizer.learning_rate == 0.002
        assert cfg.model == "brock_mirman"  # default
        assert cfg.episodes == 1000  # default

    def test_with_overrides_does_not_mutate_original(self):
        original = TrainConfig()
        new = original.with_overrides({"episodes": "500"})
        assert original.episodes == 1000
        assert new.episodes == 500


# ---------------------------------------------------------------------------
# load_config integration
# ---------------------------------------------------------------------------
class TestLoadConfigIntegration:
    def test_all_three_priority_levels(self):
        yaml_content = """
model: disaster
episodes: 2000
optimizer:
  name: mao
  learning_rate: 0.001
"""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        ) as f:
            f.write(yaml_content)
            f.flush()
            cfg = load_config(
                config_path=f.name,
                overrides={"episodes": "100"},
                episodes=500,  # CLI kwarg, overridden by --set
            )
        os.unlink(f.name)
        assert cfg.model == "disaster"  # from YAML
        assert cfg.episodes == 100  # --set wins over CLI and YAML
        assert cfg.optimizer.name == "mao"  # from YAML

    def test_cli_shortcut_optimizer_maps_to_name(self):
        # 'name' CLI kwarg maps to optimizer.name
        cfg = load_config(name="ngd")
        assert cfg.optimizer.name == "ngd"

    def test_load_config_validates(self):
        with pytest.raises(ValueError, match="episodes must be > 0"):
            load_config(episodes=0)

    def test_load_config_yaml_invalid_raises(self):
        yaml_content = """
episodes: -5
"""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        ) as f:
            f.write(yaml_content)
            f.flush()
            with pytest.raises(ValueError, match="episodes must be > 0"):
                load_config(config_path=f.name)
        os.unlink(f.name)

    def test_load_config_unknown_cli_kwarg_raises(self):
        with pytest.raises(ValueError, match="Unknown CLI config key 'bogus_param'"):
            load_config(bogus_param=42)

    def test_load_config_yaml_unknown_key_raises(self):
        yaml_content = """
model: brock_mirman
precision: fp64
"""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        ) as f:
            f.write(yaml_content)
            f.flush()
            with pytest.raises(
                ValueError, match="(?s)Unknown keys in config.*precision"
            ):
                load_config(config_path=f.name)
        os.unlink(f.name)


# ---------------------------------------------------------------------------
# Strict key validation
# ---------------------------------------------------------------------------
class TestStrictKeyValidation:
    def test_with_overrides_unknown_key_raises(self):
        cfg = TrainConfig()
        with pytest.raises(ValueError, match="Unknown keys in config overrides"):
            cfg.with_overrides({"bogus_key": 42})

    def test_with_overrides_unknown_dot_key_raises(self):
        cfg = TrainConfig()
        with pytest.raises(
            ValueError, match="(?s)Unknown keys in config overrides.*optimizer.momentum"
        ):
            cfg.with_overrides({"optimizer.momentum": 0.9})

    def test_with_overrides_valid_keys_pass(self):
        cfg = TrainConfig()
        new = cfg.with_overrides(
            {
                "episodes": "500",
                "optimizer.learning_rate": "0.01",
                "network.activation": "relu",
            }
        )
        assert new.episodes == 500
        assert new.optimizer.learning_rate == 0.01
        assert new.network.activation == "relu"

    def test_yaml_unknown_optimizer_key_raises(self):
        yaml_content = """
optimizer:
  name: adam
  schedule: cosine
"""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        ) as f:
            f.write(yaml_content)
            f.flush()
            with pytest.raises(
                ValueError, match="(?s)Unknown keys in optimizer.*schedule"
            ):
                TrainConfig.from_yaml(f.name)
        os.unlink(f.name)

    def test_yaml_unknown_network_key_raises(self):
        yaml_content = """
network:
  type: mlp
  batch_norm: true
"""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        ) as f:
            f.write(yaml_content)
            f.flush()
            with pytest.raises(
                ValueError, match="(?s)Unknown keys in network.*batch_norm"
            ):
                TrainConfig.from_yaml(f.name)
        os.unlink(f.name)

    def test_multiple_unknown_keys_all_reported(self):
        with pytest.raises(ValueError, match="foo.*\n.*bar|bar.*\n.*foo"):
            TrainConfig.from_dict({"foo": 1, "bar": 2})

    def test_valid_config_dict_passes(self):
        cfg = TrainConfig.from_dict(
            {
                "model": "disaster",
                "episodes": 500,
                "optimizer": {"name": "ngd", "learning_rate": 0.01},
                "network": {"hidden_sizes": [128, 64]},
                "composite_loss": {"anchor_weight": 0.1},
            }
        )
        assert cfg.model == "disaster"
        assert cfg.optimizer.name == "ngd"
        assert cfg.network.hidden_sizes == (128, 64)
        assert cfg.composite_loss.anchor_weight == 0.1


# ---------------------------------------------------------------------------
# Type validation
# ---------------------------------------------------------------------------
class TestTypeValidation:
    # -- TrainConfig --
    def test_episodes_string_non_numeric_raises(self):
        with pytest.raises(TypeError, match="episodes.*expected int.*got str"):
            TrainConfig(episodes="hello")

    def test_episodes_list_raises(self):
        with pytest.raises(TypeError, match="episodes.*expected int"):
            TrainConfig(episodes=[1, 2])

    def test_episodes_bool_raises(self):
        with pytest.raises(TypeError, match="episodes.*expected int.*got bool"):
            TrainConfig(episodes=True)

    def test_episodes_string_numeric_coerced(self):
        cfg = TrainConfig(episodes="500")
        assert cfg.episodes == 500
        assert isinstance(cfg.episodes, int)

    def test_model_int_raises(self):
        with pytest.raises(TypeError, match="TrainConfig.model.*expected str.*got int"):
            TrainConfig(model=42)

    def test_verbose_int_raises(self):
        with pytest.raises(
            TypeError, match="TrainConfig.verbose.*expected bool.*got int"
        ):
            TrainConfig(verbose=1)

    def test_verbose_string_raises(self):
        with pytest.raises(
            TypeError, match="TrainConfig.verbose.*expected bool.*got str"
        ):
            TrainConfig(verbose="yes")

    def test_batch_size_float_coerced(self):
        # float 64.0 can coerce to int 64
        cfg = TrainConfig(batch_size=64.0)
        assert cfg.batch_size == 64
        assert isinstance(cfg.batch_size, int)

    def test_loss_weights_string_raises(self):
        with pytest.raises(
            TypeError, match="TrainConfig.loss_weights.*expected Optional"
        ):
            TrainConfig(loss_weights="1.0,0.5")

    # -- OptimizerConfig --
    def test_learning_rate_string_non_numeric_raises(self):
        with pytest.raises(
            TypeError, match="optimizer.learning_rate.*expected float.*got str"
        ):
            OptimizerConfig(learning_rate="fast")

    def test_learning_rate_list_raises(self):
        with pytest.raises(TypeError, match="optimizer.learning_rate.*expected float"):
            OptimizerConfig(learning_rate=[0.01])

    def test_learning_rate_bool_raises(self):
        with pytest.raises(
            TypeError, match="optimizer.learning_rate.*expected float.*got bool"
        ):
            OptimizerConfig(learning_rate=True)

    def test_name_int_raises(self):
        with pytest.raises(
            TypeError, match="OptimizerConfig.name.*expected str.*got int"
        ):
            OptimizerConfig(name=42)

    def test_block_size_string_coerced(self):
        cfg = OptimizerConfig(block_size="128")
        assert cfg.block_size == 128

    # -- NetworkConfig --
    def test_hidden_sizes_string_raises(self):
        with pytest.raises(
            TypeError, match="NetworkConfig.hidden_sizes.*expected Tuple"
        ):
            NetworkConfig(hidden_sizes="64,64")

    def test_hidden_sizes_int_raises(self):
        with pytest.raises(
            TypeError, match="NetworkConfig.hidden_sizes.*expected Tuple"
        ):
            NetworkConfig(hidden_sizes=64)

    def test_activation_int_raises(self):
        with pytest.raises(
            TypeError, match="NetworkConfig.activation.*expected str.*got int"
        ):
            NetworkConfig(activation=0)

    def test_multi_head_int_raises(self):
        with pytest.raises(
            TypeError, match="NetworkConfig.multi_head.*expected bool.*got int"
        ):
            NetworkConfig(multi_head=1)

    # -- CompositeLossConfig --
    def test_anchor_weight_list_raises(self):
        with pytest.raises(
            TypeError, match="composite_loss.anchor_weight.*expected float"
        ):
            CompositeLossConfig(anchor_weight=[0.1])

    def test_anchor_weight_bool_raises(self):
        with pytest.raises(
            TypeError, match="composite_loss.anchor_weight.*expected float.*got bool"
        ):
            CompositeLossConfig(anchor_weight=True)

    def test_n_anchor_points_float_coerced(self):
        cfg = CompositeLossConfig(n_anchor_points=64.0)
        assert cfg.n_anchor_points == 64
        assert isinstance(cfg.n_anchor_points, int)

    # -- YAML type errors --
    def test_yaml_wrong_type_raises(self):
        yaml_content = """
episodes: [1, 2, 3]
"""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        ) as f:
            f.write(yaml_content)
            f.flush()
            with pytest.raises(TypeError, match="episodes.*expected int"):
                TrainConfig.from_yaml(f.name)
        os.unlink(f.name)

    def test_yaml_optimizer_wrong_type_raises(self):
        yaml_content = """
optimizer:
  learning_rate: [0.01, 0.001]
"""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        ) as f:
            f.write(yaml_content)
            f.flush()
            with pytest.raises(
                TypeError, match="optimizer.learning_rate.*expected float"
            ):
                TrainConfig.from_yaml(f.name)
        os.unlink(f.name)
