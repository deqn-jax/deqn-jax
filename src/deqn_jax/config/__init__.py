"""Structured configuration for DEQN-JAX training.

Three-plus nested Pydantic models with YAML loading and CLI override merging.
Priority: --set overrides > CLI args > YAML file > defaults.

Split into a package (config/) for readability; this module re-exports the full
public surface so ``from deqn_jax.config import TrainConfig`` etc. keep working.
"""

from deqn_jax.config.io import (
    _check_unknown_keys,
    _config_to_flat_dict,
    _flat_dict_to_config,
    _infer_type,
    load_config,
)
from deqn_jax.config.loss import CompositeLossConfig, MomentMatchingConfig
from deqn_jax.config.network import NetworkConfig
from deqn_jax.config.optimizer import OptimizerConfig
from deqn_jax.config.replay import ReplayBufferConfig
from deqn_jax.config.train import TrainConfig

__all__ = [
    "OptimizerConfig",
    "CompositeLossConfig",
    "MomentMatchingConfig",
    "ReplayBufferConfig",
    "NetworkConfig",
    "TrainConfig",
    "load_config",
    "_config_to_flat_dict",
    "_flat_dict_to_config",
    "_check_unknown_keys",
    "_infer_type",
]
