"""Structured configuration for DEQN-JAX training.

Seven Pydantic models: ``TrainConfig`` and the six blocks nested in it
(``OptimizerConfig``, ``NetworkConfig``, ``CompositeLossConfig``,
``ReplayBufferConfig``, ``CoverageConfig``, ``MomentMatchingConfig``), with
YAML loading and CLI override merging.
Priority: --set overrides > CLI args > YAML file > defaults.

This module re-exports the public surface of the ``config`` package, so
``from deqn_jax.config import TrainConfig`` works.
"""

from deqn_jax.config.coverage import CoverageConfig
from deqn_jax.config.io import (
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
    "CoverageConfig",
    "MomentMatchingConfig",
    "ReplayBufferConfig",
    "NetworkConfig",
    "TrainConfig",
    "load_config",
    "_infer_type",
]
