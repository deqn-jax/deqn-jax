"""Pure JAX implementation of Deep Equilibrium Networks for economics."""

from deqn_jax.types import ModelSpec, TrainState, ReweightState, make_reweight_state
from deqn_jax.training.trainer import train, train_from_config
from deqn_jax.config import TrainConfig, OptimizerConfig, NetworkConfig, load_config

__version__ = "0.1.0"
__all__ = [
    "ModelSpec",
    "TrainState",
    "ReweightState",
    "make_reweight_state",
    "train",
    "train_from_config",
    "TrainConfig",
    "OptimizerConfig",
    "NetworkConfig",
    "load_config",
]
