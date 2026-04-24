"""Pure JAX implementation of Deep Equilibrium Networks for economics."""

from deqn_jax.config import NetworkConfig, OptimizerConfig, TrainConfig, load_config
from deqn_jax.training.trainer import train, train_from_config
from deqn_jax.types import ModelSpec, ReweightState, TrainState, make_reweight_state

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
