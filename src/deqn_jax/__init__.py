"""Pure JAX implementation of Deep Equilibrium Networks for economics."""

from deqn_jax.types import ModelSpec, TrainState
from deqn_jax.training.trainer import train

__version__ = "0.1.0"
__all__ = ["ModelSpec", "TrainState", "train"]
