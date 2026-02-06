"""Neural network architectures using Equinox."""

from deqn_jax.networks.mlp import MLP, ResMLP, create_mlp
from deqn_jax.networks.lstm import LSTMPolicy, create_lstm

__all__ = ["MLP", "create_mlp", "LSTMPolicy", "create_lstm"]
