"""Neural network architectures using Equinox."""

from deqn_jax.networks.linear_plus_mlp import LinearPlusMLP, create_linear_plus_mlp
from deqn_jax.networks.lstm import LSTMPolicy, create_lstm
from deqn_jax.networks.mlp import MLP, ResMLP, create_mlp
from deqn_jax.networks.transformer import TransformerPolicy, create_transformer

__all__ = [
    "MLP",
    "ResMLP",
    "create_mlp",
    "LSTMPolicy",
    "create_lstm",
    "TransformerPolicy",
    "create_transformer",
    "LinearPlusMLP",
    "create_linear_plus_mlp",
]
