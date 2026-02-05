"""MLP policy network using Equinox."""

from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


class MLP(eqx.Module):
    """Multi-layer perceptron for policy approximation.

    Outputs are optionally bounded using sigmoid scaling:
        output = lower + (upper - lower) * sigmoid(raw_output)

    Attributes:
        layers: List of linear layers
        activation: Activation function (applied between layers)
        output_lower: Lower bounds for outputs [n_outputs]
        output_upper: Upper bounds for outputs [n_outputs]
    """

    layers: list
    activation: Callable = eqx.field(static=True)
    output_lower: Optional[Array]
    output_upper: Optional[Array]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: Callable = jax.nn.tanh,
        output_lower: Optional[Array] = None,
        output_upper: Optional[Array] = None,
        *,
        key: Array,
    ):
        """Initialize MLP.

        Args:
            in_features: Number of input features (n_states)
            out_features: Number of output features (n_policies)
            hidden_sizes: Tuple of hidden layer sizes
            activation: Activation function (default: tanh)
            output_lower: Lower bounds for outputs (optional)
            output_upper: Upper bounds for outputs (optional)
            key: JAX PRNG key for initialization
        """
        self.activation = activation
        self.output_lower = output_lower
        self.output_upper = output_upper

        # Build layers
        sizes = [in_features] + list(hidden_sizes) + [out_features]
        keys = jax.random.split(key, len(sizes) - 1)

        self.layers = []
        for i, (in_size, out_size, k) in enumerate(
            zip(sizes[:-1], sizes[1:], keys)
        ):
            self.layers.append(eqx.nn.Linear(in_size, out_size, key=k))

    def _forward_single(self, x: Array) -> Array:
        """Forward pass for single input [in_features]."""
        # Forward through hidden layers with activation
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        # Output layer (no activation before bounds)
        x = self.layers[-1](x)

        # Apply output bounds if specified
        if self.output_lower is not None and self.output_upper is not None:
            # Sigmoid scaling to [lower, upper]
            x = self.output_lower + (self.output_upper - self.output_lower) * jax.nn.sigmoid(x)
        elif self.output_lower is not None:
            # Softplus for lower bound only
            x = self.output_lower + jax.nn.softplus(x)
        elif self.output_upper is not None:
            # Negative softplus for upper bound only
            x = self.output_upper - jax.nn.softplus(-x)

        return x

    def __call__(self, x: Array) -> Array:
        """Forward pass.

        Args:
            x: Input tensor [batch, in_features] or [in_features]

        Returns:
            Output tensor [batch, out_features] or [out_features]
        """
        if x.ndim == 1:
            return self._forward_single(x)
        else:
            # vmap over batch dimension for Equinox Linear compatibility
            return jax.vmap(self._forward_single)(x)


def create_mlp(
    n_states: int,
    n_policies: int,
    hidden_sizes: Sequence[int] = (64, 64),
    activation: str = "tanh",
    policy_lower: Optional[Array] = None,
    policy_upper: Optional[Array] = None,
    *,
    key: Array,
) -> MLP:
    """Factory function to create MLP with common configurations.

    Args:
        n_states: Number of state variables (input dimension)
        n_policies: Number of policy variables (output dimension)
        hidden_sizes: Tuple of hidden layer sizes
        activation: Activation name ("tanh", "relu", "gelu", "silu")
        policy_lower: Lower bounds for policy outputs
        policy_upper: Upper bounds for policy outputs
        key: JAX PRNG key

    Returns:
        Initialized MLP model
    """
    activation_fns = {
        "tanh": jax.nn.tanh,
        "relu": jax.nn.relu,
        "gelu": jax.nn.gelu,
        "silu": jax.nn.silu,
        "softplus": jax.nn.softplus,
    }

    act_fn = activation_fns.get(activation, jax.nn.tanh)

    return MLP(
        in_features=n_states,
        out_features=n_policies,
        hidden_sizes=hidden_sizes,
        activation=act_fn,
        output_lower=policy_lower,
        output_upper=policy_upper,
        key=key,
    )
