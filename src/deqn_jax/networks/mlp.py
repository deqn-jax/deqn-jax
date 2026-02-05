"""MLP policy network using Equinox."""

from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


ACTIVATION_FNS = {
    "tanh": jax.nn.tanh,
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
    "silu": jax.nn.silu,
    "softplus": jax.nn.softplus,
}

INIT_FNS = {
    "xavier_normal": jax.nn.initializers.glorot_normal(),
    "xavier_uniform": jax.nn.initializers.glorot_uniform(),
    "he_normal": jax.nn.initializers.he_normal(),
    "he_uniform": jax.nn.initializers.he_uniform(),
    "lecun_normal": jax.nn.initializers.lecun_normal(),
}


def _resolve_activation(name: str) -> Callable:
    """Resolve activation name to callable."""
    return ACTIVATION_FNS.get(name, jax.nn.tanh)


def _apply_init(layer: eqx.nn.Linear, init_fn: Callable, key: Array) -> eqx.nn.Linear:
    """Re-initialize a Linear layer's weights using the given initializer."""
    new_weight = init_fn(key, layer.weight.shape)
    return eqx.tree_at(lambda l: l.weight, layer, new_weight)


class MLP(eqx.Module):
    """Multi-layer perceptron for policy approximation.

    Outputs are optionally bounded using sigmoid scaling:
        output = lower + (upper - lower) * sigmoid(raw_output)

    Attributes:
        layers: List of linear layers
        activations: Per-layer activation functions (one per hidden layer)
        output_lower: Lower bounds for outputs [n_outputs]
        output_upper: Upper bounds for outputs [n_outputs]
    """

    layers: list
    activations: tuple = eqx.field(static=True)
    output_lower: Optional[Array]
    output_upper: Optional[Array]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activations: Sequence[Callable] = (jax.nn.tanh, jax.nn.tanh),
        output_lower: Optional[Array] = None,
        output_upper: Optional[Array] = None,
        init: str = "default",
        *,
        key: Array,
    ):
        self.activations = tuple(activations)
        self.output_lower = output_lower
        self.output_upper = output_upper

        # Build layers
        sizes = [in_features] + list(hidden_sizes) + [out_features]
        n_layers = len(sizes) - 1
        use_custom_init = init != "default" and init in INIT_FNS

        if use_custom_init:
            # Need extra keys for re-initialization
            all_keys = jax.random.split(key, 2 * n_layers)
            layer_keys = all_keys[:n_layers]
            init_keys = all_keys[n_layers:]
        else:
            layer_keys = jax.random.split(key, n_layers)

        self.layers = []
        for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            layer = eqx.nn.Linear(in_size, out_size, key=layer_keys[i])
            if use_custom_init:
                layer = _apply_init(layer, INIT_FNS[init], init_keys[i])
            self.layers.append(layer)

    def _forward_single(self, x: Array) -> Array:
        """Forward pass for single input [in_features]."""
        # Forward through hidden layers with per-layer activation
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activations[i](layer(x))

        # Output layer (no activation before bounds)
        x = self.layers[-1](x)

        # Apply output bounds if specified
        if self.output_lower is not None and self.output_upper is not None:
            x = self.output_lower + (self.output_upper - self.output_lower) * jax.nn.sigmoid(x)
        elif self.output_lower is not None:
            x = self.output_lower + jax.nn.softplus(x)
        elif self.output_upper is not None:
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
            return jax.vmap(self._forward_single)(x)


def create_mlp(
    n_states: int,
    n_policies: int,
    hidden_sizes: Sequence[int] = (64, 64),
    activation: str = "tanh",
    activations: Optional[Sequence[str]] = None,
    init: str = "default",
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
        activation: Activation name for all hidden layers (default: "tanh")
        activations: Per-layer activation names (overrides activation if set)
        init: Weight initialization ("xavier_normal", "xavier_uniform",
              "he_normal", "he_uniform", "lecun_normal", "default")
        policy_lower: Lower bounds for policy outputs
        policy_upper: Upper bounds for policy outputs
        key: JAX PRNG key

    Returns:
        Initialized MLP model
    """
    n_hidden = len(hidden_sizes)

    # Resolve per-layer activations
    if activations is not None:
        if len(activations) != n_hidden:
            raise ValueError(
                f"activations length ({len(activations)}) must match "
                f"hidden_sizes length ({n_hidden})"
            )
        act_fns = tuple(_resolve_activation(a) for a in activations)
    else:
        act_fn = _resolve_activation(activation)
        act_fns = tuple(act_fn for _ in range(n_hidden))

    return MLP(
        in_features=n_states,
        out_features=n_policies,
        hidden_sizes=hidden_sizes,
        activations=act_fns,
        output_lower=policy_lower,
        output_upper=policy_upper,
        init=init,
        key=key,
    )
