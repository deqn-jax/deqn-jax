"""MLP policy network using Equinox."""

from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
from jax import Array

from deqn_jax.networks.common import (
    INIT_FNS,
    _apply_bounds,
    _apply_init,
    _normalize_input,
    _resolve_activation,
    _sanitize_upper,
    _to_tuple,
)


class MLP(eqx.Module):
    """Multi-layer perceptron for policy approximation.

    Outputs are optionally bounded per element by ``common._apply_bounds``,
    which dispatches on whether that output has a finite upper bound:

        finite upper:   lower + (upper - lower) * sigmoid(raw)
        infinite upper: lower + softplus(raw)
        lower is None:  raw (no bounding at all)

    The per-output choice is frozen at construction into the static
    ``_has_upper`` mask, so a model may mix the two forms.

    Attributes:
        layers: List of linear layers
        activations: Per-layer activation functions (one per hidden layer)
        output_lower: Lower bounds for outputs [n_outputs]
        output_upper: Upper bounds for outputs [n_outputs]
        input_shift: Input normalization shift (subtracted) [n_inputs]
        input_scale: Input normalization scale (divided) [n_inputs]
    """

    layers: list
    activations: tuple = eqx.field(static=True)
    # Static fields below: stored as tuple-of-floats so the optimizer
    # can't write to them via Adam-family second-moment updates. See
    # networks/common.py for rationale.
    output_lower: Optional[tuple] = eqx.field(static=True)
    output_upper: Optional[tuple] = eqx.field(
        static=True
    )  # inf replaced with safe finite values
    _has_upper: Optional[tuple] = eqx.field(static=True)  # per-output sigmoid mask
    input_shift: Optional[tuple] = eqx.field(static=True)
    input_scale: Optional[tuple] = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activations: Sequence[Callable] = (jax.nn.tanh, jax.nn.tanh),
        output_lower: Optional[Array] = None,
        output_upper: Optional[Array] = None,
        input_shift: Optional[Array] = None,
        input_scale: Optional[Array] = None,
        init: str = "default",
        *,
        key: Array,
    ):
        self.activations = tuple(activations)
        self.output_lower = _to_tuple(output_lower)
        safe_upper, mask = _sanitize_upper(output_upper, output_lower)
        self.output_upper = safe_upper
        self._has_upper = mask
        self.input_shift = _to_tuple(input_shift)
        self.input_scale = _to_tuple(input_scale)

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
                layer = _apply_init(layer, INIT_FNS[init], init_keys[i])  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
            self.layers.append(layer)

    def _forward_single(self, x: Array) -> Array:
        """Forward pass for single input [in_features]."""
        x = _normalize_input(x, self.input_shift, self.input_scale)

        # Forward through hidden layers with per-layer activation
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activations[i](layer(x))

        # Output layer (no activation before bounds)
        x = self.layers[-1](x)

        x = _apply_bounds(x, self.output_lower, self.output_upper, self._has_upper)

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
    input_shift: Optional[Array] = None,
    input_scale: Optional[Array] = None,
    *,
    key: Array,
) -> eqx.Module:
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
        input_shift: Input normalization shift, subtracted before the first
            layer. Paired with input_scale; pass both or neither.
        input_scale: Input normalization scale, divided after the shift.
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
        input_shift=input_shift,
        input_scale=input_scale,
        init=init,
        key=key,
    )
