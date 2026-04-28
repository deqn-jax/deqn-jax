"""MLP policy network using Equinox."""

from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
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

    Outputs are optionally bounded using sigmoid scaling:
        output = lower + (upper - lower) * sigmoid(raw_output)

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


class ResMLP(eqx.Module):
    """MLP with residual (skip) connections between hidden layers.

    Each hidden layer computes: h_{i+1} = act(W_i @ h_i + b_i) + proj(h_i)
    where proj is identity if sizes match, or a learned linear projection
    if hidden sizes differ.

    This improves gradient flow and lets the network learn corrections
    rather than full mappings — helpful for multi-equation PINNs.
    """

    layers: list
    skip_projs: list  # Linear projections for size mismatches (or None)
    activations: tuple = eqx.field(static=True)
    output_lower: Optional[tuple] = eqx.field(static=True)
    output_upper: Optional[tuple] = eqx.field(static=True)
    _has_upper: Optional[tuple] = eqx.field(static=True)
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

        sizes = [in_features] + list(hidden_sizes) + [out_features]
        n_layers = len(sizes) - 1
        use_custom_init = init != "default" and init in INIT_FNS

        # Keys: layers + skip projections (separate split to not change MLP PRNG)
        layer_keys = jax.random.split(key, n_layers)
        skip_key = jax.random.fold_in(key, 999)
        skip_keys = jax.random.split(skip_key, n_layers)

        if use_custom_init:
            init_key = jax.random.fold_in(key, 998)
            init_keys = jax.random.split(init_key, n_layers)

        self.layers = []
        self.skip_projs = []
        for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            layer = eqx.nn.Linear(in_size, out_size, key=layer_keys[i])
            if use_custom_init:
                layer = _apply_init(layer, INIT_FNS[init], init_keys[i])  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
            self.layers.append(layer)

            # Skip connection for hidden layers (not the output layer)
            if i < n_layers - 1:
                if in_size == out_size:
                    self.skip_projs.append(None)  # identity
                else:
                    self.skip_projs.append(
                        eqx.nn.Linear(
                            in_size, out_size, use_bias=False, key=skip_keys[i]
                        )
                    )
            else:
                self.skip_projs.append(None)  # no skip for output layer

    def _forward_single(self, x: Array) -> Array:
        if self.input_shift is not None:
            x = (x - jax.lax.stop_gradient(self.input_shift)) / jax.lax.stop_gradient(
                self.input_scale
            )

        for i, layer in enumerate(self.layers[:-1]):
            residual = x
            x = self.activations[i](layer(x))
            # Add skip connection
            proj = self.skip_projs[i]
            if proj is not None:
                x = x + proj(residual)
            else:
                x = x + residual

        # Output layer (no skip, no activation before bounds)
        x = self.layers[-1](x)

        x = _apply_bounds(x, self.output_lower, self.output_upper, self._has_upper)

        return x

    def __call__(self, x: Array) -> Array:
        if x.ndim == 1:
            return self._forward_single(x)
        else:
            return jax.vmap(self._forward_single)(x)


class MultiHeadMLP(eqx.Module):
    """MLP with separate output heads per policy variable.

    Shared trunk → per-policy linear heads. Gives each policy its own
    output parameters, reducing gradient interference between equations
    that depend on different policies.
    """

    trunk_layers: list
    heads: list  # list of eqx.nn.Linear, one per output
    activations: tuple = eqx.field(static=True)
    output_lower: Optional[tuple] = eqx.field(static=True)
    output_upper: Optional[tuple] = eqx.field(static=True)
    _has_upper: Optional[tuple] = eqx.field(static=True)
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

        # Build trunk (hidden layers only, no output layer)
        sizes = [in_features] + list(hidden_sizes)
        n_trunk = len(sizes) - 1
        use_custom_init = init != "default" and init in INIT_FNS

        key, *trunk_keys = jax.random.split(key, n_trunk + 1)
        if use_custom_init:
            key, *init_keys = jax.random.split(key, n_trunk + 1)

        self.trunk_layers = []
        for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            layer = eqx.nn.Linear(in_size, out_size, key=trunk_keys[i])
            if use_custom_init:
                layer = _apply_init(layer, INIT_FNS[init], init_keys[i])  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
            self.trunk_layers.append(layer)

        # Build per-policy output heads: each hidden_sizes[-1] → 1
        head_keys = jax.random.split(key, out_features)
        self.heads = [
            eqx.nn.Linear(hidden_sizes[-1], 1, key=head_keys[i])
            for i in range(out_features)
        ]

    def _forward_single(self, x: Array) -> Array:
        if self.input_shift is not None:
            x = (x - jax.lax.stop_gradient(self.input_shift)) / jax.lax.stop_gradient(
                self.input_scale
            )

        for i, layer in enumerate(self.trunk_layers):
            x = self.activations[i](layer(x))

        # Each head produces one scalar, concat into [out_features]
        raw = jnp.concatenate([head(x) for head in self.heads], axis=-1)

        raw = _apply_bounds(raw, self.output_lower, self.output_upper, self._has_upper)

        return raw

    def __call__(self, x: Array) -> Array:
        if x.ndim == 1:
            return self._forward_single(x)
        else:
            return jax.vmap(self._forward_single)(x)


class ActorCriticMLP(eqx.Module):
    """Shared-trunk policy + value network for actor-critic DEQN.

    Shared MLP trunk feeds two heads:
      * ``policy_heads`` — one ``eqx.nn.Linear(hidden, 1)`` per policy
        variable, concatenated and bounded (sigmoid scaling) to produce
        the policy vector. Same shape as ``MultiHeadMLP``.
      * ``value_head`` — single ``eqx.nn.Linear(hidden, 1)`` returning
        a scalar V(s). **No bounds**: sigmoid clamping would saturate
        ∂V/∂s, which is the quantity the policy FOCs reference.

    ``__call__(x)`` returns the policy vector so existing
    ``policy_fn(state)`` call sites work unchanged. Use ``.value(x)`` to
    get the scalar critic output (and ``jax.grad(model.value)(state)``
    for the value gradient).
    """

    trunk_layers: list
    policy_heads: list  # list of eqx.nn.Linear(hidden, 1), one per policy
    value_head: eqx.nn.Linear
    activations: tuple = eqx.field(static=True)
    output_lower: Optional[tuple] = eqx.field(static=True)
    output_upper: Optional[tuple] = eqx.field(static=True)
    _has_upper: Optional[tuple] = eqx.field(static=True)
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

        # Build trunk (hidden layers only)
        sizes = [in_features] + list(hidden_sizes)
        n_trunk = len(sizes) - 1
        use_custom_init = init != "default" and init in INIT_FNS

        key, *trunk_keys = jax.random.split(key, n_trunk + 1)
        if use_custom_init:
            key, *init_keys = jax.random.split(key, n_trunk + 1)

        self.trunk_layers = []
        for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            layer = eqx.nn.Linear(in_size, out_size, key=trunk_keys[i])
            if use_custom_init:
                layer = _apply_init(layer, INIT_FNS[init], init_keys[i])  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
            self.trunk_layers.append(layer)

        # Per-policy heads + a separate value head. Distinct PRNG splits so
        # adding the value head doesn't shift the policy-head init relative
        # to MultiHeadMLP for the same seed.
        head_key, value_key = jax.random.split(key, 2)
        head_keys = jax.random.split(head_key, out_features)
        self.policy_heads = [
            eqx.nn.Linear(hidden_sizes[-1], 1, key=head_keys[i])
            for i in range(out_features)
        ]
        self.value_head = eqx.nn.Linear(hidden_sizes[-1], 1, key=value_key)

    def _trunk_single(self, x: Array) -> Array:
        """Run shared trunk on a single input. Returns hidden activations."""
        x = _normalize_input(x, self.input_shift, self.input_scale)
        for i, layer in enumerate(self.trunk_layers):
            x = self.activations[i](layer(x))
        return x

    def _policy_single(self, x: Array) -> Array:
        h = self._trunk_single(x)
        raw = jnp.concatenate([head(h) for head in self.policy_heads], axis=-1)
        return _apply_bounds(raw, self.output_lower, self.output_upper, self._has_upper)

    def _value_single(self, x: Array) -> Array:
        h = self._trunk_single(x)
        # value_head outputs shape [1]; squeeze to scalar so jax.grad works.
        return self.value_head(h).squeeze(-1)

    def policy(self, x: Array) -> Array:
        """Policy output: [n_policies] for [in_features], [batch, n_policies] for [batch, in_features]."""
        if x.ndim == 1:
            return self._policy_single(x)
        return jax.vmap(self._policy_single)(x)

    def value(self, x: Array) -> Array:
        """Scalar value output: scalar for [in_features], [batch] for [batch, in_features]."""
        if x.ndim == 1:
            return self._value_single(x)
        return jax.vmap(self._value_single)(x)

    def __call__(self, x: Array) -> Array:
        """Default forward: returns policy. Existing call sites work unchanged."""
        return self.policy(x)


def create_mlp(
    n_states: int,
    n_policies: int,
    hidden_sizes: Sequence[int] = (64, 64),
    activation: str = "tanh",
    activations: Optional[Sequence[str]] = None,
    init: str = "default",
    policy_lower: Optional[Array] = None,
    policy_upper: Optional[Array] = None,
    multi_head: bool = False,
    skip_connections: bool = False,
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
        skip_connections: Use residual connections between hidden layers
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

    if multi_head:
        cls = MultiHeadMLP
    elif skip_connections:
        cls = ResMLP
    else:
        cls = MLP
    return cls(
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


def create_actor_critic_mlp(
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
) -> ActorCriticMLP:
    """Factory function for the shared-trunk actor-critic MLP.

    Same shape and bounding semantics as ``create_mlp(multi_head=True)``
    for the policy heads, plus an unbounded scalar value head.
    """
    n_hidden = len(hidden_sizes)

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

    return ActorCriticMLP(
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
