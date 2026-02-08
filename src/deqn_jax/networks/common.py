"""Shared utilities for policy network architectures.

Provides common bounding, activation, and initialization helpers used
by MLP, LSTM, and Transformer policy networks.
"""

from typing import Callable, Optional, Sequence, Tuple

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


def _sanitize_upper(output_upper: Optional[Array], output_lower: Optional[Array]):
    """Replace inf in output_upper with safe finite values + boolean mask.

    Returns (sanitized_upper, has_upper_mask) where:
    - sanitized_upper has inf replaced with lower+1 (safe placeholder)
    - has_upper_mask is a boolean tuple (static, no inf in pytree)
    """
    if output_upper is None:
        return None, None
    mask = jnp.isfinite(output_upper)
    if output_lower is not None:
        safe = jnp.where(mask, output_upper, output_lower + 1.0)
    else:
        safe = jnp.where(mask, output_upper, jnp.zeros_like(output_upper))
    return safe, tuple(bool(m) for m in mask)


def _apply_bounds(
    x: Array,
    output_lower: Optional[Array],
    output_upper: Optional[Array],
    has_upper_mask: Optional[tuple],
) -> Array:
    """Apply per-element output bounding.

    Supports mixed softplus/sigmoid per output:
        - has_upper_mask[i] = True  -> sigmoid: lo + (hi - lo) * sigmoid(x)
        - has_upper_mask[i] = False -> softplus: lo + softplus(x)
        - lower None -> no bounding
    """
    if output_lower is None:
        return x

    lo = jax.lax.stop_gradient(output_lower)

    if output_upper is None:
        return lo + jax.nn.softplus(x)

    hi = jax.lax.stop_gradient(output_upper)
    has_upper = jnp.array(has_upper_mask)

    sigmoid_out = lo + (hi - lo) * jax.nn.sigmoid(x)
    softplus_out = lo + jax.nn.softplus(x)
    return jnp.where(has_upper, sigmoid_out, softplus_out)


def _normalize_input(
    x: Array,
    input_shift: Optional[Array],
    input_scale: Optional[Array],
) -> Array:
    """Apply frozen input normalization."""
    if input_shift is not None:
        x = (x - jax.lax.stop_gradient(input_shift)) / jax.lax.stop_gradient(input_scale)
    return x
