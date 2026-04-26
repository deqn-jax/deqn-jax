"""Shared utilities for policy network architectures.

Provides common bounding, activation, and initialization helpers used
by MLP, LSTM, and Transformer policy networks.

Bound and normalization parameters (``output_lower``, ``output_upper``,
``input_shift``, ``input_scale``) live as **tuples of floats** on
networks, marked ``eqx.field(static=True)`` so they're excluded from
the trainable pytree. This is structural protection against a
previously-observed bug where Adam-family optimizers' second-moment
running averages picked up NaN from a single bad gradient step and
then poisoned these supposedly-frozen fields via ``eqx.apply_updates``.
``stop_gradient`` only zeros the *forward* gradient; it doesn't stop
the optimizer from writing to the leaf. Static-tuple storage does.

The ``_to_tuple`` / ``_to_array`` helpers convert at construction and
forward respectively. JIT hoists the tuple→array conversion inside
the forward as a constant — zero runtime cost.
"""

from typing import Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


def _to_tuple(x) -> Optional[Tuple[float, ...]]:
    """Convert an Array / list to a tuple of floats; ``None`` passes through.

    Used by every network's ``__init__`` to freeze bound and normalization
    parameters into a hashable static field. Float casts via ``float(v)``
    so the resulting tuple is JAX-trace-hashable across runs.
    """
    if x is None:
        return None
    return tuple(float(v) for v in x)


def _to_array(x, dtype=None) -> Array:
    """Convert a static tuple / list / Array to a JAX array for math.

    Caller must guard against ``None`` first (we narrow `Optional[tuple]`
    fields with an `is not None` check at the use site, then pass the
    value here). Inside JIT the tuple is hashable static metadata, so
    XLA constant-folds the conversion.
    """
    if isinstance(x, Array):
        return x
    return jnp.asarray(x, dtype=dtype)


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


def _sanitize_upper(output_upper, output_lower):
    """Replace inf in output_upper with safe finite values + boolean mask.

    Accepts Array or tuple/list inputs (the network ``__init__`` may pass
    either). Returns ``(sanitized_upper_tuple, has_upper_mask)``:
      - ``sanitized_upper_tuple``: tuple of floats, inf entries replaced
        with ``lower + 1.0`` (or ``0.0`` if no lower). Tuple form lets
        callers store as ``eqx.field(static=True)``.
      - ``has_upper_mask``: tuple of bools, one per output dimension.
    """
    if output_upper is None:
        return None, None
    upper_arr = _to_array(output_upper)
    lower_arr = _to_array(output_lower) if output_lower is not None else None
    mask = jnp.isfinite(upper_arr)
    if lower_arr is not None:
        safe = jnp.where(mask, upper_arr, lower_arr + 1.0)
    else:
        safe = jnp.where(mask, upper_arr, jnp.zeros_like(upper_arr))
    return _to_tuple(safe), tuple(bool(m) for m in mask)


def _apply_bounds(
    x: Array,
    output_lower,
    output_upper,
    has_upper_mask: Optional[tuple],
) -> Array:
    """Apply per-element output bounding.

    Supports mixed softplus/sigmoid per output:
        - ``has_upper_mask[i] = True``  → sigmoid: ``lo + (hi - lo) * sigmoid(x)``
        - ``has_upper_mask[i] = False`` → softplus: ``lo + softplus(x)``
        - ``lower`` is ``None`` → no bounding (raw passthrough)

    ``output_lower`` / ``output_upper`` may be Array or tuple. The
    sigmoid branch uses a "safe" upper to keep the *forward* and the
    *backward* both NaN-free even if the stored upper somehow contains
    NaN: the dead branch's ``hi - lo`` would otherwise propagate NaN
    via the where-vjp's ``0 * NaN = NaN`` rule.
    """
    if output_lower is None:
        return x

    lo = jax.lax.stop_gradient(_to_array(output_lower))

    if output_upper is None:
        return lo + jax.nn.softplus(x)

    hi = jax.lax.stop_gradient(_to_array(output_upper))
    has_upper = jnp.array(has_upper_mask)

    # Defensive: replace any non-finite bound entries with safe placeholders
    # so that the dead sigmoid branch (where has_upper is False) cannot
    # NaN-poison the gradient via where's reverse-mode rule.
    hi_safe = jnp.where(jnp.isfinite(hi), hi, lo + 1.0)
    sigmoid_out = lo + (hi_safe - lo) * jax.nn.sigmoid(x)
    softplus_out = lo + jax.nn.softplus(x)
    return jnp.where(has_upper, sigmoid_out, softplus_out)


def _normalize_input(
    x: Array,
    input_shift,
    input_scale,
) -> Array:
    """Apply frozen input normalization.

    ``shift`` and ``scale`` are paired — callers that pass one always
    pass the other. Both may be Array or tuple.
    """
    if input_shift is not None and input_scale is not None:
        shift = jax.lax.stop_gradient(_to_array(input_shift))
        scale = jax.lax.stop_gradient(_to_array(input_scale))
        x = (x - shift) / scale
    return x
