"""Transformer policy network using Equinox.

Takes a history window of states [B, H, D] and outputs policy [B, P]
for the current (last) timestep. Supports:
- Learned positional embeddings
- Pre-LN transformer blocks (LayerNorm before attention/FFN)
- Mixed softplus/sigmoid output bounding (via common._apply_bounds)
"""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.networks.common import (
    _apply_bounds,
    _normalize_input,
    _sanitize_upper,
)


class MultiHeadSelfAttention(eqx.Module):
    """Simple multi-head self-attention without dropout.

    Avoids eqx.nn.MultiheadAttention whose Dropout module has a
    non-static `inference` boolean that breaks jax.jit tracing.
    """

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    o_proj: eqx.nn.Linear
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, hidden_dim: int, num_heads: int, *, key: Array):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = eqx.nn.Linear(hidden_dim, hidden_dim, key=k1)
        self.k_proj = eqx.nn.Linear(hidden_dim, hidden_dim, key=k2)
        self.v_proj = eqx.nn.Linear(hidden_dim, hidden_dim, key=k3)
        self.o_proj = eqx.nn.Linear(hidden_dim, hidden_dim, key=k4)

    def __call__(self, x: Array) -> Array:
        """Self-attention on [seq_len, hidden_dim]."""
        seq_len = x.shape[0]
        # Project Q, K, V: [S, D] -> [S, D]
        q = jax.vmap(self.q_proj)(x)
        k = jax.vmap(self.k_proj)(x)
        v = jax.vmap(self.v_proj)(x)

        # Reshape to [S, H, head_dim] then transpose to [H, S, head_dim]
        q = q.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        k = k.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        v = v.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)

        # Scaled dot-product attention: [H, S, S]
        scale = jnp.sqrt(jnp.array(self.head_dim, dtype=q.dtype))
        attn_weights = jnp.matmul(q, k.transpose(0, 2, 1)) / scale
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        # Weighted values: [H, S, head_dim]
        attn_out = jnp.matmul(attn_weights, v)

        # Reshape back: [H, S, head_dim] -> [S, D]
        attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, -1)

        # Output projection
        return jax.vmap(self.o_proj)(attn_out)


class TransformerBlock(eqx.Module):
    """Pre-LN Transformer block: LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual."""

    attn: MultiHeadSelfAttention
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm
    ffn_up: eqx.nn.Linear
    ffn_down: eqx.nn.Linear

    def __init__(self, hidden_dim: int, num_heads: int, *, key: Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.attn = MultiHeadSelfAttention(hidden_dim, num_heads, key=k1)
        self.ln1 = eqx.nn.LayerNorm(hidden_dim)
        self.ln2 = eqx.nn.LayerNorm(hidden_dim)
        ffn_dim = hidden_dim * 4
        self.ffn_up = eqx.nn.Linear(hidden_dim, ffn_dim, key=k2)
        self.ffn_down = eqx.nn.Linear(ffn_dim, hidden_dim, key=k3)

    def __call__(self, x: Array) -> Array:
        """Forward pass for single sequence [seq_len, hidden_dim]."""
        # Pre-LN self-attention + residual
        normed = jax.vmap(self.ln1)(x)
        attn_out = self.attn(normed)
        x = x + attn_out

        # Pre-LN FFN + residual
        normed = jax.vmap(self.ln2)(x)
        ffn_out = jax.vmap(lambda h: self.ffn_down(jax.nn.gelu(self.ffn_up(h))))(normed)
        x = x + ffn_out

        return x


class TransformerPolicy(eqx.Module):
    """Transformer for history-dependent policy approximation.

    Input: [batch, history_len, n_states] or [history_len, n_states]
    Output: [batch, n_policies] or [n_policies]

    Architecture:
        input [H, D] -> normalize -> Linear -> [H, hidden_dim]
        + learned positional embeddings
        -> N TransformerBlocks
        -> extract last timestep [hidden_dim]
        -> Linear -> [n_policies] -> bounds
    """

    input_proj: eqx.nn.Linear
    pos_embed: Array  # [max_history, hidden_dim]
    blocks: list  # list of TransformerBlock
    final_ln: eqx.nn.LayerNorm
    output_proj: eqx.nn.Linear
    hidden_dim: int = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    history_len: int = eqx.field(static=True)
    output_lower: Optional[Array]
    output_upper: Optional[Array]
    _has_upper: Optional[tuple] = eqx.field(static=True)
    input_shift: Optional[Array]
    input_scale: Optional[Array]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        num_heads: int = 4,
        history_len: int = 10,
        output_lower: Optional[Array] = None,
        output_upper: Optional[Array] = None,
        input_shift: Optional[Array] = None,
        input_scale: Optional[Array] = None,
        *,
        key: Array,
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.history_len = history_len
        self.output_lower = output_lower
        safe_upper, mask = _sanitize_upper(output_upper, output_lower)
        self.output_upper = safe_upper
        self._has_upper = mask
        self.input_shift = input_shift
        self.input_scale = input_scale

        # Split keys: input_proj, pos_embed, each block, final_ln, output_proj
        keys = jax.random.split(key, n_layers + 3)

        self.input_proj = eqx.nn.Linear(in_features, hidden_dim, key=keys[0])
        self.pos_embed = jax.random.normal(keys[1], (history_len, hidden_dim)) * 0.02

        self.blocks = [
            TransformerBlock(hidden_dim, num_heads, key=keys[i + 2])
            for i in range(n_layers)
        ]

        self.final_ln = eqx.nn.LayerNorm(hidden_dim)
        self.output_proj = eqx.nn.Linear(hidden_dim, out_features, key=keys[-1])

    def _forward_single(self, x: Array) -> Array:
        """Forward pass for single sequence [history_len, n_states]."""
        # Normalize each timestep
        x = jax.vmap(
            lambda x_t: _normalize_input(x_t, self.input_shift, self.input_scale)
        )(x)

        # Project to hidden dim: [H, D] -> [H, hidden_dim]
        x = jax.vmap(self.input_proj)(x)

        # Add positional embeddings
        x = x + self.pos_embed[: x.shape[0]]

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm on last timestep
        last = self.final_ln(x[-1])

        # Output projection + bounds
        raw = self.output_proj(last)
        return _apply_bounds(raw, self.output_lower, self.output_upper, self._has_upper)

    def __call__(self, x: Array) -> Array:
        """Forward pass.

        Args:
            x: [batch, history_len, n_states] or [history_len, n_states]

        Returns:
            [batch, n_policies] or [n_policies]
        """
        if x.ndim == 2:
            return self._forward_single(x)
        else:
            return jax.vmap(self._forward_single)(x)


def create_transformer(
    n_states: int,
    n_policies: int,
    hidden_dim: int = 64,
    n_layers: int = 2,
    num_heads: int = 4,
    history_len: int = 10,
    policy_lower: Optional[Array] = None,
    policy_upper: Optional[Array] = None,
    input_shift: Optional[Array] = None,
    input_scale: Optional[Array] = None,
    *,
    key: Array,
) -> TransformerPolicy:
    """Factory function to create Transformer policy network.

    Args:
        n_states: Number of state variables
        n_policies: Number of policy variables
        hidden_dim: Transformer hidden dimension (must be divisible by num_heads)
        n_layers: Number of transformer blocks
        num_heads: Number of attention heads
        history_len: Length of history window
        policy_lower: Lower bounds for outputs
        policy_upper: Upper bounds for outputs
        input_shift: Input normalization shift
        input_scale: Input normalization scale
        key: PRNG key

    Returns:
        Initialized TransformerPolicy
    """
    return TransformerPolicy(
        in_features=n_states,
        out_features=n_policies,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        num_heads=num_heads,
        history_len=history_len,
        output_lower=policy_lower,
        output_upper=policy_upper,
        input_shift=input_shift,
        input_scale=input_scale,
        key=key,
    )
