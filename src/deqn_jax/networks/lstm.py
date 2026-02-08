"""Multi-layer LSTM policy network using Equinox.

Takes a history window of states [B, H, D] and outputs policy [B, P]
for the current (last) timestep. Supports:
- Multi-layer stacked LSTM with configurable depth
- Input normalization (frozen via stop_gradient)
- Mixed softplus/sigmoid output bounding (via common._apply_bounds)
"""

from typing import Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.networks.common import (
    _apply_bounds,
    _normalize_input,
    _sanitize_upper,
)


class LSTMPolicy(eqx.Module):
    """Multi-layer LSTM for history-dependent policy approximation.

    Input: [batch, history_len, n_states] or [history_len, n_states]
    Output: [batch, n_policies] or [n_policies]

    Architecture:
        input [D] -> Linear -> hidden_size
        -> LSTMCell layer 1 -> ... -> LSTMCell layer L
        -> final hidden state -> Linear -> n_policies -> bounds
    """

    input_proj: eqx.nn.Linear
    cells: list  # list of eqx.nn.LSTMCell
    output_proj: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)
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
        hidden_sizes: Sequence[int] = (64,),
        history_len: int = 10,
        output_lower: Optional[Array] = None,
        output_upper: Optional[Array] = None,
        input_shift: Optional[Array] = None,
        input_scale: Optional[Array] = None,
        *,
        key: Array,
    ):
        self.history_len = history_len
        self.output_lower = output_lower
        safe_upper, mask = _sanitize_upper(output_upper, output_lower)
        self.output_upper = safe_upper
        self._has_upper = mask
        self.input_shift = input_shift
        self.input_scale = input_scale

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        else:
            hidden_sizes = list(hidden_sizes)
        self.hidden_size = hidden_sizes[0]
        self.n_layers = len(hidden_sizes)

        # Split keys: input_proj, each LSTM cell, output_proj
        keys = jax.random.split(key, self.n_layers + 2)

        # Input projection: n_states -> hidden_size
        self.input_proj = eqx.nn.Linear(in_features, hidden_sizes[0], key=keys[0])

        # Stacked LSTM cells
        self.cells = []
        for i in range(self.n_layers):
            cell_in = hidden_sizes[i]  # first cell takes from input_proj
            cell_hidden = hidden_sizes[i]
            if i > 0:
                cell_in = hidden_sizes[i - 1]
            # LSTMCell(input_size, hidden_size)
            # For layer 0: input is hidden_sizes[0] (from input_proj)
            # For layer i>0: input is hidden_sizes[i-1] (from previous layer's h)
            self.cells.append(
                eqx.nn.LSTMCell(cell_in, cell_hidden, key=keys[i + 1])
            )

        # Output projection: last hidden_size -> n_policies
        self.output_proj = eqx.nn.Linear(
            hidden_sizes[-1], out_features, key=keys[-1]
        )

    def _forward_single(self, x: Array) -> Array:
        """Forward pass for single sequence [history_len, n_states]."""
        seq_len = x.shape[0]

        # Normalize each timestep's input
        def norm_step(x_t):
            return _normalize_input(x_t, self.input_shift, self.input_scale)

        x = jax.vmap(norm_step)(x)  # [H, D]

        # Project input: [H, D] -> [H, hidden_size]
        x = jax.vmap(lambda x_t: jax.nn.tanh(self.input_proj(x_t)))(x)

        # Scan through each layer sequentially
        layer_input = x  # [H, hidden_sizes[0]]
        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            h_size = cell.hidden_size

            init_state = (jnp.zeros(h_size), jnp.zeros(h_size))

            def scan_fn(carry, x_t, cell=cell):
                (h, c) = cell(x_t, carry)
                return (h, c), h

            (final_h, _), all_h = jax.lax.scan(scan_fn, init_state, layer_input)

            # Next layer takes this layer's hidden states as input
            layer_input = all_h  # [H, hidden_sizes[layer_idx]]

        # Use final hidden state from last layer
        raw = self.output_proj(final_h)
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


def create_lstm(
    n_states: int,
    n_policies: int,
    hidden_sizes: Sequence[int] = (64,),
    history_len: int = 10,
    policy_lower: Optional[Array] = None,
    policy_upper: Optional[Array] = None,
    input_shift: Optional[Array] = None,
    input_scale: Optional[Array] = None,
    *,
    key: Array,
) -> LSTMPolicy:
    """Factory function to create multi-layer LSTM policy network.

    Args:
        n_states: Number of state variables
        n_policies: Number of policy variables
        hidden_sizes: Tuple of hidden sizes (one per LSTM layer)
        history_len: Length of history window
        policy_lower: Lower bounds for outputs
        policy_upper: Upper bounds for outputs
        input_shift: Input normalization shift
        input_scale: Input normalization scale
        key: PRNG key

    Returns:
        Initialized LSTMPolicy
    """
    return LSTMPolicy(
        in_features=n_states,
        out_features=n_policies,
        hidden_sizes=hidden_sizes,
        history_len=history_len,
        output_lower=policy_lower,
        output_upper=policy_upper,
        input_shift=input_shift,
        input_scale=input_scale,
        key=key,
    )
