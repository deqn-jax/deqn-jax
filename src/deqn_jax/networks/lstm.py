"""LSTM policy network using Equinox for history-dependent policies."""

from typing import Callable, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


class LSTMPolicy(eqx.Module):
    """LSTM network for history-dependent policy approximation.

    Takes sequence of states and outputs policy for current state.

    Input: [batch, seq_len, n_states]
    Output: [batch, n_policies]
    """

    lstm: eqx.nn.LSTMCell
    linear_in: eqx.nn.Linear
    linear_out: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)
    output_lower: Optional[Array]
    output_upper: Optional[Array]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_size: int = 64,
        output_lower: Optional[Array] = None,
        output_upper: Optional[Array] = None,
        *,
        key: Array,
    ):
        """Initialize LSTM policy network.

        Args:
            in_features: Number of state variables
            out_features: Number of policy variables
            hidden_size: LSTM hidden state dimension
            output_lower: Lower bounds for outputs
            output_upper: Upper bounds for outputs
            key: PRNG key
        """
        k1, k2, k3 = jax.random.split(key, 3)

        self.hidden_size = hidden_size
        self.output_lower = output_lower
        self.output_upper = output_upper

        # Input projection
        self.linear_in = eqx.nn.Linear(in_features, hidden_size, key=k1)

        # LSTM cell
        self.lstm = eqx.nn.LSTMCell(hidden_size, hidden_size, key=k2)

        # Output projection
        self.linear_out = eqx.nn.Linear(hidden_size, out_features, key=k3)

    def __call__(self, x: Array) -> Array:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq_len, in_features] or [seq_len, in_features]

        Returns:
            Output tensor [batch, out_features] or [out_features]
        """
        # Handle unbatched input
        squeeze = x.ndim == 2
        if squeeze:
            x = x[None, :, :]

        batch_size, seq_len, _ = x.shape

        # Process sequence
        def scan_fn(carry, x_t):
            h, c = carry
            x_proj = jax.nn.tanh(self.linear_in(x_t))
            h, c = self.lstm(x_proj, (h, c))
            return (h, c), h

        # Initial hidden state
        init_h = jnp.zeros((batch_size, self.hidden_size))
        init_c = jnp.zeros((batch_size, self.hidden_size))

        # Scan over sequence (vmap over batch)
        def process_batch(x_seq):
            init_state = (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size))
            (final_h, _), _ = jax.lax.scan(
                lambda carry, x_t: (
                    self.lstm(jax.nn.tanh(self.linear_in(x_t)), carry),
                    carry[0],
                ),
                init_state,
                x_seq,
            )
            return final_h

        # vmap over batch dimension
        final_h = jax.vmap(process_batch)(x)  # [batch, hidden_size]

        # Output projection
        out = self.linear_out(final_h)

        # Apply bounds
        if self.output_lower is not None and self.output_upper is not None:
            out = self.output_lower + (self.output_upper - self.output_lower) * jax.nn.sigmoid(out)

        if squeeze:
            out = out[0]

        return out


def create_lstm(
    n_states: int,
    n_policies: int,
    hidden_size: int = 64,
    policy_lower: Optional[Array] = None,
    policy_upper: Optional[Array] = None,
    *,
    key: Array,
) -> LSTMPolicy:
    """Factory function to create LSTM policy network.

    Args:
        n_states: Number of state variables
        n_policies: Number of policy variables
        hidden_size: LSTM hidden dimension
        policy_lower: Lower bounds for outputs
        policy_upper: Upper bounds for outputs
        key: PRNG key

    Returns:
        Initialized LSTMPolicy
    """
    return LSTMPolicy(
        in_features=n_states,
        out_features=n_policies,
        hidden_size=hidden_size,
        output_lower=policy_lower,
        output_upper=policy_upper,
        key=key,
    )
