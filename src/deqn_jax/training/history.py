"""History utilities for sequence-based policy networks (LSTM, Transformer).

Provides functions for managing sliding history windows that sequence
architectures need to condition on recent state trajectories.
"""

import jax
import jax.numpy as jnp
from jax import Array


def get_history_len(net) -> int:
    """Get the history length of a policy network.

    Returns net.history_len if the attribute exists, else 1 (Markovian).
    """
    return getattr(net, "history_len", 1)


def shift_history(history: Array, new_state: Array) -> Array:
    """Shift history window: drop oldest, append new state.

    Args:
        history: [B, H, D] or [H, D] history window
        new_state: [B, D] or [D] new state to append

    Returns:
        Updated history window with same shape as input
    """
    if history.ndim == 3:
        # [B, H, D] -> drop first timestep, append new_state
        return jnp.concatenate([history[:, 1:, :], new_state[:, None, :]], axis=1)
    else:
        # [H, D] -> unbatched
        return jnp.concatenate([history[1:, :], new_state[None, :]], axis=0)


def make_constant_history(state: Array, history_len: int) -> Array:
    """Create a constant history window by tiling the state.

    Args:
        state: [B, D] or [D] state to tile
        history_len: Number of history steps

    Returns:
        [B, H, D] or [H, D] history window with state repeated
    """
    if state.ndim == 2:
        # [B, D] -> [B, H, D]
        return jnp.tile(state[:, None, :], (1, history_len, 1))
    else:
        # [D] -> [H, D]
        return jnp.tile(state[None, :], (history_len, 1))


def build_history_windows(trajectory: Array, history_len: int) -> Array:
    """Build sliding history windows from a trajectory.

    Args:
        trajectory: [T, B, D] state trajectory from episode
        history_len: Window size H

    Returns:
        [N, H, D] array of history windows where N = (T - H + 1) * B
        Each window is a contiguous H-step subsequence from one batch element.
    """
    T, B, D = trajectory.shape

    # For each valid starting index, extract a window
    # Valid starts: 0, 1, ..., T - H
    n_windows = T - history_len + 1

    def extract_window(start_idx):
        # [H, B, D] slice from trajectory
        window = jax.lax.dynamic_slice(
            trajectory,
            (start_idx, 0, 0),
            (history_len, B, D),
        )
        # Transpose to [B, H, D]
        return jnp.transpose(window, (1, 0, 2))

    # vmap over starting indices -> [n_windows, B, H, D]
    starts = jnp.arange(n_windows)
    all_windows = jax.vmap(extract_window)(starts)

    # Reshape to [(n_windows * B), H, D]
    return all_windows.reshape(-1, history_len, D)
