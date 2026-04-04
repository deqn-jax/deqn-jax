"""Episode simulation using lax.scan for efficient trajectory generation.

An episode consists of:
1. Initialize states (from distribution or previous episode)
2. Simulate T steps: s_{t+1} = step(s_t, π(s_t), ε_t)
3. Collect trajectory for training

Using lax.scan makes the entire episode JIT-compilable.
"""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import Array, lax

from deqn_jax.types import EpisodeState, ModelSpec


def simulate_step(
    model: ModelSpec,
    policy_fn: Callable[[Array], Array],
    state: Array,
    key: Array,
) -> Tuple[Array, Array]:
    """Simulate one step of the economic model.

    Args:
        model: Model specification
        policy_fn: Policy network
        state: Current state [batch, n_states]
        key: PRNG key for shock

    Returns:
        Tuple of (next_state, shock_used)
    """
    batch_size = state.shape[0]

    # Sample shock
    shock = jax.random.normal(key, (batch_size, model.n_shocks))

    # Get policy
    policy = policy_fn(state)

    # Transition (soft clip is baked into model step_fn for disaster model)
    next_state = model.step_fn(state, policy, shock, model.constants)

    return next_state, shock


def run_episode(
    model: ModelSpec,
    policy_fn: Callable[[Array], Array],
    init_state: Array,
    key: Array,
    episode_length: int = 100,
) -> Tuple[Array, Array]:
    """Run a full episode and collect trajectory.

    Uses lax.scan for efficient JIT compilation.

    Args:
        model: Model specification
        policy_fn: Policy network
        init_state: Initial state [batch, n_states]
        key: PRNG key
        episode_length: Number of steps in episode

    Returns:
        Tuple of:
            trajectory: States visited [episode_length, batch, n_states]
            final_state: Last state [batch, n_states]
    """

    def scan_fn(carry: EpisodeState, _) -> Tuple[EpisodeState, Array]:
        """Single step for lax.scan."""
        state, key = carry.state, carry.key
        key, step_key = jax.random.split(key)

        next_state, _ = simulate_step(model, policy_fn, state, step_key)

        new_carry = EpisodeState(state=next_state, key=key)
        return new_carry, state  # Output current state (before transition)

    # Initialize
    init_carry = EpisodeState(state=init_state, key=key)

    # Run episode
    final_carry, trajectory = lax.scan(
        scan_fn,
        init_carry,
        None,
        length=episode_length,
    )

    return trajectory, final_carry.state


def sample_initial_states(
    model: ModelSpec,
    key: Array,
    batch_size: int,
) -> Array:
    """Sample initial states for episode.

    If model has init_state_fn, uses that. Otherwise samples
    uniformly around steady state (or uses defaults).

    Args:
        model: Model specification
        key: PRNG key
        batch_size: Number of states to sample

    Returns:
        Initial states [batch, n_states]
    """
    if model.init_state_fn is not None:
        return model.init_state_fn(key, batch_size, model.constants)

    # Default: sample around steady state if available
    if model.steady_state_fn is not None:
        ss_state, _ = model.steady_state_fn(model.constants)
        # Add small perturbation
        noise = jax.random.normal(key, (batch_size, model.n_states)) * 0.1
        return ss_state + noise

    # Fallback: uniform in reasonable range
    return jax.random.uniform(
        key,
        (batch_size, model.n_states),
        minval=0.1,
        maxval=2.0,
    )


def run_episode_with_history(
    model: ModelSpec,
    policy_fn: Callable[[Array], Array],
    init_state: Array,
    key: Array,
    episode_length: int = 100,
    history_len: int = 10,
) -> Tuple[Array, Array]:
    """Run episode with history-aware policy (LSTM/Transformer).

    Carries a sliding history window [B, H, D] through the episode.
    At each step: policy = policy_fn(history), then shift_history.

    History is initialized as constant (init_state repeated H times).

    Args:
        model: Model specification
        policy_fn: Policy network taking [B, H, D] -> [B, P]
        init_state: Initial state [batch, n_states]
        key: PRNG key
        episode_length: Number of steps in episode
        history_len: History window size

    Returns:
        Tuple of:
            trajectory: States visited [episode_length, batch, n_states]
            final_state: Last state [batch, n_states]
    """
    from deqn_jax.training.history import make_constant_history, shift_history

    batch_size = init_state.shape[0]

    # Initialize history: repeat init_state across time axis
    init_history = make_constant_history(init_state, history_len)  # [B, H, D]

    class HistoryCarry(Tuple):
        pass

    def scan_fn(carry, _):
        state, history, key = carry
        key, step_key = jax.random.split(key)

        # Get policy from history window
        policy = policy_fn(history)

        # Sample shock and step
        shock = jax.random.normal(step_key, (batch_size, model.n_shocks))
        next_state = model.step_fn(state, policy, shock, model.constants)

        # Shift history
        next_history = shift_history(history, next_state)

        return (next_state, next_history, key), state  # output current state

    init_carry = (init_state, init_history, key)

    (final_state, _, _), trajectory = lax.scan(
        scan_fn,
        init_carry,
        None,
        length=episode_length,
    )

    return trajectory, final_state


def simulate_trajectory(
    model: ModelSpec,
    policy_fn: Callable[[Array], Array],
    key: Array,
    batch_size: int = 64,
    episode_length: int = 100,
) -> Tuple[Array, Array]:
    """Convenience function: sample initial states and run episode.

    Args:
        model: Model specification
        policy_fn: Policy network
        key: PRNG key
        batch_size: Batch size
        episode_length: Episode length

    Returns:
        Tuple of (trajectory, final_state)
    """
    key, init_key = jax.random.split(key)
    init_state = sample_initial_states(model, init_key, batch_size)
    return run_episode(model, policy_fn, init_state, key, episode_length)
