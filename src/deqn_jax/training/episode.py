"""Episode simulation using lax.scan for efficient trajectory generation.

An episode consists of:
1. Initialize states (from distribution or previous episode)
2. Simulate T steps: s_{t+1} = step(s_t, π(s_t), ε_t)
3. Collect trajectory for training

Using lax.scan makes the entire episode JIT-compilable.
"""

from typing import Any, Callable, Optional, Tuple

import jax
from jax import Array, lax

from deqn_jax.training.shocks import simulation_step
from deqn_jax.types import EpisodeState, ModelSpec


def simulate_step(
    model: ModelSpec,
    policy_fn: Callable[[Array], Array],
    state: Array,
    key: Array,
    shock_scale: Any = 1.0,
    shock_mask: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Simulate one step of the economic model.

    Delegates to ``deqn_jax.training.shocks.simulation_step`` so that
    training rollouts, evaluation paths, and IRF paths all use one
    shock-drawing contract: Gaussian shocks scaled by ``shock_scale``,
    masked by ``shock_mask``, and optionally a Bernoulli disaster
    indicator passed to step_fn when the model supports it.

    Args:
        model: Model specification
        policy_fn: Policy network
        state: Current state [batch, n_states]
        key: PRNG key for shock
        shock_scale: Curriculum multiplier on all shock draws (scalar
            or per-dimension vector); 0 freezes rollouts deterministically.
        shock_mask: Optional per-dimension 0/1 mask over shocks.

    Returns:
        Tuple of (next_state, shock_used)
    """
    return simulation_step(
        model,
        policy_fn,
        state,
        key,
        shock_scale=shock_scale,
        shock_mask=shock_mask,
    )


def run_episode(
    model: ModelSpec,
    policy_fn: Callable[[Array], Array],
    init_state: Array,
    key: Array,
    episode_length: int = 100,
    shock_scale: Any = 1.0,
    shock_mask: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Run a full episode and collect trajectory.

    Uses lax.scan for efficient JIT compilation. ``shock_scale`` and
    ``shock_mask`` are threaded through to every ``simulate_step`` call
    so that the training-time shock conventions apply uniformly across
    the episode (not just at the loss evaluation).

    Args:
        model: Model specification
        policy_fn: Policy network
        init_state: Initial state [batch, n_states]
        key: PRNG key
        episode_length: Number of steps in episode
        shock_scale: Curriculum shock ramp. Scalar or per-dim vector.
        shock_mask: Optional per-dimension 0/1 mask over shocks.

    Returns:
        Tuple of:
            trajectory: States visited [episode_length, batch, n_states]
            final_state: Last state [batch, n_states]
    """

    def scan_fn(carry: EpisodeState, _) -> Tuple[EpisodeState, Array]:
        """Single step for lax.scan."""
        state, key = carry.state, carry.key
        key, step_key = jax.random.split(key)

        next_state, _ = simulate_step(
            model,
            policy_fn,
            state,
            step_key,
            shock_scale=shock_scale,
            shock_mask=shock_mask,
        )

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
    shock_scale: Any = 1.0,
    shock_mask: Optional[Array] = None,
    init_history: Optional[Array] = None,
) -> Tuple[Array, Array, Array]:
    """Run episode with history-aware policy (LSTM/Transformer).

    Carries a sliding history window [B, H, D] through the episode.
    At each step: policy = policy_fn(history), then shift_history.

    If ``init_history`` is None, the window is initialized as a constant
    tile of ``init_state`` — appropriate for an episodic fresh start. If
    ``init_history`` is passed (from ``TrainState.history_state``), the
    window is continued from the previous rollout's final window, so
    recurrent policies see continuous ergodic trajectories across cycles
    rather than a constant-prefix cold start every time.

    Args:
        model: Model specification
        policy_fn: Policy network taking [B, H, D] -> [B, P]
        init_state: Initial state [batch, n_states]
        key: PRNG key
        episode_length: Number of steps in episode
        history_len: History window size
        init_history: Optional existing history window [B, H, D] to
            continue from; default None rebuilds a constant window.

    Returns:
        Tuple of:
            trajectory: States visited [episode_length, batch, n_states]
            final_state: Last state [batch, n_states]
            final_history: Final history window [batch, H, n_states]
                (to persist across rollouts; feed back as ``init_history``).
    """
    from deqn_jax.training.history import make_constant_history, shift_history

    if init_history is None:
        init_history = make_constant_history(init_state, history_len)  # [B, H, D]

    def scan_fn(carry, _):
        state, history, key = carry
        key, step_key = jax.random.split(key)

        # History-aware policy eval, same shock contract as run_episode.
        def _history_policy_fn(s):
            return policy_fn(history)

        next_state, _ = simulate_step(
            model,
            _history_policy_fn,
            state,
            step_key,
            shock_scale=shock_scale,
            shock_mask=shock_mask,
        )

        # Shift history
        next_history = shift_history(history, next_state)

        return (next_state, next_history, key), state  # output current state

    init_carry = (init_state, init_history, key)

    (final_state, final_history, _), trajectory = lax.scan(
        scan_fn,
        init_carry,
        None,
        length=episode_length,
    )

    return trajectory, final_state, final_history


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
