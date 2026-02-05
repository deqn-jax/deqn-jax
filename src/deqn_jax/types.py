"""Core type definitions for DEQN-JAX.

Uses NamedTuples for pytree compatibility with JAX transformations.
"""

from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import jax.numpy as jnp
from jax import Array


class ModelSpec(NamedTuple):
    """Specification for an economic model.

    A ModelSpec defines everything needed to train a DEQN:
    - Dimensions (states, policies, shocks)
    - Economic equations (equilibrium conditions)
    - State transitions (dynamics)
    - Constants (calibration parameters)

    All functions should be pure and JAX-compatible.

    Example:
        MODEL = ModelSpec(
            name="brock_mirman",
            n_states=2,
            n_policies=1,
            n_shocks=1,
            state_names=["k", "z"],
            policy_names=["sav_rate"],
            constants={"alpha": 1/3, "beta": 0.95, ...},
            equations_fn=equations,
            step_fn=step,
            steady_state_fn=compute_steady_state,
        )
    """

    name: str
    n_states: int
    n_policies: int
    n_shocks: int

    # Function signatures:
    # equations_fn(state, policy, next_state, next_policy, constants) -> Dict[str, Array]
    equations_fn: Callable[..., Dict[str, Array]]

    # step_fn(state, policy, shock, constants) -> next_state
    step_fn: Callable[..., Array]

    # Model constants (calibration parameters)
    constants: Dict[str, float]

    # Optional metadata
    state_names: Optional[Tuple[str, ...]] = None
    policy_names: Optional[Tuple[str, ...]] = None
    equation_names: Optional[Tuple[str, ...]] = None

    # Optional: compute steady state for warm-starting
    # steady_state_fn(constants) -> (ss_state, ss_policy)
    steady_state_fn: Optional[Callable[..., Tuple[Array, Array]]] = None

    # Optional: initial state distribution sampler
    # init_state_fn(key, batch_size, constants) -> state
    init_state_fn: Optional[Callable[..., Array]] = None

    # Policy bounds (for output activation)
    policy_lower: Optional[Array] = None
    policy_upper: Optional[Array] = None


class ReweightState(NamedTuple):
    """Running statistics for adaptive loss reweighting.

    Used by lr_annealing and relobralo strategies to track
    per-equation loss history for dynamic weight adjustment.

    Attributes:
        running_max: EMA of per-equation losses (for lr_annealing)
        prev_losses: Previous step losses (for relobralo)
        init_losses: First step losses (for relobralo)
        initialized: Whether init_losses has been set
    """

    running_max: Array   # [n_eq]
    prev_losses: Array   # [n_eq]
    init_losses: Array   # [n_eq]
    initialized: Array   # scalar bool


def make_reweight_state(n_equations: int) -> "ReweightState":
    """Create initial reweight state for n equations."""
    return ReweightState(
        running_max=jnp.ones(n_equations),
        prev_losses=jnp.zeros(n_equations),
        init_losses=jnp.zeros(n_equations),
        initialized=jnp.array(False),
    )


class TrainState(NamedTuple):
    """Immutable training state for JAX-compatible training loops.

    All state needed for training is bundled here so that train_step
    can be a pure function suitable for jax.jit.

    Attributes:
        params: Equinox model (pytree of parameters)
        opt_state: Optax optimizer state
        episode_state: Current state batch for episode simulation
        key: JAX PRNG key
        step: Current training step
        episode: Current episode number
        loss_weights: Per-equation loss weights [n_eq]
        reweight_state: Adaptive reweighting running statistics
    """

    params: Any  # Equinox model
    opt_state: Any  # Optax optimizer state
    episode_state: Array  # Current states [batch, n_states]
    key: Array  # JAX PRNG key
    step: int
    episode: int
    loss_weights: Array  # [n_eq] per-equation weights
    reweight_state: ReweightState  # adaptive reweighting state


class EpisodeState(NamedTuple):
    """State carried through episode simulation via lax.scan.

    Attributes:
        state: Current economic state [batch, n_states]
        key: PRNG key for shock sampling
    """

    state: Array
    key: Array


class Metrics(NamedTuple):
    """Training metrics from a single step/episode.

    Attributes:
        loss: Mean squared residual
        residuals: Dict of per-equation mean squared residuals
        grad_norm: Gradient L2 norm (optional)
    """

    loss: float
    residuals: Optional[Dict[str, float]] = None
    grad_norm: Optional[float] = None
