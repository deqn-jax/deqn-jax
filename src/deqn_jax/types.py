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

    # Optional: compute derived quantities for monitoring
    # definitions_fn(state, policy, constants) -> Dict[str, Array]
    definitions_fn: Optional[Callable[..., Dict[str, Any]]] = None

    # Policy bounds (for output activation)
    policy_lower: Optional[Array] = None
    policy_upper: Optional[Array] = None

    # Optional: clip states for simulation safety (eval/irf only, NOT training)
    # clip_state_fn(state) -> state
    clip_state_fn: Optional[Callable[..., Array]] = None

    # Optional: box barrier penalty on states (added to loss)
    # state_barrier_fn(state) -> penalty [batch]
    state_barrier_fn: Optional[Callable[..., Array]] = None

    # Optional: shock names for diagnostics/logging
    shock_names: Optional[Tuple[str, ...]] = None

    # Optional: called every ``log_every`` episodes during training, after
    # scalar/histogram logging. Side-effect only (writes plots, logs to TB,
    # etc.). Signature:
    #     cycle_hook(state: TrainState, model: ModelSpec, episode: int) -> None
    # The hook should close over any configuration it needs (output dir,
    # logger, etc.) at model-construction time. See DEQN-MAO's
    # model-level ``Hooks.py`` for the reference pattern; our version
    # differs only in that the plotting primitives themselves live in the
    # shared ``deqn_jax.plots`` module and the hook composes them.
    cycle_hook: Optional[Callable[..., None]] = None

    # Optional declarative bound specs. Format (matches DEQN-MAO upstream):
    #     {"name": {"lower": float, "upper": float,
    #               "penalty_lower": float, "penalty_upper": float}}
    # When set, the loss picks up a soft-penalty term
    #     penalty_lower * mean(max(0, lower - value) ** 2)
    # for each bounded variable (analogous for upper). Missing penalty
    # coefficients default to 1/bound**2 (upstream convention). Use
    # state_bounds for states and definition_bounds for derived quantities
    # computed via ``definitions_fn``. Hard-clipped policies are enforced
    # via ``policy_lower``/``policy_upper`` with the activation layer,
    # separately from this soft mechanism.
    state_bounds: Optional[Dict[str, Dict[str, float]]] = None
    definition_bounds: Optional[Dict[str, Dict[str, float]]] = None

    # Optional: called once before training starts, given the loaded
    # model and the resolved TrainConfig. Returns a (possibly modified)
    # ModelSpec for the trainer to use. Use this to wire config-time
    # decisions into the model -- e.g. disaster swaps `steady_state_fn`
    # to its risky-SS variant when ``constants['p_disaster'] > 0`` and
    # ``config.use_risky_steady_state`` allows it. Called outside JIT,
    # so plain Python branching is fine. Default ``None`` is a no-op
    # (model is used as-declared).
    #
    # Signature: setup_fn(model: ModelSpec, config) -> ModelSpec
    setup_fn: Optional[Callable[..., "ModelSpec"]] = None

    # Optional: called every ``log_every`` cycles in the Python-level
    # logging path, given the model and current training-batch quantities,
    # to return a dict of scalar diagnostics that the trainer prepends
    # to TensorBoard / W&B with the model's namespace prefix. Lets a
    # model expose its own per-equation decompositions, ratio
    # diagnostics, soft-floor saturation fractions, etc. without the
    # framework knowing about the model's internals. Failure is
    # tolerated -- if the hook raises, the trainer logs a warning and
    # continues.
    #
    # Signature:
    #   scalar_diagnostics_fn(
    #       model: ModelSpec,
    #       policy_fn: Callable,        # eqx Module or wrapper for sequence nets
    #       states: Array,              # [batch, n_states] training batch
    #       policy_out: Array,          # [batch, n_policies] policy at states
    #       defs: Dict[str, Array],     # definitions at (states, policy_out)
    #   ) -> Dict[str, float]
    scalar_diagnostics_fn: Optional[Callable[..., Dict[str, float]]] = None

    # Optional: model-specific auxiliary terms for the composite loss
    # (``loss_type='composite'``). Lets a model contribute extra
    # ``aux_*``-keyed losses without the framework knowing about that
    # model's definitions or solver internals. Called inside
    # ``make_composite_loss``'s closure after barrier losses, with the
    # batch-level ``defs`` dict and a kwargs dict of relevant
    # ``CompositeLossConfig`` weights. Returns ``(aux_entries,
    # total_contribution)``: ``aux_entries`` is merged into
    # ``eq_losses`` (so adaptive reweighting / logging see the
    # individual unweighted scalars under their ``aux_*`` keys);
    # ``total_contribution`` is added directly to the running total
    # (the hook applies its own weighting).
    #
    # Used by the disaster model to add Newton-step diagnostic losses
    # (``aux_newton_cond``, ``aux_newton_resid``) that read disaster-
    # specific definitions (``newton_h_prime``, ``newton_residual``).
    # Other models leave this ``None``.
    #
    # Signature:
    #   composite_aux_fn(
    #       model: ModelSpec,
    #       defs: Dict[str, Array],     # batch-level definitions
    #       data,                       # CompositeData (linearization + SS)
    #       weights: Dict[str, float],  # subset of CompositeLossConfig weights
    #   ) -> Tuple[Dict[str, Array], Array]
    composite_aux_fn: Optional[Callable[..., Tuple[Dict[str, Array], Array]]] = None


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
        target_params: Frozen policy copy for target-network style training
        aux_params: Slot for a second trainable module (e.g. value network
            in actor-critic, critic network, learned expectation operator).
            None by default. Default training loop ignores it; only loss
            functions that know about ``aux_params`` will use it.
        aux_opt_state: Optimizer state for ``aux_params`` if trained with
            its own optimizer. ``None`` if aux is trained jointly with the
            primary optimizer.
        history_state: Sliding history window ``[batch, H, n_states]`` for
            sequence policies (LSTM/Transformer, ``network.history_len > 1``).
            Persists across rollouts so recurrent training sees continuous
            ergodic trajectories rather than rebuilding a constant window
            at every cycle. ``None`` for MLP models (``history_len == 1``).
    """

    params: Any  # Equinox model
    opt_state: Any  # Optax optimizer state
    episode_state: Array  # Current states [batch, n_states]
    key: Array  # JAX PRNG key
    step: int
    episode: int
    loss_weights: Array  # [n_eq] per-equation weights
    reweight_state: ReweightState  # adaptive reweighting state
    target_params: Any = None  # Frozen policy for target network (DQN-style)
    aux_params: Any = None      # Auxiliary trainable module (value net, critic, ...)
    aux_opt_state: Any = None   # Optimizer state for aux_params if separate
    history_state: Any = None   # [batch, H, n_states] for sequence policies, else None


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
