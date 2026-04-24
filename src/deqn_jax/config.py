"""Structured configuration for DEQN-JAX training.

Three nested Pydantic models with YAML loading and CLI override merging.

Priority: --set overrides > CLI args > YAML file > defaults
"""

from __future__ import annotations

import copy
from difflib import get_close_matches
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

# ---------------------------------------------------------------------------
# Base class: wraps Pydantic ValidationError → ValueError / TypeError
# ---------------------------------------------------------------------------

class _ConfigBase(BaseModel):
    """Shared base that converts Pydantic ValidationError into the
    ValueError / TypeError that existing callers expect."""

    model_config = ConfigDict(extra="forbid")

    def __init__(self, **data: Any):
        try:
            super().__init__(**data)
        except ValidationError as exc:
            _reraise_validation_error(exc, type(self).__name__)

    @classmethod
    def model_validate(cls, obj: Any, **kwargs):
        try:
            return super().model_validate(obj, **kwargs)
        except ValidationError as exc:
            _reraise_validation_error(exc, cls.__name__)


def _reraise_validation_error(exc: ValidationError, cls_name: str):
    """Convert a Pydantic ValidationError into ValueError or TypeError.

    Preserves the original error messages that tests match against.
    """
    errors = exc.errors()
    if len(errors) == 1:
        err = errors[0]
        msg = err.get("msg", "")
        loc = ".".join(str(p) for p in err.get("loc", ()))
        err_type = err.get("type", "")

        # Our custom validators raise with "Value error, <message>"
        if msg.startswith("Value error, "):
            raise ValueError(msg[len("Value error, "):]) from None

        # Bool rejection from our validators
        if "expected" in msg and "got bool" in msg:
            raise TypeError(msg) from None

        # Pydantic's built-in type errors
        if err_type in (
            "int_type", "int_parsing", "float_type", "float_parsing",
            "string_type", "bool_type", "tuple_type", "list_type",
        ):
            expected = _pydantic_type_to_name(err_type)
            inp = err.get("input")
            actual = type(inp).__name__
            raise TypeError(
                f"{cls_name}.{loc}: expected {expected}, "
                f"got {actual} ({inp!r})"
            ) from None

        # extra_forbidden (unknown key via ConfigDict(extra='forbid'))
        if err_type == "extra_forbidden":
            raise ValueError(str(exc)) from None

    # Multiple errors or unrecognised pattern — use first error
    first = errors[0]
    msg = first.get("msg", str(exc))
    if msg.startswith("Value error, "):
        raise ValueError(msg[len("Value error, "):]) from None
    raise ValueError(str(exc)) from None


def _pydantic_type_to_name(err_type: str) -> str:
    """Map Pydantic error type codes to human-readable type names."""
    mapping = {
        "int_type": "int",
        "int_parsing": "int",
        "float_type": "float",
        "float_parsing": "float",
        "string_type": "str",
        "bool_type": "bool",
        "tuple_type": "Tuple[int, ...]",
        "list_type": "List[float]",
    }
    return mapping.get(err_type, err_type)


# ---------------------------------------------------------------------------
# Shared coercion helpers (called from field_validator one-liners)
# ---------------------------------------------------------------------------

def _coerce_float(v: Any, prefix: str) -> Any:
    """Reject bool, coerce str→float, reject list/dict/tuple."""
    if isinstance(v, bool):
        raise TypeError(f"{prefix}: expected float, got bool ({v!r})")
    if isinstance(v, (list, dict, tuple)):
        raise TypeError(
            f"{prefix}: expected float, got {type(v).__name__} ({v!r})"
        )
    if isinstance(v, str):
        try:
            return float(v)
        except (TypeError, ValueError):
            raise TypeError(f"{prefix}: expected float, got str ({v!r})") from None
    return v


def _coerce_int(v: Any, prefix: str) -> Any:
    """Reject bool, coerce str→int, coerce float→int."""
    if isinstance(v, bool):
        raise TypeError(f"{prefix}: expected int, got bool ({v!r})")
    if isinstance(v, str):
        try:
            return int(v)
        except (TypeError, ValueError):
            raise TypeError(f"{prefix}: expected int, got str ({v!r})") from None
    if isinstance(v, float):
        return int(v)
    return v


def _coerce_optional_float(v: Any, prefix: str) -> Any:
    """Like _coerce_float but passes None through."""
    if v is None:
        return v
    return _coerce_float(v, prefix)


def _coerce_optional_int(v: Any, prefix: str) -> Any:
    """Like _coerce_int but passes None through."""
    if v is None:
        return v
    return _coerce_int(v, prefix)


# ---------------------------------------------------------------------------
# OptimizerConfig
# ---------------------------------------------------------------------------

class OptimizerConfig(_ConfigBase):
    """Optimizer configuration."""

    model_config = ConfigDict(extra="forbid", coerce_numbers_to_str=False)

    name: str = Field(default="adam", description="Optimizer name. Options: `adam`, `sgd`, `adamw`, `lion`, `muon`, `ngd`, `shampoo`, `lbfgs`, `mao`, `mao_kfac`, `gn`, `lm`.")
    learning_rate: float = Field(default=1e-3, description="Peak learning rate (or constant LR when `lr_schedule='constant'`).")
    grad_clip: Optional[float] = Field(default=None, description="Global gradient-norm clipping. None disables.")
    weight_decay: float = Field(default=0.0, description="L2 weight decay (used by adamw / adam / sgd).")
    beta1: float = Field(default=0.9, description="Adam / MAO first-moment decay.")
    beta2: float = Field(default=0.999, description="Adam / MAO second-moment decay.")
    epsilon: float = Field(default=1e-8, description="Adam / MAO numerical floor.")
    damping: float = Field(default=1e-4, description="NGD preconditioner damping (adds to Fisher diagonal).")
    decay: float = Field(default=0.999, description="NGD / Shampoo preconditioner EMA decay.")
    block_size: int = Field(default=64, description="Shampoo Kronecker block size.")
    precond_update_freq: int = Field(default=10, description="Shampoo preconditioner update frequency.")
    memory_size: int = Field(default=10, description="L-BFGS history size.")
    ns_steps: int = Field(default=5, description="Muon Newton-Schulz iteration count.")
    lr_schedule: str = Field(default="constant", description="LR schedule: `constant`, `cosine`, or `reduce_on_plateau`.")
    lr_warmup: int = Field(default=0, description="Linear warmup episodes before `lr_schedule` kicks in.")
    lr_min_factor: float = Field(default=0.0, description="Minimum LR as a fraction of peak (cosine / reduce_on_plateau floor).")

    lr_reduce_factor: float = Field(default=0.5, description="ReduceLROnPlateau: multiply LR by this factor on plateau.")
    lr_reduce_patience: int = Field(default=500, description="ReduceLROnPlateau: episodes without improvement before decay.")
    lr_reduce_cooldown: int = Field(default=100, description="ReduceLROnPlateau: episodes to wait after a decay before resuming monitoring.")
    lr_reduce_min_delta: float = Field(default=1e-6, description="ReduceLROnPlateau: minimum loss drop that counts as improvement.")
    # Lower bound on LR as a fraction of initial. Reusing lr_min_factor
    # (already present for cosine) keeps config surface small.

    VALID_NAMES: ClassVar[frozenset] = frozenset({
        "adam", "sgd", "adamw", "lion", "muon",
        "ngd", "shampoo", "lbfgs", "mao", "mao_kfac", "gn", "lm",
    })
    VALID_LR_SCHEDULES: ClassVar[frozenset] = frozenset({"constant", "cosine", "reduce_on_plateau"})

    @field_validator(
        "learning_rate", "weight_decay", "beta1", "beta2", "epsilon",
        "damping", "decay", "lr_min_factor",
        "lr_reduce_factor", "lr_reduce_min_delta",
        mode="before",
    )
    @classmethod
    def _coerce_float_reject_bool(cls, v, info):
        return _coerce_float(v, f"optimizer.{info.field_name}")

    @field_validator("grad_clip", mode="before")
    @classmethod
    def _coerce_grad_clip(cls, v, info):
        return _coerce_optional_float(v, f"optimizer.{info.field_name}")

    @field_validator(
        "block_size", "precond_update_freq", "memory_size", "ns_steps",
        "lr_warmup",
        "lr_reduce_patience", "lr_reduce_cooldown",
        mode="before",
    )
    @classmethod
    def _coerce_int_reject_bool(cls, v, info):
        return _coerce_int(v, f"optimizer.{info.field_name}")

    @field_validator("name", mode="before")
    @classmethod
    def _check_name_type(cls, v):
        if not isinstance(v, str):
            raise TypeError(
                f"OptimizerConfig.name: expected str, got {type(v).__name__} ({v!r})"
            )
        return v

    @model_validator(mode="after")
    def _validate_ranges(self):
        if self.name not in self.VALID_NAMES:
            raise ValueError(
                f"Unknown optimizer '{self.name}'. "
                f"Valid: {sorted(self.VALID_NAMES)}"
            )
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError(f"grad_clip must be > 0, got {self.grad_clip}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if not (0 < self.beta1 < 1):
            raise ValueError(f"beta1 must be in (0, 1), got {self.beta1}")
        if not (0 < self.beta2 < 1):
            raise ValueError(f"beta2 must be in (0, 1), got {self.beta2}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if self.name in {"gn", "lm"}:
            if self.damping < 0:
                raise ValueError(f"damping must be >= 0 for {self.name}, got {self.damping}")
        elif self.damping <= 0:
            raise ValueError(f"damping must be > 0, got {self.damping}")
        if not (0 < self.decay < 1):
            raise ValueError(f"decay must be in (0, 1), got {self.decay}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")
        if self.precond_update_freq <= 0:
            raise ValueError(f"precond_update_freq must be > 0, got {self.precond_update_freq}")
        if self.memory_size <= 0:
            raise ValueError(f"memory_size must be > 0, got {self.memory_size}")
        if self.ns_steps <= 0:
            raise ValueError(f"ns_steps must be > 0, got {self.ns_steps}")
        if self.lr_schedule not in self.VALID_LR_SCHEDULES:
            raise ValueError(
                f"Unknown lr_schedule '{self.lr_schedule}'. "
                f"Valid: {sorted(self.VALID_LR_SCHEDULES)}"
            )
        if self.lr_warmup < 0:
            raise ValueError(f"lr_warmup must be >= 0, got {self.lr_warmup}")
        if not (0 <= self.lr_min_factor <= 1):
            raise ValueError(f"lr_min_factor must be in [0, 1], got {self.lr_min_factor}")
        return self


# ---------------------------------------------------------------------------
# CompositeLossConfig
# ---------------------------------------------------------------------------

class CompositeLossConfig(_ConfigBase):
    """Composite loss configuration (anchor + Jacobian + barrier + Newton terms).

    Anchor and Jacobian losses decay with shock_scale during curriculum
    (most useful near SS, fade as stochastic domain expands).
    Barrier and Newton losses don't decay (always useful for feasibility).
    """

    model_config = ConfigDict(extra="forbid")

    anchor_weight: float = Field(default=0.1, description="Weight on the anchor loss (||π_net(x) - π_lin(x)||² at sampled anchor points near SS).")
    jac_weight: float = Field(default=0.01, description="Weight on the Jacobian-match loss (||J_net(SS) - P||² at the steady state).")
    jac_anchor_weight: float = Field(default=0.0, description="Weight on the per-anchor Jacobian match (||J_net(x_i) - P||² averaged over anchors). 0 = off. ~d× more expensive than `jac_weight`.")
    barrier_weight: float = Field(default=0.01, description="Weight on economic feasibility barriers (net worth, leverage, consumption positivity).")
    newton_weight: float = Field(default=0.01, description="Weight on Newton-step auxiliary losses (condition number, residual) for kink-approximation stabilization.")
    n_anchor_points: int = Field(default=64, description="Number of anchor points sampled near SS at setup time (deterministic).")
    anchor_sigma: float = Field(default=1.0, description="Scale of the Gaussian spread around SS for anchor-point sampling.")
    leverage_mult: float = Field(default=5.0, description="Leverage barrier fires when `L > leverage_mult * L_ss`. Higher = more permissive.")
    aux_decay_floor: float = Field(default=0.2, description="Minimum retained weight of anchor+jac auxiliaries as curriculum progresses. Set to 1.0 to keep aux terms fully active throughout.")

    @field_validator(
        "anchor_weight", "jac_weight", "jac_anchor_weight", "barrier_weight", "newton_weight",
        "anchor_sigma", "leverage_mult", "aux_decay_floor",
        mode="before",
    )
    @classmethod
    def _coerce_float_reject_bool(cls, v, info):
        return _coerce_float(v, f"composite_loss.{info.field_name}")

    @field_validator("n_anchor_points", mode="before")
    @classmethod
    def _coerce_int_reject_bool(cls, v, info):
        return _coerce_int(v, f"composite_loss.{info.field_name}")

    @model_validator(mode="after")
    def _validate_ranges(self):
        for name in ("anchor_weight", "jac_weight", "jac_anchor_weight", "barrier_weight", "newton_weight"):
            val = getattr(self, name)
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val}")
        if self.n_anchor_points <= 0:
            raise ValueError(f"n_anchor_points must be > 0, got {self.n_anchor_points}")
        if self.anchor_sigma <= 0:
            raise ValueError(f"anchor_sigma must be > 0, got {self.anchor_sigma}")
        if self.leverage_mult <= 0:
            raise ValueError(f"leverage_mult must be > 0, got {self.leverage_mult}")
        if not (0 <= self.aux_decay_floor <= 1):
            raise ValueError(f"aux_decay_floor must be in [0, 1], got {self.aux_decay_floor}")
        return self


# ---------------------------------------------------------------------------
# NetworkConfig
# ---------------------------------------------------------------------------

class NetworkConfig(_ConfigBase):
    """Neural network configuration."""

    model_config = ConfigDict(extra="forbid")

    VALID_TYPES: ClassVar[frozenset] = frozenset({"mlp", "lstm", "transformer", "linear_plus_mlp"})
    VALID_ACTIVATIONS: ClassVar[frozenset] = frozenset({"tanh", "relu", "gelu", "silu", "softplus"})
    VALID_INITS: ClassVar[frozenset] = frozenset({
        "default", "xavier_normal", "xavier_uniform",
        "he_normal", "he_uniform", "lecun_normal",
    })

    type: str = Field(default="mlp", description="Network architecture: `mlp` (feedforward), `lstm`, `transformer`, or `linear_plus_mlp`.")
    hidden_sizes: Tuple[int, ...] = Field(default=(64, 64), description="Hidden layer widths. E.g. `(64, 64)` = two 64-unit hidden layers.")
    activation: str = Field(default="tanh", description="Per-layer activation: `tanh`, `relu`, `gelu`, `silu`, `sigmoid`, `softplus`.")
    activations: Optional[Tuple[str, ...]] = Field(default=None, description="Per-layer activations if different per layer. None = use `activation` uniformly. Length = `len(hidden_sizes)`.")
    init: str = Field(default="default", description="Weight init scheme: `default` (Equinox default), `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`.")
    multi_head: bool = Field(default=False, description="If True, use separate output heads per policy dimension (experimental).")
    skip_connections: bool = Field(default=False, description="If True, add residual connections between matching-width hidden layers.")
    history_len: int = Field(default=1, description="History window length for sequence policies. 1 = MLP (no history). >1 = LSTM / Transformer.")
    num_heads: int = Field(default=4, description="Transformer: attention heads per layer.")
    n_layers: int = Field(default=2, description="Transformer: number of transformer blocks.")
    init_scale: float = Field(default=0.0, description="`linear_plus_mlp` only: init scale of the MLP delta's final layer. 0.0 = policy starts exactly at the linear solution.")

    use_zlb_feature: bool = Field(default=False, description="`linear_plus_mlp` + disaster only: prepend `(R_lag - R_lb)` as an extra feature for the delta MLP. Experimental.")

    @field_validator("hidden_sizes", mode="before")
    @classmethod
    def _coerce_hidden_sizes(cls, v):
        if isinstance(v, list):
            return tuple(v)
        if isinstance(v, str) or isinstance(v, int):
            raise TypeError(
                f"NetworkConfig.hidden_sizes: expected Tuple[int, ...], "
                f"got {type(v).__name__} ({v!r})"
            )
        return v

    @field_validator("activations", mode="before")
    @classmethod
    def _coerce_activations(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_validator("history_len", "num_heads", "n_layers", mode="before")
    @classmethod
    def _coerce_int_reject_bool(cls, v, info):
        return _coerce_int(v, f"network.{info.field_name}")

    @field_validator("init_scale", mode="before")
    @classmethod
    def _coerce_init_scale(cls, v, info):
        return _coerce_float(v, f"network.{info.field_name}")

    @field_validator("type", "activation", "init", mode="before")
    @classmethod
    def _check_str_type(cls, v, info):
        if not isinstance(v, str):
            raise TypeError(
                f"NetworkConfig.{info.field_name}: expected str, got {type(v).__name__} ({v!r})"
            )
        return v

    @field_validator("multi_head", "skip_connections", "use_zlb_feature", mode="before")
    @classmethod
    def _check_bool_type(cls, v, info):
        if not isinstance(v, bool):
            raise TypeError(
                f"NetworkConfig.{info.field_name}: expected bool, got {type(v).__name__} ({v!r})"
            )
        return v

    @model_validator(mode="after")
    def _validate_ranges(self):
        if self.type not in self.VALID_TYPES:
            raise ValueError(
                f"Unknown network type '{self.type}'. "
                f"Valid: {sorted(self.VALID_TYPES)}"
            )
        if self.activation not in self.VALID_ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{self.activation}'. "
                f"Valid: {sorted(self.VALID_ACTIVATIONS)}"
            )
        if self.init not in self.VALID_INITS:
            raise ValueError(
                f"Unknown init '{self.init}'. "
                f"Valid: {sorted(self.VALID_INITS)}"
            )
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes must be non-empty")
        if any(s <= 0 for s in self.hidden_sizes):
            raise ValueError(f"All hidden_sizes must be > 0, got {self.hidden_sizes}")
        if self.activations is not None and len(self.activations) != len(self.hidden_sizes):
            raise ValueError(
                f"activations length ({len(self.activations)}) must match "
                f"hidden_sizes length ({len(self.hidden_sizes)})"
            )
        if self.history_len < 1:
            raise ValueError(f"history_len must be >= 1, got {self.history_len}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {self.num_heads}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {self.n_layers}")
        if self.type == "transformer":
            hidden_dim = self.hidden_sizes[0]
            if hidden_dim % self.num_heads != 0:
                raise ValueError(
                    f"For transformer, hidden_dim ({hidden_dim}) must be divisible "
                    f"by num_heads ({self.num_heads})"
                )
        return self


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------

class TrainConfig(_ConfigBase):
    """Complete training configuration.

    Supports construction from:
    - Direct keyword arguments
    - YAML file via from_yaml()
    - Dictionary via from_dict()
    - Overrides via with_overrides()
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(default="brock_mirman", description="Name of the registered model to train; see `deqn-jax list` for valid choices.")
    episodes: int = Field(default=1000, description="Number of outer training cycles (rollout + minibatch sweep).")
    batch_size: int = Field(default=64, description="Minibatch size used for each gradient step.")
    episode_length: int = Field(default=100, description="Trajectory length per rollout (T). With T=1 you must set `initialize_each_episode=True` (see validator).")
    mc_samples: int = Field(default=5, description="Monte Carlo shock samples per state for the residual expectation. Ignored when `expectation_type='gauss_hermite'`.")
    seed: int = Field(default=42, description="Top-level PRNG seed. Controls network init and the rollout/loss shock streams.")

    network: NetworkConfig = Field(default_factory=NetworkConfig, description="Policy network architecture; see NetworkConfig.")
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig, description="Optimizer and LR schedule; see OptimizerConfig.")

    loss_type: str = Field(default="mse", description="`mse` = base residual MSE. `composite` = base + anchor + Jacobian + barriers + Newton (disaster-style). Composite is rejected at startup with MAO / GN / LM / LBFGS / PCGrad.")
    composite_loss: CompositeLossConfig = Field(default_factory=CompositeLossConfig, description="Composite-loss weights; only active when `loss_type='composite'`.")

    loss_choice: str = Field(default="mse", description="Residual aggregation over batch elements: `mse` or `huber`. Applied AFTER the shock expectation. Huber caps gradient at ±huber_delta and helps when rare pathological states dominate.")
    huber_delta: float = Field(default=1.0, description="Cutoff for Huber loss (`loss_choice='huber'`). Ignored for `loss_choice='mse'`.")

    warm_start: bool = Field(default=False, description="If True, run L-BFGS pre-fit of the network to the steady-state policy before gradient-based training. Speeds early convergence; can mask Euler-equation bugs.")
    warm_start_linearize: bool = Field(default=False, description="If True, linearize the model around SS and use the Blanchard-Kahn P matrix to seed the network's Jacobian at SS. Advanced.")
    warm_start_dynare: Optional[str] = Field(default=None, description="Path to a Dynare output file to seed warm-start linearization. Rare.")
    loss_weights: Optional[List[float]] = Field(default=None, description="Manual per-equation weight vector of length `n_equations`. Default None = uniform weight 1.0.")
    loss_reweight: str = Field(default="none", description="Adaptive reweighting: `none` (default), `lr_annealing` (inverse-EMA), `relobralo` (softmax of loss ratios).")
    reweight_alpha: float = Field(default=0.9, description="EMA decay for `lr_annealing` / `relobralo`. Higher = slower adaptation.")

    log_every: int = Field(default=100, description="Episodes between console / TensorBoard scalar logs and cycle_hook invocations.")
    verbose: bool = Field(default=True, description="If False, suppress console output (the CLI `-q` flag sets this).")
    fp64: bool = Field(default=False, description="Enable JAX x64 mode for higher numerical precision. Applied at `train_from_config` entry.")

    tensorboard_dir: Optional[str] = Field(default=None, description="Directory for TensorBoard event files. None disables TB logging.")
    wandb_project: Optional[str] = Field(default=None, description="W&B project name. None disables W&B logging.")
    checkpoint_dir: Optional[str] = Field(default=None, description="Directory to save checkpoints (`checkpoint_<episode>.eqx` + `checkpoint_best.eqx` + `config.yaml`). None disables.")
    checkpoint_every: Optional[int] = Field(default=None, description="Episodes between periodic checkpoints. None = no periodic checkpoints (only best is saved).")
    max_checkpoints: Optional[int] = Field(default=None, description="Keep only the N most recent periodic checkpoints (best is never deleted).")

    gradient_surgery: str = Field(default="none", description="Multi-equation gradient conflict resolution: `none` or `pcgrad` (projecting conflicting gradients).")
    resume: Optional[str] = Field(default=None, description="Path to a `.eqx` checkpoint to resume from. Reads the sibling `config.yaml` to rebuild the correct pytree template.")
    switch_optimizer: Optional[str] = Field(default=None, description="If set, switch to this optimizer name at `switch_episode`. Old optimizer state is discarded; new optimizer is initialized from resumed params.")
    switch_episode: Optional[int] = Field(default=None, description="Episode at which to activate `switch_optimizer` and `switch_lr`.")
    switch_lr: Optional[float] = Field(default=None, description="Learning rate for the switched optimizer. None = keep the original optimizer's LR.")

    early_stop_patience: Optional[int] = Field(default=None, description="Stop training if loss hasn't improved by `early_stop_min_delta` for this many episodes. None = no early stopping.")
    early_stop_min_delta: float = Field(default=1e-6, description="Minimum absolute loss improvement counted against `early_stop_patience`.")

    curriculum_episodes: int = Field(default=0, description="Ramp `shock_scale` linearly from `curriculum_start` to 1.0 over this many episodes. 0 = no curriculum.")
    curriculum_start: float = Field(default=0.1, description="Initial `shock_scale` when curriculum is active.")
    ss_reset_frac: float = Field(default=0.0, description="Fraction of batch re-initialized to SS-neighborhood each rollout (prevents trajectory drift). Orthogonal to `initialize_each_episode`.")

    initialize_each_episode: bool = Field(
        default=False,
        description=(
            "If True, replace episode_state with a fresh `init_state_fn` draw "
            "at the start of every rollout cycle (non-ergodic training, matches "
            "DEQN-MAO's flag of the same name). False = continue trajectory "
            "across cycles (ergodic). Required True when `episode_length=1`."
        ),
    )

    expectation_type: str = Field(
        default="mc",
        description=(
            "How to integrate over shocks in the residual: `mc` (antithetic "
            "Monte Carlo, uses `mc_samples`) or `quadrature`/`gh`/`gauss_hermite` "
            "(deterministic tensor-product grid, uses `n_quadrature_points`)."
        ),
    )
    n_quadrature_points: int = Field(
        default=3,
        description="Quadrature points per shock dimension when `expectation_type='gauss_hermite'`. Total nodes = n_quadrature_points^n_shocks.",
    )

    barrier_weight: float = Field(default=0.0, description="Legacy state-barrier penalty weight. 0 disables. Prefer `definition_bounds` on the ModelSpec for new models.")
    shock_mask: Optional[List[float]] = Field(
        default=None,
        description=(
            "Per-dimension multiplicative mask over shocks (length must equal "
            "`model.n_shocks`). Values in [0, 1]; 0 zeroes that shock entirely. "
            "Applied to BOTH the residual expectation and the rollout state path."
        ),
    )

    target_update_every: int = Field(default=0, description="Target-network update interval in episodes. 0 disables target network entirely.")
    target_tau: float = Field(default=1.0, description="Polyak averaging coefficient for target-network update. 1.0 = hard copy, <1 = soft update toward current params.")

    constants: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-run override of model.constants (e.g. `{p_disaster: 0.02}`). Merges into the model's built-in calibration.",
    )

    use_risky_steady_state: bool = Field(
        default=True,
        description=(
            "If True and `p_disaster > 0`, anchor composite loss and "
            "linearization at the risky SS (E_d[F]=0) instead of deterministic SS. "
            "Set False to force deterministic SS anchor under disaster risk "
            "(for ablation)."
        ),
    )

    save_best_checkpoint: bool = Field(
        default=True,
        description=(
            "If True and `checkpoint_dir` is set, persist `checkpoint_best.eqx` "
            "on every loss improvement (after `curriculum_episodes` grace period). "
            "Guards against rare huge-gradient events corrupting the latest snapshot."
        ),
    )

    n_epochs_per_rollout: int = Field(
        default=1,
        description=(
            "DEQN cycle: per outer iteration, 1 rollout fills a trajectory of "
            "(`sim_batch` × `episode_length`) states, then we do `n_epochs_per_rollout` "
            "sweeps over it. Default 1 matches DEQN-MAO's run_cycle."
        ),
    )
    n_minibatches_per_epoch: Optional[int] = Field(
        default=None,
        description=(
            "Minibatches per sweep. None = all available (full-trajectory sweep). "
            "Set to 1 for the legacy one-grad-per-rollout behavior."
        ),
    )

    sorted_within_batch: bool = Field(
        default=False,
        description=(
            "Minibatch shuffle policy. False = IID shuffle across all "
            "(episode_length × sim_batch) samples. True = each minibatch is a "
            "contiguous temporal slice of a single trajectory (RL-style); batch "
            "order shuffled, intra-batch order preserved. MLP-only."
        ),
    )

    sim_batch: Optional[int] = Field(
        default=None,
        description=(
            "Number of parallel simulation trajectories in the rollout. None "
            "(default) = same as `batch_size`. Setting `sim_batch > batch_size` "
            "decouples trajectory count from gradient minibatch size — larger "
            "pool = more representative ergodic distribution per cycle."
        ),
    )

    VALID_LOSS_TYPES: ClassVar[frozenset] = frozenset({"mse", "composite"})
    VALID_LOSS_CHOICES: ClassVar[frozenset] = frozenset({"mse", "huber"})
    VALID_LOSS_REWEIGHTS: ClassVar[frozenset] = frozenset({"none", "lr_annealing", "relobralo"})
    VALID_GRADIENT_SURGERY: ClassVar[frozenset] = frozenset({"none", "pcgrad"})
    VALID_EXPECTATION_TYPES: ClassVar[frozenset] = frozenset({"mc", "quadrature", "gh", "gauss_hermite"})

    # -- before-mode validators for type coercion --

    @field_validator("model", mode="before")
    @classmethod
    def _check_model_type(cls, v):
        if not isinstance(v, str):
            raise TypeError(
                f"TrainConfig.model: expected str, got {type(v).__name__} ({v!r})"
            )
        return v

    @field_validator(
        "episodes", "batch_size", "episode_length", "mc_samples", "seed",
        "log_every", "curriculum_episodes", "n_quadrature_points",
        "target_update_every", "n_epochs_per_rollout",
        mode="before",
    )
    @classmethod
    def _coerce_int_reject_bool(cls, v, info):
        return _coerce_int(v, info.field_name)

    @field_validator("n_minibatches_per_epoch", mode="before")
    @classmethod
    def _coerce_n_minibatches(cls, v):
        return _coerce_optional_int(v, "n_minibatches_per_epoch")

    @field_validator("sim_batch", mode="before")
    @classmethod
    def _coerce_sim_batch(cls, v):
        return _coerce_optional_int(v, "sim_batch")

    @field_validator(
        "reweight_alpha", "early_stop_min_delta", "curriculum_start",
        "ss_reset_frac", "barrier_weight", "target_tau", "huber_delta",
        mode="before",
    )
    @classmethod
    def _coerce_float_reject_bool(cls, v, info):
        return _coerce_float(v, info.field_name)

    @field_validator("switch_lr", mode="before")
    @classmethod
    def _coerce_switch_lr(cls, v, info):
        return _coerce_optional_float(v, info.field_name)

    @field_validator("switch_episode", "checkpoint_every", "max_checkpoints",
                     "early_stop_patience", mode="before")
    @classmethod
    def _coerce_optional_int_fields(cls, v, info):
        return _coerce_optional_int(v, info.field_name)

    @field_validator("verbose", "warm_start", "warm_start_linearize", "fp64",
                     "use_risky_steady_state",
                     "save_best_checkpoint", mode="before")
    @classmethod
    def _check_bool_type(cls, v, info):
        if not isinstance(v, bool):
            raise TypeError(
                f"TrainConfig.{info.field_name}: expected bool, got {type(v).__name__} ({v!r})"
            )
        return v

    @field_validator("loss_weights", mode="before")
    @classmethod
    def _check_loss_weights_type(cls, v):
        if v is not None and not isinstance(v, list):
            raise TypeError(
                f"TrainConfig.loss_weights: expected Optional[List[float]], "
                f"got {type(v).__name__} ({v!r})"
            )
        return v

    @field_validator("constants", mode="before")
    @classmethod
    def _check_constants_type(cls, v):
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise TypeError(
                f"TrainConfig.constants: expected Dict[str, float], "
                f"got {type(v).__name__} ({v!r})"
            )
        for k, val in v.items():
            if not isinstance(k, str):
                raise TypeError(
                    f"TrainConfig.constants: keys must be str, got {type(k).__name__} ({k!r})"
                )
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(
                    f"TrainConfig.constants[{k!r}]: expected number, "
                    f"got {type(val).__name__} ({val!r})"
                )
        return v

    @field_validator("optimizer", mode="before")
    @classmethod
    def _coerce_optimizer_str(cls, v):
        """Allow `optimizer: "adam"` shorthand in YAML."""
        if isinstance(v, str):
            return OptimizerConfig(name=v)
        return v

    @field_validator("network", mode="before")
    @classmethod
    def _coerce_network_str(cls, v):
        """Allow `network: "mlp"` shorthand in YAML."""
        if isinstance(v, str):
            return NetworkConfig(type=v)
        return v

    @model_validator(mode="after")
    def _validate_ranges(self):
        if not self.model or not isinstance(self.model, str):
            raise ValueError(f"model must be a non-empty string, got {self.model!r}")
        if self.episodes <= 0:
            raise ValueError(f"episodes must be > 0, got {self.episodes}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.episode_length <= 0:
            raise ValueError(f"episode_length must be > 0, got {self.episode_length}")
        if self.mc_samples <= 0:
            raise ValueError(f"mc_samples must be > 0, got {self.mc_samples}")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        if self.loss_type not in self.VALID_LOSS_TYPES:
            raise ValueError(
                f"Unknown loss_type '{self.loss_type}'. "
                f"Valid: {sorted(self.VALID_LOSS_TYPES)}"
            )
        if self.loss_choice not in self.VALID_LOSS_CHOICES:
            raise ValueError(
                f"Unknown loss_choice '{self.loss_choice}'. "
                f"Valid: {sorted(self.VALID_LOSS_CHOICES)}"
            )
        if self.huber_delta <= 0:
            raise ValueError(f"huber_delta must be > 0, got {self.huber_delta}")
        if self.loss_reweight not in self.VALID_LOSS_REWEIGHTS:
            raise ValueError(
                f"Unknown loss_reweight '{self.loss_reweight}'. "
                f"Valid: {sorted(self.VALID_LOSS_REWEIGHTS)}"
            )
        if not (0 < self.reweight_alpha < 1):
            raise ValueError(f"reweight_alpha must be in (0, 1), got {self.reweight_alpha}")
        if self.gradient_surgery not in self.VALID_GRADIENT_SURGERY:
            raise ValueError(
                f"Unknown gradient_surgery '{self.gradient_surgery}'. "
                f"Valid: {sorted(self.VALID_GRADIENT_SURGERY)}"
            )
        if self.expectation_type not in self.VALID_EXPECTATION_TYPES:
            raise ValueError(
                f"Unknown expectation_type '{self.expectation_type}'. "
                f"Valid: {sorted(self.VALID_EXPECTATION_TYPES)}"
            )
        if self.n_quadrature_points <= 0:
            raise ValueError(f"n_quadrature_points must be > 0, got {self.n_quadrature_points}")
        if self.log_every <= 0:
            raise ValueError(f"log_every must be > 0, got {self.log_every}")
        if self.curriculum_episodes < 0:
            raise ValueError(f"curriculum_episodes must be >= 0, got {self.curriculum_episodes}")
        if self.curriculum_episodes > 0 and not (0 < self.curriculum_start <= 1):
            raise ValueError(
                f"curriculum_start must be in (0, 1] when curriculum is active, "
                f"got {self.curriculum_start}"
            )
        if self.early_stop_min_delta < 0:
            raise ValueError(f"early_stop_min_delta must be >= 0, got {self.early_stop_min_delta}")
        if self.switch_optimizer is not None and self.switch_episode is None:
            raise ValueError(
                "switch_episode must be set when switch_optimizer is specified"
            )
        if self.checkpoint_every is not None and self.checkpoint_every <= 0:
            raise ValueError(
                f"checkpoint_every must be > 0, got {self.checkpoint_every}"
            )
        if self.network is not None and self.network.history_len > self.episode_length:
            raise ValueError(
                f"history_len ({self.network.history_len}) must be <= episode_length "
                f"({self.episode_length}), otherwise no training windows can be formed"
            )
        if self.loss_weights is not None:
            if any(w < 0 for w in self.loss_weights):
                raise ValueError(
                    f"All loss_weights must be >= 0, got {self.loss_weights}"
                )
        if self.shock_mask is not None:
            if not all(0 <= m <= 1 for m in self.shock_mask):
                raise ValueError(
                    f"All shock_mask values must be in [0, 1], got {self.shock_mask}"
                )
        if self.target_update_every < 0:
            raise ValueError(f"target_update_every must be >= 0, got {self.target_update_every}")
        if not (0 < self.target_tau <= 1):
            raise ValueError(f"target_tau must be in (0, 1], got {self.target_tau}")
        if self.n_epochs_per_rollout < 1:
            raise ValueError(f"n_epochs_per_rollout must be >= 1, got {self.n_epochs_per_rollout}")
        if self.n_minibatches_per_epoch is not None and self.n_minibatches_per_epoch < 1:
            raise ValueError(
                f"n_minibatches_per_epoch must be >= 1 or None, got {self.n_minibatches_per_epoch}"
            )
        return self

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        """Create config from a flat or nested dictionary.

        Handles nested ``optimizer:`` and ``network:`` sub-dicts.
        """
        d = copy.deepcopy(d)

        # Extract nested sub-configs
        opt_dict = d.pop("optimizer", {})
        net_dict = d.pop("network", {})
        comp_dict = d.pop("composite_loss", {})

        # If optimizer is a plain string, treat as name
        if isinstance(opt_dict, str):
            opt_dict = {"name": opt_dict}
        if isinstance(net_dict, str):
            net_dict = {"type": net_dict}

        # Convert hidden_sizes list to tuple
        if "hidden_sizes" in net_dict and isinstance(net_dict["hidden_sizes"], list):
            net_dict["hidden_sizes"] = tuple(net_dict["hidden_sizes"])

        # Convert activations list to tuple
        if "activations" in net_dict and isinstance(net_dict["activations"], list):
            net_dict["activations"] = tuple(net_dict["activations"])

        # Convert loss_weights list (YAML gives lists)
        if "loss_weights" in d and isinstance(d["loss_weights"], list):
            d["loss_weights"] = list(d["loss_weights"])

        # Validate: reject unknown keys (with did-you-mean suggestions)
        opt_fields = set(OptimizerConfig.model_fields.keys())
        net_fields = set(NetworkConfig.model_fields.keys())
        comp_fields = set(CompositeLossConfig.model_fields.keys())
        train_fields = set(TrainConfig.model_fields.keys())

        _check_unknown_keys(set(opt_dict.keys()), opt_fields, "optimizer")
        _check_unknown_keys(set(net_dict.keys()), net_fields, "network")
        _check_unknown_keys(set(comp_dict.keys()), comp_fields, "composite_loss")
        _check_unknown_keys(set(d.keys()), train_fields, "config")

        return cls(
            optimizer=OptimizerConfig(**opt_dict),
            network=NetworkConfig(**net_dict),
            composite_loss=CompositeLossConfig(**comp_dict),
            **{k: v for k, v in d.items() if k in train_fields},
        )

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        """Load config from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def with_overrides(self, overrides: Dict[str, Any]) -> "TrainConfig":
        """Return a new config with dot-notation overrides applied.

        Example:
            config.with_overrides({"optimizer.learning_rate": 0.01, "episodes": 500})
        """
        d = _config_to_flat_dict(self)
        for key, val in overrides.items():
            val = _infer_type(val)
            d[key] = val
        return _flat_dict_to_config(d)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested dictionary."""
        return self.model_dump()

    def to_yaml(self, path: str) -> None:
        """Write config to a YAML file."""
        import yaml

        d = self.to_dict()
        # Convert tuples to lists for YAML readability
        if "network" in d and "hidden_sizes" in d["network"]:
            d["network"]["hidden_sizes"] = list(d["network"]["hidden_sizes"])
        if "network" in d and d["network"].get("activations") is not None:
            d["network"]["activations"] = list(d["network"]["activations"])

        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Helpers (kept from original)
# ---------------------------------------------------------------------------

def _check_unknown_keys(
    provided: Set[str],
    valid: Set[str],
    context: str,
) -> None:
    """Raise ValueError if *provided* contains keys not in *valid*.

    Includes "did you mean?" suggestions using difflib.
    """
    unknown = provided - valid
    if not unknown:
        return
    parts = []
    for key in sorted(unknown):
        matches = get_close_matches(key, sorted(valid), n=3, cutoff=0.6)
        if matches:
            parts.append(f"  '{key}' (did you mean: {', '.join(repr(m) for m in matches)}?)")
        else:
            parts.append(f"  '{key}'")
    raise ValueError(
        f"Unknown keys in {context}:\n"
        + "\n".join(parts)
        + f"\nValid keys: {sorted(valid)}"
    )


def _config_to_flat_dict(config: TrainConfig) -> Dict[str, Any]:
    """Flatten a TrainConfig into dot-notation keys."""
    flat: Dict[str, Any] = {}
    for name in TrainConfig.model_fields:
        val = getattr(config, name)
        if name == "optimizer":
            for of in OptimizerConfig.model_fields:
                flat[f"optimizer.{of}"] = getattr(val, of)
        elif name == "network":
            for nf in NetworkConfig.model_fields:
                flat[f"network.{nf}"] = getattr(val, nf)
        elif name == "composite_loss":
            for cf in CompositeLossConfig.model_fields:
                flat[f"composite_loss.{cf}"] = getattr(val, cf)
        else:
            flat[name] = val
    return flat


def _flat_dict_to_config(flat: Dict[str, Any]) -> TrainConfig:
    """Reconstruct TrainConfig from flat dot-notation dict."""
    opt_kw: Dict[str, Any] = {}
    net_kw: Dict[str, Any] = {}
    comp_kw: Dict[str, Any] = {}
    train_kw: Dict[str, Any] = {}

    opt_fields = set(OptimizerConfig.model_fields.keys())
    net_fields = set(NetworkConfig.model_fields.keys())
    comp_fields = set(CompositeLossConfig.model_fields.keys())
    train_fields = set(TrainConfig.model_fields.keys()) - {"optimizer", "network", "composite_loss"}

    # Build set of all valid flat keys for validation
    valid_flat_keys = set(train_fields)
    valid_flat_keys |= {f"optimizer.{n}" for n in opt_fields}
    valid_flat_keys |= {f"network.{n}" for n in net_fields}
    valid_flat_keys |= {f"composite_loss.{n}" for n in comp_fields}

    _check_unknown_keys(set(flat.keys()), valid_flat_keys, "config overrides")

    for key, val in flat.items():
        if key.startswith("optimizer."):
            opt_kw[key[len("optimizer."):]] = val
        elif key.startswith("network."):
            net_kw[key[len("network."):]] = val
        elif key.startswith("composite_loss."):
            comp_kw[key[len("composite_loss."):]] = val
        else:
            train_kw[key] = val

    return TrainConfig(
        optimizer=OptimizerConfig(**opt_kw),
        network=NetworkConfig(**net_kw),
        composite_loss=CompositeLossConfig(**comp_kw),
        **train_kw,
    )


def _infer_type(val: Any) -> Any:
    """Infer Python type from string value (for CLI --set overrides)."""
    if not isinstance(val, str):
        return val
    # Booleans
    if val.lower() in ("true", "yes"):
        return True
    if val.lower() in ("false", "no", "none"):
        if val.lower() == "none":
            return None
        return False
    # Try int then float
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    # Tuple-like for hidden_sizes
    if "," in val and all(c.isdigit() or c in ", " for c in val):
        return tuple(int(x.strip()) for x in val.split(",") if x.strip())
    return val


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, str]] = None,
    **cli_kwargs: Any,
) -> TrainConfig:
    """Load config with full priority merging.

    Priority: overrides (--set) > cli_kwargs > YAML > defaults
    """
    # Start from YAML or defaults
    if config_path:
        config = TrainConfig.from_yaml(config_path)
    else:
        config = TrainConfig()

    # Apply CLI keyword arguments (non-None only)
    if cli_kwargs:
        flat = _config_to_flat_dict(config)
        for key, val in cli_kwargs.items():
            if val is None:
                continue
            # Map flat CLI keys to dot-notation
            if key in flat:
                flat[key] = val
            elif f"optimizer.{key}" in flat:
                flat[f"optimizer.{key}"] = val
            elif f"network.{key}" in flat:
                flat[f"network.{key}"] = val
            else:
                matches = get_close_matches(key, sorted(flat.keys()), n=3, cutoff=0.6)
                hint = f" (did you mean: {', '.join(repr(m) for m in matches)}?)" if matches else ""
                raise ValueError(f"Unknown CLI config key '{key}'{hint}")
        config = _flat_dict_to_config(flat)

    # Apply --set overrides last (highest priority)
    if overrides:
        config = config.with_overrides(overrides)

    return config
