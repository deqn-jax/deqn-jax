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

    name: str = "adam"
    learning_rate: float = Field(default=1e-3)
    grad_clip: Optional[float] = None
    weight_decay: float = Field(default=0.0)
    # Adam / MAO
    beta1: float = Field(default=0.9)
    beta2: float = Field(default=0.999)
    epsilon: float = Field(default=1e-8)
    # NGD
    damping: float = Field(default=1e-4)
    decay: float = Field(default=0.999)
    # Shampoo
    block_size: int = Field(default=64)
    precond_update_freq: int = Field(default=10)
    # L-BFGS
    memory_size: int = Field(default=10)
    # Muon
    ns_steps: int = Field(default=5)
    # LR schedule
    lr_schedule: str = "constant"
    lr_warmup: int = Field(default=0)
    lr_min_factor: float = Field(default=0.0)

    VALID_NAMES: ClassVar[frozenset] = frozenset({
        "adam", "sgd", "adamw", "lion", "muon",
        "ngd", "shampoo", "lbfgs", "mao", "mao_kfac", "gn", "lm",
    })
    VALID_LR_SCHEDULES: ClassVar[frozenset] = frozenset({"constant", "cosine"})

    @field_validator(
        "learning_rate", "weight_decay", "beta1", "beta2", "epsilon",
        "damping", "decay", "lr_min_factor",
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
        if self.damping <= 0:
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

    anchor_weight: float = Field(default=0.1)
    jac_weight: float = Field(default=0.01)
    barrier_weight: float = Field(default=0.01)
    newton_weight: float = Field(default=0.01)
    n_anchor_points: int = Field(default=64)
    anchor_sigma: float = Field(default=1.0)
    leverage_mult: float = Field(default=5.0)
    aux_decay_floor: float = Field(default=0.2)

    @field_validator(
        "anchor_weight", "jac_weight", "barrier_weight", "newton_weight",
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
        for name in ("anchor_weight", "jac_weight", "barrier_weight", "newton_weight"):
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

    type: str = "mlp"
    hidden_sizes: Tuple[int, ...] = (64, 64)
    activation: str = "tanh"
    activations: Optional[Tuple[str, ...]] = None
    init: str = "default"
    multi_head: bool = False
    skip_connections: bool = False
    history_len: int = Field(default=1)
    num_heads: int = Field(default=4)
    n_layers: int = Field(default=2)
    # For linear_plus_mlp: scale of the MLP delta's final layer at init.
    # 0.0 means policy starts exactly at the linear solution.
    init_scale: float = Field(default=0.0)

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

    @field_validator("multi_head", "skip_connections", mode="before")
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

    model: str = "brock_mirman"
    episodes: int = Field(default=1000)
    batch_size: int = Field(default=64)
    episode_length: int = Field(default=100)
    mc_samples: int = Field(default=5)
    seed: int = Field(default=42)

    network: NetworkConfig = Field(default_factory=NetworkConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)

    loss_type: str = "mse"
    composite_loss: CompositeLossConfig = Field(default_factory=CompositeLossConfig)

    warm_start: bool = False
    warm_start_linearize: bool = False
    warm_start_dynare: Optional[str] = None
    loss_weights: Optional[List[float]] = None
    loss_reweight: str = "none"
    reweight_alpha: float = Field(default=0.9)

    log_every: int = Field(default=100)
    verbose: bool = True
    fp64: bool = False

    tensorboard_dir: Optional[str] = None
    wandb_project: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    checkpoint_every: Optional[int] = None
    max_checkpoints: Optional[int] = None

    rescale_equations: bool = False
    gradient_surgery: str = "none"
    resume: Optional[str] = None
    switch_optimizer: Optional[str] = None
    switch_episode: Optional[int] = None
    switch_lr: Optional[float] = None

    early_stop_patience: Optional[int] = None
    early_stop_min_delta: float = Field(default=1e-6)

    curriculum_episodes: int = Field(default=0)
    curriculum_start: float = Field(default=0.1)
    ss_reset_frac: float = Field(default=0.0)

    expectation_type: str = "mc"
    n_quadrature_points: int = Field(default=3)

    barrier_weight: float = Field(default=0.0)
    shock_mask: Optional[List[float]] = None

    target_update_every: int = Field(default=0)  # 0=off, >0=update target net every N episodes
    target_tau: float = Field(default=1.0)        # 1.0=hard copy, <1=Polyak averaging

    VALID_LOSS_TYPES: ClassVar[frozenset] = frozenset({"mse", "composite"})
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
        "target_update_every",
        mode="before",
    )
    @classmethod
    def _coerce_int_reject_bool(cls, v, info):
        return _coerce_int(v, info.field_name)

    @field_validator(
        "reweight_alpha", "early_stop_min_delta", "curriculum_start",
        "ss_reset_frac", "barrier_weight", "target_tau",
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
                     "rescale_equations", mode="before")
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
