"""Structured configuration for DEQN-JAX training.

Three nested dataclasses with YAML loading and CLI override merging.

Priority: --set overrides > CLI args > YAML file > defaults
"""

from dataclasses import dataclass, field, fields, asdict
from difflib import get_close_matches
from typing import Any, Dict, List, Optional, Set, Tuple, get_type_hints, get_origin, get_args, Union
import copy


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    name: str = "adam"
    learning_rate: float = 1e-3
    grad_clip: Optional[float] = None
    weight_decay: float = 0.0
    # Adam / MAO
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    # NGD
    damping: float = 1e-4
    decay: float = 0.999
    # Shampoo
    block_size: int = 64
    precond_update_freq: int = 10
    # L-BFGS
    memory_size: int = 10
    # Muon
    ns_steps: int = 5
    # LR schedule
    lr_schedule: str = "constant"   # "constant" or "cosine"
    lr_warmup: int = 0              # warmup episodes before decay
    lr_min_factor: float = 0.0      # min LR = learning_rate * lr_min_factor

    VALID_NAMES = frozenset({
        "adam", "sgd", "adamw", "lion", "muon",
        "ngd", "shampoo", "lbfgs", "mao", "mao_kfac", "gn", "lm",
    })
    VALID_LR_SCHEDULES = frozenset({"constant", "cosine"})

    def __post_init__(self):
        """Coerce YAML string values to proper numeric types."""
        self.learning_rate = _coerce(self.learning_rate, float, "optimizer.learning_rate")
        if self.grad_clip is not None:
            self.grad_clip = _coerce(self.grad_clip, float, "optimizer.grad_clip")
        self.weight_decay = _coerce(self.weight_decay, float, "optimizer.weight_decay")
        self.beta1 = _coerce(self.beta1, float, "optimizer.beta1")
        self.beta2 = _coerce(self.beta2, float, "optimizer.beta2")
        self.epsilon = _coerce(self.epsilon, float, "optimizer.epsilon")
        self.damping = _coerce(self.damping, float, "optimizer.damping")
        self.decay = _coerce(self.decay, float, "optimizer.decay")
        self.block_size = _coerce(self.block_size, int, "optimizer.block_size")
        self.precond_update_freq = _coerce(self.precond_update_freq, int, "optimizer.precond_update_freq")
        self.memory_size = _coerce(self.memory_size, int, "optimizer.memory_size")
        self.ns_steps = _coerce(self.ns_steps, int, "optimizer.ns_steps")
        self.lr_warmup = _coerce(self.lr_warmup, int, "optimizer.lr_warmup")
        self.lr_min_factor = _coerce(self.lr_min_factor, float, "optimizer.lr_min_factor")
        _validate_field_types(self)
        self.validate()

    def validate(self):
        """Validate optimizer configuration values."""
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


@dataclass
class CompositeLossConfig:
    """Composite loss configuration (anchor + Jacobian + barrier + Newton terms).

    Anchor and Jacobian losses decay with shock_scale during curriculum
    (most useful near SS, fade as stochastic domain expands).
    Barrier and Newton losses don't decay (always useful for feasibility).
    """

    anchor_weight: float = 0.1
    jac_weight: float = 0.01
    barrier_weight: float = 0.01
    newton_weight: float = 0.01
    n_anchor_points: int = 64  # Fixed sample points near SS
    anchor_sigma: float = 1.0  # Spread of anchor points (in ergodic std devs)
    leverage_mult: float = 5.0
    aux_decay_floor: float = 0.2  # Minimum anchor/jac weight fraction (0=full decay, 1=no decay)

    def __post_init__(self):
        """Coerce YAML string values to proper numeric types."""
        self.anchor_weight = _coerce(self.anchor_weight, float, "composite_loss.anchor_weight")
        self.jac_weight = _coerce(self.jac_weight, float, "composite_loss.jac_weight")
        self.barrier_weight = _coerce(self.barrier_weight, float, "composite_loss.barrier_weight")
        self.newton_weight = _coerce(self.newton_weight, float, "composite_loss.newton_weight")
        self.n_anchor_points = _coerce(self.n_anchor_points, int, "composite_loss.n_anchor_points")
        self.anchor_sigma = _coerce(self.anchor_sigma, float, "composite_loss.anchor_sigma")
        self.leverage_mult = _coerce(self.leverage_mult, float, "composite_loss.leverage_mult")
        self.aux_decay_floor = _coerce(self.aux_decay_floor, float, "composite_loss.aux_decay_floor")
        _validate_field_types(self)
        self.validate()

    def validate(self):
        """Validate composite loss configuration values."""
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


@dataclass
class NetworkConfig:
    """Neural network configuration."""

    VALID_TYPES = frozenset({"mlp", "lstm", "transformer"})
    VALID_ACTIVATIONS = frozenset({"tanh", "relu", "gelu", "silu", "softplus"})
    VALID_INITS = frozenset({
        "default", "xavier_normal", "xavier_uniform",
        "he_normal", "he_uniform", "lecun_normal",
    })

    type: str = "mlp"
    hidden_sizes: Tuple[int, ...] = (64, 64)
    activation: str = "tanh"
    activations: Optional[Tuple[str, ...]] = None  # per-layer override
    init: str = "default"
    multi_head: bool = False  # separate output head per policy
    skip_connections: bool = False  # residual connections between hidden layers
    history_len: int = 1  # 1=Markovian (MLP), >1=sequence (LSTM/Transformer)
    num_heads: int = 4  # Transformer attention heads
    n_layers: int = 2  # Transformer/LSTM depth (separate from hidden_sizes for Transformer)

    def __post_init__(self):
        """Coerce and validate network configuration."""
        if isinstance(self.hidden_sizes, list):
            self.hidden_sizes = tuple(self.hidden_sizes)
        if isinstance(self.activations, list):
            self.activations = tuple(self.activations)
        self.history_len = _coerce(self.history_len, int, "network.history_len")
        self.num_heads = _coerce(self.num_heads, int, "network.num_heads")
        self.n_layers = _coerce(self.n_layers, int, "network.n_layers")
        _validate_field_types(self)
        self.validate()

    def validate(self):
        """Validate network configuration values."""
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


@dataclass
class TrainConfig:
    """Complete training configuration.

    Supports construction from:
    - Direct keyword arguments
    - YAML file via from_yaml()
    - Dictionary via from_dict()
    - Overrides via with_overrides()
    """

    model: str = "brock_mirman"
    episodes: int = 1000
    batch_size: int = 64
    episode_length: int = 100
    mc_samples: int = 5
    seed: int = 42

    network: NetworkConfig = field(default_factory=NetworkConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    loss_type: str = "mse"  # "mse" or "composite"
    composite_loss: CompositeLossConfig = field(default_factory=CompositeLossConfig)

    warm_start: bool = False
    warm_start_linearize: bool = False  # Use linearized (Blanchard-Kahn) warm start
    warm_start_dynare: Optional[str] = None  # Path to Dynare results dir (ghx/ghu CSVs)
    loss_weights: Optional[List[float]] = None
    loss_reweight: str = "none"
    reweight_alpha: float = 0.9

    log_every: int = 100
    verbose: bool = True
    fp64: bool = False

    tensorboard_dir: Optional[str] = None
    wandb_project: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    checkpoint_every: Optional[int] = None
    max_checkpoints: Optional[int] = None

    rescale_equations: bool = False
    gradient_surgery: str = "none"  # "none" or "pcgrad"
    resume: Optional[str] = None
    switch_optimizer: Optional[str] = None
    switch_episode: Optional[int] = None
    switch_lr: Optional[float] = None

    early_stop_patience: Optional[int] = None  # Stop if no improvement for N episodes
    early_stop_min_delta: float = 1e-6  # Minimum improvement to count as progress

    curriculum_episodes: int = 0  # Ramp shock_scale from curriculum_start to 1.0 over N episodes
    curriculum_start: float = 0.1  # Initial shock scale for curriculum
    ss_reset_frac: float = 0.0  # Fraction of batch to reset to SS-neighborhood each episode

    expectation_type: str = "mc"  # "mc" or "quadrature" (Gauss-Hermite)
    n_quadrature_points: int = 3  # Points per dimension (3^5=243 nodes for 5 shocks)

    episode_soft_clip: bool = True   # Differentiable soft clip in episode trajectories
    barrier_weight: float = 0.0      # State barrier penalty weight (0 = off)

    VALID_LOSS_TYPES = frozenset({"mse", "composite"})
    VALID_LOSS_REWEIGHTS = frozenset({"none", "lr_annealing", "relobralo"})
    VALID_GRADIENT_SURGERY = frozenset({"none", "pcgrad"})
    VALID_EXPECTATION_TYPES = frozenset({"mc", "quadrature", "gh", "gauss_hermite"})

    def __post_init__(self):
        """Coerce YAML string values to proper numeric types."""
        self.episodes = _coerce(self.episodes, int, "episodes")
        self.batch_size = _coerce(self.batch_size, int, "batch_size")
        self.episode_length = _coerce(self.episode_length, int, "episode_length")
        self.mc_samples = _coerce(self.mc_samples, int, "mc_samples")
        self.seed = _coerce(self.seed, int, "seed")
        self.reweight_alpha = _coerce(self.reweight_alpha, float, "reweight_alpha")
        self.log_every = _coerce(self.log_every, int, "log_every")
        self.curriculum_episodes = _coerce(self.curriculum_episodes, int, "curriculum_episodes")
        self.curriculum_start = _coerce(self.curriculum_start, float, "curriculum_start")
        self.ss_reset_frac = _coerce(self.ss_reset_frac, float, "ss_reset_frac")
        self.early_stop_min_delta = _coerce(self.early_stop_min_delta, float, "early_stop_min_delta")
        self.n_quadrature_points = _coerce(self.n_quadrature_points, int, "n_quadrature_points")
        self.barrier_weight = _coerce(self.barrier_weight, float, "barrier_weight")
        if self.switch_lr is not None:
            self.switch_lr = _coerce(self.switch_lr, float, "switch_lr")
        if self.switch_episode is not None:
            self.switch_episode = _coerce(self.switch_episode, int, "switch_episode")
        if self.checkpoint_every is not None:
            self.checkpoint_every = _coerce(self.checkpoint_every, int, "checkpoint_every")
        if self.max_checkpoints is not None:
            self.max_checkpoints = _coerce(self.max_checkpoints, int, "max_checkpoints")
        if self.early_stop_patience is not None:
            self.early_stop_patience = _coerce(self.early_stop_patience, int, "early_stop_patience")
        _validate_field_types(self)
        self.validate()

    def validate(self):
        """Validate training configuration values."""
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

        # Validate: reject unknown keys
        opt_fields = {f.name for f in fields(OptimizerConfig)}
        net_fields = {f.name for f in fields(NetworkConfig)}
        comp_fields = {f.name for f in fields(CompositeLossConfig)}
        train_fields = {f.name for f in fields(TrainConfig)}

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
        return asdict(self)

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


def _coerce(value: Any, target: type, field_name: str) -> Any:
    """Coerce *value* to *target* type with a clear error on failure.

    Handles the common YAML case where numbers arrive as strings.
    Booleans are never coerced to int/float (``True`` is not ``1``).
    """
    if isinstance(value, target) and not (target in (int, float) and isinstance(value, bool)):
        return value
    # Reject bool -> int/float coercion
    if isinstance(value, bool):
        raise TypeError(
            f"{field_name}: expected {target.__name__}, got bool ({value!r})"
        )
    try:
        return target(value)
    except (TypeError, ValueError):
        raise TypeError(
            f"{field_name}: expected {target.__name__}, "
            f"got {type(value).__name__} ({value!r})"
        ) from None


def _type_name(tp: type) -> str:
    """Human-readable name for a type annotation."""
    origin = get_origin(tp)
    if origin is Union:
        args = get_args(tp)
        # Optional[X] is Union[X, None]
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and type(None) in args:
            return f"Optional[{_type_name(non_none[0])}]"
        return " | ".join(_type_name(a) for a in args)
    if origin is list:
        (inner,) = get_args(tp)
        return f"List[{_type_name(inner)}]"
    if origin is tuple:
        args = get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return f"Tuple[{_type_name(args[0])}, ...]"
        return f"Tuple[{', '.join(_type_name(a) for a in args)}]"
    return getattr(tp, "__name__", str(tp))


def _check_type(value: Any, tp: type) -> bool:
    """Check if *value* matches type annotation *tp*.

    Handles: str, int, float, bool, Optional[X], Tuple[X, ...],
    List[X], and nested dataconfig classes.
    """
    origin = get_origin(tp)

    # Optional[X] = Union[X, None]
    if origin is Union:
        return any(_check_type(value, arg) for arg in get_args(tp))

    # List[X]
    if origin is list:
        if not isinstance(value, list):
            return False
        (inner,) = get_args(tp)
        return all(_check_type(v, inner) for v in value)

    # Tuple[X, ...]
    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        args = get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return all(_check_type(v, args[0]) for v in value)
        if len(args) != len(value):
            return False
        return all(_check_type(v, a) for v, a in zip(value, args))

    # NoneType
    if tp is type(None):
        return value is None

    # bool must be checked before int (bool is subclass of int)
    if tp is bool:
        return isinstance(value, bool)
    if tp is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if tp is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    return isinstance(value, tp)


def _validate_field_types(obj: Any) -> None:
    """Validate all dataclass field types on *obj*.

    Call after coercion in ``__post_init__`` to catch type mismatches
    with clear error messages.  Skips class-var frozensets (VALID_*).
    """
    hints = get_type_hints(type(obj))
    for f in fields(type(obj)):
        tp = hints.get(f.name)
        if tp is None:
            continue
        value = getattr(obj, f.name)
        if not _check_type(value, tp):
            raise TypeError(
                f"{type(obj).__name__}.{f.name}: expected {_type_name(tp)}, "
                f"got {type(value).__name__} ({value!r})"
            )


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
    for f in fields(TrainConfig):
        val = getattr(config, f.name)
        if f.name == "optimizer":
            for of in fields(OptimizerConfig):
                flat[f"optimizer.{of.name}"] = getattr(val, of.name)
        elif f.name == "network":
            for nf in fields(NetworkConfig):
                flat[f"network.{nf.name}"] = getattr(val, nf.name)
        elif f.name == "composite_loss":
            for cf in fields(CompositeLossConfig):
                flat[f"composite_loss.{cf.name}"] = getattr(val, cf.name)
        else:
            flat[f.name] = val
    return flat


def _flat_dict_to_config(flat: Dict[str, Any]) -> TrainConfig:
    """Reconstruct TrainConfig from flat dot-notation dict."""
    opt_kw: Dict[str, Any] = {}
    net_kw: Dict[str, Any] = {}
    comp_kw: Dict[str, Any] = {}
    train_kw: Dict[str, Any] = {}

    opt_fields = {f.name for f in fields(OptimizerConfig)}
    net_fields = {f.name for f in fields(NetworkConfig)}
    comp_fields = {f.name for f in fields(CompositeLossConfig)}
    train_fields = {f.name for f in fields(TrainConfig)} - {"optimizer", "network", "composite_loss"}

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
