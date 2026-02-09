"""Structured configuration for DEQN-JAX training.

Three nested dataclasses with YAML loading and CLI override merging.

Priority: --set overrides > CLI args > YAML file > defaults
"""

from dataclasses import dataclass, field, fields, asdict
from typing import Any, Dict, List, Optional, Tuple
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

    def __post_init__(self):
        """Coerce YAML string values to proper numeric types."""
        self.learning_rate = float(self.learning_rate)
        if self.grad_clip is not None:
            self.grad_clip = float(self.grad_clip)
        self.weight_decay = float(self.weight_decay)
        self.beta1 = float(self.beta1)
        self.beta2 = float(self.beta2)
        self.epsilon = float(self.epsilon)
        self.lr_min_factor = float(self.lr_min_factor)


@dataclass
class CompositeLossConfig:
    """Composite loss configuration (anchor + Jacobian + barrier + Newton terms)."""

    anchor_weight: float = 1.0
    jac_weight: float = 0.1
    barrier_weight: float = 0.01
    newton_weight: float = 0.01
    n_anchor_points: int = 64
    anchor_sigma: float = 1.0
    leverage_mult: float = 5.0

    def __post_init__(self):
        """Coerce YAML string values to proper numeric types."""
        self.anchor_weight = float(self.anchor_weight)
        self.jac_weight = float(self.jac_weight)
        self.barrier_weight = float(self.barrier_weight)
        self.newton_weight = float(self.newton_weight)
        self.n_anchor_points = int(self.n_anchor_points)
        self.anchor_sigma = float(self.anchor_sigma)
        self.leverage_mult = float(self.leverage_mult)


@dataclass
class NetworkConfig:
    """Neural network configuration."""

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

    expectation_type: str = "mc"  # "mc" or "quadrature" (Gauss-Hermite)
    n_quadrature_points: int = 3  # Points per dimension (3^5=243 nodes for 5 shocks)

    def __post_init__(self):
        """Coerce YAML string values to proper numeric types."""
        if self.switch_lr is not None:
            self.switch_lr = float(self.switch_lr)
        if self.switch_episode is not None:
            self.switch_episode = int(self.switch_episode)
        self.curriculum_start = float(self.curriculum_start)
        self.early_stop_min_delta = float(self.early_stop_min_delta)

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

        # Filter to known fields only
        opt_fields = {f.name for f in fields(OptimizerConfig)}
        net_fields = {f.name for f in fields(NetworkConfig)}
        comp_fields = {f.name for f in fields(CompositeLossConfig)}
        train_fields = {f.name for f in fields(TrainConfig)}

        opt_kw = {k: v for k, v in opt_dict.items() if k in opt_fields}
        net_kw = {k: v for k, v in net_dict.items() if k in net_fields}
        comp_kw = {k: v for k, v in comp_dict.items() if k in comp_fields}
        train_kw = {k: v for k, v in d.items() if k in train_fields}

        return cls(
            optimizer=OptimizerConfig(**opt_kw),
            network=NetworkConfig(**net_kw),
            composite_loss=CompositeLossConfig(**comp_kw),
            **train_kw,
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

    for key, val in flat.items():
        if key.startswith("optimizer."):
            subkey = key[len("optimizer."):]
            if subkey in opt_fields:
                opt_kw[subkey] = val
        elif key.startswith("network."):
            subkey = key[len("network."):]
            if subkey in net_fields:
                net_kw[subkey] = val
        elif key.startswith("composite_loss."):
            subkey = key[len("composite_loss."):]
            if subkey in comp_fields:
                comp_kw[subkey] = val
        elif key in train_fields:
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
        config = _flat_dict_to_config(flat)

    # Apply --set overrides last (highest priority)
    if overrides:
        config = config.with_overrides(overrides)

    return config
