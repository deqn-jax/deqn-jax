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
    # K-FAC
    kfac_damping: float = 1e-3
    kfac_update_freq: int = 10


@dataclass
class NetworkConfig:
    """Neural network configuration."""

    type: str = "mlp"
    hidden_sizes: Tuple[int, ...] = (64, 64)
    activation: str = "tanh"


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

    warm_start: bool = False
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

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        """Create config from a flat or nested dictionary.

        Handles nested ``optimizer:`` and ``network:`` sub-dicts.
        """
        d = copy.deepcopy(d)

        # Extract nested sub-configs
        opt_dict = d.pop("optimizer", {})
        net_dict = d.pop("network", {})

        # If optimizer is a plain string, treat as name
        if isinstance(opt_dict, str):
            opt_dict = {"name": opt_dict}
        if isinstance(net_dict, str):
            net_dict = {"type": net_dict}

        # Convert hidden_sizes list to tuple
        if "hidden_sizes" in net_dict and isinstance(net_dict["hidden_sizes"], list):
            net_dict["hidden_sizes"] = tuple(net_dict["hidden_sizes"])

        # Convert loss_weights list (YAML gives lists)
        if "loss_weights" in d and isinstance(d["loss_weights"], list):
            d["loss_weights"] = list(d["loss_weights"])

        # Filter to known fields only
        opt_fields = {f.name for f in fields(OptimizerConfig)}
        net_fields = {f.name for f in fields(NetworkConfig)}
        train_fields = {f.name for f in fields(TrainConfig)}

        opt_kw = {k: v for k, v in opt_dict.items() if k in opt_fields}
        net_kw = {k: v for k, v in net_dict.items() if k in net_fields}
        train_kw = {k: v for k, v in d.items() if k in train_fields}

        return cls(
            optimizer=OptimizerConfig(**opt_kw),
            network=NetworkConfig(**net_kw),
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
        else:
            flat[f.name] = val
    return flat


def _flat_dict_to_config(flat: Dict[str, Any]) -> TrainConfig:
    """Reconstruct TrainConfig from flat dot-notation dict."""
    opt_kw: Dict[str, Any] = {}
    net_kw: Dict[str, Any] = {}
    train_kw: Dict[str, Any] = {}

    opt_fields = {f.name for f in fields(OptimizerConfig)}
    net_fields = {f.name for f in fields(NetworkConfig)}
    train_fields = {f.name for f in fields(TrainConfig)} - {"optimizer", "network"}

    for key, val in flat.items():
        if key.startswith("optimizer."):
            subkey = key[len("optimizer."):]
            if subkey in opt_fields:
                opt_kw[subkey] = val
        elif key.startswith("network."):
            subkey = key[len("network."):]
            if subkey in net_fields:
                net_kw[subkey] = val
        elif key in train_fields:
            train_kw[key] = val

    return TrainConfig(
        optimizer=OptimizerConfig(**opt_kw),
        network=NetworkConfig(**net_kw),
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
