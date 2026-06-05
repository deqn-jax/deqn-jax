"""Config (de)serialization: YAML load, --set flat-dict merge, type inference."""

from __future__ import annotations

from difflib import get_close_matches
from typing import Any, Dict, Optional, Set

from deqn_jax.config.loss import CompositeLossConfig, MomentMatchingConfig
from deqn_jax.config.network import NetworkConfig
from deqn_jax.config.optimizer import OptimizerConfig
from deqn_jax.config.replay import ReplayBufferConfig
from deqn_jax.config.train import TrainConfig

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
            parts.append(
                f"  '{key}' (did you mean: {', '.join(repr(m) for m in matches)}?)"
            )
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
        elif name == "replay_buffer":
            for rf in ReplayBufferConfig.model_fields:
                flat[f"replay_buffer.{rf}"] = getattr(val, rf)
        elif name == "moment_matching":
            for mf in MomentMatchingConfig.model_fields:
                flat[f"moment_matching.{mf}"] = getattr(val, mf)
        else:
            flat[name] = val
    return flat


def _flat_dict_to_config(flat: Dict[str, Any]) -> TrainConfig:
    """Reconstruct TrainConfig from flat dot-notation dict."""
    opt_kw: Dict[str, Any] = {}
    net_kw: Dict[str, Any] = {}
    comp_kw: Dict[str, Any] = {}
    replay_kw: Dict[str, Any] = {}
    mom_kw: Dict[str, Any] = {}
    train_kw: Dict[str, Any] = {}

    opt_fields = set(OptimizerConfig.model_fields.keys())
    net_fields = set(NetworkConfig.model_fields.keys())
    comp_fields = set(CompositeLossConfig.model_fields.keys())
    replay_fields = set(ReplayBufferConfig.model_fields.keys())
    mom_fields = set(MomentMatchingConfig.model_fields.keys())
    train_fields = set(TrainConfig.model_fields.keys()) - {
        "optimizer",
        "network",
        "composite_loss",
        "replay_buffer",
        "moment_matching",
    }

    # Build set of all valid flat keys for validation
    valid_flat_keys = set(train_fields)
    valid_flat_keys |= {f"optimizer.{n}" for n in opt_fields}
    valid_flat_keys |= {f"network.{n}" for n in net_fields}
    valid_flat_keys |= {f"composite_loss.{n}" for n in comp_fields}
    valid_flat_keys |= {f"replay_buffer.{n}" for n in replay_fields}
    valid_flat_keys |= {f"moment_matching.{n}" for n in mom_fields}

    _check_unknown_keys(set(flat.keys()), valid_flat_keys, "config overrides")

    for key, val in flat.items():
        if key.startswith("optimizer."):
            opt_kw[key[len("optimizer.") :]] = val
        elif key.startswith("network."):
            net_kw[key[len("network.") :]] = val
        elif key.startswith("composite_loss."):
            comp_kw[key[len("composite_loss.") :]] = val
        elif key.startswith("replay_buffer."):
            replay_kw[key[len("replay_buffer.") :]] = val
        elif key.startswith("moment_matching."):
            mom_kw[key[len("moment_matching.") :]] = val
        else:
            train_kw[key] = val

    return TrainConfig(
        optimizer=OptimizerConfig(**opt_kw),
        network=NetworkConfig(**net_kw),
        composite_loss=CompositeLossConfig(**comp_kw),
        replay_buffer=ReplayBufferConfig(**replay_kw),
        moment_matching=MomentMatchingConfig(**mom_kw),
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
                hint = (
                    f" (did you mean: {', '.join(repr(m) for m in matches)}?)"
                    if matches
                    else ""
                )
                raise ValueError(f"Unknown CLI config key '{key}'{hint}")
        config = _flat_dict_to_config(flat)

    # Apply --set overrides last (highest priority)
    if overrides:
        config = config.with_overrides(overrides)

    return config
