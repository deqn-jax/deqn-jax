"""Config (de)serialization: YAML load, --set flat-dict merge, type inference."""

from __future__ import annotations

from difflib import get_close_matches
from typing import Any, Dict, Optional, Set

from deqn_jax.config._base import _ConfigBase
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


def _nested_blocks() -> Dict[str, type]:
    """``{field name: config class}`` for every nested ``_ConfigBase`` block on
    TrainConfig (optimizer, network, composite_loss, replay_buffer,
    moment_matching, coverage, ...). Derived from the model, so a new block is
    reachable through ``--set block.field`` without touching this module."""
    blocks: Dict[str, type] = {}
    for name, field in TrainConfig.model_fields.items():
        ann = field.annotation
        if isinstance(ann, type) and issubclass(ann, _ConfigBase):
            blocks[name] = ann
    return blocks


def _config_to_flat_dict(config: TrainConfig) -> Dict[str, Any]:
    """Flatten a TrainConfig into dot-notation keys."""
    blocks = _nested_blocks()
    flat: Dict[str, Any] = {}
    for name in TrainConfig.model_fields:
        val = getattr(config, name)
        if name in blocks:
            for sub in blocks[name].model_fields:
                flat[f"{name}.{sub}"] = getattr(val, sub)
        else:
            flat[name] = val
    return flat


def _flat_dict_to_config(flat: Dict[str, Any]) -> TrainConfig:
    """Reconstruct TrainConfig from flat dot-notation dict."""
    blocks = _nested_blocks()
    block_kw: Dict[str, Dict[str, Any]] = {name: {} for name in blocks}
    train_kw: Dict[str, Any] = {}

    train_fields = set(TrainConfig.model_fields.keys()) - set(blocks)
    valid_flat_keys = set(train_fields)
    for name, cls in blocks.items():
        valid_flat_keys |= {f"{name}.{sub}" for sub in cls.model_fields}

    _check_unknown_keys(set(flat.keys()), valid_flat_keys, "config overrides")

    for key, val in flat.items():
        block, dot, sub = key.partition(".")
        if dot and block in blocks:
            block_kw[block][sub] = val
        else:
            train_kw[key] = val

    return TrainConfig(
        **{name: cls(**block_kw[name]) for name, cls in blocks.items()},
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
