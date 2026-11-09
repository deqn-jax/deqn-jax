"""Helpers shared by the CLI commands."""

from pathlib import Path


def enable_fp64(fp64: bool, config_path=None) -> None:
    """Switch JAX to 64-bit if ``fp64`` is set or the YAML at ``config_path`` asks.

    Call before importing any ``deqn_jax`` module that builds arrays (models,
    model folders, training, evaluation): an array made earlier keeps the
    precision that was in force when it was made.
    """
    if not fp64 and config_path is not None and Path(config_path).exists():
        import yaml

        with open(config_path) as f:
            fp64 = bool((yaml.safe_load(f) or {}).get("fp64", False))
    if fp64:
        import jax

        jax.config.update("jax_enable_x64", True)


def enable_fp64_from_config(args):
    """Enable fp64 if the checkpoint's config (or ``--config``) requires it."""
    config_path = getattr(args, "config", None)
    if config_path is None:
        config_path = Path(args.checkpoint).parent / "config.yaml"
    enable_fp64(False, config_path)
