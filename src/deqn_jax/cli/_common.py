"""Helpers shared by the CLI commands."""


def enable_fp64_from_config(args):
    """Enable fp64 if checkpoint config requires it."""
    from pathlib import Path

    import yaml

    config_path = getattr(args, "config", None)
    if config_path is None:
        ckpt_dir = Path(args.checkpoint).parent
        config_path = str(ckpt_dir / "config.yaml")
    if Path(config_path).exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        if cfg.get("fp64", False):
            import jax

            jax.config.update("jax_enable_x64", True)
