"""Command-line interface for DEQN-JAX."""

import argparse
import os
import sys

# JAX preallocates 75% of GPU VRAM by default — disable so multiple
# runs can share one GPU and small models don't waste memory.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DEQN-JAX: Pure JAX Deep Equilibrium Networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "model",
        type=str,
        nargs="?",
        default=None,
        help="Model to train (brock_mirman, disaster)",
    )
    train_parser.add_argument(
        "-n", "--episodes",
        type=int,
        default=None,
        help="Number of training episodes (default: 1000)",
    )
    train_parser.add_argument(
        "--hidden",
        type=str,
        default=None,
        help="Hidden layer sizes, comma-separated (default: 64,64)",
    )
    train_parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (default: 1e-3)",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: 64)",
    )
    train_parser.add_argument(
        "--episode-length",
        type=int,
        default=None,
        help="Steps per episode (default: 100)",
    )
    train_parser.add_argument(
        "--mc-samples",
        type=int,
        default=None,
        help="Monte Carlo samples (default: 5)",
    )
    train_parser.add_argument(
        "-o", "--optimizer",
        type=str,
        default=None,
        help="Optimizer name (default: adam). Use 'deqn-jax optimizers' to list.",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: 42)",
    )
    train_parser.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="Log frequency (default: 100)",
    )
    train_parser.add_argument(
        "--warm-start",
        action="store_true",
        default=None,
        help="Initialize from steady state using L-BFGS",
    )
    train_parser.add_argument(
        "--fp64",
        action="store_true",
        default=None,
        help="Use float64 precision",
    )
    train_parser.add_argument(
        "--gradient-surgery",
        choices=["none", "pcgrad"],
        default=None,
        help="Gradient surgery method for multi-equation conflict resolution",
    )
    train_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output",
    )

    # Config options
    train_parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="YAML config file",
    )
    train_parser.add_argument(
        "-s", "--set",
        type=str,
        action="append",
        default=None,
        dest="overrides",
        metavar="KEY=VAL",
        help="Override config (repeatable, dot-notation). E.g. --set optimizer.learning_rate=0.01",
    )

    # Optimizer-specific options
    train_parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Gradient clipping norm (default: none)",
    )
    train_parser.add_argument(
        "--loss-weights",
        type=str,
        default=None,
        help="Manual equation weights, comma-separated (e.g. '1.0,0.5')",
    )
    train_parser.add_argument(
        "--loss-reweight",
        type=str,
        default=None,
        choices=["none", "lr_annealing", "relobralo"],
        help="Adaptive loss reweighting strategy (default: none)",
    )
    train_parser.add_argument(
        "--reweight-alpha",
        type=float,
        default=None,
        help="EMA decay for adaptive reweighting (default: 0.9)",
    )
    train_parser.add_argument(
        "--tensorboard",
        type=str,
        default=None,
        metavar="DIR",
        help="TensorBoard log directory",
    )
    train_parser.add_argument(
        "--wandb",
        type=str,
        default=None,
        metavar="PROJECT",
        help="W&B project name",
    )
    train_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint save directory",
    )
    train_parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Checkpoint interval (episodes)",
    )
    train_parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=None,
        help="Keep only N most recent checkpoints",
    )
    train_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help="Resume from checkpoint .eqx file",
    )
    train_parser.add_argument(
        "--switch-optimizer",
        type=str,
        default=None,
        help="Switch to this optimizer mid-training",
    )
    train_parser.add_argument(
        "--switch-episode",
        type=int,
        default=None,
        help="Episode at which to switch optimizer",
    )
    train_parser.add_argument(
        "--switch-lr",
        type=float,
        default=None,
        help="Learning rate for switched optimizer",
    )
    train_parser.add_argument(
        "--lr-schedule",
        type=str,
        default=None,
        choices=["constant", "cosine"],
        help="LR schedule (default: constant)",
    )
    train_parser.add_argument(
        "--lr-warmup",
        type=int,
        default=None,
        help="Warmup episodes before LR decay (default: 0)",
    )
    train_parser.add_argument(
        "--lr-min-factor",
        type=float,
        default=None,
        help="Min LR as fraction of peak (default: 0.0)",
    )

    # List command
    subparsers.add_parser("list", help="List available models")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show model details")
    info_parser.add_argument("model", type=str, help="Model name")

    # Optimizers command
    subparsers.add_parser("optimizers", help="List available optimizers")

    # IRF command
    irf_parser = subparsers.add_parser("irf", help="Run impulse response functions")
    irf_parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to checkpoint .eqx file",
    )
    irf_parser.add_argument(
        "--shock", "-s",
        type=str,
        action="append",
        dest="shocks",
        help="Shock name (eps, mu_ups, mu_z, g, m_p). Repeatable. Default: all.",
    )
    irf_parser.add_argument(
        "--shock-size",
        type=float,
        default=1.0,
        help="Shock size in std devs (default: 1.0)",
    )
    irf_parser.add_argument(
        "--horizon",
        type=int,
        default=40,
        help="Periods after shock (default: 40)",
    )
    irf_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config YAML (auto-detected from checkpoint dir if omitted)",
    )
    irf_parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (default: irf_results)",
    )
    irf_parser.add_argument(
        "--girf",
        action="store_true",
        help="Generalized IRF: subtract a no-shock baseline trajectory "
             "(same initial state, zero shocks) from the shocked path. "
             "Required for nonlinear models where the initial state is not "
             "a fixed point of step(·, 0, d=0) — e.g. disaster with risky SS.",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model accuracy")
    eval_parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to checkpoint .eqx file",
    )
    eval_parser.add_argument(
        "--periods", "-n",
        type=int,
        default=10_000,
        help="Simulation length (default: 10,000)",
    )
    eval_parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed (default: 123)",
    )
    eval_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config YAML (auto-detected from checkpoint dir if omitted)",
    )
    eval_parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Label for output (default: checkpoint dir name)",
    )
    eval_parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for CSV results",
    )

    # Check command
    subparsers.add_parser("check", help="Check installation")

    # Init-config command
    init_parser = subparsers.add_parser("init-config", help="Generate default config file")
    init_parser.add_argument("output", nargs="?", default="train.yaml", help="Output path (default: train.yaml)")

    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "list":
        run_list()
    elif args.command == "info":
        run_info(args)
    elif args.command == "optimizers":
        run_optimizers()
    elif args.command == "irf":
        run_irf_command(args)
    elif args.command == "evaluate":
        run_evaluate_command(args)
    elif args.command == "check":
        run_check()
    elif args.command == "init-config":
        run_init_config(args)
    else:
        parser.print_help()
        sys.exit(1)


def run_train(args):
    """Run training."""
    # Set precision before importing JAX — check both CLI flag and config file
    if args.fp64:
        import jax
        jax.config.update("jax_enable_x64", True)
    elif hasattr(args, "config") and args.config:
        # Check if YAML config has fp64: true (before full config load)
        import yaml
        with open(args.config) as _f:
            _raw = yaml.safe_load(_f) or {}
        if _raw.get("fp64", False):
            import jax
            jax.config.update("jax_enable_x64", True)

    from deqn_jax.config import load_config
    from deqn_jax.training.trainer import train_from_config

    # Parse --set overrides
    overrides = {}
    if args.overrides:
        for item in args.overrides:
            if "=" not in item:
                print(f"Error: --set values must be KEY=VAL, got '{item}'", file=sys.stderr)
                sys.exit(1)
            key, val = item.split("=", 1)
            overrides[key] = val

    # Build CLI kwargs (non-None values only)
    cli_kwargs = {}
    if args.model is not None:
        cli_kwargs["model"] = args.model
    if args.episodes is not None:
        cli_kwargs["episodes"] = args.episodes
    if args.hidden is not None:
        cli_kwargs["network.hidden_sizes"] = tuple(int(x) for x in args.hidden.split(","))
    if args.lr is not None:
        cli_kwargs["optimizer.learning_rate"] = args.lr
    if args.batch_size is not None:
        cli_kwargs["batch_size"] = args.batch_size
    if args.episode_length is not None:
        cli_kwargs["episode_length"] = args.episode_length
    if args.mc_samples is not None:
        cli_kwargs["mc_samples"] = args.mc_samples
    if args.optimizer is not None:
        cli_kwargs["optimizer.name"] = args.optimizer
    if args.seed is not None:
        cli_kwargs["seed"] = args.seed
    if args.log_every is not None:
        cli_kwargs["log_every"] = args.log_every
    if args.warm_start:
        cli_kwargs["warm_start"] = True
    if args.fp64:
        cli_kwargs["fp64"] = True
    if getattr(args, "gradient_surgery", None) is not None:
        cli_kwargs["gradient_surgery"] = args.gradient_surgery
    if args.quiet:
        cli_kwargs["verbose"] = False
    if args.grad_clip is not None:
        cli_kwargs["optimizer.grad_clip"] = args.grad_clip
    if args.loss_reweight is not None:
        cli_kwargs["loss_reweight"] = args.loss_reweight
    if args.reweight_alpha is not None:
        cli_kwargs["reweight_alpha"] = args.reweight_alpha
    if args.tensorboard is not None:
        cli_kwargs["tensorboard_dir"] = args.tensorboard
    if args.wandb is not None:
        cli_kwargs["wandb_project"] = args.wandb
    if args.checkpoint_dir is not None:
        cli_kwargs["checkpoint_dir"] = args.checkpoint_dir
    if args.checkpoint_every is not None:
        cli_kwargs["checkpoint_every"] = args.checkpoint_every
    if args.max_checkpoints is not None:
        cli_kwargs["max_checkpoints"] = args.max_checkpoints
    if args.resume is not None:
        cli_kwargs["resume"] = args.resume
    if args.switch_optimizer is not None:
        cli_kwargs["switch_optimizer"] = args.switch_optimizer
    if args.switch_episode is not None:
        cli_kwargs["switch_episode"] = args.switch_episode
    if args.switch_lr is not None:
        cli_kwargs["switch_lr"] = args.switch_lr
    if args.lr_schedule is not None:
        cli_kwargs["optimizer.lr_schedule"] = args.lr_schedule
    if args.lr_warmup is not None:
        cli_kwargs["optimizer.lr_warmup"] = args.lr_warmup
    if args.lr_min_factor is not None:
        cli_kwargs["optimizer.lr_min_factor"] = args.lr_min_factor

    # Load config with priority: --set > CLI > YAML > defaults
    config = load_config(
        config_path=args.config,
        overrides=overrides,
        **cli_kwargs,
    )

    # Parse loss_weights (special case -- list type)
    if args.loss_weights is not None:
        config = config.model_copy(
            update={"loss_weights": [float(x) for x in args.loss_weights.split(",")]}
        )

    # Validate model is set
    if config.model is None or config.model == "brock_mirman" and args.model is None and args.config is None:
        pass  # default is fine

    train_from_config(config)


def _enable_fp64_from_config(args):
    """Enable fp64 if checkpoint config requires it."""
    import yaml
    from pathlib import Path
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


def run_irf_command(args):
    """Run impulse response functions."""
    _enable_fp64_from_config(args)
    from deqn_jax.irf import run_irf_cli
    run_irf_cli(args)


def run_evaluate_command(args):
    """Run model evaluation suite."""
    _enable_fp64_from_config(args)
    from deqn_jax.evaluate import run_evaluate_cli
    run_evaluate_cli(args)


def run_list():
    """List available models."""
    from deqn_jax.models import list_models

    models = list_models()

    print("Available models:")
    print()
    for name, desc in models:
        print(f"  {name:20s} - {desc}")
    print()
    print("Usage: deqn-jax train <model> [options]")


def run_info(args):
    """Show model details."""
    from deqn_jax.models import load_model

    try:
        model = load_model(args.model)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    w = 60
    print("=" * w)
    print(f"Model: {model.name}")
    print("=" * w)

    print(f"\nStates ({model.n_states}):")
    for name in (model.state_names or []):
        print(f"  {name}")

    print(f"\nPolicies ({model.n_policies}):")
    if model.state_names and model.policy_lower is not None and model.policy_upper is not None:
        import jax.numpy as jnp
        for i, name in enumerate(model.policy_names or []):
            lo = float(model.policy_lower[i])
            hi = float(model.policy_upper[i])
            print(f"  {name:20s} [{lo:.4g}, {hi:.4g}]")
    else:
        for name in (model.policy_names or []):
            print(f"  {name}")

    print(f"\nEquations ({len(model.equation_names or ())}):")
    for name in (model.equation_names or []):
        print(f"  {name}")

    print(f"\nShocks: {model.n_shocks}")
    print(f"Steady state: {'yes' if model.steady_state_fn else 'no'}")

    print(f"\nConstants ({len(model.constants)}):")
    for k, v in model.constants.items():
        print(f"  {k:20s} = {v}")

    print()


def run_check():
    """Check installation."""
    import jax

    print(f"JAX:     {jax.__version__}")
    print(f"Devices: {jax.devices()}")

    x = jax.numpy.ones((2, 2))
    print(f"Ops:     OK (sum={float(jax.numpy.sum(x))})")

    import equinox
    print(f"Equinox: {equinox.__version__}")

    import optax
    print(f"Optax:   {optax.__version__}")

    from deqn_jax.models import list_models
    names = [n for n, _ in list_models()]
    print(f"Models:  {names}")

    from deqn_jax.optimizers import list_optimizers
    print(f"Optims:  {list_optimizers()}")

    print("\nAll checks passed!")


def run_init_config(args):
    """Generate a default config file."""
    from deqn_jax.config import TrainConfig

    config = TrainConfig()
    config.to_yaml(args.output)
    print(f"Created {args.output}")


def run_optimizers():
    """List registered optimizers."""
    from deqn_jax.optimizers import list_optimizers, OptimizerKind
    from deqn_jax.optimizers.registry import _REGISTRY

    print("Available optimizers:")
    print()
    for name in list_optimizers():
        _, kind = _REGISTRY[name]
        kind_str = f"({kind.value})"
        print(f"  {name:12s} {kind_str}")
    print()
    print("Usage: deqn-jax train <model> -o <optimizer>")
    print("   or: deqn-jax train --config config.yaml --set optimizer.name=<optimizer>")


if __name__ == "__main__":
    main()
