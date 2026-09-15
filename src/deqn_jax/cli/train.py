"""``deqn-jax train``."""

import sys


def add_parser(subparsers):
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "model",
        type=str,
        nargs="?",
        default=None,
        help="Model to train; see 'deqn-jax list' for available models.",
    )
    train_parser.add_argument(
        "-n",
        "--episodes",
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
        "--lr",
        "--learning-rate",
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
        "-o",
        "--optimizer",
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
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress output",
    )

    # Config options
    train_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="YAML config file",
    )
    train_parser.add_argument(
        "-s",
        "--set",
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
    train_parser.set_defaults(func=run)


def run(args):
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
                print(
                    f"Error: --set values must be KEY=VAL, got '{item}'",
                    file=sys.stderr,
                )
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
        cli_kwargs["network.hidden_sizes"] = tuple(
            int(x) for x in args.hidden.split(",")
        )
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

    train_from_config(config)
