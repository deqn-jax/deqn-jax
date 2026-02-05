"""Command-line interface for DEQN-JAX."""

import argparse
import sys


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
        choices=["brock_mirman", "disaster"],
        help="Model to train",
    )
    train_parser.add_argument(
        "-n", "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes (default: 1000)",
    )
    train_parser.add_argument(
        "--hidden",
        type=str,
        default="64,64",
        help="Hidden layer sizes, comma-separated (default: 64,64)",
    )
    train_parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    train_parser.add_argument(
        "--episode-length",
        type=int,
        default=100,
        help="Steps per episode (default: 100)",
    )
    train_parser.add_argument(
        "--mc-samples",
        type=int,
        default=5,
        help="Monte Carlo samples (default: 5)",
    )
    train_parser.add_argument(
        "-o", "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd", "adamw"],
        help="Optimizer (default: adam)",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    train_parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Log frequency (default: 100)",
    )
    train_parser.add_argument(
        "--warm-start",
        action="store_true",
        help="Initialize from steady state using L-BFGS",
    )
    train_parser.add_argument(
        "--fp64",
        action="store_true",
        help="Use float64 precision",
    )
    train_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output",
    )

    # New options
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
        default="none",
        choices=["none", "lr_annealing", "relobralo"],
        help="Adaptive loss reweighting strategy (default: none)",
    )
    train_parser.add_argument(
        "--reweight-alpha",
        type=float,
        default=0.9,
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

    # List command
    list_parser = subparsers.add_parser("list", help="List available models")

    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "list":
        run_list()
    else:
        parser.print_help()
        sys.exit(1)


def run_train(args):
    """Run training."""
    # Set precision before importing JAX
    if args.fp64:
        import jax
        jax.config.update("jax_enable_x64", True)

    from deqn_jax.training.trainer import train

    # Parse hidden sizes
    hidden_sizes = tuple(int(x) for x in args.hidden.split(","))

    # Parse loss weights
    loss_weights = None
    if args.loss_weights is not None:
        loss_weights = [float(x) for x in args.loss_weights.split(",")]

    # Train
    params, history = train(
        model_name=args.model,
        episodes=args.episodes,
        hidden_sizes=hidden_sizes,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        episode_length=args.episode_length,
        mc_samples=args.mc_samples,
        optimizer=args.optimizer,
        warm_start=args.warm_start,
        seed=args.seed,
        log_every=args.log_every,
        verbose=not args.quiet,
        grad_clip=args.grad_clip,
        loss_weights=loss_weights,
        loss_reweight=args.loss_reweight,
        reweight_alpha=args.reweight_alpha,
        tensorboard_dir=args.tensorboard,
        wandb_project=args.wandb,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
    )


def run_list():
    """List available models."""
    models = [
        ("brock_mirman", "Brock-Mirman (1972) optimal growth model"),
        ("disaster", "NK-DSGE with financial frictions (coming soon)"),
    ]

    print("Available models:")
    print()
    for name, desc in models:
        print(f"  {name:20s} - {desc}")
    print()
    print("Usage: deqn-jax train <model> [options]")


if __name__ == "__main__":
    main()
