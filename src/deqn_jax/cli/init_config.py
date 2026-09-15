"""``deqn-jax init-config``."""


def add_parser(subparsers):
    init_parser = subparsers.add_parser(
        "init-config", help="Generate default config file"
    )
    init_parser.add_argument(
        "output",
        nargs="?",
        default="train.yaml",
        help="Output path (default: train.yaml)",
    )
    init_parser.set_defaults(func=run)


def run(args):
    """Generate a default config file."""
    from deqn_jax.config import TrainConfig

    config = TrainConfig()
    config.to_yaml(args.output)
    print(f"Created {args.output}")
