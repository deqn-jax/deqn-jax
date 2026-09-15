"""Command-line interface for DEQN-JAX: one module per subcommand.

Each command module exposes ``add_parser(subparsers)`` (its argparse
subparser, with ``func`` bound to its handler) and imports its heavy
dependencies lazily inside the handler, so ``deqn-jax --help`` stays fast.
"""

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
    from deqn_jax import __version__

    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"deqn-jax {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    from deqn_jax.cli import evaluate, init_config, irf, models, train

    train.add_parser(subparsers)
    models.add_parser(subparsers)
    irf.add_parser(subparsers)
    evaluate.add_parser(subparsers)
    init_config.add_parser(subparsers)

    args = parser.parse_args()
    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        sys.exit(1)
    handler(args)


if __name__ == "__main__":
    main()
