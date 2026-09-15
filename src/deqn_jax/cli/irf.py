"""``deqn-jax irf``."""

from deqn_jax.cli._common import enable_fp64_from_config


def add_parser(subparsers):
    irf_parser = subparsers.add_parser("irf", help="Run impulse response functions")
    irf_parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to checkpoint .eqx file",
    )
    irf_parser.add_argument(
        "--shock",
        "-s",
        type=str,
        action="append",
        dest="shocks",
        help="Shock name; see 'deqn-jax info <model>' for valid names. "
        "Repeatable. Default: all model shocks.",
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
        "--output",
        "-o",
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
    irf_parser.set_defaults(func=run)


def run(args):
    """Run impulse response functions."""
    enable_fp64_from_config(args)
    from deqn_jax.evaluate.irf import run_irf_cli

    run_irf_cli(args)
