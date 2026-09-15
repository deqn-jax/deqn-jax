"""``deqn-jax evaluate``."""

from deqn_jax.cli._common import enable_fp64_from_config


def add_parser(subparsers):
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate trained model accuracy"
    )
    eval_parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to checkpoint .eqx file",
    )
    eval_parser.add_argument(
        "--periods",
        "-n",
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
        "--dynare-dir",
        type=str,
        default=None,
        help="Directory with dynare_moments.csv / dynare_ghx.csv / dynare_ghu.csv "
        "/ irf_e_<shock>.csv. When provided, runs moment, linearization (ghx), "
        "and IRF comparisons against the Dynare perturbation reference.",
    )
    eval_parser.add_argument(
        "--dynare-irf-girf",
        action="store_true",
        help="Use generalized IRF (GIRF) for the Dynare IRF comparison; "
        "matches the nonlinear-policy IRF semantics of `deqn-jax irf --girf`. "
        "Default deterministic IRF mirrors Dynare's first-order perturbation IRFs.",
    )
    eval_parser.add_argument(
        "--dynare-ghx-perturb",
        type=float,
        default=1.0e-3,
        help="Std of Gaussian perturbation around SS used by the ghx jacobian "
        "comparator (default: 1e-3). Set to 0 to evaluate at exact SS, which "
        "may hit gradient discontinuities at sigmoid/soft-floor boundaries.",
    )
    eval_parser.set_defaults(func=run)


def run(args):
    """Run model evaluation suite."""
    enable_fp64_from_config(args)
    from deqn_jax.evaluate import run_evaluate_cli

    run_evaluate_cli(args)
