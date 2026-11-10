"""``deqn-jax verify``: is a model folder's run a solution? Writes report.html."""

import sys


def add_parser(subparsers):
    p = subparsers.add_parser(
        "verify",
        help="Verify a model folder's latest run and write its report with figures",
    )
    p.add_argument("folder", help="Model folder (model.py, train.yaml, runs/)")
    p.add_argument("--run", default=None, help="Run number, e.g. 03 (default: latest)")
    p.add_argument(
        "--exact",
        action="store_true",
        help="Before training: run the check on exact.py's policy instead of a run. "
        "Errors near 1e-7 confirm model.py's equations and training range.",
    )
    p.set_defaults(func=run)


def _verify_exact(folder):
    from deqn_jax.evaluate.solution import check_solution, print_solution_check
    from deqn_jax.models.folder import (
        folder_model_name,
        load_exact_policy,
        load_model_file,
    )

    model, module = load_model_file(folder, folder_model_name(folder))
    exact = load_exact_policy(folder)
    if exact is None:
        sys.exit(f"{folder}/exact.py does not exist")
    check = check_solution(
        lambda s: exact(s, model.constants),
        model,
        exact,
        getattr(module, "SHOCK_NODES_PER_DIM", None),
    )
    print("Check of the exact solution (exact.py) against model.py:")
    print_solution_check(check)
    sys.exit(0 if check.solved else 1)


def run(args):
    if args.exact:
        _verify_exact(args.folder)
    from deqn_jax.evaluate.report import check_run, write_report
    from deqn_jax.evaluate.solution import print_solution_check
    from deqn_jax.models.folder import find_run

    run_dir = find_run(args.folder, args.run)
    check, net, model = check_run(run_dir)
    report = write_report(run_dir, check, net, model)
    print(f"Run: {run_dir}")
    print_solution_check(check)
    print(f"\nReport: {report}\nFigures: {run_dir / 'figures'}")
    sys.exit(0 if check.solved else 1)
