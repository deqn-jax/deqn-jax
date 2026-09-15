"""``deqn-jax list`` / ``info`` / ``optimizers`` / ``check``."""

import sys


def add_parser(subparsers):
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.set_defaults(func=lambda args: run_list())
    info_parser = subparsers.add_parser("info", help="Show model details")
    info_parser.add_argument("model", type=str, help="Model name")
    info_parser.set_defaults(func=run_info)
    opt_parser = subparsers.add_parser("optimizers", help="List available optimizers")
    opt_parser.set_defaults(func=lambda args: run_optimizers())
    check_parser = subparsers.add_parser("check", help="Check installation")
    check_parser.set_defaults(func=lambda args: run_check())


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
    for name in model.state_names or []:
        print(f"  {name}")

    print(f"\nPolicies ({model.n_policies}):")
    if (
        model.state_names
        and model.policy_lower is not None
        and model.policy_upper is not None
    ):
        for i, name in enumerate(model.policy_names or []):
            lo = float(model.policy_lower[i])
            hi = float(model.policy_upper[i])
            print(f"  {name:20s} [{lo:.4g}, {hi:.4g}]")
    else:
        for name in model.policy_names or []:
            print(f"  {name}")

    print(f"\nEquations ({len(model.equation_names or ())}):")
    for name in model.equation_names or []:
        print(f"  {name}")

    print(f"\nShocks ({model.n_shocks}):")
    for name in model.shock_names or []:
        print(f"  {name}")
    print(f"Steady state: {'yes' if model.steady_state_fn else 'no'}")

    print(f"\nConstants ({len(model.constants)}):")
    for k, v in model.constants.items():
        print(f"  {k:20s} = {v}")

    print()


def run_optimizers():
    """List registered optimizers."""
    from deqn_jax.optimizers import list_optimizers
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
