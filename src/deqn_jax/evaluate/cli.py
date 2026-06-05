"""evaluate CLI entry point."""

import csv
import os
from pathlib import Path

import jax
import yaml

from deqn_jax.evaluate.diagnostics import (
    euler_equation_errors,
    market_clearing_errors,
    print_euler_errors,
    print_moments,
    simulated_moments,
    stability_check,
)
from deqn_jax.evaluate.dynare import (
    compare_to_dynare_ghx,
    compare_to_dynare_irfs,
    compare_to_dynare_moments,
    print_dynare_comparison,
)
from deqn_jax.irf import load_policy_from_checkpoint

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_evaluate_cli(args):
    """CLI handler for 'deqn-jax evaluate'."""
    # Enable fp64 if config says so
    ckpt_dir = Path(args.checkpoint).parent
    config_path = args.config or str(ckpt_dir / "config.yaml")
    if Path(config_path).exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        if cfg.get("fp64", False):
            jax.config.update("jax_enable_x64", True)

    print(f"Loading checkpoint: {args.checkpoint}")
    policy_net, model = load_policy_from_checkpoint(
        args.checkpoint, config_path if Path(config_path).exists() else None
    )
    n_params = sum(p.size for p in jax.tree.leaves(policy_net))
    print(f"Model: {model.name}, params: {n_params}")

    label = args.label or Path(args.checkpoint).parent.name

    # 1. Stability check
    print("\n[1/4] Stability check...")
    stab = stability_check(policy_net, model, n_periods=args.periods, seed=args.seed)
    print(f"  NaN-free:          {'YES' if stab['nan_free'] else 'NO'}")
    print(f"  Bound-hitting:     {stab['bound_hit_pct']:.1f}% of outputs")
    print(f"  Max SS deviation:  {stab['max_ss_deviation_pct']:.1f}%")
    print(f"  Stable:            {'YES' if stab['stable'] else 'NO'}")

    if not stab["nan_free"]:
        print("\n  FATAL: Simulation produced NaN. Cannot continue evaluation.")
        return

    # 2. Euler equation errors
    print(f"\n[2/4] Euler equation errors ({args.periods} periods)...")
    ee_result = euler_equation_errors(
        policy_net, model, n_periods=args.periods, seed=args.seed
    )
    print_euler_errors(ee_result, label=label)

    # 3. Market clearing
    print("[3/4] Market clearing...")
    mc = market_clearing_errors(
        policy_net, model, n_periods=args.periods, seed=args.seed
    )
    if "error" not in mc:
        print(f"  {mc['equation']}:")
        print(
            f"    Mean |error|:  {mc['mean_abs']:.2e} (log10: {mc['mean_log10']:.2f})"
        )
        print(f"    Max  |error|:  {mc['max_abs']:.2e} (log10: {mc['max_log10']:.2f})")

    # 4. Simulated moments
    n_steps = 5 if getattr(args, "dynare_dir", None) else 4
    print(f"\n[4/{n_steps}] Simulated moments...")
    moments = simulated_moments(
        policy_net, model, n_periods=args.periods, seed=args.seed
    )
    print_moments(moments, label=label, n_periods=args.periods)

    # 5. Dynare comparison (optional)
    if getattr(args, "dynare_dir", None):
        print(f"\n[5/{n_steps}] Dynare comparison...")
        moments_diff = compare_to_dynare_moments(
            policy_net,
            model,
            args.dynare_dir,
            n_periods=args.periods,
            seed=args.seed,
        )
        ghx_diff = compare_to_dynare_ghx(
            policy_net,
            model,
            args.dynare_dir,
            perturb_sigma=getattr(args, "dynare_ghx_perturb", 1.0e-3),
        )
        irf_diff = compare_to_dynare_irfs(
            policy_net,
            model,
            args.dynare_dir,
            horizon=40,
            use_girf=getattr(args, "dynare_irf_girf", False),
        )
        print_dynare_comparison(moments_diff, ghx_diff, irf_diff, label=label)

    # Save results
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        # Save Euler errors CSV
        residuals = ee_result["residuals"]
        eq_names = ee_result["equation_names"]
        csv_path = os.path.join(args.output, "euler_errors.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(eq_names)
            for row in residuals:
                writer.writerow([float(v) for v in row])
        print(f"\nSaved Euler errors to {csv_path}")

        # Save moments CSV
        csv_path = os.path.join(args.output, "moments.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["variable", "ss", "mean", "std", "min", "max", "dev_pct"])
            for name, m in moments.items():
                writer.writerow(
                    [
                        name,
                        m["ss"],
                        m["mean"],
                        m["std"],
                        m["min"],
                        m["max"],
                        m["mean_dev_pct"],
                    ]
                )
        print(f"Saved moments to {csv_path}")
