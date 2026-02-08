"""Evaluation suite for trained DEQN models.

Implements the standard accuracy metrics from the computational economics
literature (Azinovic et al. 2022, Den Haan & Marcet 1994):

1. Euler Equation Errors — log10 residuals along simulated path
2. Impulse Response Functions — economy's response to shocks
3. Market Clearing — resource constraint satisfaction
4. Simulated Moments — compare ergodic moments to steady state

Usage:
    deqn-jax evaluate --checkpoint path/to/checkpoint.eqx [--periods 10000]
"""

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import yaml
from jax import Array

from deqn_jax.irf import load_policy_from_checkpoint


# ---------------------------------------------------------------------------
# 1. Euler Equation Errors
# ---------------------------------------------------------------------------

def euler_equation_errors(
    policy_net: eqx.Module,
    model,
    n_periods: int = 10_000,
    seed: int = 123,
    burn_in: int = 500,
) -> Dict[str, Array]:
    """Simulate a long stochastic path and compute Euler residuals everywhere.

    This is the gold standard for DEQN accuracy (Azinovic et al. 2022).
    Reports log10(|residual|) distribution.

    Args:
        policy_net: Trained policy network
        model: ModelSpec
        n_periods: Length of simulation (default: 10,000)
        seed: Random seed for shock draws
        burn_in: Discard first N periods (reach ergodic distribution)

    Returns:
        Dict with:
            "residuals": [n_periods - burn_in, n_equations] array of log10 |residuals|
            "equation_names": list of equation names
            "states": [n_periods - burn_in, n_states] simulated states
    """
    constants = model.constants
    ss_state, ss_policy = model.steady_state_fn(constants)
    n_shocks = model.n_shocks

    state = ss_state[None, :]  # [1, n_states]
    key = jax.random.PRNGKey(seed)

    eq_names = list(model.equation_names) if model.equation_names else []
    n_eq = len(eq_names)

    # JIT-compile the simulation step for speed
    @eqx.filter_jit
    def _sim_step(state, shock):
        policy = policy_net(state)
        if policy.ndim == 1:
            policy = policy[None, :]
        next_state = model.step_fn(state, policy, shock, constants)
        next_policy = policy_net(next_state)
        if next_policy.ndim == 1:
            next_policy = next_policy[None, :]
        residuals = model.equations_fn(state, policy, next_state, next_policy, constants)
        row = jnp.stack([residuals[name][0] if residuals[name].ndim > 0
                        else residuals[name] for name in eq_names])
        return next_state, row, state[0]

    all_residuals = []
    all_states = []

    for t in range(n_periods):
        key, shock_key = jax.random.split(key)
        shock = jax.random.normal(shock_key, (1, n_shocks))

        next_state, row, st = _sim_step(state, shock)

        if t >= burn_in:
            all_residuals.append(row)
            all_states.append(st)

        state = next_state

    residuals_array = jnp.stack(all_residuals)  # [T, n_eq]
    states_array = jnp.stack(all_states)  # [T, n_states]

    return {
        "residuals": residuals_array,
        "equation_names": eq_names,
        "states": states_array,
    }


def print_euler_errors(result: Dict, label: str = ""):
    """Print Euler equation error table in the standard format."""
    residuals = result["residuals"]  # [T, n_eq]
    eq_names = result["equation_names"]

    # log10(|residual|), clamp to avoid log(0)
    log_errors = jnp.log10(jnp.maximum(jnp.abs(residuals), 1e-20))

    header = f"Euler Equation Errors (log10)"
    if label:
        header += f" — {label}"
    print(f"\n{header}")
    print("=" * 100)
    print(f"{'Equation':>30s}  {'Mean':>7s}  {'p50':>7s}  {'p95':>7s}  {'p99':>7s}  {'p99.9':>7s}  {'Max':>7s}  {'Grade':>12s}")
    print("-" * 100)

    for i, name in enumerate(eq_names):
        col = log_errors[:, i]
        mean_val = float(jnp.mean(col))
        p50 = float(jnp.percentile(col, 50))
        p95 = float(jnp.percentile(col, 95))
        p99 = float(jnp.percentile(col, 99))
        p999 = float(jnp.percentile(col, 99.9))
        max_val = float(jnp.max(col))

        # Grade based on mean
        if mean_val < -4:
            grade = "Very good"
        elif mean_val < -3:
            grade = "Good"
        elif mean_val < -2:
            grade = "Acceptable"
        else:
            grade = "POOR"

        print(f"{name:>30s}  {mean_val:>7.2f}  {p50:>7.2f}  {p95:>7.2f}  "
              f"{p99:>7.2f}  {p999:>7.2f}  {max_val:>7.2f}  {grade:>12s}")

    # Overall summary
    all_log = log_errors.flatten()
    print("-" * 100)
    print(f"{'OVERALL':>30s}  {float(jnp.mean(all_log)):>7.2f}  "
          f"{float(jnp.percentile(all_log, 50)):>7.2f}  "
          f"{float(jnp.percentile(all_log, 95)):>7.2f}  "
          f"{float(jnp.percentile(all_log, 99)):>7.2f}  "
          f"{float(jnp.percentile(all_log, 99.9)):>7.2f}  "
          f"{float(jnp.max(all_log)):>7.2f}")
    print()

    # Interpretation
    overall_mean = float(jnp.mean(all_log))
    overall_max = float(jnp.max(all_log))
    print(f"  Mean log10 error: {overall_mean:.2f} → {10**overall_mean:.1e} "
          f"({'<0.1% Good' if overall_mean < -3 else '<1% Acceptable' if overall_mean < -2 else 'POOR >1%'})")
    print(f"  Max  log10 error: {overall_max:.2f} → {10**overall_max:.1e} "
          f"({'<1% Good' if overall_max < -2 else '<10% Acceptable' if overall_max < -1 else 'POOR >10%'})")


# ---------------------------------------------------------------------------
# 2. Market Clearing (resource constraint)
# ---------------------------------------------------------------------------

def market_clearing_errors(
    policy_net: eqx.Module,
    model,
    n_periods: int = 10_000,
    seed: int = 123,
    burn_in: int = 500,
) -> Dict[str, float]:
    """Check resource constraint satisfaction along simulated path.

    For the disaster model: Y = C + I + G + monitoring_costs

    Returns dict with mean/max absolute and relative errors.
    """
    # Resource constraint is eq11 in the disaster model
    result = euler_equation_errors(policy_net, model, n_periods, seed, burn_in)
    residuals = result["residuals"]
    eq_names = result["equation_names"]

    # Find resource constraint equation
    rc_idx = None
    for i, name in enumerate(eq_names):
        if "resource" in name.lower():
            rc_idx = i
            break

    if rc_idx is None:
        return {"error": "No resource constraint equation found"}

    rc_residuals = residuals[:, rc_idx]
    return {
        "equation": eq_names[rc_idx],
        "mean_abs": float(jnp.mean(jnp.abs(rc_residuals))),
        "max_abs": float(jnp.max(jnp.abs(rc_residuals))),
        "mean_log10": float(jnp.mean(jnp.log10(jnp.maximum(jnp.abs(rc_residuals), 1e-20)))),
        "max_log10": float(jnp.max(jnp.log10(jnp.maximum(jnp.abs(rc_residuals), 1e-20)))),
    }


# ---------------------------------------------------------------------------
# 3. Simulated Moments
# ---------------------------------------------------------------------------

def simulated_moments(
    policy_net: eqx.Module,
    model,
    n_periods: int = 10_000,
    seed: int = 123,
    burn_in: int = 500,
) -> Dict[str, Dict[str, float]]:
    """Compute ergodic moments from long simulation.

    Returns moments for each state and policy variable:
    mean, std, min, max, and deviation from steady state.
    """
    constants = model.constants
    ss_state, ss_policy = model.steady_state_fn(constants)
    n_shocks = model.n_shocks

    state = ss_state[None, :]
    key = jax.random.PRNGKey(seed)

    state_names = list(model.state_names)
    policy_names = list(model.policy_names)

    @eqx.filter_jit
    def _sim_step(state, shock):
        policy = policy_net(state)
        if policy.ndim == 1:
            policy = policy[None, :]
        next_state = model.step_fn(state, policy, shock, constants)
        return next_state, state[0], policy[0]

    all_states = []
    all_policies = []

    for t in range(n_periods):
        key, shock_key = jax.random.split(key)
        shock = jax.random.normal(shock_key, (1, n_shocks))

        next_state, st, pol = _sim_step(state, shock)

        if t >= burn_in:
            all_states.append(st)
            all_policies.append(pol)

        state = next_state

    states = jnp.stack(all_states)    # [T, n_states]
    policies = jnp.stack(all_policies)  # [T, n_policies]

    moments = {}
    for i, name in enumerate(state_names):
        col = states[:, i]
        ss_val = float(ss_state[i])
        moments[name] = {
            "mean": float(jnp.mean(col)),
            "std": float(jnp.std(col)),
            "min": float(jnp.min(col)),
            "max": float(jnp.max(col)),
            "ss": ss_val,
            "mean_dev_pct": float((jnp.mean(col) - ss_val) / abs(ss_val) * 100) if abs(ss_val) > 0.01 else 0.0,
        }

    for i, name in enumerate(policy_names):
        col = policies[:, i]
        ss_val = float(ss_policy[i])
        moments[name] = {
            "mean": float(jnp.mean(col)),
            "std": float(jnp.std(col)),
            "min": float(jnp.min(col)),
            "max": float(jnp.max(col)),
            "ss": ss_val,
            "mean_dev_pct": float((jnp.mean(col) - ss_val) / abs(ss_val) * 100) if abs(ss_val) > 0.01 else 0.0,
        }

    return moments


def print_moments(moments: Dict[str, Dict[str, float]], label: str = ""):
    """Print simulated moments table."""
    header = "Simulated Moments (10,000 periods)"
    if label:
        header += f" — {label}"
    print(f"\n{header}")
    print("=" * 95)
    print(f"{'Variable':>20s}  {'SS':>8s}  {'Mean':>8s}  {'Std':>8s}  "
          f"{'Min':>8s}  {'Max':>8s}  {'Dev%':>7s}")
    print("-" * 95)

    for name, m in moments.items():
        dev = m["mean_dev_pct"]
        flag = " !" if abs(dev) > 10 else "  " if abs(dev) > 5 else ""
        print(f"{name:>20s}  {m['ss']:>8.4f}  {m['mean']:>8.4f}  {m['std']:>8.4f}  "
              f"{m['min']:>8.4f}  {m['max']:>8.4f}  {dev:>+6.1f}%{flag}")


# ---------------------------------------------------------------------------
# 4. Stability check — does the economy survive?
# ---------------------------------------------------------------------------

def stability_check(
    policy_net: eqx.Module,
    model,
    n_periods: int = 10_000,
    seed: int = 123,
) -> Dict[str, bool]:
    """Check if the simulated economy remains stable.

    Returns flags for common pathologies:
    - bound_hitting: policies hitting bounds frequently
    - divergence: state variables drifting away from SS
    - nan: any NaN in simulation
    """
    constants = model.constants
    ss_state, ss_policy = model.steady_state_fn(constants)
    n_shocks = model.n_shocks

    state = ss_state[None, :]
    key = jax.random.PRNGKey(seed)

    policy_names = list(model.policy_names)
    policy_lower = model.policy_lower
    policy_upper = model.policy_upper

    margin = 0.01 * (policy_upper - policy_lower) if (policy_lower is not None and policy_upper is not None) else None

    @eqx.filter_jit
    def _sim_step(state, shock):
        policy = policy_net(state)
        if policy.ndim == 1:
            policy = policy[None, :]
        next_state = model.step_fn(state, policy, shock, constants)
        return next_state, policy

    bound_hits = 0
    total_outputs = 0
    has_nan = False

    for t in range(n_periods):
        key, shock_key = jax.random.split(key)
        shock = jax.random.normal(shock_key, (1, n_shocks))

        next_state, policy = _sim_step(state, shock)

        # Check NaN
        if jnp.any(jnp.isnan(policy)) or jnp.any(jnp.isnan(state)):
            has_nan = True
            break

        # Check bound hitting (within 1% of bounds)
        if margin is not None:
            p = policy[0]
            near_lower = jnp.sum(p < policy_lower + margin)
            near_upper = jnp.sum(p > policy_upper - margin)
            bound_hits += int(near_lower + near_upper)
            total_outputs += len(policy_names)

        state = next_state

    # Check final state deviation from SS
    final_state = state[0] if state.ndim == 2 else state
    ss_dev = jnp.abs(final_state - ss_state) / jnp.maximum(jnp.abs(ss_state), 1e-8)
    max_dev = float(jnp.max(ss_dev))

    bound_pct = bound_hits / max(total_outputs, 1) * 100

    return {
        "nan_free": not has_nan,
        "bound_hit_pct": bound_pct,
        "max_ss_deviation_pct": max_dev * 100,
        "stable": not has_nan and bound_pct < 20 and max_dev < 5,
    }


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
    mc = market_clearing_errors(policy_net, model, n_periods=args.periods, seed=args.seed)
    if "error" not in mc:
        print(f"  {mc['equation']}:")
        print(f"    Mean |error|:  {mc['mean_abs']:.2e} (log10: {mc['mean_log10']:.2f})")
        print(f"    Max  |error|:  {mc['max_abs']:.2e} (log10: {mc['max_log10']:.2f})")

    # 4. Simulated moments
    print(f"\n[4/4] Simulated moments...")
    moments = simulated_moments(
        policy_net, model, n_periods=args.periods, seed=args.seed
    )
    print_moments(moments, label=label)

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
                writer.writerow([name, m["ss"], m["mean"], m["std"],
                               m["min"], m["max"], m["mean_dev_pct"]])
        print(f"Saved moments to {csv_path}")
