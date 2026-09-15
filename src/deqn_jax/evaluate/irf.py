"""Impulse Response Function (IRF) computation for DEQN-JAX.

Loads a trained checkpoint and simulates the economy's response to shocks.
This is the real quality metric — MSE loss doesn't tell you if the policy
functions produce economically sensible dynamics.

Usage:
    deqn-jax irf path/to/checkpoint.eqx [--shock <name>] [--horizon 40]
    (shock names: ``deqn-jax info <model>``)
"""

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import yaml
from jax import Array

from deqn_jax.training.checkpointing import load_policy_from_checkpoint

# ---------------------------------------------------------------------------
# Core IRF simulation
# ---------------------------------------------------------------------------


def run_irf(
    policy_net: eqx.Module,
    model,
    shock_name: str,
    shock_size: float = 1.0,
    horizon: int = 40,
) -> Dict[str, List[float]]:
    """Run impulse response from steady state.

    Args:
        policy_net: Trained policy network
        model: ModelSpec with dynamics, equations, etc.
        shock_name: Which shock to hit (a name from ``model.shock_names``)
        shock_size: Shock magnitude in std devs (default: 1σ)
        horizon: Number of periods to simulate after shock

    Returns:
        Dict mapping variable names to time series lists.
        Keys: "period", state names, policy names, definition names, equation names.
    """
    constants = model.constants
    ss_state, ss_policy = model.steady_state_fn(constants)

    # Discrete-chain models: a Gaussian "1σ shock" doesn't apply because
    # the shock support is categorical. Refuse cleanly with a pointer to
    # the planned target_z extension; users wanting impulse-response-like
    # diagnostics on a discrete model should hand-roll a forced-z trajectory.
    if (
        getattr(model, "transition_matrix", None) is not None
        and getattr(model, "z_state_idx", None) is not None
    ):
        raise NotImplementedError(
            "run_irf does not support discrete-chain models (model.transition_matrix "
            "is set). Continuous-shock IRFs do not apply when the shock is a "
            "categorical index. To diagnose impulse responses on a discrete model, "
            "hand-roll a trajectory that forces the desired z-state for one period "
            "and then evolves freely under the chain. A target_z= argument may be "
            "added in a future release."
        )

    # State is [1, n_states] for batched dynamics
    state = ss_state[None, :] if ss_state.ndim == 1 else ss_state
    n_shocks = model.n_shocks

    # Shock index mapping
    shock_names = (
        list(model.shock_names)
        if model.shock_names
        else [f"shock_{i}" for i in range(n_shocks)]
    )
    if shock_name not in shock_names:
        raise ValueError(f"Unknown shock '{shock_name}'. Choose from: {shock_names}")
    shock_idx = shock_names.index(shock_name)

    # Collect results
    results: Dict[str, List[float]] = {"period": []}

    # State/policy/equation names
    state_names = list(model.state_names)
    policy_names = list(model.policy_names)
    eq_names = list(model.equation_names) if model.equation_names else []
    for name in state_names + policy_names + eq_names:
        results[name] = []

    # Pre-compute definition names if available
    has_defs = model.definitions_fn is not None
    def_names: List[str] = []
    if has_defs:
        # Probe definitions to get the keys
        _probe_defs = model.definitions_fn(state, policy_net(state), constants)  # pyright: ignore[reportCallIssue]  # ty: ignore[call-non-callable]
        def_names = list(_probe_defs.keys())
        for name in def_names:
            results[name] = []

    def record(
        t: int,
        st: Array,
        pol: Array,
        defs: Optional[Dict] = None,
        residuals: Optional[Dict] = None,
    ):
        results["period"].append(t)
        st_flat = st[0] if st.ndim == 2 else st
        pol_flat = pol[0] if pol.ndim == 2 else pol

        for i, name in enumerate(state_names):
            results[name].append(float(st_flat[i]))
        for i, name in enumerate(policy_names):
            results[name].append(float(pol_flat[i]))

        for name in def_names:
            if defs is not None and name in defs:
                v = defs[name]
                v = v[0] if hasattr(v, "ndim") and v.ndim > 0 else v
                results[name].append(float(v))
            else:
                results[name].append(float("nan"))

        for name in eq_names:
            if residuals is not None and name in residuals:
                v = residuals[name]
                v = v[0] if hasattr(v, "ndim") and v.ndim > 0 else v
                results[name].append(float(v))
            else:
                results[name].append(float("nan"))

    # ---- Period 0: record pre-shock state ----
    policy = policy_net(state)  # pyright: ignore[reportCallIssue]  # ty: ignore[call-non-callable]
    if policy.ndim == 1:
        policy = policy[None, :]

    defs = None
    if has_defs:
        defs = model.definitions_fn(state, policy, constants)

    record(0, state, policy, defs)

    # ---- Period 1: apply shock ----
    shock = jnp.zeros((1, n_shocks))
    shock = shock.at[0, shock_idx].set(shock_size)
    next_state = model.step_fn(state, policy, shock, constants)
    state = (
        model.clip_state_fn(next_state)
        if model.clip_state_fn is not None
        else next_state
    )

    # ---- Periods 2..horizon: deterministic (zero shock every period) ----
    # Shares the rollout loop with the stochastic eval paths; the
    # deterministic variant consumes no PRNG and feeds a zero shock.
    # Imported here, not at module scope: ``evaluate/cli.py`` imports
    # ``load_policy_from_checkpoint`` from this module, so a top-level
    # import of the evaluate package would be circular.
    from deqn_jax.evaluate.simulate import deterministic_rollout

    def step(state, zero_shock, _d):
        policy = policy_net(state)  # pyright: ignore[reportCallIssue]  # ty: ignore[call-non-callable]
        if policy.ndim == 1:
            policy = policy[None, :]

        # Compute definitions and residuals
        defs = None
        residuals = None
        if has_defs:
            defs = model.definitions_fn(state, policy, constants)

        # For residuals, we need next state + next policy
        next_state = model.step_fn(state, policy, zero_shock, constants)
        next_policy = policy_net(next_state)  # pyright: ignore[reportCallIssue]  # ty: ignore[call-non-callable]
        if next_policy.ndim == 1:
            next_policy = next_policy[None, :]
        if model.equations_fn is not None:
            residuals = model.equations_fn(
                state, policy, next_state, next_policy, constants
            )
        return next_state, state, policy, defs, residuals

    def record_period(i, outputs):
        _next_state, st, pol, defs, residuals = outputs
        record(i + 1, st, pol, defs, residuals)

    deterministic_rollout(model, state, horizon, step, record_period)

    return results


def run_girf(
    policy_net,
    model,
    shock_name: str,
    shock_size: float = 1.0,
    horizon: int = 40,
) -> Dict[str, List[float]]:
    """Generalized IRF: response = shocked path − no-shock path, same start state.

    Fixes the bug where ``run_irf`` compared the shocked trajectory against the
    initial SS alone. When the SS in use is the **risky** SS, the no-shock
    trajectory drifts away from SS on its own because risky_SS is defined by
    ``E_d[F] = 0`` under the disaster mixture, not by
    ``step(SS, 0, d=0) = SS``. The plain-IRF output conflates that drift with
    the shock response. GIRF subtracts a matched no-shock counterfactual so
    only the shock response survives.

    Returns a dict with the same schema as ``run_irf`` but the recorded state,
    policy, and definition series are *deviations* ``shocked − baseline``.
    Per-period scalars (``period``) are unchanged; equation residuals are
    recorded from the shocked path (they are exact residuals, no baseline
    concept).

    Args:
        policy_net, model, shock_name, shock_size, horizon: as in ``run_irf``.

    Returns:
        Dict with ``period`` and per-variable deviation series of length
        ``horizon + 1`` (t = 0..horizon), plus the private marker
        ``_mode = "girf"``. Keys starting with ``_`` are metadata, not
        series: ``save_irf_csv`` turns the marker into the CSV's ``mode``
        column and ``print_irf_summary`` skips them, so a GIRF table cannot
        be mislabelled by a caller that forgets to say which mode it ran.
    """
    shocked = run_irf(policy_net, model, shock_name, shock_size, horizon)
    baseline = run_irf(policy_net, model, shock_name, 0.0, horizon)

    out: Dict[str, Any] = {"_mode": "girf"}
    for key, series in shocked.items():
        if key == "period":
            out[key] = list(series)
            continue
        base = baseline.get(key)
        if base is None or len(base) != len(series):
            # Fall back to raw shocked value if baseline is missing.
            out[key] = list(series)
            continue
        out[key] = [s - b for s, b in zip(series, base)]
    return out


# ---------------------------------------------------------------------------
# Loading checkpoint → policy network
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_irf_csv(results: Dict[str, List[float]], path: str):
    """Save IRF results to CSV.

    The mode is read off the results dict itself (``run_girf`` sets
    ``_mode = "girf"``), never supplied by the caller, so the label cannot
    disagree with the numbers. It lands in a trailing ``mode`` column: ``0``
    for a plain IRF, ``1`` for a GIRF (deviations from the matched no-shock
    baseline). The flag is numeric so readers that parse every field as a
    float (``scripts/dev/make_plots.py``) keep working. Other ``_``-prefixed keys
    are metadata too and are not written.
    """
    keys = [k for k in results if not k.startswith("_")]
    n_rows = len(results["period"])
    mode_flag = 1 if results.get("_mode") == "girf" else 0

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([*keys, "mode"])
        for i in range(n_rows):
            writer.writerow([*(results[k][i] for k in keys), mode_flag])


def print_irf_summary(results: Dict[str, List[float]], shock_name: str):
    """Print a concise IRF summary to stdout."""
    periods = results["period"]
    # Find which keys are states, policies, equations
    # (equations have "eq" prefix)
    eq_keys = [k for k in results if k.startswith("eq")]
    var_keys = [
        k
        for k in results
        if k not in ("period",) and k not in eq_keys and not k.startswith("_")
    ]

    print(f"\nIRF: 1σ shock to {shock_name}, {len(periods)} periods")
    print("=" * 70)

    # Show key variables at t=0, t=1 (impact), t=5, t=20, t=last
    show_t = [0, 1, 5, 20, min(40, periods[-1])]
    show_t = [t for t in show_t if t in periods]
    t_indices = [periods.index(t) for t in show_t]

    # Print header
    header = f"{'Variable':>25s}" + "".join(f"  t={t:>3d}" for t in show_t)
    print(header)
    print("-" * len(header))

    for name in var_keys:
        vals = results[name]
        row = f"{name:>25s}"
        for idx in t_indices:
            row += f"  {vals[idx]:>8.4f}"
        print(row)

    # Print max Euler residuals
    if eq_keys:
        print(f"\n{'Euler residuals':>25s}")
        print("-" * len(header))
        for name in eq_keys:
            vals = results[name]
            # Skip t=0 (no residual at pre-shock)
            valid = [abs(v) for i, v in enumerate(vals) if results["period"][i] > 0]
            if valid:
                row = f"{name:>25s}"
                for idx in t_indices:
                    if results["period"][idx] > 0 and idx < len(vals):
                        row += f"  {vals[idx]:>8.1e}"
                    else:
                        row += f"  {'---':>8s}"
                print(row)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_irf_cli(args):
    """CLI handler for 'deqn-jax irf'."""
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
    print(
        f"Model: {model.name}, params: {sum(p.size for p in jax.tree.leaves(policy_net))}"
    )

    # Run IRF for each shock
    if args.shocks:
        shocks = args.shocks
    elif model.shock_names:
        shocks = list(model.shock_names)
    else:
        shocks = [f"shock_{i}" for i in range(model.n_shocks)]
    outdir = args.output or "irf_results"
    os.makedirs(outdir, exist_ok=True)

    use_girf = getattr(args, "girf", False)
    runner = run_girf if use_girf else run_irf
    label = (
        "GIRF (shocked − no-shock baseline)"
        if use_girf
        else "IRF (shocked path, no baseline)"
    )
    print(f"\nMode: {label}")

    for shock_name in shocks:
        print(f"\n{'=' * 70}")
        print(f"Shock: {shock_name} ({args.shock_size}σ)")
        print(f"{'=' * 70}")

        results = runner(
            policy_net,
            model,
            shock_name=shock_name,
            shock_size=args.shock_size,
            horizon=args.horizon,
        )

        # Save CSV. Plain IRF keeps ``irf_<shock>.csv`` (the name
        # scripts/dev/make_plots.py reads); GIRF gets its own basename so a
        # second run in the same outdir can't silently overwrite the first.
        # Both carry the numeric ``mode`` column.
        basename = f"irf_{shock_name}_girf.csv" if use_girf else f"irf_{shock_name}.csv"
        csv_path = os.path.join(outdir, basename)
        save_irf_csv(results, csv_path)
        print(f"Saved: {csv_path}")

        # Print summary
        print_irf_summary(results, shock_name)

    print(f"\nAll {'GIRF' if use_girf else 'IRF'} results saved to {outdir}/")
