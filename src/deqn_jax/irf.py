"""Impulse Response Function (IRF) computation for DEQN-JAX.

Loads a trained checkpoint and simulates the economy's response to shocks.
This is the real quality metric — MSE loss doesn't tell you if the policy
functions produce economically sensible dynamics.

Usage:
    deqn-jax irf --checkpoint path/to/checkpoint.eqx [--shock eps] [--horizon 40]
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

from deqn_jax.models import load_model
from deqn_jax.types import TrainState


# ---------------------------------------------------------------------------
# Core IRF simulation
# ---------------------------------------------------------------------------

def run_irf(
    policy_net: eqx.Module,
    model,
    shock_name: str,
    shock_size: float = 1.0,
    horizon: int = 40,
    warmup: int = 0,
) -> Dict[str, List[float]]:
    """Run impulse response from steady state.

    Args:
        policy_net: Trained policy network
        model: ModelSpec with dynamics, equations, etc.
        shock_name: Which shock to hit ("eps", "mu_ups", "mu_z", "g", "m_p")
        shock_size: Shock magnitude in std devs (default: 1σ)
        horizon: Number of periods to simulate after shock
        warmup: Deterministic warmup periods before shock (0 = start from SS)

    Returns:
        Dict mapping variable names to time series lists.
        Keys: "period", state names, policy names, definition names, equation names.
    """
    constants = model.constants
    ss_state, ss_policy = model.steady_state_fn(constants)

    # State is [1, n_states] for batched dynamics
    state = ss_state[None, :] if ss_state.ndim == 1 else ss_state
    n_shocks = model.n_shocks

    # Shock index mapping
    shock_names = list(model.shock_names) if model.shock_names else [f"shock_{i}" for i in range(n_shocks)]
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
        _probe_defs = model.definitions_fn(state, policy_net(state), constants)
        def_names = list(_probe_defs.keys())
        for name in def_names:
            results[name] = []

    def record(t: int, st: Array, pol: Array, defs: Optional[Dict] = None,
               residuals: Optional[Dict] = None):
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
                v = v[0] if hasattr(v, 'ndim') and v.ndim > 0 else v
                results[name].append(float(v))
            else:
                results[name].append(float('nan'))

        for name in eq_names:
            if residuals is not None and name in residuals:
                v = residuals[name]
                v = v[0] if hasattr(v, 'ndim') and v.ndim > 0 else v
                results[name].append(float(v))
            else:
                results[name].append(float('nan'))

    # ---- Warmup: deterministic steps from SS ----
    zero_shock = jnp.zeros((1, n_shocks))
    for t in range(-warmup, 0):
        policy = policy_net(state)
        if policy.ndim == 1:
            policy = policy[None, :]
        record(t, state, policy)
        next_state = model.step_fn(state, policy, zero_shock, constants)
        state = model.clip_state_fn(next_state) if model.clip_state_fn is not None else next_state

    # ---- Period 0: record pre-shock state ----
    policy = policy_net(state)
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
    state = model.clip_state_fn(next_state) if model.clip_state_fn is not None else next_state

    # ---- Periods 2..horizon: deterministic ----
    for t in range(1, horizon + 1):
        policy = policy_net(state)
        if policy.ndim == 1:
            policy = policy[None, :]

        # Compute definitions and residuals
        defs = None
        residuals = None
        if has_defs:
            defs = model.definitions_fn(state, policy, constants)

        # For residuals, we need next state + next policy
        next_state = model.step_fn(state, policy, zero_shock, constants)
        next_policy = policy_net(next_state)
        if next_policy.ndim == 1:
            next_policy = next_policy[None, :]
        if model.equations_fn is not None:
            residuals = model.equations_fn(state, policy, next_state, next_policy, constants)

        record(t, state, policy, defs, residuals)
        state = model.clip_state_fn(next_state) if model.clip_state_fn is not None else next_state

    return results


def run_girf(
    policy_net,
    model,
    shock_name: str,
    shock_size: float = 1.0,
    horizon: int = 40,
    warmup: int = 0,
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
        policy_net, model, shock_name, shock_size, horizon, warmup: as in ``run_irf``.

    Returns:
        Dict with ``period`` and per-variable deviation series of length
        ``horizon + 1`` (t = 0..horizon).
    """
    shocked = run_irf(policy_net, model, shock_name, shock_size, horizon, warmup)
    baseline = run_irf(policy_net, model, shock_name, 0.0, horizon, warmup)

    out: Dict[str, List[float]] = {}
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

def load_policy_from_checkpoint(
    checkpoint_path: str,
    config_path: Optional[str] = None,
) -> Tuple[eqx.Module, object]:
    """Load trained policy network from checkpoint.

    Args:
        checkpoint_path: Path to .eqx checkpoint file
        config_path: Path to config.yaml (auto-detected from checkpoint dir if None)

    Returns:
        (policy_net, model) tuple
    """
    # Auto-detect config
    if config_path is None:
        ckpt_dir = Path(checkpoint_path).parent
        config_path = str(ckpt_dir / "config.yaml")
        if not Path(config_path).exists():
            raise FileNotFoundError(
                f"No config.yaml found in {ckpt_dir}. "
                f"Pass --config explicitly."
            )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Enable fp64 if checkpoint was trained with it
    if cfg.get("fp64", False):
        jax.config.update("jax_enable_x64", True)

    model = load_model(cfg["model"])

    # Extract network config
    net_cfg = cfg.get("network", {})
    hidden_sizes = tuple(net_cfg.get("hidden_sizes", [64, 64]))
    activation = net_cfg.get("activation", "tanh")
    activations = net_cfg.get("activations", None)
    init = net_cfg.get("init", "default")
    multi_head = net_cfg.get("multi_head", False)
    skip_connections = net_cfg.get("skip_connections", False)
    net_type = net_cfg.get("type", "mlp")
    history_len = net_cfg.get("history_len", 1)
    num_heads = net_cfg.get("num_heads", 4)
    n_layers = net_cfg.get("n_layers", 2)

    key = jax.random.PRNGKey(0)  # doesn't matter, will be overwritten

    # Deserialize — the checkpoint is a full TrainState, we need just params
    # Build a template TrainState to match the checkpoint structure
    from deqn_jax.config import load_config, OptimizerConfig
    from deqn_jax.training.trainer import create_train_state
    from deqn_jax.optimizers import create_optimizer

    n_equations = len(model.equation_names) if model.equation_names else model.n_policies

    # Parse loss_weights from config
    loss_weights = cfg.get("loss_weights", None)

    opt_cfg_dict = cfg.get("optimizer", {"name": "adam"})
    # If checkpoint was saved after optimizer switch, use the switched optimizer
    switch_opt = cfg.get("switch_optimizer", None)
    switch_ep = cfg.get("switch_episode", 0)
    # Extract episode number from checkpoint filename (e.g. checkpoint_010000.eqx)
    ckpt_ep = 0
    try:
        ckpt_ep = int(Path(checkpoint_path).stem.split("_")[-1])
    except (ValueError, IndexError):
        pass
    if switch_opt and ckpt_ep >= switch_ep:
        opt_cfg_dict = dict(opt_cfg_dict)
        opt_cfg_dict["name"] = switch_opt
        if cfg.get("switch_lr") is not None:
            opt_cfg_dict["learning_rate"] = cfg["switch_lr"]
    opt_cfg = OptimizerConfig(**{
        k: v for k, v in opt_cfg_dict.items()
        if k in OptimizerConfig.model_fields
    })

    from deqn_jax.config import NetworkConfig
    net_config = NetworkConfig(
        type=net_type,
        hidden_sizes=hidden_sizes,
        activation=activation,
        activations=activations,
        init=init,
        multi_head=multi_head,
        skip_connections=skip_connections,
        history_len=history_len,
        num_heads=num_heads,
        n_layers=n_layers,
    )

    template_state, _, _ = create_train_state(
        model, key,
        hidden_sizes=hidden_sizes,
        batch_size=cfg.get("batch_size", 64),
        loss_weights=loss_weights,
        n_equations=n_equations,
        optimizer_config=opt_cfg,
        network_config=net_config,
    )

    state = eqx.tree_deserialise_leaves(checkpoint_path, template_state)
    policy_net = state.params

    # Restore correct bounds (old checkpoints may have drifted bounds
    # due to a bug where output_lower/output_upper were trainable)
    if model.policy_lower is not None and hasattr(policy_net, 'output_lower'):
        if model.policy_upper is not None:
            policy_net = eqx.tree_at(
                lambda net: (net.output_lower, net.output_upper),
                policy_net,
                (model.policy_lower, model.policy_upper),
            )
        else:
            policy_net = eqx.tree_at(
                lambda net: net.output_lower,
                policy_net,
                model.policy_lower,
            )

    return policy_net, model


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_irf_csv(results: Dict[str, List[float]], path: str):
    """Save IRF results to CSV."""
    keys = list(results.keys())
    n_rows = len(results["period"])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(n_rows):
            writer.writerow([results[k][i] for k in keys])


def print_irf_summary(results: Dict[str, List[float]], shock_name: str):
    """Print a concise IRF summary to stdout."""
    periods = results["period"]
    # Find which keys are states, policies, equations
    # (equations have "eq" prefix)
    eq_keys = [k for k in results if k.startswith("eq")]
    var_keys = [k for k in results if k not in ("period",) and k not in eq_keys]

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
    print(f"Model: {model.name}, params: {sum(p.size for p in jax.tree.leaves(policy_net))}")

    # Run IRF for each shock
    shocks = args.shocks if args.shocks else ["eps", "mu_ups", "mu_z", "g", "m_p"]
    outdir = args.output or "irf_results"
    os.makedirs(outdir, exist_ok=True)

    use_girf = getattr(args, "girf", False)
    runner = run_girf if use_girf else run_irf
    label = "GIRF (shocked − no-shock baseline)" if use_girf else "IRF (shocked path, no baseline)"
    print(f"\nMode: {label}")

    for shock_name in shocks:
        print(f"\n{'=' * 70}")
        print(f"Shock: {shock_name} ({args.shock_size}σ)")
        print(f"{'=' * 70}")

        results = runner(
            policy_net, model,
            shock_name=shock_name,
            shock_size=args.shock_size,
            horizon=args.horizon,
        )

        # Save CSV
        suffix = "girf" if use_girf else "irf"
        csv_path = os.path.join(outdir, f"{suffix}_{shock_name}.csv")
        save_irf_csv(results, csv_path)
        print(f"Saved: {csv_path}")

        # Print summary
        print_irf_summary(results, shock_name)

    print(f"\nAll {'GIRF' if use_girf else 'IRF'} results saved to {outdir}/")
