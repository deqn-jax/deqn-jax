"""Console reporting helpers for the training entry points.

Header / footer banners, residual tables, per-episode scalar/histogram
logging, the verbose console progress line, and the small formatting
utilities they depend on. Pure presentation / logging orchestration --
no training numerics. Lives separately from ``trainer.py`` so the
orchestrator file is shorter and the print/log formatting can evolve
without churning the train loop.
"""

import re
import time
from typing import Any, Dict, Optional, Tuple

import equinox as eqx
import jax
import optax

from deqn_jax.training.history import make_constant_history
from deqn_jax.types import ModelSpec, TrainState


def strip_eq_prefix(name: str) -> str:
    """Strip 'eq1_', 'eq12_' numeric prefixes from equation names."""
    return re.sub(r"^eq\d+_", "", name)


def print_residual_table(items: list, n_cols: int = 3):
    """Print residuals as an aligned multi-column table."""
    name_width = max(len(n) for n, _ in items)
    rows = (len(items) + n_cols - 1) // n_cols
    for r in range(rows):
        parts = []
        for c in range(n_cols):
            idx = r + c * rows
            if idx < len(items):
                n, v = items[idx]
                parts.append(f"{n:<{name_width}} {v:>9.2e}")
        print("    " + "   ".join(parts))


def count_params(model: eqx.Module) -> int:
    """Count trainable parameters in an Equinox model."""
    params = eqx.filter(model, eqx.is_array)
    leaves = jax.tree_util.tree_leaves(params)
    return sum(x.size for x in leaves)


def _network_shape_str(model_spec: ModelSpec, hidden_sizes) -> str:
    """Format network shape as e.g. '2 -> 64 -> 64 -> 1'."""
    if isinstance(hidden_sizes, int):
        hidden_sizes = (hidden_sizes,)
    sizes = [model_spec.n_states] + list(hidden_sizes) + [model_spec.n_policies]
    return " → ".join(str(s) for s in sizes)


def _format_time(seconds: float) -> str:
    """Format seconds as a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def print_header(
    model_spec: ModelSpec,
    optimizer: str,
    learning_rate: float,
    hidden_sizes: Tuple[int, ...],
    n_params: int,
    batch_size: int,
    mc_samples: int,
    warm_start: bool,
    grad_clip: Optional[float],
    loss_reweight: str,
    fp64: bool,
    lr_schedule: str = "constant",
    lr_warmup: int = 0,
    lr_min_factor: float = 0.0,
    net_type: str = "mlp",
    history_len: int = 1,
):
    """Print rich training header."""
    eq_names = list(model_spec.equation_names or [])
    if not eq_names:
        eq_names = ["(auto-discovered)"]

    precision = "float64" if fp64 else "float32"
    ws_str = "yes" if warm_start else "no"
    clip_str = str(grad_clip) if grad_clip else "none"
    reweight_str = loss_reweight if loss_reweight != "none" else "none"

    w = 60
    print("=" * w)
    print("DEQN-JAX Training")
    print("=" * w)
    print(f"  Model:           {model_spec.name}")
    learning_rate = float(learning_rate)
    lr_str = f"lr={learning_rate:.0e}"
    if lr_schedule != "constant":
        lr_str += f", {lr_schedule}"
        if lr_warmup > 0:
            lr_str += f", warmup={lr_warmup}"
        lr_str += f", min={learning_rate * float(lr_min_factor):.0e}"
    print(f"  Optimizer:       {optimizer} ({lr_str})")
    print(f"  Precision:       {precision}")
    net_str = _network_shape_str(model_spec, hidden_sizes)
    if net_type != "mlp":
        net_str = f"{net_type.upper()} {net_str} (H={history_len})"
    print(f"  Network:         {net_str}")
    print(f"  Parameters:      {n_params:,}")
    print(f"  Batch size:      {batch_size}")
    print(f"  Expectations:    {mc_samples} MC samples")
    print(f"  Warm start:      {ws_str}")
    if grad_clip:
        print(f"  Grad clip:       {clip_str}")
    if loss_reweight != "none":
        print(f"  Reweighting:     {reweight_str}")
    print("=" * w)


def print_final(
    elapsed: float,
    episodes: int,
    final_loss: float,
    final_residuals: Optional[Dict[str, float]],
):
    """Print final training summary."""
    eps_per_sec = episodes / elapsed if elapsed > 0 else 0
    w = 60
    print("=" * w)
    print(f"Training complete in {_format_time(elapsed)} ({eps_per_sec:.0f} ep/s)")
    print(f"Final loss: {final_loss:.2e}")
    if final_residuals:
        eq_items = [
            (strip_eq_prefix(k), float(v))
            for k, v in final_residuals.items()
            if not k.startswith("aux_")
        ]
        aux_items = [
            (k[4:], float(v))
            for k, v in final_residuals.items()
            if k.startswith("aux_")
        ]
        if len(eq_items) <= 3:
            for n, v in eq_items:
                print(f"  {n}: {v:.2e}")
        else:
            print_residual_table(eq_items)
        if aux_items:
            print("  aux: " + "  ".join(f"{n}={v:.2e}" for n, v in aux_items))
    print("=" * w)


def log_episode(
    config,
    model: ModelSpec,
    state: TrainState,
    metrics,
    ep_num: int,
    total_episodes: int,
    start_episode: int,
    t_start: float,
    current_lr: float,
    history_len: int,
    logger,
) -> None:
    """Emit scalar + histogram logs and run cycle/scalar_diagnostics hooks.

    Computes ``policy_out`` once for both policy and derived histograms
    so the JIT-eval cost is paid once per logging episode.
    """
    import numpy as np

    elapsed = time.perf_counter() - t_start
    eps_done = ep_num - start_episode
    ep_per_sec = eps_done / elapsed if elapsed > 0 else 0
    loss_val = float(metrics.loss)
    grad_val = float(metrics.grad_norm)

    param_norm = float(optax.global_norm(eqx.filter(state.params, eqx.is_array)))

    log_dict: Dict[str, Any] = {
        "train/loss": loss_val,
        "train/grad_norm": grad_val,
        "train/param_norm": param_norm,
        "train/ep_per_sec": ep_per_sec,
        "train/lr": current_lr,
    }
    if metrics.residuals:
        for k, v in metrics.residuals.items():
            if k.startswith("aux_"):
                log_dict[f"aux/{k[4:]}"] = float(v)
            else:
                log_dict[f"eq/{k}"] = float(v)
    if config.loss_reweight != "none" and model.equation_names:
        for i, name in enumerate(model.equation_names):
            log_dict[f"weights/{name}"] = float(state.loss_weights[i])

    hist_dict: Dict[str, Any] = {}
    ep_states = state.episode_state  # [batch, n_states]

    if model.state_names:
        for i, name in enumerate(model.state_names):
            hist_dict[f"state/{name}"] = np.asarray(ep_states[:, i])

    # For sequence nets, approximate with constant history at current state.
    if history_len > 1:
        ep_history = make_constant_history(ep_states, history_len)
        policy_out = jax.vmap(state.params)(ep_history)
    else:
        policy_out = jax.vmap(state.params)(ep_states)
    if model.policy_names:
        for i, name in enumerate(model.policy_names):
            hist_dict[f"policy/{name}"] = np.asarray(policy_out[:, i])

    if model.definitions_fn is not None:
        # Bind to local to keep narrowing inside the lambda body.
        defs_fn = model.definitions_fn
        defs = jax.vmap(lambda s, p: defs_fn(s, p, model.constants))(
            ep_states, policy_out
        )
        for name, vals in defs.items():
            hist_dict[f"derived/{name}"] = np.asarray(vals)

        if model.scalar_diagnostics_fn is not None:
            if history_len > 1:
                _diag_policy_fn = lambda s: state.params(
                    make_constant_history(s[None], history_len)[0]
                )
            else:
                _diag_policy_fn = state.params
            try:
                diag = model.scalar_diagnostics_fn(
                    model,
                    _diag_policy_fn,
                    ep_states,
                    policy_out,
                    defs,
                )
                for dk, dv in diag.items():
                    log_dict[dk] = float(dv)
            except Exception as exc:
                import warnings

                warnings.warn(f"scalar_diagnostics_fn raised at ep {ep_num}: {exc}")

    logger.log_scalars(log_dict, step=ep_num)

    # Filter out arrays with NaN/Inf (early training can produce these).
    hist_dict = {
        k: v for k, v in hist_dict.items() if np.isfinite(v).all() and v.size > 0
    }
    if hist_dict:
        logger.log_histograms(hist_dict, step=ep_num)

    # Model-provided cycle hook (plots, custom diagnostics). Errors are
    # swallowed so a bad plot doesn't kill training.
    if model.cycle_hook is not None:
        try:
            model.cycle_hook(state, model, ep_num)
        except Exception as exc:
            import warnings

            warnings.warn(f"cycle_hook raised at ep {ep_num}: {exc}")


def print_episode_progress(
    metrics,
    ep_num: int,
    total_episodes: int,
    ep_width: int,
    start_episode: int,
    t_start: float,
) -> None:
    """Console summary line + residual table (verbose path)."""
    elapsed = time.perf_counter() - t_start
    eps_done = ep_num - start_episode
    ep_per_sec = eps_done / elapsed if elapsed > 0 else 0
    loss_val = float(metrics.loss)
    grad_val = float(metrics.grad_norm)
    residuals = metrics.residuals or {}

    print(
        f"  [{ep_num:>{ep_width}}/{total_episodes}] "
        f"loss={loss_val:.2e} | grad={grad_val:.2e} | {ep_per_sec:.0f} ep/s"
    )

    if residuals:
        eq_items = [
            (strip_eq_prefix(k), float(v))
            for k, v in residuals.items()
            if not k.startswith("aux_")
        ]
        aux_items = [
            (k[4:], float(v)) for k, v in residuals.items() if k.startswith("aux_")
        ]
        if len(eq_items) <= 3:
            print("    " + "  ".join(f"{n}={v:.2e}" for n, v in eq_items))
        else:
            print_residual_table(eq_items)
        if aux_items:
            print("    aux: " + "  ".join(f"{n}={v:.2e}" for n, v in aux_items))
