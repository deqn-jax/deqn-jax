"""Console reporting helpers for the training entry points.

Header / footer banners, residual tables, and the small formatting
utilities they depend on. Pure presentation -- no JAX side effects, no
file I/O. Lives separately from ``trainer.py`` so the orchestrator
file is shorter and the print formatting can evolve without churning
the train loop.
"""

import re
from typing import Dict, Optional, Tuple

import equinox as eqx
import jax

from deqn_jax.types import ModelSpec


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
