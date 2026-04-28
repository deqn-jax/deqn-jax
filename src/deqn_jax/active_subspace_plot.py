"""Visualization of trained policies on their active subspace.

Companion to ``active_subspace.py``: takes the eigendecomposition of
the gradient covariance and produces matplotlib figures that recover
the "look at the policy surface" diagnostic that low-d HJB-PINN
methods take for granted.

Three figure types:

  1. ``plot_policy_subspace_2d`` — single output's heatmap on the
     top-2 active directions, with visited-state scatter and
     (optionally) Dynare's linear policy contour overlaid on the
     same projection. This is the "see the shape" plot.

  2. ``plot_subspace_spectrum`` — per-policy eigenvalue bar chart in
     a multi-panel grid. The "how many directions does each output
     actually use" diagnostic.

  3. ``save_active_subspace_report`` — driver that runs both for an
     entire trained policy and writes the figures to a directory.

matplotlib is lazy-imported so importing this module on a host
without a display works (the import only runs when you actually call
a plot function).
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.active_subspace import (
    estimate_active_subspace,
    policy_grid_on_subspace,
    project_states,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_direction(
    eigenvector: Array,
    state_names: Sequence[str],
    n_top: int = 3,
) -> str:
    """Render an eigenvector as ``±0.71 k_lag ±0.42 z + …`` for axis labels.

    Reports the top-``n_top`` state contributions by absolute weight.
    Compact enough for an axis label, informative enough to read.
    """
    weights = jnp.asarray(eigenvector)
    order = jnp.argsort(jnp.abs(weights))[::-1]
    parts: List[str] = []
    for k in range(min(n_top, len(state_names))):
        idx = int(order[k])
        w = float(weights[idx])
        sign = "+" if w >= 0 else "−"
        parts.append(f"{sign}{abs(w):.2f}·{state_names[idx]}")
    return " ".join(parts)


def _resolve_grid_range(
    proj_visited: Array,
    z_index: int,
    sigma_mult: float = 3.0,
) -> Tuple[float, float]:
    """Pick a grid range covering the visited-state cloud at ±sigma_mult σ."""
    col = proj_visited[:, z_index]
    mu = float(jnp.mean(col))
    sd = float(jnp.std(col))
    return (mu - sigma_mult * sd, mu + sigma_mult * sd)


# ---------------------------------------------------------------------------
# Single-output 2-d figure
# ---------------------------------------------------------------------------


def plot_policy_subspace_2d(
    policy_fn: Callable[[Array], Array],
    states: Array,
    output_idx: int,
    state_names: Sequence[str],
    policy_names: Sequence[str],
    ss_state: Array,
    *,
    n_pts: int = 60,
    sigma_mult: float = 3.0,
    overlay_visited: bool = True,
    overlay_linear: Optional[Array] = None,
    figsize: Tuple[float, float] = (8.0, 6.0),
):
    """2-d heatmap of ``policy_fn(s)[output_idx]`` on its active subspace.

    Args:
        policy_fn: trained policy, ``[n_states] → [n_policies]``.
        states: ``[N, n_states]`` ergodic-trajectory sample. The active
            subspace is estimated on these; the same states are scattered
            on the figure so the visited region is visible.
        output_idx: which policy output to plot.
        state_names, policy_names: name tuples from the model spec.
        ss_state: steady-state center of the projection.
        n_pts: grid resolution.
        sigma_mult: grid extends ±sigma_mult × std of projected visited
            states in each direction. Captures the visited cloud comfortably.
        overlay_visited: scatter the projected visited states.
        overlay_linear: optional ``[n_policies, n_states]`` Dynare-style
            linearization. When provided, contours of the linear approx
            for the same output are drawn on top of the network's
            heatmap so any deviation is visible.
        figsize: matplotlib figure size.

    Returns:
        ``(fig, ax, info_dict)`` where ``info_dict`` includes the
        eigenvalues + axis labels — useful for downstream callers that
        want to drop the figure into a custom layout.
    """
    import matplotlib.pyplot as plt

    from deqn_jax.active_subspace import effective_dimensionality

    sub = estimate_active_subspace(policy_fn, states, output_idx)
    eigenvalues = sub["eigenvalues"]
    eigenvectors = sub["eigenvectors"]
    # Annotate eff_dim here so the figure title is informative even when
    # the caller hasn't gone through summarize_subspace_per_policy.
    sub["effective_dim"] = effective_dimensionality(eigenvalues, threshold=0.95)
    v1 = eigenvectors[:, 0]
    v2 = eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else None

    proj_visited = project_states(
        states, jnp.stack([v1, v2 if v2 is not None else v1], axis=1), center=ss_state
    )

    if v2 is None:
        # Degenerate case: 1-d. Plot a 1-d sweep.
        z1_range = _resolve_grid_range(proj_visited, 0, sigma_mult)
        sweep = policy_grid_on_subspace(
            policy_fn,
            ss_state,
            v1,
            None,
            output_idx,
            grid_range=z1_range,
            n_pts=n_pts,
        )
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(sweep["z1"], sweep["values"], lw=2)
        if overlay_visited:
            visited_vals = jax.vmap(lambda s: policy_fn(s)[output_idx])(states)
            ax.scatter(proj_visited[:, 0], visited_vals, s=4, alpha=0.3, color="black")
        ax.set_xlabel(f"z₁  ({_format_direction(v1, state_names)})")
        ax.set_ylabel(policy_names[output_idx])
        ax.set_title(f"{policy_names[output_idx]} (1-d effective subspace)")
        info = {"eigenvalues": eigenvalues, "shape": "1d"}
        return fig, ax, info

    # 2-d heatmap.
    z1_range = _resolve_grid_range(proj_visited, 0, sigma_mult)
    z2_range = _resolve_grid_range(proj_visited, 1, sigma_mult)
    common_range = (
        min(z1_range[0], z2_range[0]),
        max(z1_range[1], z2_range[1]),
    )
    grid = policy_grid_on_subspace(
        policy_fn,
        ss_state,
        v1,
        v2,
        output_idx,
        grid_range=common_range,
        n_pts=n_pts,
    )
    fig, ax = plt.subplots(figsize=figsize)
    pcm = ax.pcolormesh(
        grid["z1"],
        grid["z2"],
        grid["values"],
        cmap="viridis",
        shading="auto",
    )
    fig.colorbar(pcm, ax=ax, label=policy_names[output_idx])

    if overlay_visited:
        ax.scatter(
            proj_visited[:, 0],
            proj_visited[:, 1],
            s=2,
            alpha=0.25,
            color="white",
            edgecolors="none",
        )

    if overlay_linear is not None:
        # Linear approximation along the same z1, z2 directions: at state
        # s = ss + z1·v1 + z2·v2, the linear policy_i is
        # ss_policy_i + (P_i · v1) z1 + (P_i · v2) z2 — a plane.
        # Draw contours of that plane to compare with the heatmap.
        P = jnp.asarray(overlay_linear)
        # We need ss_policy[output_idx]; the caller provides P only, so
        # compute the linear value at the four corners by direct
        # evaluation: linear(s) = ss_policy + P @ (s - ss_state). Without
        # ss_policy we can only show the *deviations* from the network at
        # the SS column; that's enough to see if the linear gradient
        # alignment matches.
        z1_grid = grid["z1"]
        z2_grid = grid["z2"]
        a1 = float(jnp.dot(P[output_idx], v1))
        a2 = float(jnp.dot(P[output_idx], v2))
        Z1, Z2 = jnp.meshgrid(z1_grid, z2_grid, indexing="xy")
        linear_dev = a1 * Z1 + a2 * Z2  # deviation from SS under linearization
        # Overlay linear-contours of the deviation. The network's
        # heatmap is in absolute units (centered at policy(SS)); the
        # contour spacing is what matters for "is the local slope right",
        # not the absolute level, so we plot ``linear_dev`` directly.
        cs = ax.contour(
            z1_grid,
            z2_grid,
            linear_dev,
            colors="white",
            linewidths=0.7,
            linestyles="dashed",
            alpha=0.7,
        )
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.2g")
        ax.text(
            0.02,
            0.98,
            "white dashes: linear (Dynare ghx)",
            transform=ax.transAxes,
            color="white",
            fontsize=8,
            verticalalignment="top",
            alpha=0.85,
        )

    ax.set_xlabel(f"z₁  ({_format_direction(v1, state_names)})")
    ax.set_ylabel(f"z₂  ({_format_direction(v2, state_names)})")
    eff_dim = sub.get("effective_dim", "?")
    pr = sub.get("participation_ratio", float("nan"))
    ax.set_title(f"{policy_names[output_idx]} (eff_dim={eff_dim}, part_ratio={pr:.2f})")
    info = {
        "eigenvalues": eigenvalues,
        "shape": "2d",
        "z1_label": _format_direction(v1, state_names),
        "z2_label": _format_direction(v2, state_names),
    }
    return fig, ax, info


# ---------------------------------------------------------------------------
# Per-policy eigenvalue spectrum (multi-panel)
# ---------------------------------------------------------------------------


def plot_subspace_spectrum(
    summary: Dict[str, Dict[str, Any]],
    *,
    cols: int = 4,
    figsize_per_panel: Tuple[float, float] = (3.0, 2.0),
    log_scale: bool = True,
):
    """Per-policy eigenvalue bar chart in a grid.

    Reads the output of ``summarize_subspace_per_policy`` and draws one
    subplot per output, eigenvalues sorted descending. Log-scale by
    default since the spectrum often spans many orders of magnitude.
    """
    import matplotlib.pyplot as plt

    n = len(summary)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * figsize_per_panel[0], rows * figsize_per_panel[1]),
        squeeze=False,
    )
    items = list(summary.items())
    for i, (name, sub) in enumerate(items):
        ax = axes[i // cols][i % cols]
        evals = jnp.asarray(sub["eigenvalues"])
        ax.bar(range(len(evals)), evals, width=0.8, color="steelblue")
        if log_scale:
            ax.set_yscale("symlog", linthresh=max(float(jnp.max(evals)) * 1e-8, 1e-12))
        eff_dim = sub.get("effective_dim", "?")
        pr = sub.get("participation_ratio", float("nan"))
        ax.set_title(f"{name}\n(eff_dim={eff_dim}, pr={pr:.1f})", fontsize=9)
        ax.tick_params(axis="both", labelsize=7)
    # Hide unused panels.
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Driver: dump a full report for a checkpoint
# ---------------------------------------------------------------------------


def save_active_subspace_report(
    policy_fn: Callable[[Array], Array],
    states: Array,
    state_names: Sequence[str],
    policy_names: Sequence[str],
    ss_state: Array,
    out_dir: str,
    *,
    overlay_linear: Optional[Array] = None,
    n_pts: int = 60,
    threshold: float = 0.95,
    label: str = "",
) -> Dict[str, str]:
    """Produce one PNG per output + a summary spectrum panel.

    Layout under ``out_dir``:
        spectrum.png                  # multi-panel bar chart, all outputs
        policy_<name>.png             # 2-d heatmap per output
        summary.json                  # eigenvalue + eff_dim per output

    Returns a dict mapping artifact name → path.
    """
    import json

    import matplotlib.pyplot as plt

    from deqn_jax.active_subspace import (
        summarize_subspace_per_policy,
    )

    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    summary = summarize_subspace_per_policy(
        policy_fn, states, list(policy_names), threshold=threshold
    )

    spec_fig, _ = plot_subspace_spectrum(summary)
    if label:
        spec_fig.suptitle(f"Eigenvalue spectra — {label}", y=1.02, fontsize=11)
    spec_path = os.path.join(out_dir, "spectrum.png")
    spec_fig.savefig(spec_path, dpi=150, bbox_inches="tight")
    plt.close(spec_fig)
    paths["spectrum"] = spec_path

    for i, name in enumerate(policy_names):
        try:
            fig, _, _ = plot_policy_subspace_2d(
                policy_fn,
                states,
                i,
                state_names,
                policy_names,
                ss_state,
                n_pts=n_pts,
                overlay_linear=overlay_linear,
            )
            if label:
                fig.suptitle(label, y=1.02, fontsize=10)
            pol_path = os.path.join(out_dir, f"policy_{name}.png")
            fig.savefig(pol_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths[f"policy_{name}"] = pol_path
        except Exception as e:
            paths[f"policy_{name}_error"] = f"{type(e).__name__}: {e}"

    serializable_summary: Dict[str, Dict[str, Any]] = {}
    for name, sub in summary.items():
        serializable_summary[name] = {
            "eigenvalues": [float(v) for v in sub["eigenvalues"]],
            "cumulative_variance_ratio": [
                float(v) for v in sub["cumulative_variance_ratio"]
            ],
            "participation_ratio": sub["participation_ratio"],
            "trace": sub["trace"],
            "n_finite_samples": sub["n_finite_samples"],
            "effective_dim": sub["effective_dim"],
        }
    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(serializable_summary, f, indent=2)
    paths["summary_json"] = json_path

    return paths


__all__ = [
    "plot_policy_subspace_2d",
    "plot_subspace_spectrum",
    "save_active_subspace_report",
]
