"""Plots for a single training run's `history` dict.

A ``history`` dict is what ``train_from_config`` returns as its second
output, or a per-run block from ``deqn_jax.plots.compare.parse_log``.
Expected keys:
    - ``loss``: list[float] or np.ndarray, one per logged cycle.
    - ``grad_norm``: list[float] (optional).
    - ``episodes``: list[int] for x-axis (optional; falls back to index).

Plot functions are self-contained: they apply DEQN styling internally,
but respect any ``ax`` the caller passes in.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from deqn_jax.plots._style import deqn_style, get_ax

# ---------------------------------------------------------------------------

def _x_axis(history: Mapping, fallback_len: int) -> np.ndarray:
    """Use `episodes` if present, else an index [0, n)."""
    eps = history.get("episodes")
    if eps is not None and len(eps) == fallback_len:
        return np.asarray(eps)
    return np.arange(fallback_len)


def plot_loss_curve(
    history: Mapping,
    *,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color=None,
    log_y: bool = True,
    best_marker: bool = True,
) -> plt.Axes:
    """Training loss vs cycle.

    Args:
        history: dict with ``loss`` (required) and optionally ``episodes``.
        ax: existing Axes to draw into (creates new figure if None).
        label: legend entry; if None, no label.
        color: matplotlib colour spec.
        log_y: log-scale y-axis (default True — losses span orders of magnitude).
        best_marker: mark ``history['best']`` if present as a small circle.

    Returns:
        The matplotlib Axes drawn into.
    """
    losses = np.asarray(history["loss"], dtype=float)
    x = _x_axis(history, len(losses))

    with deqn_style():
        ax = get_ax(ax, figsize=(7, 4))
        ax.plot(x, losses, color=color, label=label)
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Training loss")

        best = history.get("best")
        if best_marker and best is not None:
            loss_b, ep_b = best
            ax.plot([ep_b], [loss_b], marker="o", color=color, markersize=6)

        if label is not None:
            ax.legend(loc="best")
    return ax


def plot_grad_norm(
    history: Mapping,
    *,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color=None,
    log_y: bool = True,
) -> plt.Axes:
    """Gradient-norm trajectory over training.

    Useful for spotting gradient spikes that destabilise training, and
    for comparing grad-clip thresholds against actual gradient behaviour.
    """
    if "grad_norm" not in history:
        raise KeyError(
            "history is missing 'grad_norm'; plot_grad_norm needs it. "
            "Pre-v0.2.0 train logs may not include it."
        )
    grads = np.asarray(history["grad_norm"], dtype=float)
    x = _x_axis(history, len(grads))

    with deqn_style():
        ax = get_ax(ax, figsize=(7, 4))
        ax.plot(x, grads, color=color, label=label)
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Gradient ℓ² norm")
        if label is not None:
            ax.legend(loc="best")
    return ax


def plot_per_equation_residuals(
    residuals_by_eq: Mapping[str, Sequence[float]],
    *,
    episodes: Optional[Sequence[int]] = None,
    ax: Optional[plt.Axes] = None,
    log_y: bool = True,
    max_eqs: Optional[int] = None,
) -> plt.Axes:
    """Overlay per-equation residual trajectories.

    Takes a dict ``{equation_name: [residual per cycle]}``. Produced
    naturally if you accumulate ``metrics.residuals`` per cycle during
    training, or reconstruct from TensorBoard logs.

    Args:
        residuals_by_eq: per-equation time series.
        episodes: optional x-axis values (cycle numbers).
        ax: existing Axes.
        log_y: log-scale y-axis.
        max_eqs: show only the first N equations (by insertion order). Useful
            for the disaster model where 11 overlaid lines get cluttered.

    Returns:
        The Axes drawn into.
    """
    if not residuals_by_eq:
        raise ValueError("residuals_by_eq is empty")

    items = list(residuals_by_eq.items())
    if max_eqs is not None:
        items = items[:max_eqs]

    with deqn_style():
        ax = get_ax(ax, figsize=(8, 5))
        cmap = plt.cm.tab20
        for i, (eq_name, series) in enumerate(items):
            series = np.asarray(series, dtype=float)
            x = np.asarray(episodes) if episodes is not None else np.arange(len(series))
            ax.plot(x, series, color=cmap(i % 20), label=eq_name, lw=1.1)
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Per-equation residual")
        ax.legend(loc="upper right", ncol=2, fontsize=7)
    return ax
