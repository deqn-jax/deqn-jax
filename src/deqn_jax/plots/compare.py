"""Multi-run comparison plots + training-log parsing.

``parse_log`` turns a deqn-jax training log file into one or more
``history`` dicts keyed by run name. Supports the standard marker
convention ("==== name starting ...") that our sweep scripts use, and a
simpler single-run helper.

The plot functions here take dicts of histories and overlay them — the
high-level view for ablation studies.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np

from deqn_jax.plots._style import deqn_style, get_ax

# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

_RUN_RE = re.compile(r"==== (\S+) (starting|finished).*$", re.M)
_STEP_RE = re.compile(r"\[(\d+)/(\d+)\] loss=([\d.eE+-]+) \| grad=([\d.eE+-]+)")
_BEST_RE = re.compile(r"Best checkpoint: ([\d.eE+-]+) at episode (\d+)")


def _extract_history_from_block(block: str) -> Dict:
    eps, losses, grads = [], [], []
    for ep_s, _, loss_s, grad_s in _STEP_RE.findall(block):
        eps.append(int(ep_s))
        losses.append(float(loss_s))
        grads.append(float(grad_s))
    best_match = _BEST_RE.search(block)
    best = (
        (float(best_match.group(1)), int(best_match.group(2))) if best_match else None
    )
    return {
        "episodes": np.asarray(eps),
        "loss": np.asarray(losses),
        "grad_norm": np.asarray(grads),
        "best": best,
    }


def parse_log(path) -> Dict[str, Dict]:
    """Parse a training log that may contain multiple runs.

    Run boundaries are detected by lines matching:

        ==== <run_name> starting ... ====
        ...
        ==== <run_name> finished ... ====

    Returns ``{run_name: history_dict}``. History dicts have keys
    ``episodes``, ``loss``, ``grad_norm``, ``best`` (None if absent).
    """
    text = Path(path).read_text()
    runs: Dict[str, Dict] = {}
    markers = [(m.group(1), m.group(2), m.start()) for m in _RUN_RE.finditer(text)]
    starts = [(n, p) for (n, ev, p) in markers if ev == "starting"]
    ends = {n: p for (n, ev, p) in markers if ev == "finished"}
    for name, start in starts:
        end = ends.get(name, len(text))
        runs[name] = _extract_history_from_block(text[start:end])
    return runs


def parse_log_single(path, run_name: str) -> Dict[str, Dict]:
    """Parse a log file containing a single run; assign it ``run_name``.

    Convenience for logs that don't use the ``==== starting ====`` markers
    or whose marker wording the outer runner mangled.
    """
    text = Path(path).read_text()
    return {run_name: _extract_history_from_block(text)}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_multi_run_loss(
    runs: Mapping[str, Mapping],
    *,
    ax: Optional[plt.Axes] = None,
    log_y: bool = True,
    best_marker: bool = True,
    palette=None,
) -> plt.Axes:
    """Overlay loss curves for multiple runs, one line per run.

    Args:
        runs: ``{run_label: history_dict}``. Key is used as the legend label.
        ax: existing Axes.
        log_y: log-scale y-axis.
        best_marker: show ``history['best']`` as a point per run.
        palette: iterable of colours. If None, uses matplotlib defaults.
    """
    with deqn_style():
        ax = get_ax(ax, figsize=(8, 5))
        items = list(runs.items())
        colors = palette if palette is not None else [None] * len(items)
        for (label, history), color in zip(items, colors):
            losses = np.asarray(history.get("loss", []), dtype=float)
            if len(losses) == 0:
                continue
            x = np.asarray(history.get("episodes", np.arange(len(losses))))
            ax.plot(x, losses, color=color, label=label)
            if best_marker and history.get("best") is not None:
                loss_b, ep_b = history["best"]
                ax.plot([ep_b], [loss_b], marker="o", color=color, markersize=6)
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Training loss")
        ax.legend(loc="upper right", fontsize=9)
    return ax


def plot_schedule_alignment(
    runs_with_updates: Mapping[str, tuple],
    *,
    ax: Optional[plt.Axes] = None,
    log_x: bool = True,
    log_y: bool = True,
) -> plt.Axes:
    """Loss vs **total gradient updates** (not cycles).

    Exposes same-compute comparisons across runs that use different
    schedules (single-update vs sweep-based, etc.). X-axis is
    ``cycle × updates_per_cycle``.

    Args:
        runs_with_updates: ``{label: (history_dict, updates_per_cycle)}``.
    """
    with deqn_style():
        ax = get_ax(ax, figsize=(8, 5))
        for label, (history, updates_per_cycle) in runs_with_updates.items():
            losses = np.asarray(history.get("loss", []), dtype=float)
            if len(losses) == 0:
                continue
            cycles = np.asarray(history.get("episodes", np.arange(len(losses))))
            updates = cycles * updates_per_cycle
            ax.plot(updates, losses, label=label)
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Total gradient updates")
        ax.set_ylabel("Training loss")
        ax.legend(loc="upper right", fontsize=9)
    return ax
