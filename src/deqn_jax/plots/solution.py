"""Figures for a ``SolutionCheck`` (``deqn_jax.evaluate.solution``).

Each function takes the check and returns a ``Figure``; the panel titles and
suptitles are written for a reader who is an economist, not a programmer.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from deqn_jax.evaluate.solution import MEAN_ERROR_MAX, WORST_ERROR_MAX, SolutionCheck
from deqn_jax.plots._style import deqn_style

NETWORK, EXACT, BAND = "C0", "C1", "0.6"
MAX_PANELS = 8


def _grid(n: int, width: float = 3.2, height: float = 2.5):
    cols = min(max(n, 1), 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(max(width * cols, 8.0), height * rows + 0.7), squeeze=False
    )
    for ax in axes.flat[n:]:
        ax.axis("off")
    return fig, list(axes.flat[:n])


def _finish(fig, title: str):
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_equation_errors(check: SolutionCheck) -> plt.Figure:
    """Histogram of log10 |error| per equation, with the two gate lines."""
    with deqn_style():
        names = check.equation_names[:MAX_PANELS]
        fig, axes = _grid(len(names))
        for i, (ax, name) in enumerate(zip(axes, names)):
            errors = check.errors[:, i]
            errors = errors[np.isfinite(errors)]
            if errors.size:
                ax.hist(np.log10(np.maximum(errors, 1e-16)), bins=60, color=NETWORK)
            else:
                ax.text(0.5, 0.5, "no finite errors", ha="center", va="center")
            ax.axvline(np.log10(MEAN_ERROR_MAX), color="green", ls="--", label="0.1%")
            ax.axvline(np.log10(WORST_ERROR_MAX), color="red", ls="--", label="1%")
            ax.set_title(name)
            ax.set_xlabel("error, powers of ten (-3 = 0.1%)")
            ax.set_yticks([])
            if i == 0:
                ax.legend()
        return _finish(fig, "How far each equilibrium condition is from holding")


def _span(values: np.ndarray, rel: float = 0.05) -> tuple:
    """Axis limits at least ``rel`` of the level wide, so a nearly constant
    policy is not blown up into noise."""
    values = values[np.isfinite(values)]
    if not values.size:
        return 0.0, 1.0
    lo, hi = float(values.min()), float(values.max())
    mid = (lo + hi) / 2
    half = max(hi - lo, rel * abs(mid), 1e-12) / 2 * 1.1
    return mid - half, mid + half


def plot_policies_on_path(
    check: SolutionCheck, x_state: Optional[str] = None
) -> plt.Figure:
    """Each policy at the simulated states, against one state (default: the first)."""
    with deqn_style():
        j = check.state_names.index(x_state) if x_state else 0
        x = check.states[:, j]
        names = check.policy_names[:MAX_PANELS]
        fig, axes = _grid(len(names))
        for i, (ax, name) in enumerate(zip(axes, names)):
            ax.scatter(
                x, check.policies[:, i], s=3, color=NETWORK, alpha=0.35, label="network"
            )
            shown = check.policies[:, i]
            if check.exact is not None:
                ax.scatter(
                    x, check.exact[:, i], s=1, color=EXACT, alpha=0.6, label="exact"
                )
                shown = np.concatenate([shown, check.exact[:, i]])
            ax.set_ylim(_span(shown))
            ax.set_title(name)
            ax.set_xlabel(check.state_names[j])
            if i == 0:
                ax.legend(markerscale=4)
        return _finish(
            fig,
            f"Decisions at the states the economy visits, against "
            f"{check.state_names[j]}",
        )


def plot_against_exact(check: SolutionCheck) -> Optional[plt.Figure]:
    """Network against the known solution; points on the diagonal agree."""
    if check.exact is None:
        return None
    with deqn_style():
        names = check.policy_names[:MAX_PANELS]
        fig, axes = _grid(len(names), width=2.8, height=2.8)
        for ax, i in zip(axes, range(len(names))):
            exact, net = check.exact[:, i], check.policies[:, i]
            lim = _span(np.concatenate([exact, net]))
            ax.plot(lim, lim, color=EXACT, lw=1)
            ax.scatter(exact, net, s=2, color=NETWORK, alpha=0.4)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_title(names[i])
            ax.set_xlabel("exact")
            ax.set_ylabel("network")
        return _finish(fig, "Network against the exact solution (diagonal = identical)")


def plot_simulated_states(check: SolutionCheck, periods: int = 200) -> plt.Figure:
    """A stretch of the simulation, with the training range as a grey band."""
    with deqn_style():
        names = check.state_names[:MAX_PANELS]
        fig, axes = _grid(len(names), height=2.1)
        for i, (ax, name) in enumerate(zip(axes, names)):
            ax.plot(check.states[:periods, i], color=NETWORK)
            if check.region is not None:
                ax.axhspan(
                    check.region[0][i], check.region[1][i], color=BAND, alpha=0.15
                )
            ax.set_title(name)
            ax.set_xlabel("period")
        return _finish(
            fig,
            f"A simulated history ({periods} periods); "
            "grey = range the network was trained on",
        )
