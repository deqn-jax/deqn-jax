"""Internal styling helpers for DEQN-JAX plots.

Centralises rcParams tweaks and colour choices so every plot function
gets the same visual language. Callers can pass their own ``ax`` to
escape the context; otherwise each function applies these defaults.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import matplotlib.pyplot as plt

# Compact sans-serif, thin grid, small marks. Roughly matches matplotlib's
# "seaborn-v0_8-whitegrid" but without importing seaborn.
_RC_DEFAULTS = {
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.4,
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 110,
}


@contextmanager
def deqn_style() -> Iterator[None]:
    """Temporarily apply DEQN plot styling. Restores rcParams on exit."""
    with plt.rc_context(_RC_DEFAULTS):
        yield


def get_ax(ax: Optional["plt.Axes"] = None, figsize=(6, 4)) -> "plt.Axes":
    """Return the given ``ax`` or create a new Figure+Axes at ``figsize``."""
    if ax is not None:
        return ax
    _, ax_new = plt.subplots(figsize=figsize)
    return ax_new


# Colour palettes -----------------------------------------------------------

def p_disaster_palette(p_values):
    """Ordered colours for a p_disaster sweep (increasing p → darker)."""
    import numpy as np
    n = len(p_values)
    # Avoid the very light end of viridis so the lowest p is still visible.
    return plt.cm.viridis(np.linspace(0.15, 0.9, max(n, 2)))[:n]
