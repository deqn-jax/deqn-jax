"""Plotting utilities for DEQN-JAX.

Each ``plot_*`` function is a pure function of its data plus optional
styling arguments. All return a ``matplotlib.axes.Axes`` (or a ``Figure``
for multi-panel plots) so callers can compose them into their own
layouts. ``matplotlib`` is an optional dependency — install via
``uv pip install -e ".[plotting]"``.

Design principles
-----------------
- **Data, not I/O**: functions take pre-parsed history dicts, trained
  policy networks, or simulated trajectories — never file paths.
- **Ax-composable**: pass ``ax=None`` to create a standalone figure,
  or ``ax=your_axes`` to draw into a subplot you already have.
- **Log-parsing lives in** ``deqn_jax.plots.compare`` for multi-run
  use; not baked into individual plot functions.

Public entry points
-------------------
Training diagnostics from a ``history`` dict returned by
``train_from_config``::

    from deqn_jax.plots import plot_loss_curve, plot_grad_norm

IRF / GIRF from an ``irf_results`` dict returned by
``deqn_jax.irf.run_irf`` / ``run_girf``::

    from deqn_jax.plots import plot_irf_grid

Multi-run comparison from parsed log files::

    from deqn_jax.plots.compare import parse_log, plot_multi_run_loss

See individual module docstrings for the exact data contracts.
"""

from deqn_jax.plots.compare import (
    parse_log,
    parse_log_single,
    plot_multi_run_loss,
    plot_schedule_alignment,
)
from deqn_jax.plots.irf import (
    plot_irf_grid,
)
from deqn_jax.plots.training import (
    plot_grad_norm,
    plot_loss_curve,
    plot_per_equation_residuals,
)

__all__ = [
    "plot_loss_curve",
    "plot_per_equation_residuals",
    "plot_grad_norm",
    "plot_irf_grid",
    "parse_log",
    "parse_log_single",
    "plot_multi_run_loss",
    "plot_schedule_alignment",
]
