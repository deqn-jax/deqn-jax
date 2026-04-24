"""IRF / GIRF grid plots.

Takes the ``Dict[str, List[float]]`` output of ``deqn_jax.irf.run_irf``
or ``run_girf`` (or a parsed CSV with the same schema) and produces a
shock × variable grid of trajectories.

Grid semantics: each row is one shock's IRF, each column is one model
variable. All panels show percent deviation from ``t=0`` (the pre-shock
baseline). For GIRF, the reported series is already ``shocked −
no-shock``, so deviations are small by construction at long horizons
when the shock dies out.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from deqn_jax.plots._style import deqn_style

# ---------------------------------------------------------------------------

def _pct_deviation(series: np.ndarray) -> np.ndarray:
    """Percent deviation of each point from series[0]."""
    ss = series[0]
    return (series - ss) / (abs(ss) + 1e-12) * 100


def plot_irf_grid(
    irf_by_shock: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    variables: Sequence[str] = ("y_gdp", "pi", "R", "i", "c", "K_p"),
    shock_labels: Optional[Mapping[str, str]] = None,
    title: str = "Impulse responses (% dev from t=0)",
    mark_t1: bool = True,
    row_height: float = 2.0,
    col_width: float = 3.0,
) -> plt.Figure:
    """Render a shock × variable grid of IRFs.

    Args:
        irf_by_shock: ``{shock_name: {var_name: series, ...}}``. ``series``
            must include ``period``-indexed values and one entry per target
            variable in ``variables``. In practice this is what you'd get
            by packing the CSV output of ``run_irf``.
        variables: which columns to plot. Default picks a compact macro
            summary (y_gdp, pi, R, i, c, K_p).
        shock_labels: pretty-print for row labels. Defaults to shock name.
        title: figure suptitle.
        mark_t1: annotate the t=1 value on each panel. Useful because the
            "initial response" is where IRF shape is most interpretable;
            later periods can be drift-dominated in non-linear solutions.
        row_height / col_width: per-cell figure size in inches.

    Returns:
        The Figure object.
    """
    shocks = list(irf_by_shock.keys())
    nrows = len(shocks)
    ncols = len(variables)
    if nrows == 0:
        raise ValueError("irf_by_shock is empty")

    default_labels = {
        "eps": r"Transitory TFP ($\varepsilon$)",
        "mu_ups": r"Investment-specific ($\mu_\Upsilon$)",
        "mu_z": r"Permanent growth ($\mu_z$)",
        "g": r"Government spending ($g$)",
        "m_p": r"Monetary policy ($m^p$)",
    }
    labels = {**default_labels, **(shock_labels or {})}

    with deqn_style():
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(col_width * ncols, row_height * nrows),
            sharex=True, squeeze=False,
        )

        for i, shock in enumerate(shocks):
            shock_data = irf_by_shock[shock]
            # Figure out the x axis from whichever variable has `period`.
            period_series = shock_data.get("period")
            if period_series is None:
                # Default: use index for the first variable's length.
                first_var = variables[0]
                period_series = list(range(len(shock_data[first_var])))
            periods = np.asarray(period_series, dtype=int)

            for j, var in enumerate(variables):
                ax = axes[i, j]
                if var not in shock_data:
                    ax.axis("off")
                    continue
                series = np.asarray(shock_data[var], dtype=float)
                dev = _pct_deviation(series)
                ax.plot(periods, dev, lw=1.3)
                ax.axhline(0, color="gray", lw=0.5, ls="--")
                if mark_t1 and len(dev) > 1:
                    ax.annotate(
                        f"t=1: {dev[1]:+.2f}%",
                        xy=(periods[1], dev[1]),
                        xytext=(6, dev[1]),
                        fontsize=7,
                        arrowprops=dict(arrowstyle="->", lw=0.4),
                    )
                if i == 0:
                    ax.set_title(var, fontsize=9)
                if j == 0:
                    ax.set_ylabel(labels.get(shock, shock), fontsize=8)
                ax.tick_params(labelsize=7)

        for ax in axes[-1]:
            ax.set_xlabel("Period", fontsize=8)

        fig.suptitle(title, fontsize=11, y=1.00)
        fig.tight_layout()
    return fig
