"""Smoke tests for the plotting module.

Verify each plotting function runs without exceptions on small synthetic
data and produces a non-empty image. Uses the Agg backend so tests work
headless. Does not check pixel fidelity.
"""

from __future__ import annotations

import io

# Headless backend for tests.
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deqn_jax.plots import (
    parse_log,
    parse_log_single,
    plot_grad_norm,
    plot_irf_grid,
    plot_loss_curve,
    plot_multi_run_loss,
    plot_per_equation_residuals,
    plot_schedule_alignment,
)


def _fig_has_pixels(fig) -> bool:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=60)
    return buf.tell() > 1000  # a non-empty PNG is at least ~1 KB


def _fake_history(n: int = 50) -> dict:
    """Synthetic training history resembling train_from_config output."""
    eps = np.arange(1, n + 1)
    losses = 0.1 * np.exp(-eps / 20.0) + 1e-4 + 1e-5 * np.random.randn(n)
    grads = 0.5 * np.exp(-eps / 30.0) + 1e-2 + 1e-3 * np.random.randn(n)
    return {
        "episodes": eps,
        "loss": np.abs(losses),
        "grad_norm": np.abs(grads),
        "best": (float(np.abs(losses).min()), int(np.argmin(np.abs(losses))) + 1),
    }


def test_plot_loss_curve_smoke():
    ax = plot_loss_curve(_fake_history(), label="demo")
    assert _fig_has_pixels(ax.figure)
    plt.close(ax.figure)


def test_plot_grad_norm_smoke():
    ax = plot_grad_norm(_fake_history(), label="demo")
    assert _fig_has_pixels(ax.figure)
    plt.close(ax.figure)


def test_plot_grad_norm_missing_key():
    h = _fake_history()
    del h["grad_norm"]
    with pytest.raises(KeyError):
        plot_grad_norm(h)


def test_plot_per_equation_residuals_smoke():
    n = 20
    data = {
        "eq1": 0.05 * np.exp(-np.arange(n) / 10) + 1e-4,
        "eq2": 0.02 * np.exp(-np.arange(n) / 15) + 1e-4,
        "eq3": 0.1 * np.exp(-np.arange(n) / 5) + 1e-4,
    }
    ax = plot_per_equation_residuals(data, episodes=np.arange(1, n + 1))
    assert _fig_has_pixels(ax.figure)
    plt.close(ax.figure)


def test_plot_per_equation_residuals_empty_raises():
    with pytest.raises(ValueError):
        plot_per_equation_residuals({})


def test_plot_multi_run_loss_smoke():
    runs = {
        "baseline": _fake_history(),
        "variant A": _fake_history(),
        "variant B": _fake_history(),
    }
    ax = plot_multi_run_loss(runs)
    assert _fig_has_pixels(ax.figure)
    plt.close(ax.figure)


def test_plot_schedule_alignment_smoke():
    runs = {
        "old schedule": (_fake_history(), 1),
        "new sweep": (_fake_history(), 20),
    }
    ax = plot_schedule_alignment(runs)
    assert _fig_has_pixels(ax.figure)
    plt.close(ax.figure)


def test_plot_irf_grid_smoke():
    n = 40
    periods = list(range(n))

    def fake_irf_row():
        t = np.arange(n)
        return {
            "period": periods,
            "y_gdp": (3.0 + 0.01 * np.exp(-t / 10)).tolist(),
            "pi": (1.01 + 0.001 * np.exp(-t / 10)).tolist(),
            "R": (1.02 - 0.002 * np.exp(-t / 10)).tolist(),
            "i": (0.79 + 0.005 * np.exp(-t / 10)).tolist(),
            "c": (1.59 - 0.002 * np.exp(-t / 10)).tolist(),
            "K_p": (4.83 - 0.01 * np.exp(-t / 10)).tolist(),
        }

    irf_by_shock = {
        "eps": fake_irf_row(),
        "m_p": fake_irf_row(),
    }
    fig = plot_irf_grid(irf_by_shock, title="Smoke-test IRFs")
    assert _fig_has_pixels(fig)
    plt.close(fig)


def test_parse_log_single(tmp_path):
    log_text = """
==== my_run starting Mon Jan 1 ====
  [100/300] loss=3.50e-04 | grad=2.21e-01 | 3 ep/s
  [200/300] loss=1.50e-04 | grad=1.21e-01 | 3 ep/s
  [300/300] loss=1.20e-04 | grad=1.00e-01 | 3 ep/s
Best checkpoint: 1.10e-04 at episode 287
==== my_run finished exit=0 Mon Jan 1 ====
"""
    log_path = tmp_path / "log.txt"
    log_path.write_text(log_text)
    runs = parse_log_single(log_path, "solo")
    assert "solo" in runs
    h = runs["solo"]
    assert len(h["loss"]) == 3
    assert h["loss"][-1] == pytest.approx(1.20e-04)
    assert h["best"] == (pytest.approx(1.10e-04), 287)


def test_parse_log_multi(tmp_path):
    log_text = """
==== runA starting ... ====
  [100/200] loss=5.00e-04 | grad=1.00e-01 | 3 ep/s
  [200/200] loss=2.00e-04 | grad=5.00e-02 | 3 ep/s
Best checkpoint: 1.80e-04 at episode 190
==== runA finished exit=0 ... ====
==== runB starting ... ====
  [100/200] loss=4.00e-04 | grad=1.00e-01 | 3 ep/s
  [200/200] loss=1.00e-04 | grad=4.00e-02 | 3 ep/s
==== runB finished exit=0 ... ====
"""
    log_path = tmp_path / "log.txt"
    log_path.write_text(log_text)
    runs = parse_log(log_path)
    assert set(runs.keys()) == {"runA", "runB"}
    assert runs["runA"]["best"] is not None
    assert runs["runB"]["best"] is None
