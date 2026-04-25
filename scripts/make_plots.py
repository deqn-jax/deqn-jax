"""Make a handful of useful plots from our experimental runs.

Produces PNGs in plots/. Parses training logs for loss trajectories and
reads IRF CSVs for dynamics. Intentionally self-contained (stdlib +
numpy + matplotlib) — no new dependencies, no fancy styling.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

RUN_RE = re.compile(r"==== (\S+) (starting|finished).*$", re.M)
STEP_RE = re.compile(r"\[(\d+)/(\d+)\] loss=([\d.eE+-]+) \| grad=([\d.eE+-]+)")
BEST_RE = re.compile(r"Best checkpoint: ([\d.eE+-]+) at episode (\d+)")


def parse_log_single(path: Path, run_name: str) -> Dict[str, Dict]:
    """Parse a log file containing a single run; assign it the given name."""
    text = path.read_text() if path.exists() else ""
    eps, losses, grads = [], [], []
    for ep_s, _, loss_s, grad_s in STEP_RE.findall(text):
        eps.append(int(ep_s))
        losses.append(float(loss_s))
        grads.append(float(grad_s))
    best_match = BEST_RE.search(text)
    best = (
        (float(best_match.group(1)), int(best_match.group(2))) if best_match else None
    )
    return {
        run_name: {
            "episodes": np.array(eps),
            "losses": np.array(losses),
            "grads": np.array(grads),
            "best": best,
        }
    }


def parse_log(path: Path) -> Dict[str, Dict]:
    """Return {run_name: {episodes: [...], losses: [...], grads: [...], final, best}}."""
    text = path.read_text()
    runs: Dict[str, Dict] = {}
    # Find run blocks by looking at start/finish markers.
    markers = [(m.group(1), m.group(2), m.start()) for m in RUN_RE.finditer(text)]
    starts = [(n, p) for (n, ev, p) in markers if ev == "starting"]
    ends = {n: p for (n, ev, p) in markers if ev == "finished"}
    for name, start in starts:
        end = ends.get(name, len(text))
        block = text[start:end]
        eps, losses, grads = [], [], []
        for ep_s, _, loss_s, grad_s in STEP_RE.findall(block):
            eps.append(int(ep_s))
            losses.append(float(loss_s))
            grads.append(float(grad_s))
        best_match = BEST_RE.search(block)
        best = (
            (float(best_match.group(1)), int(best_match.group(2)))
            if best_match
            else None
        )
        runs[name] = {
            "episodes": np.array(eps),
            "losses": np.array(losses),
            "grads": np.array(grads),
            "best": best,
        }
    return runs


# ---------------------------------------------------------------------------
# IRF parsing
# ---------------------------------------------------------------------------


def parse_irf_csv(path: Path) -> Tuple[List[str], np.ndarray]:
    """Return (column_names, table[periods, cols])."""
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [list(map(float, r)) for r in reader]
    return header, np.array(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_loss_curves(data_root: Path) -> Path:
    """Compare loss trajectories across p_disaster values + schedule+arch ablations."""
    zlb_runs = parse_log(data_root / "zlb.log")
    sweep_runs = parse_log_single(data_root / "zlb_sweep.log", "disaster_p10_zlb")
    v030_runs = parse_log(data_root / "v030.log")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: p sweep at old schedule.
    palette = plt.cm.viridis(np.linspace(0.15, 0.9, 6))
    p_order = [
        ("disaster_p0_zlb", "p = 0 (baseline)"),
        ("disaster_p001_zlb", "p = 0.001"),
        ("disaster_p005_zlb", "p = 0.005"),
        ("disaster_p02_zlb", "p = 0.02"),
        ("disaster_p05_zlb", "p = 0.05"),
        ("disaster_p10_zlb", "p = 0.1"),
    ]
    for (name, label), color in zip(p_order, palette):
        run = zlb_runs.get(name)
        if run is None or len(run["episodes"]) == 0:
            continue
        ax1.plot(run["episodes"], run["losses"], color=color, label=label, lw=1.4)
    ax1.set_yscale("log")
    ax1.set_xlabel("Cycle (outer iteration)")
    ax1.set_ylabel("Training loss (log scale)")
    ax1.set_title(
        "ZLB sweep across p_disaster\n(old schedule: 1 grad update / cycle, 3000 cycles)"
    )
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(alpha=0.3)

    # Right: at p=0.1, old schedule vs new sweep vs ReLU vs regime-feat.
    series = [
        (
            zlb_runs.get("disaster_p10_zlb"),
            "tanh, old schedule (3000 × 1 update)",
            "C0",
            "-",
        ),
        (
            sweep_runs.get("disaster_p10_zlb"),
            "tanh, new sweep (500 × 20 updates)",
            "C1",
            "-",
        ),
        (v030_runs.get("disaster_p10_zlb_relu"), "ReLU, new sweep", "C2", "-"),
        (
            v030_runs.get("disaster_p10_zlb_zlbfeat"),
            "tanh + regime feat, new sweep",
            "C3",
            "-",
        ),
    ]
    for run, label, color, ls in series:
        if run is None or len(run["episodes"]) == 0:
            continue
        ax2.plot(
            run["episodes"], run["losses"], color=color, label=label, lw=1.4, ls=ls
        )
        if run["best"] is not None:
            loss_b, ep_b = run["best"]
            ax2.plot(ep_b, loss_b, marker="o", color=color, markersize=6)
    ax2.set_yscale("log")
    ax2.set_xlabel("Cycle")
    ax2.set_ylabel("Training loss (log scale)")
    ax2.set_title(
        "p = 0.1 ablations: schedule & architecture\n(circle = best-checkpoint minimum)"
    )
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = OUT / "loss_curves.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_irf(data_root: Path) -> Path:
    """4-panel IRF plot for the canonical shocks."""
    irf_dir = data_root / "irf_p10_zlb"
    shocks = ["eps", "mu_ups", "mu_z", "g", "m_p"]
    shock_labels = {
        "eps": "Transitory TFP ($\\varepsilon$)",
        "mu_ups": "Investment-specific ($\\mu_\\Upsilon$)",
        "mu_z": "Permanent growth ($\\mu_z$)",
        "g": "Government spending ($g$)",
        "m_p": "Monetary policy ($m^p$)",
    }
    keyvars = ["y_gdp", "pi", "R", "i", "c", "K_p"]

    fig, axes = plt.subplots(
        len(shocks),
        len(keyvars),
        figsize=(3.2 * len(keyvars), 2.0 * len(shocks)),
        sharex=True,
    )
    for i, shock in enumerate(shocks):
        path = irf_dir / f"irf_{shock}.csv"
        if not path.exists():
            continue
        header, data = parse_irf_csv(path)
        periods = data[:, 0].astype(int)
        for j, var in enumerate(keyvars):
            if var not in header:
                continue
            col = header.index(var)
            series = data[:, col]
            ss = series[0]  # pre-shock value
            dev_pct = (series - ss) / (abs(ss) + 1e-12) * 100
            ax = axes[i, j]
            ax.plot(periods, dev_pct, lw=1.3, color="C0")
            ax.axhline(0, color="gray", lw=0.5, ls="--")
            if i == 0:
                ax.set_title(var, fontsize=9)
            if j == 0:
                ax.set_ylabel(shock_labels[shock], fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2, lw=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("Period", fontsize=8)
    fig.suptitle(
        "Impulse response functions (% deviation from pre-shock) — p=0.1 ZLB",
        fontsize=11,
        y=1.00,
    )
    fig.tight_layout()
    out = OUT / "irfs.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_schedule_alignment(data_root: Path) -> Path:
    """Visualise schedule alignment: loss vs total gradient updates (not cycles)."""
    zlb_runs = parse_log(data_root / "zlb.log")
    sweep_runs = parse_log_single(data_root / "zlb_sweep.log", "disaster_p10_zlb")
    v030_runs = parse_log(data_root / "v030.log")

    fig, ax = plt.subplots(figsize=(8, 5))
    series = [
        (
            zlb_runs.get("disaster_p10_zlb"),
            1,
            "tanh, old schedule (1 update/cycle)",
            "C0",
        ),
        (
            sweep_runs.get("disaster_p10_zlb"),
            20,
            "tanh, new sweep (20 updates/cycle)",
            "C1",
        ),
        (v030_runs.get("disaster_p10_zlb_relu"), 20, "ReLU, new sweep", "C2"),
        (
            v030_runs.get("disaster_p10_zlb_zlbfeat"),
            20,
            "tanh + regime feat, new sweep",
            "C3",
        ),
    ]
    for run, updates_per_cycle, label, color in series:
        if run is None or len(run["episodes"]) == 0:
            continue
        updates = run["episodes"] * updates_per_cycle
        ax.plot(updates, run["losses"], color=color, label=label, lw=1.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total gradient updates")
    ax.set_ylabel("Training loss")
    ax.set_title(
        "Loss vs total gradient updates — schedule alignment & architecture ablations (p=0.1)"
    )
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = OUT / "schedule_alignment.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default="/tmp/deqn_plot_data",
        help="Directory with zlb.log, zlb_sweep.log, v030.log, irf_p10_zlb/",
    )
    args = ap.parse_args()
    data_root = Path(args.data)

    print("Parsing logs and plotting…")
    p1 = plot_loss_curves(data_root)
    print(f"  wrote {p1}")
    p2 = plot_schedule_alignment(data_root)
    print(f"  wrote {p2}")
    p3 = plot_irf(data_root)
    print(f"  wrote {p3}")
    print("Done.")


if __name__ == "__main__":
    main()
