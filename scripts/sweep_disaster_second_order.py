"""Sequential sweep launcher for disaster second-order optimizer comparison.

Runs each (optimizer, learning_rate) cell in series in the same Python
process (so JIT cache benefits where possible) and persists per-run
metrics to ``runs/sweep_so/<run_id>/result.json``. Idempotent: skips
runs whose result.json already exists.

Stage 1 grid (22 runs):
    Adam-LR-style optimizers (4) × {1e-2, 1e-3, 1e-4} = 12
    Curvature-step-size optimizers (3) × {0.1, 0.5, 1.0} = 9
    LBFGS (1) × {1.0} = 1

Wall-clock target: ~45 min on a GB10 post-JIT.

Run inside the NGC JAX container; the container path is mounted at
/workspace via ``scripts/run_in_container.sh``.

Usage (from repo root, on the DGX, inside the container):
    python scripts/sweep_disaster_second_order.py
    python scripts/sweep_disaster_second_order.py --only ngd_lr1e-3
    python scripts/sweep_disaster_second_order.py --resume   # default
    python scripts/sweep_disaster_second_order.py --redo     # overwrite results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = REPO_ROOT / "configs/sweeps/disaster_so_base.yaml"
SWEEP_RUNS_DIR = REPO_ROOT / "runs/sweep_so"
SWEEP_CKPT_DIR = REPO_ROOT / "checkpoints/sweep_so"
WANDB_PROJECT = "deqn-disaster-second-order"


# (run_id, override_dict). Overrides are dot-paths into the YAML config.
SWEEP_GRID: List[Tuple[str, Dict[str, object]]] = []

# Adam-style learning-rate semantics: small steps, momentum-like preconditioner.
for opt in ("ngd", "mao", "shampoo", "muon"):
    for lr in (1e-2, 1e-3, 1e-4):
        SWEEP_GRID.append(
            (
                f"{opt}_lr{lr:g}",
                {"optimizer.name": opt, "optimizer.learning_rate": lr},
            )
        )

# Curvature-step-size optimizers: full step is 1.0, damping does the regularization.
for opt in ("gn", "ign", "lm"):
    for lr in (0.1, 0.5, 1.0):
        SWEEP_GRID.append(
            (
                f"{opt}_lr{lr:g}",
                {"optimizer.name": opt, "optimizer.learning_rate": lr},
            )
        )

# LBFGS: line search internally; learning_rate caps the step.
SWEEP_GRID.append(
    ("lbfgs_lr1", {"optimizer.name": "lbfgs", "optimizer.learning_rate": 1.0})
)


def _set_dotted(cfg: dict, path: str, value: object) -> None:
    """Set ``cfg`` at dotted ``path`` to ``value``, creating intermediate dicts."""
    keys = path.split(".")
    cur = cfg
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def _build_config(run_id: str, overrides: Dict[str, object]) -> dict:
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)
    for path, value in overrides.items():
        _set_dotted(cfg, path, value)
    cfg["tensorboard_dir"] = str(SWEEP_RUNS_DIR / run_id / "tb")
    cfg["checkpoint_dir"] = str(SWEEP_CKPT_DIR / run_id)
    if os.environ.get("DEQN_DISABLE_WANDB") == "1":
        cfg["wandb_project"] = None
    else:
        cfg["wandb_project"] = WANDB_PROJECT
    return cfg


def _result_path(run_id: str) -> Path:
    return SWEEP_RUNS_DIR / run_id / "result.json"


def _write_result(run_id: str, payload: dict) -> None:
    path = _result_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _summarize_history(history: Dict[str, list]) -> Dict[str, float]:
    """Pull final + best scalar stats out of the history dict.

    train_from_config returns ``(params, history)`` where history is
    ``{"loss": [...], "grad_norm": [...]}`` — appended once per logging
    cycle. Per-eq residuals are NOT in history; the Ralph analysis
    loop reads them from TensorBoard events.
    """
    summary: Dict[str, float] = {}
    losses = history.get("loss") or []
    grad_norms = history.get("grad_norm") or []
    if losses:
        best = min(losses)
        summary["final_loss"] = float(losses[-1])
        summary["best_loss"] = float(best)
        summary["best_loss_idx"] = int(losses.index(best))
        summary["n_log_points"] = len(losses)
    if grad_norms:
        summary["final_grad_norm"] = float(grad_norms[-1])
    return summary


def _run_one(run_id: str, overrides: Dict[str, object]) -> dict:
    """Run a single sweep cell. Returns the result dict."""
    from deqn_jax.config import TrainConfig
    from deqn_jax.training.trainer import train_from_config

    cfg_dict = _build_config(run_id, overrides)
    # Per-run W&B run name via env (metrics.WandbLogger doesn't take a name kwarg).
    if cfg_dict.get("wandb_project"):
        os.environ["WANDB_RUN_NAME"] = run_id

    print(f"\n{'=' * 60}\n[{run_id}] starting", flush=True)
    print(f"  overrides: {overrides}", flush=True)

    t0 = time.perf_counter()
    payload = {
        "run_id": run_id,
        "overrides": overrides,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    try:
        cfg = TrainConfig.model_validate(cfg_dict)
        _params, history = train_from_config(cfg)
        wall = time.perf_counter() - t0
        summary = _summarize_history(history)
        payload.update({"status": "ok", "wall_seconds": wall, **summary})
        loss_str = (
            f"{summary.get('final_loss', float('nan')):.6e}"
            if "final_loss" in summary
            else "n/a"
        )
        best_str = (
            f"{summary.get('best_loss', float('nan')):.6e}"
            if "best_loss" in summary
            else "n/a"
        )
        print(
            f"[{run_id}] OK   final={loss_str}  best={best_str}  wall={wall:.1f}s",
            flush=True,
        )
    except Exception as e:
        wall = time.perf_counter() - t0
        payload.update(
            {
                "status": "error",
                "wall_seconds": wall,
                "error_type": type(e).__name__,
                "error_msg": str(e),
                "traceback": traceback.format_exc(),
            }
        )
        print(f"[{run_id}] FAIL {type(e).__name__}: {e}", flush=True)

    _write_result(run_id, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--only",
        help="Comma-separated list of run_ids to run (default: all). "
        "Useful for re-running a single config.",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Re-run cells that already have a result.json (default: skip).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the sweep grid and exit, no training.",
    )
    args = parser.parse_args()

    SWEEP_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        print(f"Sweep grid ({len(SWEEP_GRID)} runs):")
        for run_id, overrides in SWEEP_GRID:
            print(f"  {run_id:20s} {overrides}")
        return 0

    only = set(s.strip() for s in args.only.split(",")) if args.only else None
    skipped = 0
    ran = 0
    failed = 0

    for run_id, overrides in SWEEP_GRID:
        if only is not None and run_id not in only:
            continue
        if not args.redo and _result_path(run_id).exists():
            print(f"[{run_id}] skip (result.json exists)", flush=True)
            skipped += 1
            continue
        result = _run_one(run_id, overrides)
        ran += 1
        if result.get("status") != "ok":
            failed += 1

    print(
        f"\nSweep done: ran={ran}, skipped={skipped}, failed={failed}, "
        f"total_grid={len(SWEEP_GRID)}",
        flush=True,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
