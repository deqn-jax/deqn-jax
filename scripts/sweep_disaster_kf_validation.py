"""Validation sweep: K/F-anchor + moment-matching vs vanilla MLP on disaster.

Grid (21 cells):
    mlp_baseline                     × seeds {0, 1, 2}    = 3
    kf_anchor_no_moment              × seeds {0, 1, 2}    = 3
    kf_anchor_moment_w0.01           × seeds {0, 1, 2}    = 3
    kf_anchor_moment_w0.05           × seeds {0, 1, 2}    = 3
    kf_anchor_moment_w0.10           × seeds {0, 1, 2}    = 3
    linear_plus_mlp                  × seeds {0, 1, 2}    = 3
    linear_plus_mlp_kfmask           × seeds {0, 1, 2}    = 3

The linear_plus_mlp arms are the missing baselines from the original
2026-04-27 run: residual ansatz π = π_BK + δ_θ with δ_θ zero-init.
The kfmask variant additionally pins the four Calvo Phillips-curve
auxiliaries (F_p, K_p, F_w, K_w) to the BK linearization (subsumes
the kf_anchored_mlp arm with a stronger prior on the 7 free outputs).

Each run trains 5000 episodes on disaster (adam, lr=1e-3) under
``configs/sweeps/disaster_kf_validation.yaml``. After training, the
launcher loads the best checkpoint and runs the full Dynare comparison
(moments + ghx + IRFs), saving the per-run diff numbers into
``result.json`` alongside the loss curve. This produces a single
artifact per cell that the analysis step can consume directly — no
separate eval pass needed.

Wall-clock target on GB10: ~30 min/cell post-JIT, ~10-11 hr total
for the full 21-cell grid.

Idempotent (skips cells with an existing ``result.json``). Same
scripts/run_sweep_in_container.sh wrapper reuses the NGC JAX container.

Usage:
    python scripts/sweep_disaster_kf_validation.py
    python scripts/sweep_disaster_kf_validation.py --only kf_anchor_no_moment_seed0
    python scripts/sweep_disaster_kf_validation.py --list
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
BASE_CONFIG = REPO_ROOT / "configs/sweeps/disaster_kf_validation.yaml"
SWEEP_RUNS_DIR = REPO_ROOT / "runs/sweep_kf"
SWEEP_CKPT_DIR = REPO_ROOT / "checkpoints/sweep_kf"
DYNARE_DIR = REPO_ROOT / "dynare/results"
WANDB_PROJECT = "deqn-disaster-kf-validation"


def _build_grid() -> List[Tuple[str, Dict[str, object]]]:
    grid: List[Tuple[str, Dict[str, object]]] = []
    seeds = (0, 1, 2)
    for seed in seeds:
        grid.append(
            (
                f"mlp_baseline_seed{seed}",
                {
                    "seed": seed,
                    "network.type": "mlp",
                    "moment_matching.enabled": False,
                },
            )
        )
        grid.append(
            (
                f"kf_anchor_no_moment_seed{seed}",
                {
                    "seed": seed,
                    "network.type": "kf_anchored_mlp",
                    "moment_matching.enabled": False,
                },
            )
        )
        for w in (0.01, 0.05, 0.10):
            tag = f"kf_anchor_moment_w{w:g}_seed{seed}"
            grid.append(
                (
                    tag,
                    {
                        "seed": seed,
                        "network.type": "kf_anchored_mlp",
                        "moment_matching.enabled": True,
                        "moment_matching.weight": w,
                    },
                )
            )
        # Pure residual ansatz (no kf mask): π = π_BK + δ_θ on every output.
        # network.kf_names=[] explicitly overrides the NetworkConfig default
        # which would otherwise enable the K/F mask.
        grid.append(
            (
                f"linear_plus_mlp_seed{seed}",
                {
                    "seed": seed,
                    "network.type": "linear_plus_mlp",
                    "network.kf_names": [],
                    "moment_matching.enabled": False,
                },
            )
        )
        # Residual ansatz + K/F mask: π = π_BK + δ_θ everywhere except
        # F_p, K_p, F_w, K_w which stay exactly π_BK forever.
        grid.append(
            (
                f"linear_plus_mlp_kfmask_seed{seed}",
                {
                    "seed": seed,
                    "network.type": "linear_plus_mlp",
                    "network.kf_names": ["F_p", "K_p", "F_w", "K_w"],
                    "moment_matching.enabled": False,
                },
            )
        )
    return grid


SWEEP_GRID = _build_grid()


def _set_dotted(cfg: dict, path: str, value: object) -> None:
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
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def _summarize_history(history: Dict[str, list]) -> Dict[str, float]:
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


def _run_dynare_eval(checkpoint_dir: Path, n_periods: int = 2000) -> Dict[str, object]:
    """Load best.eqx and run the Dynare comparison; return diffs."""
    from deqn_jax.evaluate import (
        compare_to_dynare_ghx,
        compare_to_dynare_irfs,
        compare_to_dynare_moments,
    )
    from deqn_jax.irf import load_policy_from_checkpoint

    best_path = checkpoint_dir / "checkpoint_best.eqx"
    if not best_path.exists():
        return {"eval_skipped": "no checkpoint_best.eqx"}

    try:
        policy, model = load_policy_from_checkpoint(str(best_path))
    except Exception as e:
        return {"eval_skipped": f"load failed: {type(e).__name__}: {e}"}

    out: Dict[str, object] = {}
    try:
        mom = compare_to_dynare_moments(
            policy, model, str(DYNARE_DIR), n_periods=n_periods
        )
        out["moments"] = {
            "median_abs_mean_diff_pct": mom.get("median_abs_mean_diff_pct"),
            "median_abs_std_diff_pct": mom.get("median_abs_std_diff_pct"),
            "max_abs_mean_diff_pct": mom.get("max_abs_mean_diff_pct"),
            "max_abs_std_diff_pct": mom.get("max_abs_std_diff_pct"),
            "n_compared": mom.get("n_compared"),
        }
    except Exception as e:
        out["moments_error"] = f"{type(e).__name__}: {e}"

    try:
        ghx = compare_to_dynare_ghx(policy, model, str(DYNARE_DIR))
        out["ghx"] = {
            "frobenius": ghx.get("frobenius"),
            "frobenius_relative": ghx.get("frobenius_relative"),
            "j_net_nonfinite_entries": ghx.get("j_net_nonfinite_entries"),
        }
    except Exception as e:
        out["ghx_error"] = f"{type(e).__name__}: {e}"

    try:
        irf = compare_to_dynare_irfs(policy, model, str(DYNARE_DIR), horizon=20)
        per_shock = irf.get("per_shock", {})
        out["irf"] = {
            shock: {
                "max_abs_overall": payload.get("max_abs_overall"),
                "n_vars": payload.get("n_vars"),
            }
            for shock, payload in per_shock.items()
        }
        out["irf_shocks_skipped"] = irf.get("shocks_skipped", [])
    except Exception as e:
        out["irf_error"] = f"{type(e).__name__}: {e}"

    return out


def _run_one(run_id: str, overrides: Dict[str, object]) -> dict:
    from deqn_jax.config import TrainConfig
    from deqn_jax.training.trainer import train_from_config

    cfg_dict = _build_config(run_id, overrides)
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

        eval_out = _run_dynare_eval(SWEEP_CKPT_DIR / run_id)
        payload["dynare_eval"] = eval_out

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
        mean_diff = None
        std_diff = None
        if isinstance(eval_out.get("moments"), dict):
            mean_diff = eval_out["moments"].get("median_abs_mean_diff_pct")
            std_diff = eval_out["moments"].get("median_abs_std_diff_pct")
        moments_str = (
            f"|Δmean|={mean_diff:.1f}%  |Δstd|={std_diff:.1f}%"
            if mean_diff is not None and std_diff is not None
            else "moments=n/a"
        )
        print(
            f"[{run_id}] OK   final={loss_str}  best={best_str}  {moments_str}  "
            f"wall={wall:.1f}s",
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
        help="Comma-separated list of run_ids to run (default: all).",
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
            print(f"  {run_id:42s} {overrides}")
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
