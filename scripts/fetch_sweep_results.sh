#!/usr/bin/env bash
# Pull disaster sweep results from the DGX to the local repo.
#
# Mirrors only:
#   - runs/sweep_so/*/result.json   (small, cheap)
#   - runs/sweep_so/*/tb/           (TensorBoard events; needed for curve analysis)
#
# Skipped:
#   - checkpoints/sweep_so/*        (large; not needed for analysis)
#   - runs/sweep_so/.wandb/*        (wandb cache)
#
# Usage:
#   bash scripts/fetch_sweep_results.sh
#
# Env:
#   DGX_HOST    defaults to anna@130.223.169.108
#   DGX_REPO    defaults to /home/anna/projects/deqn-jax

set -euo pipefail

DGX_HOST="${DGX_HOST:-anna@130.223.169.108}"
DGX_REPO="${DGX_REPO:-/home/anna/projects/deqn-jax}"
LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$LOCAL_REPO/runs/sweep_so"

echo "[fetch] $DGX_HOST:$DGX_REPO/runs/sweep_so/  →  $LOCAL_REPO/runs/sweep_so/"

rsync -az --info=stats1 \
    --include='*/' \
    --include='result.json' \
    --include='tb/***' \
    --exclude='*' \
    "$DGX_HOST:$DGX_REPO/runs/sweep_so/" \
    "$LOCAL_REPO/runs/sweep_so/"

n_results=$(find "$LOCAL_REPO/runs/sweep_so" -name 'result.json' -type f 2>/dev/null | wc -l | tr -d ' ')
echo "[fetch] $n_results result.json files locally now"
