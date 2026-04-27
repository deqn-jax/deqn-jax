#!/usr/bin/env bash
# Pull a disaster sweep's results from the DGX to the local repo.
#
# Mirrors only:
#   - runs/<sweep>/*/result.json   (small, cheap)
#   - runs/<sweep>/*/tb/           (TensorBoard events; needed for curve analysis)
#
# Skipped:
#   - checkpoints/<sweep>/*        (large; not needed for analysis)
#   - runs/<sweep>/.wandb/*        (wandb cache)
#
# Usage:
#   bash scripts/fetch_sweep_results.sh                  # default: sweep_so
#   SWEEP=sweep_kf bash scripts/fetch_sweep_results.sh   # K/F validation sweep
#
# Env:
#   DGX_HOST    defaults to anna@130.223.169.108
#   DGX_REPO    defaults to /home/anna/projects/deqn-jax
#   SWEEP       sweep subdir name under runs/ (default: sweep_so)

set -euo pipefail

DGX_HOST="${DGX_HOST:-anna@130.223.169.108}"
DGX_REPO="${DGX_REPO:-/home/anna/projects/deqn-jax}"
SWEEP="${SWEEP:-sweep_so}"
LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$LOCAL_REPO/runs/$SWEEP"

echo "[fetch] $DGX_HOST:$DGX_REPO/runs/$SWEEP/  →  $LOCAL_REPO/runs/$SWEEP/"

rsync -az --info=stats1 \
    --include='*/' \
    --include='result.json' \
    --include='tb/***' \
    --exclude='*' \
    "$DGX_HOST:$DGX_REPO/runs/$SWEEP/" \
    "$LOCAL_REPO/runs/$SWEEP/"

n_results=$(find "$LOCAL_REPO/runs/$SWEEP" -name 'result.json' -type f 2>/dev/null | wc -l | tr -d ' ')
echo "[fetch] $n_results result.json files locally now"
