#!/usr/bin/env bash
# 5-seed EWM sweep, 4 arms: irbc_plain (unstable baseline), irbc_ewm
# (coverage), irbc (BK-anchor composite, post-KKT-fix re-measure), and
# irbc_ewm_anchor (composition: coverage + composite). Sequential; each
# run saves checkpoint_best.eqx + checkpoint_004000.eqx (~final) +
# config.yaml under runs/ewm_sweep/<tag>/. Completed tags are skipped,
# so re-running after adding arms only trains the new ones.
# Evaluate afterwards with scripts/ewm_stress_table.py (rho(SS) +
# held-out stress/base grids).
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
mkdir -p runs/ewm_sweep logs

for cfg in irbc_plain irbc_ewm irbc irbc_ewm_anchor; do
  for s in 0 1 2 3 4; do
    tag="${cfg}_s${s}"
    if [ -f "runs/ewm_sweep/${tag}/checkpoint_004000.eqx" ]; then
      echo "=== ${tag} already done, skipping ==="
      continue
    fi
    echo "=== ${tag} start $(date) ==="
    uv run deqn-jax train --config "configs/${cfg}.yaml" \
      --set seed="${s}" \
      --set checkpoint_dir="runs/ewm_sweep/${tag}" \
      --set checkpoint_every=4000 \
      > "logs/${tag}.log" 2>&1
    echo "=== ${tag} done rc=$? $(date) ==="
  done
done
echo "SWEEP COMPLETE $(date)"
