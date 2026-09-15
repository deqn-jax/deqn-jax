#!/usr/bin/env bash
# Disaster certification sweep (spec-let 1): 4 arms x 3 seeds, full recipe
# (3000 episodes each). Arms: baseline / gated anchor / ELB coverage / both.
# Sequential; each run saves checkpoint_best.eqx + config.yaml under
# runs/disaster_cert/<tag>/. Completed tags are skipped on re-run.
# Evaluate afterwards with:
#   JAX_ENABLE_X64=1 uv run python scripts/disaster_ss_probe.py \
#       --runs-dir runs/disaster_cert --json-out runs/disaster_cert/probe.json
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
mkdir -p runs/disaster_cert logs

for cfg in disaster disaster_gated disaster_elbcov disaster_gated_elbcov; do
  for s in 0 1 2; do
    tag="${cfg}_s${s}"
    if [ -f "runs/disaster_cert/${tag}/checkpoint_best.eqx" ] && \
       grep -q "Training complete" "logs/${tag}.log" 2>/dev/null; then
      echo "=== ${tag} already done, skipping ==="
      continue
    fi
    echo "=== ${tag} start $(date) ==="
    uv run deqn-jax train --config "configs/${cfg}.yaml" \
      --set seed="${s}" \
      --set checkpoint_dir="runs/disaster_cert/${tag}" \
      --set checkpoint_every=1000 \
      --set tensorboard_dir="runs/disaster_cert/${tag}_tb" \
      > "logs/${tag}.log" 2>&1
    echo "=== ${tag} done rc=$? $(date) ==="
  done
done
echo "CERT SWEEP COMPLETE $(date)"
