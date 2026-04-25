#!/usr/bin/env bash
# Run the disaster second-order sweep inside the NGC JAX container.
#
# Mounts the repo at /workspace, pip-installs project deps that aren't in the
# NGC base image (equinox, tensorboardX, wandb, tqdm, pydantic-settings, rich),
# then runs scripts/sweep_disaster_second_order.py.
#
# Env:
#   REPO_DIR        defaults to /home/anna/projects/deqn-jax
#   IMAGE           defaults to nvcr.io/nvidia/jax:26.02-py3
#   WANDB_API_KEY   optional; if unset, the launcher disables W&B
#
# Usage:
#   ./scripts/run_sweep_in_container.sh                          # full sweep
#   ./scripts/run_sweep_in_container.sh --only ngd_lr1e-3        # one cell
#   ./scripts/run_sweep_in_container.sh --list                   # dry-run grid
#   ./scripts/run_sweep_in_container.sh --redo                   # overwrite results

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/anna/projects/deqn-jax}"
IMAGE="${IMAGE:-nvcr.io/nvidia/jax:26.02-py3}"

if [ ! -d "$REPO_DIR" ]; then
    echo "REPO_DIR=$REPO_DIR not found" >&2
    exit 1
fi

# Forward all CLI args verbatim to the launcher.
EXTRA_ARGS=("$@")
PY_ARGS_QUOTED=$(printf '"%s" ' "${EXTRA_ARGS[@]}")

echo "[wrapper] image=$IMAGE"
echo "[wrapper] repo=$REPO_DIR"
echo "[wrapper] launcher args: ${EXTRA_ARGS[*]:-(none)}"

docker run --rm --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$REPO_DIR:/workspace" \
    -w /workspace \
    -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
    -e WANDB_DIR=/workspace/runs/sweep_so/.wandb \
    -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$IMAGE" \
    bash -c "
        set -euo pipefail
        echo '[setup] installing project deps in container...'
        pip install --quiet --no-deps \
            equinox tensorboardX wandb tqdm pydantic-settings rich
        # protobuf / sentry-sdk / etc. are wandb/tensorboardX deps; bring those in.
        pip install --quiet \
            'protobuf>=3.20' 'sentry-sdk>=2' 'gitpython>=3' 'platformdirs'
        pip install --quiet --no-deps -e .
        echo '[setup] python:' && python --version
        echo '[setup] jax devices:' && python -c 'import jax; print(jax.devices())'
        if [ -z \"\${WANDB_API_KEY:-}\" ]; then
            echo '[setup] WANDB_API_KEY unset — disabling W&B for this sweep'
            export DEQN_DISABLE_WANDB=1
        fi
        echo '[run] starting launcher...'
        python scripts/sweep_disaster_second_order.py $PY_ARGS_QUOTED
    "
