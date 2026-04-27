#!/usr/bin/env bash
# Run a disaster sweep inside the NGC JAX container.
#
# Mounts the repo at /workspace, pip-installs project deps that aren't in the
# NGC base image (equinox, tensorboardX, wandb, tqdm, pydantic-settings, rich),
# then runs the launcher selected by ``LAUNCHER`` (default:
# ``scripts/sweep_disaster_second_order.py``).
#
# Env:
#   REPO_DIR        defaults to /home/anna/projects/deqn-jax
#   IMAGE           defaults to nvcr.io/nvidia/jax:26.02-py3
#   LAUNCHER        path (relative to repo) of the sweep launcher to run.
#                   Defaults to scripts/sweep_disaster_second_order.py for
#                   backward compat. Set to scripts/sweep_disaster_kf_validation.py
#                   for the K/F-anchor validation sweep.
#   WANDB_DIR_NAME  per-sweep wandb subdir name (default: sweep_so)
#   WANDB_API_KEY   optional; if unset, the launcher disables W&B
#
# Usage:
#   ./scripts/run_sweep_in_container.sh                          # full sweep
#   LAUNCHER=scripts/sweep_disaster_kf_validation.py \
#     WANDB_DIR_NAME=sweep_kf ./scripts/run_sweep_in_container.sh
#   ./scripts/run_sweep_in_container.sh --only <cell>            # one cell
#   ./scripts/run_sweep_in_container.sh --list                   # dry-run grid
#   ./scripts/run_sweep_in_container.sh --redo                   # overwrite results

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/anna/projects/deqn-jax}"
IMAGE="${IMAGE:-nvcr.io/nvidia/jax:26.02-py3}"
LAUNCHER="${LAUNCHER:-scripts/sweep_disaster_second_order.py}"
WANDB_DIR_NAME="${WANDB_DIR_NAME:-sweep_so}"

if [ ! -d "$REPO_DIR" ]; then
    echo "REPO_DIR=$REPO_DIR not found" >&2
    exit 1
fi

if [ ! -f "$REPO_DIR/$LAUNCHER" ]; then
    echo "LAUNCHER=$LAUNCHER not found in $REPO_DIR" >&2
    exit 1
fi

echo "[wrapper] image=$IMAGE"
echo "[wrapper] repo=$REPO_DIR"
echo "[wrapper] launcher=$LAUNCHER"
echo "[wrapper] launcher args: ${*:-(none)}"

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

docker run --rm --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --user "$HOST_UID:$HOST_GID" \
    -v "$REPO_DIR:/workspace" \
    -w /workspace \
    -e HOME=/workspace/.docker_home \
    -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
    -e WANDB_DIR="/workspace/runs/$WANDB_DIR_NAME/.wandb" \
    -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e LAUNCHER="$LAUNCHER" \
    "$IMAGE" \
    bash -c '
        set -euo pipefail
        echo "[setup] installing project deps in container..."
        # When running with --user (non-root), pip --user installs to
        # $HOME=/workspace/.docker_home which is host-mounted and persists.
        mkdir -p "$HOME"
        pip install --quiet --user --no-deps \
            equinox tensorboardX wandb tqdm pydantic-settings rich \
            treescope orbax-checkpoint matplotlib
        pip install --quiet --user \
            "protobuf>=3.20" "sentry-sdk>=2" "gitpython>=3" "platformdirs" \
            "contourpy>=1.0" "cycler>=0.10" "fonttools>=4.0" \
            "kiwisolver>=1.3" "pyparsing>=2.4" "pillow>=8"
        pip install --quiet --user --no-deps -e .
        echo "[setup] python:" && python --version
        echo "[setup] jax devices:" && python -c "import jax; print(jax.devices())"
        if [ -z "${WANDB_API_KEY:-}" ]; then
            echo "[setup] WANDB_API_KEY unset — disabling W&B for this sweep"
            export DEQN_DISABLE_WANDB=1
        fi
        echo "[run] starting launcher: $LAUNCHER ..."
        python "$LAUNCHER" "$@"
    ' bash "$@"
