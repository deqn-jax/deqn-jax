#!/usr/bin/env bash
# Run a disaster sweep inside the NGC JAX container.
#
# Mounts the repo at /workspace, pip-installs project deps that aren't in the
# NGC base image (equinox, tensorboardX, wandb, tqdm, pydantic-settings, rich),
# then runs the launcher named by ``LAUNCHER`` with any extra arguments.
#
# Env:
#   REPO_DIR        defaults to /home/anna/projects/deqn-jax
#   IMAGE           defaults to nvcr.io/nvidia/jax:26.02-py3
#   LAUNCHER        required: path (relative to the repo) of the launcher to
#                   run inside the container. The certification sweep is
#                   scripts/dgx/cert_sweep_container.py (takes no arguments;
#                   DONE-marker resumable).
#   WANDB_DIR_NAME  per-sweep wandb subdir name (default: sweep_so)
#   WANDB_API_KEY   optional; if unset, the launcher disables W&B
#
# Usage:
#   LAUNCHER=scripts/dgx/cert_sweep_container.py ./scripts/dgx/run_sweep_in_container.sh
#
# Arguments after the script name are passed to the launcher verbatim; a
# launcher that parses none ignores them.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/anna/projects/deqn-jax}"
IMAGE="${IMAGE:-nvcr.io/nvidia/jax:26.02-py3}"
LAUNCHER="${LAUNCHER:?set LAUNCHER to the launcher path relative to the repo, e.g. scripts/dgx/cert_sweep_container.py}"
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
