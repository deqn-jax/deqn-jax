"""Checkpoint I/O for the training loop.

Owns the on-disk layout for training checkpoints:
- ``checkpoint_{episode:06d}.eqx`` — periodic snapshots, used for
  resume + NaN rollback.
- ``checkpoint_best.eqx`` (+ ``.meta``) — overwritten whenever loss
  improves, persisted for the lifetime of the run.
- ``config.yaml`` — written once on first save so resume can rebuild
  a matching state pytree even if the live config has drifted.

Lives here so ``trainer.py`` carries only the policy of *when* to
save / prune / resume, not the storage layout.
"""

import glob as glob_mod
import os
from typing import Any

import equinox as eqx

from deqn_jax.types import TrainState

BEST_CHECKPOINT_FILENAME = "checkpoint_best.eqx"
BEST_CHECKPOINT_META_FILENAME = "checkpoint_best.meta"


def best_checkpoint_path(checkpoint_dir: str) -> str:
    """Path to the best-so-far snapshot inside ``checkpoint_dir``."""
    return os.path.join(checkpoint_dir, BEST_CHECKPOINT_FILENAME)


def save_checkpoint(
    state: TrainState,
    checkpoint_dir: str,
    episode: int,
    config=None,
) -> None:
    """Save a periodic training snapshot named by episode.

    Writes ``config.yaml`` once on the first call (subsequent calls
    skip if it already exists) so resume can reconstruct the matching
    state pytree.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_{episode:06d}.eqx")
    eqx.tree_serialise_leaves(path, state)
    if config is not None:
        cfg_path = os.path.join(checkpoint_dir, "config.yaml")
        if not os.path.exists(cfg_path):
            config.to_yaml(cfg_path)


def save_best_checkpoint(
    state: TrainState,
    checkpoint_dir: str,
    episode: int,
    loss: float,
    config=None,
) -> None:
    """Overwrite the best-so-far checkpoint and record the episode/loss.

    Called whenever loss improves past the running minimum. The
    resulting file is the "best achievable" artefact across the
    whole run -- useful when training finds a good solution mid-run
    and then gets destabilised.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = best_checkpoint_path(checkpoint_dir)
    eqx.tree_serialise_leaves(path, state)
    meta_path = os.path.join(checkpoint_dir, BEST_CHECKPOINT_META_FILENAME)
    with open(meta_path, "w") as f:
        f.write(f"episode {episode}\nloss {loss:.6e}\n")
    if config is not None and getattr(config, "checkpoint_dir", None):
        cfg_path = os.path.join(checkpoint_dir, "config.yaml")
        if not os.path.exists(cfg_path):
            config.to_yaml(cfg_path)


def prune_checkpoints(checkpoint_dir: str, max_keep: int) -> None:
    """Delete oldest periodic checkpoints, keeping only the most recent ``max_keep``."""
    pattern = os.path.join(checkpoint_dir, "checkpoint_*.eqx")
    existing = sorted(glob_mod.glob(pattern))
    # Don't sweep up the best snapshot — it's not part of the periodic series.
    existing = [p for p in existing if os.path.basename(p) != BEST_CHECKPOINT_FILENAME]
    while len(existing) > max_keep:
        os.remove(existing.pop(0))


def resume_from(template_state: Any, checkpoint_path: str) -> Any:
    """Load a serialised TrainState from disk into the given template.

    Thin wrapper around ``eqx.tree_deserialise_leaves`` so trainer.py
    doesn't need to spell out the equinox call. ``template_state`` must
    have the same pytree structure as the saved state -- typically built
    from the config that produced the checkpoint.
    """
    return eqx.tree_deserialise_leaves(checkpoint_path, template_state)
