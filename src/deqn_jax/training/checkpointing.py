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
import math
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


# ---------------------------------------------------------------------------
# In-loop orchestration (when to save / refresh-rollback / fallback).
# Moved out of trainer.py so the trainer carries only the loop, not the
# checkpoint policy. ``nan`` / ``tracker`` are duck-typed (trainer's
# _NanRollback / _SaveBestTracker) so this module stays free of trainer
# imports.
# ---------------------------------------------------------------------------


def maybe_checkpoint(config, state: TrainState, nan, ep_num: int) -> None:
    """Periodic checkpoint write + refresh of the NaN-rollback snapshot.

    ``nan`` needs ``.last_good_state`` and ``.last_good_episode`` attributes.
    """
    if (
        config.checkpoint_dir is None
        or config.checkpoint_every is None
        or ep_num % config.checkpoint_every != 0
    ):
        return
    save_checkpoint(state, config.checkpoint_dir, ep_num, config=config)
    if config.max_checkpoints is not None:
        prune_checkpoints(config.checkpoint_dir, config.max_checkpoints)
    nan.last_good_state = state
    nan.last_good_episode = ep_num


def maybe_save_best(
    config, state: TrainState, tracker, loss_val: float, ep_num: int
) -> None:
    """Save best-so-far checkpoint on improvement, after the grace period.

    ``tracker`` needs ``.grace``, ``.best_loss``, ``.best_episode`` attributes.
    """
    if not (
        config.save_best_checkpoint
        and config.checkpoint_dir is not None
        and ep_num > tracker.grace
        and not math.isnan(loss_val)
        and loss_val < tracker.best_loss
    ):
        return
    tracker.best_loss = loss_val
    tracker.best_episode = ep_num
    save_best_checkpoint(state, config.checkpoint_dir, ep_num, loss_val, config=config)


def final_save_best_fallback(config, state: TrainState, nan, tracker, history) -> None:
    """End-of-training fallback when the in-loop save-best gate never fired.

    The save-best gate (``ep_num > grace AND loss_val < best_save_loss``) is
    correct for STANDARD training: the curriculum-ramp grace prevents
    artificially-low ramp losses from being labelled "best". But for a run
    whose post-grace losses are all NaN (curvature methods at aggressive
    lr/damping settle into NaN-update regions once shocks reach full
    magnitude), the gate never fires and no ``checkpoint_best.eqx`` is written
    even though we have a perfectly good ``last_good_state`` from the
    periodic-checkpoint NaN-rollback path. Without this fallback, eval tooling
    can't load anything from such runs.
    """
    if not (
        config.save_best_checkpoint
        and config.checkpoint_dir is not None
        and tracker.best_loss == float("inf")
    ):
        return
    fallback_state = nan.last_good_state if nan.last_good_state is not None else state
    # Synthesize a best-loss for meta from history if we have one; otherwise
    # leave NaN so post-hoc eval can detect it's a fallback.
    finite_losses = [v for v in history.get("loss", []) if not math.isnan(v)]
    fallback_loss = min(finite_losses) if finite_losses else float("nan")
    fallback_episode = (
        nan.last_good_episode if nan.last_good_state is not None else config.episodes
    )
    save_best_checkpoint(
        fallback_state,
        config.checkpoint_dir,
        fallback_episode,
        fallback_loss,
        config=config,
    )
    # Annotate fallback so downstream eval can distinguish from a real in-loop
    # save-best. Append rather than overwrite so the canonical episode/loss line
    # stays first.
    meta_path = os.path.join(config.checkpoint_dir, BEST_CHECKPOINT_META_FILENAME)
    with open(meta_path, "a") as f:
        f.write(
            "fallback true  # save-best gate never fired during loop "
            "(post-grace losses all NaN); persisted last_good_state\n"
        )
    if config.verbose:
        print(
            f"Best checkpoint: FALLBACK save (post-grace losses all NaN) "
            f"→ {best_checkpoint_path(config.checkpoint_dir)}"
        )
