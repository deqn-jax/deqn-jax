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
from pathlib import Path
from typing import Any, Optional, Tuple

import equinox as eqx
import jax
import yaml

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


# ---------------------------------------------------------------------------
# Loading a trained policy back (evaluation, IRFs, probes)
# ---------------------------------------------------------------------------


def load_policy_from_checkpoint(
    checkpoint_path: str,
    config_path: Optional[str] = None,
) -> Tuple[eqx.Module, object]:
    """Load trained policy network from checkpoint.

    Args:
        checkpoint_path: Path to .eqx checkpoint file
        config_path: Path to config.yaml (auto-detected from checkpoint dir if None)

    Returns:
        (policy_net, model) tuple
    """
    # Auto-detect config
    if config_path is None:
        ckpt_dir = Path(checkpoint_path).parent
        config_path = str(ckpt_dir / "config.yaml")
        if not Path(config_path).exists():
            raise FileNotFoundError(
                f"No config.yaml found in {ckpt_dir}. Pass --config explicitly."
            )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Enable fp64 if checkpoint was trained with it
    if cfg.get("fp64", False):
        jax.config.update("jax_enable_x64", True)

    from deqn_jax.models import load_model

    model = load_model(cfg["model"])

    # Extract network config
    net_cfg = cfg.get("network", {})
    hidden_sizes = tuple(net_cfg.get("hidden_sizes", [64, 64]))

    key = jax.random.PRNGKey(0)  # doesn't matter, will be overwritten

    # Deserialize — the checkpoint is a full TrainState, we need just params
    # Build a template TrainState to match the checkpoint structure
    from deqn_jax.config import OptimizerConfig
    from deqn_jax.training.trainer import create_train_state

    n_equations = (
        len(model.equation_names) if model.equation_names else model.n_policies
    )

    # Parse loss_weights from config
    loss_weights = cfg.get("loss_weights", None)

    opt_cfg_dict = cfg.get("optimizer", {"name": "adam"})
    # If checkpoint was saved after optimizer switch, use the switched optimizer
    switch_opt = cfg.get("switch_optimizer", None)
    switch_ep = cfg.get("switch_episode", 0)
    # Extract episode number from checkpoint filename (e.g. checkpoint_010000.eqx)
    ckpt_ep = 0
    try:
        ckpt_ep = int(Path(checkpoint_path).stem.split("_")[-1])
    except (ValueError, IndexError):
        pass
    if switch_opt and ckpt_ep >= switch_ep:
        opt_cfg_dict = dict(opt_cfg_dict)
        opt_cfg_dict["name"] = switch_opt
        if cfg.get("switch_lr") is not None:
            opt_cfg_dict["learning_rate"] = cfg["switch_lr"]
    from deqn_jax.config.io import _drop_removed_fields

    opt_cfg_dict = _drop_removed_fields("optimizer", dict(opt_cfg_dict))
    opt_cfg = OptimizerConfig(
        **{k: v for k, v in opt_cfg_dict.items() if k in OptimizerConfig.model_fields}
    )

    from deqn_jax.config import NetworkConfig

    # Pass through EVERY recognized network field, exactly like the
    # OptimizerConfig construction above. A hand-picked subset here silently
    # dropped STATIC fields that change the forward graph (bk_pin,
    # use_zlb_feature, reparam flags): the template net was then built with
    # a different architecture than the checkpoint was trained with, and
    # leaf deserialization can't repair a wrong graph (2026-07-11, caught
    # by an impossible bkpin probe: pi(s*) is pinned by construction, yet
    # the loaded net showed 476% SS error).
    net_cfg = _drop_removed_fields("network", dict(net_cfg))
    net_config = NetworkConfig(
        **{k: v for k, v in net_cfg.items() if k in NetworkConfig.model_fields}
    )

    # sim_batch and replay_buffer also shape the TrainState pytree
    # (episode_state carries sim_batch trajectories; replay_state is a
    # whole subtree) — same silent-drop class as the network fields above.
    from deqn_jax.config import ReplayBufferConfig

    replay_dict = cfg.get("replay_buffer") or {}
    replay_cfg = ReplayBufferConfig(
        **{k: v for k, v in replay_dict.items() if k in ReplayBufferConfig.model_fields}
    )

    template_state, _, _ = create_train_state(
        model,
        key,
        hidden_sizes=hidden_sizes,
        batch_size=cfg.get("batch_size", 64),
        loss_weights=loss_weights,
        n_equations=n_equations,
        optimizer_config=opt_cfg,
        network_config=net_config,
        sim_batch=cfg.get("sim_batch"),
        replay_config=replay_cfg,
    )

    # Deserialize through the trainer's own loader so checkpoint reading has
    # one implementation. The FULL-NetworkConfig template above is still built
    # here (see the 2026-07-11 note) — ``resume_from`` only fills its leaves.
    state = resume_from(template_state, checkpoint_path)
    policy_net = state.params

    # NB: the previous "restore correct bounds" rehab block was a fix for
    # a former bug where output_lower / output_upper drifted under Adam-
    # family second-moment updates. That bug was closed structurally in
    # ``f5041c8`` (bound + normalization fields are now ``eqx.field(static=True)``
    # tuples, excluded from the trainable pytree). New checkpoints inherit
    # the correct bounds from the template state's ``__init__`` at load
    # time, so no post-load rehab is needed — and ``eqx.tree_at`` on
    # static fields raises, since static fields aren't pytree leaves.

    return policy_net, model
