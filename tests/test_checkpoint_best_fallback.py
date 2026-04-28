"""Tests for the end-of-training save-best fallback.

The save-best gate in `_run_training_loop` is `ep_num > best_save_grace`
where `best_save_grace = max(curriculum_episodes, log_every)`. For a run
whose post-grace losses are all NaN (curvature methods that NaN-out once
shocks reach full magnitude), the gate never fires and no
`checkpoint_best.eqx` is written. The fallback at end-of-training writes
`last_good_state` to that path and appends a "fallback true" line to the
meta file so downstream eval can distinguish it from a real save-best.
"""

import os
import tempfile

import pytest

from deqn_jax.config import TrainConfig
from deqn_jax.training.trainer import train_from_config


def _base_cfg(checkpoint_dir: str, episodes: int, **overrides) -> TrainConfig:
    cfg_dict = {
        "model": "brock_mirman",
        "episodes": episodes,
        "episode_length": 4,
        "batch_size": 8,
        "sim_batch": 8,
        "mc_samples": 1,
        "fp64": False,
        "verbose": False,
        "log_every": 1,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_every": max(1, episodes // 4),
        "max_checkpoints": 2,
        "save_best_checkpoint": True,
        "network": {"type": "mlp", "hidden_sizes": [8]},
        "optimizer": {"name": "adam", "learning_rate": 1e-3},
    }
    cfg_dict.update(overrides)
    return TrainConfig.model_validate(cfg_dict)


def test_fallback_fires_when_grace_exceeds_episodes():
    """If curriculum_episodes > episodes, save-best gate never fires
    in-loop; fallback should still produce checkpoint_best.eqx."""
    with tempfile.TemporaryDirectory() as tmp:
        # episodes=8, curriculum=999 → best_save_grace=999 → no in-loop save.
        cfg = _base_cfg(tmp, episodes=8, curriculum_episodes=999)
        train_from_config(cfg)

        best_path = os.path.join(tmp, "checkpoint_best.eqx")
        meta_path = os.path.join(tmp, "checkpoint_best.meta")
        assert os.path.exists(best_path), "fallback did not write checkpoint_best.eqx"
        assert os.path.exists(meta_path), "meta file missing"
        with open(meta_path) as f:
            meta = f.read()
        assert "fallback true" in meta, (
            f"meta should contain 'fallback true' annotation: {meta!r}"
        )


def test_fallback_does_not_overwrite_real_best():
    """When save-best fires normally during training, the fallback path
    must NOT overwrite it with stale state."""
    with tempfile.TemporaryDirectory() as tmp:
        # episodes=20, curriculum=2 → grace=2 → save-best fires from ep 3 on.
        cfg = _base_cfg(tmp, episodes=20, curriculum_episodes=2)
        train_from_config(cfg)

        meta_path = os.path.join(tmp, "checkpoint_best.meta")
        assert os.path.exists(meta_path)
        with open(meta_path) as f:
            meta = f.read()
        # Real save-best meta has "episode N\nloss V" lines; the fallback
        # annotation must NOT be appended on this path.
        assert "fallback true" not in meta, (
            f"real save-best path should not be marked fallback: {meta!r}"
        )
        assert "episode " in meta and "loss " in meta


def test_fallback_skipped_when_save_disabled():
    """save_best_checkpoint=False short-circuits the fallback too — no
    file should be written."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _base_cfg(
            tmp, episodes=8, curriculum_episodes=999, save_best_checkpoint=False
        )
        train_from_config(cfg)
        assert not os.path.exists(os.path.join(tmp, "checkpoint_best.eqx"))


def test_fallback_skipped_when_no_checkpoint_dir():
    """checkpoint_dir=None → save-best path is unreachable. Smoke-only."""
    cfg = _base_cfg(checkpoint_dir="", episodes=4, curriculum_episodes=999)
    cfg = cfg.with_overrides({"checkpoint_dir": None})
    # Just assert no crash.
    train_from_config(cfg)


@pytest.mark.parametrize("curriculum", [0, 50, 999])
def test_fallback_state_loadable(curriculum: int):
    """The state written by the fallback can be loaded back via
    eqx.tree_deserialise_leaves into a matching template."""
    import equinox as eqx
    import jax.random as jr

    from deqn_jax.models import load_model
    from deqn_jax.training.trainer import create_train_state

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _base_cfg(tmp, episodes=6, curriculum_episodes=curriculum)
        train_from_config(cfg)

        best_path = os.path.join(tmp, "checkpoint_best.eqx")
        assert os.path.exists(best_path), "checkpoint_best.eqx missing"

        model = load_model(cfg.model)
        template, _, _, _ = create_train_state(
            model,
            jr.PRNGKey(cfg.seed),
            hidden_sizes=cfg.network.hidden_sizes,
            batch_size=cfg.batch_size,
            n_equations=1,
            optimizer_config=cfg.optimizer,
            network_config=cfg.network,
            sim_batch=cfg.sim_batch,
            replay_config=cfg.replay_buffer,
        )
        restored = eqx.tree_deserialise_leaves(best_path, template)
        # Just check the params field came through as a callable Module.
        assert restored.params is not None
