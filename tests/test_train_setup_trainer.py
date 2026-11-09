"""train_from_config: return value on a finished resume, logger lifetime, and
the speed in the final banner."""

import os

import equinox as eqx
import jax
import numpy as np
import pytest

from deqn_jax.config import TrainConfig
from deqn_jax.training import trainer
from deqn_jax.training.checkpointing import load_policy_from_checkpoint


def _cfg(**over) -> TrainConfig:
    base = {
        "model": "brock_mirman",
        "episodes": 3,
        "batch_size": 16,
        "seed": 0,
        "verbose": False,
        "network": {"type": "mlp", "hidden_sizes": [16]},
    }
    base.update(over)
    return TrainConfig.from_dict(base)


class _Logger:
    def __init__(self):
        self.closed = False

    def log(self, *a, **k):
        pass

    def close(self):
        self.closed = True


@pytest.fixture
def loggers(monkeypatch):
    made = []

    def _create_logger(**kwargs):
        made.append(_Logger())
        return made[-1]

    monkeypatch.setattr(trainer, "create_logger", _create_logger)
    return made


@pytest.fixture
def finished_run(tmp_path):
    ckpt_dir = str(tmp_path / "ck")
    trainer.train_from_config(
        _cfg(checkpoint_dir=ckpt_dir, checkpoint_every=3, save_best_checkpoint=False)
    )
    return os.path.join(ckpt_dir, "checkpoint_000003.eqx")


def _arrays(net):
    return [np.asarray(a) for a in jax.tree.leaves(eqx.filter(net, eqx.is_array))]


def test_resume_of_finished_run_returns_loaded_policy(finished_run, loggers):
    result = trainer.train_from_config(_cfg(resume=finished_run))

    assert result is not None
    params, history = result
    loaded, _ = load_policy_from_checkpoint(finished_run)
    for a, b in zip(_arrays(params), _arrays(loaded), strict=True):
        np.testing.assert_array_equal(a, b)
    assert history["loss"] == []
    assert all(lg.closed for lg in loggers)


def _final_banner_episodes(monkeypatch, cfg) -> int:
    seen = {}
    monkeypatch.setattr(trainer, "_print_final", lambda **kw: seen.update(kw))
    trainer.train_from_config(cfg)
    return seen["episodes"]


def test_final_banner_counts_episodes_run_after_early_stop(monkeypatch):
    # min_delta this large: episode 1 sets the best, episode 2 cannot
    # improve on it, patience 1 stops the run there.
    cfg = _cfg(
        episodes=20,
        verbose=True,
        early_stop_patience=1,
        early_stop_min_delta=1e9,
    )
    assert _final_banner_episodes(monkeypatch, cfg) == 2


def test_final_banner_counts_episodes_run_after_resume(monkeypatch, finished_run):
    cfg = _cfg(episodes=5, verbose=True, resume=finished_run)
    assert _final_banner_episodes(monkeypatch, cfg) == 2
