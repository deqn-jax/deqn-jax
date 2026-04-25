"""Tests for checkpoint resume, including optimizer switching on resume.

Simon's requested use case: train for a while with one optimizer,
stop, and restart training with a different optimizer. The checkpoint
should preserve the network weights (and training history) while the
optimizer state is freshly initialized for the new optimizer.
"""

import os

import equinox as eqx
import jax
import jax.numpy as jnp

from deqn_jax.config import NetworkConfig, OptimizerConfig, TrainConfig
from deqn_jax.training.trainer import train_from_config


def _tiny_config(
    optimizer_name: str, episodes: int, checkpoint_dir: str
) -> TrainConfig:
    """Configs small enough to train in seconds but exercise the full pipeline."""
    return TrainConfig(
        model="brock_mirman",
        episodes=episodes,
        batch_size=16,
        episode_length=10,
        mc_samples=2,
        seed=0,
        log_every=10,
        verbose=False,
        warm_start=False,
        network=NetworkConfig(
            type="mlp",
            hidden_sizes=(8,),
            activation="tanh",
        ),
        optimizer=OptimizerConfig(
            name=optimizer_name,
            learning_rate=1e-3,
        ),
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=episodes,  # save once, at the end
        max_checkpoints=1,
    )


def _latest_checkpoint(checkpoint_dir: str) -> str:
    files = sorted(
        f
        for f in os.listdir(checkpoint_dir)
        if f.startswith("checkpoint_") and f.endswith(".eqx")
    )
    assert files, f"No checkpoints saved in {checkpoint_dir}"
    return os.path.join(checkpoint_dir, files[-1])


def _params_fingerprint(params):
    """Flat parameter vector for equality checks across runs.

    Accepts an Equinox module (train_from_config returns state.params,
    which is the MLP / LinearPlusMLP itself).
    """
    leaves = jax.tree_util.tree_leaves(eqx.filter(params, eqx.is_array))
    if not leaves:
        return jnp.zeros(0)
    return jnp.concatenate([leaf.flatten() for leaf in leaves])


class TestCheckpointResume:
    def test_resume_same_optimizer(self, tmp_path):
        """Baseline: resuming with the same optimizer trains further."""
        ckpt_dir = str(tmp_path / "run1")
        cfg1 = _tiny_config("adam", episodes=20, checkpoint_dir=ckpt_dir)
        params1, _ = train_from_config(cfg1)
        fp1 = _params_fingerprint(params1)

        ckpt = _latest_checkpoint(ckpt_dir)
        assert os.path.exists(ckpt)

        cfg2 = _tiny_config("adam", episodes=40, checkpoint_dir=ckpt_dir)
        cfg2 = cfg2.model_copy(update={"resume": ckpt})
        params2, _ = train_from_config(cfg2)
        fp2 = _params_fingerprint(params2)

        assert jnp.all(jnp.isfinite(fp2))
        # Training continued: weights moved from the saved checkpoint
        assert jnp.abs(fp1 - fp2).max() > 0.0

    def test_resume_with_different_optimizer(self, tmp_path):
        """Restart an adam checkpoint under NGD, keeping the weights."""
        ckpt_dir = str(tmp_path / "run2")

        cfg_adam = _tiny_config("adam", episodes=20, checkpoint_dir=ckpt_dir)
        params_adam, _ = train_from_config(cfg_adam)
        fp_adam = _params_fingerprint(params_adam)
        ckpt = _latest_checkpoint(ckpt_dir)

        # Switching optimizer on resume: the trainer should auto-detect the
        # optimizer name change and re-initialise opt_state while keeping the
        # network weights. Config.yaml from the first run is auto-read from
        # the checkpoint directory to reconstruct the pytree template.
        cfg_ngd = _tiny_config("ngd", episodes=40, checkpoint_dir=ckpt_dir)
        cfg_ngd = cfg_ngd.model_copy(update={"resume": ckpt})
        params_ngd, _ = train_from_config(cfg_ngd)
        fp_ngd = _params_fingerprint(params_ngd)

        # Sanity: finite weights, architecture preserved (same total count),
        # training actually moved weights from the adam-final state.
        assert jnp.all(jnp.isfinite(fp_ngd))
        assert fp_adam.shape == fp_ngd.shape
        assert jnp.abs(fp_adam - fp_ngd).max() > 0.0
