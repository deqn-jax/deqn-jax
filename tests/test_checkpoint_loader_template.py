"""Loader-template completeness: fields that shape the TrainState pytree.

load_policy_from_checkpoint builds a template TrainState from the run's
config.yaml and deserializes leaves into it. Any config field that changes
the pytree (episode_state via sim_batch, the replay_state subtree via
replay_buffer) must be forwarded, or loading crashes on shape mismatch —
the silent-drop class caught 2026-07-11 with static network fields.
"""

import equinox as eqx
import jax
import pytest
import yaml

from deqn_jax.config import ReplayBufferConfig
from deqn_jax.irf import load_policy_from_checkpoint
from deqn_jax.models import load_model
from deqn_jax.training.trainer import create_train_state


def _save_run(tmp_path, sim_batch=None, replay_enabled=False):
    model = load_model("brock_mirman")
    replay_cfg = ReplayBufferConfig(enabled=replay_enabled, capacity=128)
    state, _, _ = create_train_state(
        model,
        jax.random.PRNGKey(0),
        hidden_sizes=(16,),
        batch_size=16,
        n_equations=len(model.equation_names),
        sim_batch=sim_batch,
        replay_config=replay_cfg,
    )
    ckpt = tmp_path / "checkpoint_000010.eqx"
    eqx.tree_serialise_leaves(str(ckpt), state)
    cfg = {
        "model": "brock_mirman",
        "batch_size": 16,
        "network": {"hidden_sizes": [16]},
    }
    if sim_batch is not None:
        cfg["sim_batch"] = sim_batch
    if replay_enabled:
        cfg["replay_buffer"] = {"enabled": True, "capacity": 128}
    with open(tmp_path / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    return str(ckpt)


class TestLoaderTemplateCompleteness:
    def test_plain_roundtrip(self, tmp_path):
        ckpt = _save_run(tmp_path)
        net, model = load_policy_from_checkpoint(ckpt)
        assert net is not None and model.name == "brock_mirman"

    def test_sim_batch_roundtrip(self, tmp_path):
        # sim_batch != batch_size changes episode_state's leading dim;
        # a template built without it fails leaf deserialization.
        ckpt = _save_run(tmp_path, sim_batch=48)
        net, _ = load_policy_from_checkpoint(ckpt)
        assert net is not None

    def test_replay_buffer_roundtrip(self, tmp_path):
        # replay_buffer.enabled adds the replay_state subtree to the pytree.
        ckpt = _save_run(tmp_path, replay_enabled=True)
        net, _ = load_policy_from_checkpoint(ckpt)
        assert net is not None


if __name__ == "__main__":
    pytest.main([__file__, "-q"])


def test_surrogate_state_roundtrip(tmp_path):
    """EWM world arm: aux_params (Ŵ + optimizer state) and target_params are
    part of the pytree; the loader template must rebuild them from the run's
    config.yaml or deserialization fails on shape/structure mismatch."""
    from deqn_jax.config import SurrogateConfig

    model = load_model("olg_lifecycle")
    sur = SurrogateConfig(enabled=True, width=8)
    state, _, _ = create_train_state(
        model,
        jax.random.PRNGKey(0),
        hidden_sizes=(16,),
        batch_size=16,
        n_equations=len(model.equation_names),
        surrogate_config=sur,
    )
    ckpt = tmp_path / "checkpoint_000010.eqx"
    eqx.tree_serialise_leaves(str(ckpt), state)
    with open(tmp_path / "config.yaml", "w") as f:
        yaml.safe_dump(
            {
                "model": "olg_lifecycle",
                "batch_size": 16,
                "network": {"hidden_sizes": [16]},
                "surrogate": {"enabled": True, "width": 8},
            },
            f,
        )
    net, _ = load_policy_from_checkpoint(str(ckpt))
    assert net is not None
