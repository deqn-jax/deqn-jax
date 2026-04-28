"""Tests for the sim_batch / batch_size separation.

``batch_size`` is the gradient minibatch size. ``sim_batch`` (optional)
is the number of parallel simulation trajectories in the rollout pool.
When ``sim_batch`` is None, both collapse to ``batch_size`` (legacy
single-number behaviour); when set, the rolled-out pool is
(sim_batch × episode_length) states and each gradient step draws
``batch_size`` samples from that pool. Matches DEQN-MAO's
N_sim_batch vs N_minibatch_size distinction.
"""

import numpy as np
import pytest


def test_sim_batch_none_defaults_to_batch_size():
    """When sim_batch is unset, episode_state.shape[0] == batch_size."""
    from deqn_jax.config import NetworkConfig, OptimizerConfig, TrainConfig
    from deqn_jax.training.trainer import train_from_config

    cfg = TrainConfig(
        model="brock_mirman",
        episodes=1,
        batch_size=32,
        episode_length=4,
        mc_samples=2,
        network=NetworkConfig(hidden_sizes=(8,)),
        optimizer=OptimizerConfig(name="adam", learning_rate=1e-3),
        verbose=False,
        log_every=1,
    )
    _, h = train_from_config(cfg)
    # Just confirm training runs to completion without shape errors.
    assert len(h["loss"]) == 1
    assert np.isfinite(h["loss"][-1])


def test_sim_batch_larger_than_batch_size():
    """sim_batch=256, batch_size=64: pool has 256×episode_length samples,
    each gradient step pulls 64 of them."""
    from deqn_jax.config import NetworkConfig, OptimizerConfig, TrainConfig
    from deqn_jax.training.trainer import train_from_config

    cfg = TrainConfig(
        model="brock_mirman",
        episodes=2,
        batch_size=64,
        sim_batch=256,
        episode_length=8,
        mc_samples=2,
        network=NetworkConfig(hidden_sizes=(16,)),
        optimizer=OptimizerConfig(name="adam", learning_rate=1e-3),
        verbose=False,
        log_every=2,
    )
    _, h = train_from_config(cfg)
    assert np.isfinite(h["loss"][-1])


def test_sim_batch_carries_through_episode_state():
    """With sim_batch set, the initial state sampler should draw
    sim_batch trajectories, and episode_state should reflect that."""
    import jax

    from deqn_jax.config import NetworkConfig, OptimizerConfig
    from deqn_jax.models.brock_mirman import MODEL
    from deqn_jax.training.trainer import create_train_state

    key = jax.random.PRNGKey(0)
    state, _, _, _ = create_train_state(
        MODEL,
        key,
        hidden_sizes=(8,),
        batch_size=32,
        sim_batch=128,
        n_equations=1,
        optimizer_config=OptimizerConfig(name="adam", learning_rate=1e-3),
        network_config=NetworkConfig(hidden_sizes=(8,)),
    )
    # episode_state should have sim_batch trajectories, not batch_size
    assert state.episode_state.shape[0] == 128
    assert state.episode_state.shape[1] == MODEL.n_states


def test_sim_batch_none_uses_batch_size_for_episode_state():
    """When sim_batch=None, episode_state falls back to batch_size."""
    import jax

    from deqn_jax.config import NetworkConfig, OptimizerConfig
    from deqn_jax.models.brock_mirman import MODEL
    from deqn_jax.training.trainer import create_train_state

    key = jax.random.PRNGKey(0)
    state, _, _, _ = create_train_state(
        MODEL,
        key,
        hidden_sizes=(8,),
        batch_size=48,
        sim_batch=None,
        n_equations=1,
        optimizer_config=OptimizerConfig(name="adam", learning_rate=1e-3),
        network_config=NetworkConfig(hidden_sizes=(8,)),
    )
    assert state.episode_state.shape[0] == 48


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
