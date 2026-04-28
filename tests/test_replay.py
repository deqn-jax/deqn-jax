"""Unit tests for the prioritized state-replay buffer."""

import os
import tempfile

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from deqn_jax.config import TrainConfig
from deqn_jax.models import load_model
from deqn_jax.training import replay
from deqn_jax.training.trainer import create_train_state
from deqn_jax.types import make_replay_state

# ---------------------------------------------------------------------------
# ReplayState construction
# ---------------------------------------------------------------------------


def test_make_replay_state_shapes():
    """Initial state has correct shapes, dtypes, zeros, and counters."""
    state = make_replay_state(capacity=32, n_states=5)
    assert state.buffer.shape == (32, 5)
    assert state.priorities.shape == (32,)
    assert state.buffer.dtype == jnp.float32
    assert state.priorities.dtype == jnp.float32
    assert int(state.write_idx) == 0
    assert int(state.n_filled) == 0
    assert jnp.all(state.buffer == 0)
    assert jnp.all(state.priorities == 0)


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


def test_write_basic():
    """A single write updates buffer, priorities, write_idx, and n_filled."""
    state = make_replay_state(capacity=8, n_states=3)
    samples = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    prios = jnp.array([1.0, 2.0, 3.0, 4.0])
    state = replay.write(state, samples, prios)
    assert int(state.write_idx) == 4
    assert int(state.n_filled) == 4
    np.testing.assert_array_equal(np.asarray(state.buffer[:4]), np.asarray(samples))
    np.testing.assert_array_equal(np.asarray(state.priorities[:4]), np.asarray(prios))
    # Unwritten tail still zero
    assert jnp.all(state.buffer[4:] == 0)


def test_write_ring_wraps():
    """Writes past capacity overwrite the oldest rows; write_idx wraps."""
    state = make_replay_state(capacity=8, n_states=2)
    # First write: 5 rows of value 1
    state = replay.write(
        state,
        jnp.ones((5, 2), dtype=jnp.float32),
        jnp.ones((5,), dtype=jnp.float32),
    )
    assert int(state.write_idx) == 5
    assert int(state.n_filled) == 5
    # Second write: 5 rows of value 2; positions 5,6,7,0,1 written
    state = replay.write(
        state,
        2.0 * jnp.ones((5, 2), dtype=jnp.float32),
        2.0 * jnp.ones((5,), dtype=jnp.float32),
    )
    assert int(state.write_idx) == 2  # (5 + 5) mod 8
    assert int(state.n_filled) == 8  # capped
    # Positions 5,6,7,0,1 should be 2.0; positions 2,3,4 remain 1.0
    buf = np.asarray(state.buffer)
    assert (buf[5:8] == 2.0).all()
    assert (buf[0:2] == 2.0).all()
    assert (buf[2:5] == 1.0).all()


def test_write_oversized_batch():
    """A single write of N > capacity rows keeps only the LAST capacity rows."""
    state = make_replay_state(capacity=4, n_states=2)
    # 10 rows; only the last 4 should land.
    samples = jnp.arange(20, dtype=jnp.float32).reshape(10, 2)  # rows 0..9
    prios = jnp.arange(10, dtype=jnp.float32)
    state = replay.write(state, samples, prios)
    assert int(state.n_filled) == 4
    # Check that some of the trailing input is in the buffer (positions
    # depend on how the implementation slices; just verify last-row-presence).
    buf_set = set(map(int, state.buffer[:, 0]))
    assert 18 in buf_set  # row 9 starts with 18
    assert 16 in buf_set  # row 8


def test_write_priority_set():
    """Priorities array gets exactly the values passed to write."""
    state = make_replay_state(capacity=8, n_states=2)
    prios = jnp.array([0.5, 1.5, 2.5, 3.5])
    state = replay.write(
        state,
        jnp.zeros((4, 2), dtype=jnp.float32),
        prios,
    )
    np.testing.assert_array_equal(np.asarray(state.priorities[:4]), np.asarray(prios))


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------


def test_sample_uniform_when_priorities_equal():
    """alpha>0 with equal priorities recovers (approximately) uniform draw."""
    state = make_replay_state(capacity=64, n_states=2)
    samples_in = jnp.arange(128, dtype=jnp.float32).reshape(64, 2)
    prios = jnp.ones((64,), dtype=jnp.float32)
    state = replay.write(state, samples_in, prios)

    key = jr.PRNGKey(0)
    drawn, _ = replay.sample(state, key, n=10000, alpha=0.6, eps=1e-6)
    # Each row's first column is its row index * 2; check that all 64 rows are
    # represented and counts are within reasonable spread of uniform.
    counts = np.bincount(np.asarray(drawn[:, 0]).astype(int) // 2, minlength=64)
    assert counts.min() > 0
    # Uniform expectation = 10000/64 ≈ 156. Tolerate 3x dev (sampling is noisy).
    assert counts.max() < 156 * 3


def test_sample_skewed_when_priorities_skewed():
    """High-priority entries are oversampled relative to low-priority ones."""
    state = make_replay_state(capacity=8, n_states=1)
    samples_in = jnp.arange(8, dtype=jnp.float32).reshape(8, 1)
    # Row 7 has priority 100x larger than the rest.
    prios = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0])
    state = replay.write(state, samples_in, prios)

    drawn, _ = replay.sample(state, jr.PRNGKey(7), n=2000, alpha=0.6, eps=1e-6)
    counts = np.bincount(np.asarray(drawn[:, 0]).astype(int), minlength=8)
    # Row 7 should beat each of the others by a wide margin.
    assert counts[7] > 5 * counts[:7].max()


def test_sample_respects_n_filled():
    """Samples never come from the unfilled tail of the buffer."""
    state = make_replay_state(capacity=16, n_states=1)
    # Fill only the first 4 rows. The tail (rows 4..15) is still zeros — but
    # rows 0..3 are explicitly nonzero so we can detect tail leaks.
    state = replay.write(
        state,
        jnp.array([[10.0], [20.0], [30.0], [40.0]], dtype=jnp.float32),
        jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32),
    )
    drawn, _ = replay.sample(state, jr.PRNGKey(1), n=200, alpha=0.6, eps=1e-6)
    drawn_vals = set(map(float, drawn[:, 0]))
    assert drawn_vals.issubset({10.0, 20.0, 30.0, 40.0}), (
        f"sampled outside filled prefix: {drawn_vals}"
    )


def test_sample_when_empty_does_not_crash():
    """Buffer empty + sample still returns an array of the right shape (caller gates)."""
    state = make_replay_state(capacity=8, n_states=2)
    # Without crashing — tests the eps / weights_sum stabilization.
    drawn, _ = replay.sample(state, jr.PRNGKey(2), n=4, alpha=0.6, eps=1e-6)
    assert drawn.shape == (4, 2)


# ---------------------------------------------------------------------------
# is_warm
# ---------------------------------------------------------------------------


def test_is_warm_threshold():
    """is_warm flips True only after n_filled >= floor(capacity * min_fill_frac)."""
    state = make_replay_state(capacity=10, n_states=1)
    assert not replay.is_warm(state, 10, 0.5)
    state = replay.write(
        state,
        jnp.arange(4, dtype=jnp.float32).reshape(4, 1),
        jnp.ones((4,), dtype=jnp.float32),
    )
    assert not replay.is_warm(state, 10, 0.5)  # 4 < 5
    state = replay.write(
        state,
        jnp.arange(4, dtype=jnp.float32).reshape(4, 1),
        jnp.ones((4,), dtype=jnp.float32),
    )
    assert replay.is_warm(state, 10, 0.5)  # 8 >= 5


# ---------------------------------------------------------------------------
# compute_priorities
# ---------------------------------------------------------------------------


def test_compute_priorities_finite_on_disaster():
    """compute_priorities returns finite, non-negative [N] for the disaster model."""
    model = load_model("disaster")
    assert model.steady_state_fn is not None
    ss_state, _ = model.steady_state_fn(model.constants)
    states = jnp.broadcast_to(ss_state[None, :], (8, model.n_states))
    # Build a minimal MLP with the right shape.
    from deqn_jax.networks.mlp import MLP

    net = MLP(
        in_features=model.n_states,
        out_features=model.n_policies,
        hidden_sizes=(8,),
        activations=(jax.nn.tanh,),
        key=jr.PRNGKey(0),
    )
    prios = replay.compute_priorities(model, net, states, jr.PRNGKey(1))
    assert prios.shape == (8,)
    assert bool(jnp.all(jnp.isfinite(prios)))
    assert bool(jnp.all(prios >= 0))


# ---------------------------------------------------------------------------
# Serialization round-trip (checkpoint resume)
# ---------------------------------------------------------------------------


def test_replay_state_serialization_roundtrip():
    """eqx.tree_serialise_leaves on a TrainState with replay preserves all fields."""
    cfg_dict = {
        "model": "brock_mirman",
        "episodes": 1,
        "episode_length": 4,
        "batch_size": 8,
        "sim_batch": 8,
        "mc_samples": 1,
        "fp64": False,
        "verbose": False,
        "network": {"type": "mlp", "hidden_sizes": [8]},
        "optimizer": {"name": "adam", "learning_rate": 1e-3},
        "replay_buffer": {
            "enabled": True,
            "capacity": 16,
            "min_fill_frac": 0.0,
            "mix_ratio": 0.5,
        },
    }
    cfg = TrainConfig.model_validate(cfg_dict)
    model = load_model(cfg.model)
    state, _, _, _ = create_train_state(
        model,
        jr.PRNGKey(0),
        hidden_sizes=cfg.network.hidden_sizes,
        batch_size=cfg.batch_size,
        n_equations=1,
        optimizer_config=cfg.optimizer,
        network_config=cfg.network,
        sim_batch=cfg.sim_batch,
        replay_config=cfg.replay_buffer,
    )
    # Mutate the buffer so serialization round-trip has something to compare.
    samples = jnp.arange(16, dtype=jnp.float32).reshape(8, 2)
    prios = jnp.arange(8, dtype=jnp.float32) + 0.5
    new_replay = replay.write(state.replay_state, samples, prios)
    state = state._replace(replay_state=new_replay)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.eqx")
        eqx.tree_serialise_leaves(path, state)

        # Build a fresh template for deserialization.
        template_state, _, _, _ = create_train_state(
            model,
            jr.PRNGKey(0),
            hidden_sizes=cfg.network.hidden_sizes,
            batch_size=cfg.batch_size,
            n_equations=1,
            optimizer_config=cfg.optimizer,
            network_config=cfg.network,
            sim_batch=cfg.sim_batch,
            replay_config=cfg.replay_buffer,
        )
        restored = eqx.tree_deserialise_leaves(path, template_state)

    np.testing.assert_array_equal(
        np.asarray(restored.replay_state.buffer),
        np.asarray(state.replay_state.buffer),
    )
    np.testing.assert_array_equal(
        np.asarray(restored.replay_state.priorities),
        np.asarray(state.replay_state.priorities),
    )
    assert int(restored.replay_state.write_idx) == int(state.replay_state.write_idx)
    assert int(restored.replay_state.n_filled) == int(state.replay_state.n_filled)


# ---------------------------------------------------------------------------
# Disabled-path: replay_state stays None
# ---------------------------------------------------------------------------


def test_replay_disabled_state_is_none():
    """When replay_buffer.enabled is False, TrainState.replay_state is None."""
    cfg = TrainConfig.model_validate(
        {
            "model": "brock_mirman",
            "episodes": 1,
            "episode_length": 4,
            "batch_size": 8,
            "sim_batch": 8,
            "mc_samples": 1,
            "fp64": False,
            "verbose": False,
            "network": {"type": "mlp", "hidden_sizes": [8]},
            "optimizer": {"name": "adam", "learning_rate": 1e-3},
            # default replay_buffer (enabled=False)
        }
    )
    model = load_model(cfg.model)
    state, _, _, _ = create_train_state(
        model,
        jr.PRNGKey(0),
        hidden_sizes=cfg.network.hidden_sizes,
        batch_size=cfg.batch_size,
        n_equations=1,
        optimizer_config=cfg.optimizer,
        network_config=cfg.network,
        sim_batch=cfg.sim_batch,
        replay_config=cfg.replay_buffer,
    )
    assert state.replay_state is None


# ---------------------------------------------------------------------------
# Validator guards
# ---------------------------------------------------------------------------


def test_validator_rejects_replay_with_sequence_network():
    """history_len > 1 + replay enabled raises NotImplementedError."""
    from deqn_jax.training.trainer import _validate_train_config

    cfg = TrainConfig.model_validate(
        {
            "model": "brock_mirman",
            "episodes": 2,
            "episode_length": 4,
            "batch_size": 8,
            "sim_batch": 8,
            "mc_samples": 1,
            "fp64": False,
            "verbose": False,
            "network": {"type": "lstm", "hidden_sizes": [8], "history_len": 4},
            "replay_buffer": {"enabled": True},
        }
    )
    with pytest.raises(NotImplementedError, match="history_len"):
        _validate_train_config(cfg)


def test_validator_rejects_replay_with_sorted_within_batch():
    """sorted_within_batch + replay enabled raises ValueError."""
    from deqn_jax.training.trainer import _validate_train_config

    cfg = TrainConfig.model_validate(
        {
            "model": "brock_mirman",
            "episodes": 2,
            "episode_length": 4,
            "batch_size": 8,
            "sim_batch": 8,
            "mc_samples": 1,
            "fp64": False,
            "verbose": False,
            "network": {"type": "mlp", "hidden_sizes": [8]},
            "replay_buffer": {"enabled": True},
            "sorted_within_batch": True,
        }
    )
    with pytest.raises(ValueError, match="sorted_within_batch"):
        _validate_train_config(cfg)
