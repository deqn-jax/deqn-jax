"""Tests for the sorted_within_batch flag.

When True, minibatches must be contiguous slices of single trajectories
rather than IID-shuffled samples. Verified via a stubbed grad_step that
records which samples land in each minibatch, then checking the layout
property directly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _run_cycle_capture(sorted_flag: bool, seed: int = 0):
    """Run one cycle of _make_cycle_step with a no-op grad_step that
    captures each minibatch, and return the list of minibatches observed."""
    from deqn_jax.models.brock_mirman import MODEL
    from deqn_jax.training.cycle import make_cycle_step as _make_cycle_step
    from deqn_jax.types import Metrics, TrainState, make_reweight_state

    episode_length = 8
    batch_size = 4            # = minibatch size; 8 samples per trajectory
    sim_batch = 3             # 3 parallel trajectories

    # Fake trajectory with identifiable entries: entry at [t, b, :] encodes
    # b and t as state values so we can detect layout.
    # trajectory[t, b, :] = [b, t]  with n_states = 2.
    # We use MODEL.brock_mirman (n_states=2) for signature compatibility.
    t_grid = jnp.arange(episode_length, dtype=jnp.float32)
    b_grid = jnp.arange(sim_batch, dtype=jnp.float32)
    T, B = jnp.meshgrid(t_grid, b_grid, indexing="ij")   # [T, B]
    trajectory = jnp.stack([B, T], axis=-1)              # [T, B, 2]
    assert trajectory.shape == (episode_length, sim_batch, 2)

    # Stub rollout_fn: returns the prepared trajectory and a fake final state.
    # Accepts shock_scale for API compatibility; unused in this unit test.
    # 4-tuple matches the real rollout_fn post history-state plumbing:
    # (trajectory, final_state, final_history, new_key). final_history
    # is None for this MLP-shaped test fixture.
    def rollout_fn(state, shock_scale=jnp.array(1.0)):
        return trajectory, trajectory[-1], None, state.key

    # Capture each minibatch.
    observed = []

    def grad_step(state, minibatch, lr_scale, shock_scale):
        observed.append(np.asarray(minibatch))
        return state, Metrics(
            loss=jnp.float32(0.0),
            residuals={"euler": jnp.float32(0.0)},
            grad_norm=jnp.float32(0.0),
        )

    cycle = _make_cycle_step(
        rollout_fn=rollout_fn,
        grad_step=grad_step,
        model=MODEL,
        batch_size=batch_size,
        n_epochs_per_rollout=1,
        n_minibatches_per_epoch=None,
        history_len=1,
        sorted_within_batch=sorted_flag,
    )

    # Build a minimal TrainState
    state = TrainState(
        params=None,
        opt_state=None,
        episode_state=trajectory[0],
        key=jax.random.PRNGKey(seed),
        step=0,
        episode=0,
        loss_weights=jnp.ones(1),
        reweight_state=make_reweight_state(1),
    )

    cycle(state, jnp.float32(1.0))
    return observed


def test_sorted_false_is_iid_shuffle():
    """With sorted_within_batch=False, minibatches are IID-shuffled mixes."""
    minibatches = _run_cycle_capture(sorted_flag=False, seed=0)

    # Each minibatch of size 4 should have mixed trajectory ids + mixed times
    # (probability of a single-trajectory minibatch by chance is very low).
    # Check at least one minibatch contains samples from multiple trajectories.
    multi_traj_batches = 0
    for mb in minibatches:
        traj_ids = set(np.unique(mb[:, 0]).tolist())
        if len(traj_ids) > 1:
            multi_traj_batches += 1
    assert multi_traj_batches >= 1, \
        "With sorted=False, expected minibatches to mix trajectories; none did."


def test_sorted_true_gives_single_trajectory_contiguous_segments():
    """With sorted_within_batch=True, each minibatch comes from ONE trajectory
    and is a contiguous temporal segment."""
    minibatches = _run_cycle_capture(sorted_flag=True, seed=0)

    assert len(minibatches) >= 1

    for i, mb in enumerate(minibatches):
        # mb is shape [batch_size, 2]: column 0 = trajectory id, col 1 = time.
        traj_ids = np.unique(mb[:, 0])
        assert traj_ids.size == 1, \
            f"minibatch {i} spans multiple trajectories: {traj_ids}"

        times = mb[:, 1]
        # Times should be consecutive integers (contiguous segment).
        sorted_times = np.sort(times)
        assert np.all(np.diff(sorted_times) == 1), \
            f"minibatch {i} times are not contiguous: {times}"


def test_sorted_true_shuffles_batch_order_across_seeds():
    """The *order* of batches should vary with seed, even though each batch
    is a contiguous trajectory segment."""
    a = _run_cycle_capture(sorted_flag=True, seed=0)
    b = _run_cycle_capture(sorted_flag=True, seed=42)

    # Extract the trajectory id of each minibatch.
    a_order = [int(np.unique(mb[:, 0])[0]) for mb in a]
    b_order = [int(np.unique(mb[:, 0])[0]) for mb in b]

    # Different seeds should yield different orderings (probabilistic; with
    # only 3 trajectories there's a 1/6 chance of collision — retry a few
    # times with different seed pairs if this turns out flaky in CI).
    # 3 distinct trajectories means 6 possible orderings; a single trial has
    # prob 1/6 ≈ 17% of colliding. Use two checks with different seeds to
    # drive collision probability to (1/6)^2 ≈ 3%.
    c = _run_cycle_capture(sorted_flag=True, seed=99)
    c_order = [int(np.unique(mb[:, 0])[0]) for mb in c]

    orders = {tuple(a_order), tuple(b_order), tuple(c_order)}
    assert len(orders) >= 2, \
        f"Expected distinct batch orderings across seeds, got {orders}"


def test_sorted_within_batch_config_field_plumbed_end_to_end():
    """End-to-end: training works with both values of the flag and produces
    finite losses. Smoke test at the real train_from_config level."""
    from deqn_jax.config import NetworkConfig, OptimizerConfig, TrainConfig
    from deqn_jax.training.trainer import train_from_config

    base = dict(
        model="brock_mirman",
        episodes=3,
        batch_size=16,
        episode_length=8,
        mc_samples=2,
        network=NetworkConfig(hidden_sizes=(8,)),
        optimizer=OptimizerConfig(name="adam", learning_rate=1e-3),
        verbose=False,
        log_every=3,
    )

    # sorted_within_batch=False (default)
    _, h_false = train_from_config(TrainConfig(**base, sorted_within_batch=False))
    assert np.isfinite(h_false["loss"][-1])

    # sorted_within_batch=True
    _, h_true = train_from_config(TrainConfig(**base, sorted_within_batch=True))
    assert np.isfinite(h_true["loss"][-1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
