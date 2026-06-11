"""Tests for the rollout steady-state reset (`ss_reset_frac`).

Regression for the zombie-path bug: the reset used `ep_states.at[:n_reset]`
-- a FIXED index prefix -- so batch indices >= n_reset could never be
reset. Any path that diverged to the soft-clip ceiling during a training
transient then stayed in the batch forever (observed: 55/64 paths pinned
at k=100 for 1750+ episodes in the disaster model, training loss frozen
at ~1e-1 while the policy's true ergodic loss was 4e-5). Random reset
indices bound the zombie lifetime to ~1/frac episodes.
"""

import jax
import jax.numpy as jnp
import numpy as np

from deqn_jax.training.cycle import apply_ss_reset

N_STATES = 5
BATCH = 16
SS = jnp.arange(1.0, N_STATES + 1.0)  # distinctive SS vector
SENTINEL = 999.0


def _reset_mask(key, frac=0.25):
    """Boolean mask of which batch rows were touched by one reset call."""
    states = jnp.full((BATCH, N_STATES), SENTINEL)
    out = apply_ss_reset(key, states, frac, SS)
    return np.asarray(jnp.any(out != SENTINEL, axis=1))


def test_reset_touches_exactly_n_reset_rows():
    mask = _reset_mask(jax.random.PRNGKey(0), frac=0.25)
    assert mask.sum() == int(0.25 * BATCH)


def test_reset_rows_are_near_ss():
    states = jnp.full((BATCH, N_STATES), SENTINEL)
    out = apply_ss_reset(jax.random.PRNGKey(1), states, 0.25, SS)
    touched = np.asarray(jnp.any(out != SENTINEL, axis=1))
    reset_rows = np.asarray(out)[touched]
    assert np.all(np.abs(reset_rows / np.asarray(SS)[None, :] - 1.0) <= 0.05)


def test_reset_eventually_covers_every_index():
    """Over many keys, every batch index must be resettable.

    The fixed-prefix implementation only ever resets indices
    [0, n_reset), leaving the rest of the batch immortal -- this is the
    zombie-path regression test.
    """
    covered = np.zeros(BATCH, dtype=bool)
    for seed in range(200):
        covered |= _reset_mask(jax.random.PRNGKey(seed), frac=0.25)
    assert covered.all(), (
        f"indices never reset across 200 keys: {np.flatnonzero(~covered).tolist()}"
    )


def test_reset_noop_when_frac_zero():
    states = jnp.full((BATCH, N_STATES), SENTINEL)
    out = apply_ss_reset(jax.random.PRNGKey(2), states, 0.0, SS)
    assert np.all(np.asarray(out) == SENTINEL)


def test_reset_traceable_under_jit():
    @jax.jit
    def f(key, states):
        return apply_ss_reset(key, states, 0.25, SS)

    out = f(jax.random.PRNGKey(3), jnp.full((BATCH, N_STATES), SENTINEL))
    assert np.isfinite(np.asarray(out)).all()
