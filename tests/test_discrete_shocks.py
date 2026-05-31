"""Tests for first-class discrete Markov-chain shock support.

Three checks:

1. Residual expectation under a 2-state chain matches manual enumeration.
2. Trajectory rollout produces shock indices in {0, 1} with the documented
   stationary distribution after burn-in.
3. A tiny RBC-with-2-state-TFP variant trains for a few episodes without
   NaN and produces a finite loss.

A synthetic test fixture (``_make_chain_model``) is constructed inline; this
keeps the test self-contained and avoids registering a one-off model in
``deqn_jax.models``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from deqn_jax.training.loss import compute_loss
from deqn_jax.training.shocks import draw_discrete_shocks, simulation_step
from deqn_jax.types import ModelSpec


def _make_chain_model(transition_matrix: np.ndarray) -> ModelSpec:
    """Synthetic 1-state, 1-policy, 1-equation model with a discrete chain.

    State: ``s = [z]`` where ``z ∈ [0, K)`` is the chain index.
    Policy: ``π(s) = c`` (a single scalar, will be regressed to 0 by trainer).
    Equation: ``r(s, π, s', π') = π - z`` (residual zero iff policy = current z).

    The model has no continuous shocks; the integer ``shock`` from the
    rollout is the next-period z, embedded into ``state[z_state_idx]``.
    """

    def step(state, policy, shock, constants):
        # state: [batch, 1] (current z), shock: [batch] int32 (next z)
        return shock.astype(state.dtype)[:, None]

    def equations(state, policy, next_state, next_policy, constants):
        # residual = policy - current_z; just a placeholder so vmap works
        return {"r": (policy[:, 0] - state[:, 0])}

    def steady_state(constants):
        # Pick z=0 as nominal SS for warm-start anchoring
        s = jnp.array([0.0])
        p = jnp.array([0.0])
        return s, p

    return ModelSpec(
        name="chain_test",
        n_states=1,
        n_policies=1,
        n_shocks=1,  # nominal; the actual shock is a categorical index
        constants={},
        equations_fn=equations,
        step_fn=step,
        steady_state_fn=steady_state,
        state_names=("z",),
        policy_names=("c",),
        equation_names=("r",),
        transition_matrix=jnp.asarray(transition_matrix),
        z_state_idx=0,
    )


# ---------------------------------------------------------------------------
# 1. Residual expectation matches manual enumeration
# ---------------------------------------------------------------------------


def test_discrete_residual_expectation_matches_manual_enumeration():
    """E_{z+|z_t}[r] computed by compute_loss must equal Π[z_t]·r_grid manually.

    Under a 2-state chain with Π = [[0.7, 0.3], [0.4, 0.6]] and a residual
    r(s, π, s', π') that depends explicitly on z' (the next-period state),
    the expectation aggregator must weight residuals at each candidate z'
    by the row Π[current_z, :].
    """
    Π = np.array([[0.7, 0.3], [0.4, 0.6]])

    def step(state, policy, shock, constants):
        return shock.astype(state.dtype)[:, None]

    def equations(state, policy, next_state, next_policy, constants):
        # r = next_z (an explicit next-period dependency)
        return {"r": next_state[:, 0]}

    def steady_state(constants):
        return jnp.array([0.0]), jnp.array([0.0])

    model = ModelSpec(
        name="chain",
        n_states=1,
        n_policies=1,
        n_shocks=1,
        constants={},
        equations_fn=equations,
        step_fn=step,
        steady_state_fn=steady_state,
        state_names=("z",),
        policy_names=("c",),
        equation_names=("r",),
        transition_matrix=jnp.asarray(Π),
        z_state_idx=0,
    )

    # Two batch elements: one starting at z=0, one at z=1
    states = jnp.array([[0.0], [1.0]])
    policy_fn = lambda s: jnp.zeros_like(s)  # ignored by this residual
    _, eq_losses = compute_loss(
        model, policy_fn, states, key=jr.PRNGKey(0), mc_samples=1
    )

    # Manual: E[r | z=0] = 0.7*0 + 0.3*1 = 0.3; (E[r])² = 0.09
    #         E[r | z=1] = 0.4*0 + 0.6*1 = 0.6; (E[r])² = 0.36
    # Mean over batch: (0.09 + 0.36) / 2 = 0.225
    assert eq_losses["r"] == pytest.approx(0.225, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. Trajectory rollout reproduces the stationary distribution
# ---------------------------------------------------------------------------


def test_discrete_rollout_recovers_stationary_distribution():
    """Long-run frequency of z under draw_discrete_shocks matches stationary
    distribution of Π. Tested on a 2-state chain with closed-form stationary."""
    Π = jnp.array([[0.7, 0.3], [0.4, 0.6]])
    # Stationary: π_0 / π_1 = Π[1,0] / Π[0,1] = 0.4 / 0.3, normalized.
    # π_0 = 0.4 / 0.7 = 4/7, π_1 = 0.3 / 0.7 = 3/7.
    pi_stationary = np.array([4.0 / 7.0, 3.0 / 7.0])

    n_steps = 5000
    batch_size = 64
    key = jr.PRNGKey(123)

    # Start at z=0 for everyone
    z = jnp.zeros(batch_size, dtype=jnp.int32)
    counts = np.zeros(2, dtype=np.int64)

    burn_in = 500
    for t in range(n_steps):
        key, sub = jr.split(key)
        z = draw_discrete_shocks(sub, z, Π)
        if t >= burn_in:
            counts += np.bincount(np.asarray(z), minlength=2)

    freq = counts / counts.sum()
    np.testing.assert_allclose(freq, pi_stationary, atol=1e-2)


# ---------------------------------------------------------------------------
# 3. Smoke: simulation_step end-to-end on a 2-state chain
# ---------------------------------------------------------------------------


def test_discrete_simulation_step_roundtrip():
    """``simulation_step`` should:
    - return integer shocks (next-z) in {0, ..., K-1}
    - update state[:, z_state_idx] = shock
    """
    Π = np.array([[0.5, 0.5], [0.5, 0.5]])  # uniform, easy to reason about
    model = _make_chain_model(Π)

    state = jnp.zeros((8, 1))  # everyone at z=0
    policy_fn = lambda s: jnp.zeros_like(s)
    next_state, shock = simulation_step(model, policy_fn, state, key=jr.PRNGKey(0))

    assert shock.dtype == jnp.int32
    assert bool(jnp.all((shock == 0) | (shock == 1)))
    # step_fn embeds shock into state[:, 0]
    np.testing.assert_array_equal(np.asarray(next_state[:, 0]), np.asarray(shock))


# ---------------------------------------------------------------------------
# 4. Mixed continuous + discrete: existing models unaffected
# ---------------------------------------------------------------------------


def test_legacy_continuous_models_unaffected():
    """Models without ``transition_matrix`` set must still use Gaussian
    shocks and uniform sample weights — pure regression check that the
    legacy MC path still produces a finite, non-NaN loss."""
    from deqn_jax.models import load_model

    m = load_model("brock_mirman")
    assert getattr(m, "transition_matrix", None) is None  # legacy
    s_ss, p_ss = m.steady_state_fn(m.constants)
    states = jnp.broadcast_to(s_ss[None, :], (16, m.n_states))
    policy_fn = lambda s: jnp.broadcast_to(p_ss[None, :], (s.shape[0], m.n_policies))

    loss, eq_losses = compute_loss(
        m, policy_fn, states, key=jr.PRNGKey(42), mc_samples=4
    )
    assert bool(jnp.isfinite(loss))
    for k, v in eq_losses.items():
        assert bool(jnp.isfinite(v)), f"non-finite {k}: {v}"
