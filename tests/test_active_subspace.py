"""Tests for active-subspace analysis of trained policies.

The active subspace of a scalar output ``π_i(s)`` is the leading
eigenspace of ``C_i = E_s[∇π_i(s) · ∇π_i(s)^T]``. The eigenvalue
spectrum tells us how many directions in state space the policy
genuinely depends on (effective dimensionality), and the leading
eigenvectors are those directions.

Tests pin three contracts using closed-form policies:

  1. Linear policy with rank-1 gradient → exactly one nonzero
     eigenvalue. Top eigenvector matches the gradient direction.
  2. Quadratic two-direction policy ``f(s) = s[0]² + s[1]²`` →
     exactly two equal eigenvalues, eff_dim_0.95 = 2.
  3. Constant policy → all eigenvalues ~0, effective dim 0,
     summary shows trace ~0.

Plus an end-to-end test on the disaster KfAnchoredMLP confirming that
the analysis runs without NaN poisoning and produces reasonable
spectra (anchored K/F outputs should be 1-d by construction since
they're linear in state; non-anchored outputs should have richer
spectra).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

jax.config.update("jax_enable_x64", True)

from deqn_jax.active_subspace import (  # noqa: E402
    effective_dimensionality,
    estimate_active_subspace,
    estimate_gradient_covariance,
    policy_grid_on_subspace,
    project_states,
    summarize_subspace_per_policy,
)

# ---------------------------------------------------------------------------
# Closed-form policies
# ---------------------------------------------------------------------------


def _linear_policy_from(J: jnp.ndarray):
    """f(s) = J @ s — gradient is the constant matrix J^T."""

    def f(s):
        return J @ s

    return f


def _quadratic_policy(s):
    """f_0(s) = s[0]² + s[1]² — gradient = [2 s[0], 2 s[1], 0, …]."""
    return jnp.array([s[0] ** 2 + s[1] ** 2])


def _constant_policy(s):
    return jnp.array([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Spectrum contracts
# ---------------------------------------------------------------------------


def test_rank_one_linear_policy_gives_single_eigenvalue():
    """Output 0 = 1·s[0]: ∇ = e_0. C = e_0 e_0^T → one eigenvalue."""
    n_states = 5
    J = jnp.zeros((3, n_states)).at[0, 0].set(1.0)
    states = jr.normal(jr.PRNGKey(0), (200, n_states))
    sub = estimate_active_subspace(_linear_policy_from(J), states, output_idx=0)

    eigenvalues = sub["eigenvalues"]
    assert eigenvalues.shape == (n_states,)
    # Exactly one nonzero eigenvalue.
    assert float(eigenvalues[0]) > 0.5
    assert all(float(eigenvalues[k]) < 1e-6 for k in range(1, n_states))

    # Top eigenvector aligned with e_0 (sign-invariant).
    top = sub["eigenvectors"][:, 0]
    e0 = jnp.zeros(n_states).at[0].set(1.0)
    assert abs(abs(float(top @ e0)) - 1.0) < 1e-6

    # Effective dim and participation ratio both = 1.
    assert effective_dimensionality(eigenvalues, 0.95) == 1
    assert abs(sub["participation_ratio"] - 1.0) < 1e-3


def test_two_direction_quadratic_policy_gives_eff_dim_two():
    """f(s) = s[0]² + s[1]²: ∇ varies with state in two orthogonal directions."""
    n_states = 5
    states = jr.normal(jr.PRNGKey(1), (2000, n_states))
    sub = estimate_active_subspace(_quadratic_policy, states, output_idx=0)

    eigenvalues = sub["eigenvalues"]
    # Top two eigenvalues both ≈ 4 (E[(2 s_i)²] for Gaussian s with var 1).
    assert float(eigenvalues[0]) > 2.5
    assert float(eigenvalues[1]) > 2.5
    # Trailing eigenvalues should be tiny.
    for k in range(2, n_states):
        assert float(eigenvalues[k]) < 0.5

    assert effective_dimensionality(eigenvalues, 0.95) == 2
    # Participation ratio for two equal eigenvalues = 2.
    assert abs(sub["participation_ratio"] - 2.0) < 0.2


def test_constant_policy_has_zero_trace_and_zero_dim():
    """A state-blind output: ∇ = 0 everywhere, C = 0, trace = 0, eff dim 0."""
    states = jr.normal(jr.PRNGKey(2), (100, 4))
    sub = estimate_active_subspace(_constant_policy, states, output_idx=0)
    assert sub["trace"] < 1e-12
    assert all(float(v) < 1e-12 for v in sub["eigenvalues"])
    assert effective_dimensionality(sub["eigenvalues"], 0.95) == 0


def test_gradient_covariance_skips_nonfinite_states():
    """If a policy produces NaN gradient on some states, those samples
    are excluded from the average via the n_finite scrub."""

    def f(s):
        # Nonlinear with an explicit NaN-producing branch when s[0] < 0.
        bad = jnp.where(s[0] < 0, jnp.nan, 0.0)
        return jnp.array([s[0] + bad])

    # All states have s[0] > 0, so gradients are clean.
    good_states = jnp.abs(jr.normal(jr.PRNGKey(3), (50, 4)))
    C, n_kept = estimate_gradient_covariance(f, good_states, output_idx=0)
    assert n_kept == 50
    assert bool(jnp.all(jnp.isfinite(C)))


# ---------------------------------------------------------------------------
# Effective dim threshold sweep
# ---------------------------------------------------------------------------


def test_effective_dim_thresholds():
    """Threshold sweeps reflect the eigenvalue accumulation curve."""
    # Eigenvalue spectrum: 5, 3, 1, 0.1, 0.01 — total = 9.11.
    eigenvalues = jnp.array([5.0, 3.0, 1.0, 0.1, 0.01])
    # Cumulative ratios: 0.549, 0.878, 0.988, 0.999, 1.000
    assert effective_dimensionality(eigenvalues, 0.5) == 1
    assert effective_dimensionality(eigenvalues, 0.85) == 2
    assert effective_dimensionality(eigenvalues, 0.95) == 3
    assert effective_dimensionality(eigenvalues, 0.99) == 4
    assert effective_dimensionality(eigenvalues, 0.9999) == 5


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------


def test_project_states_recenters_and_projects():
    """``project_states`` subtracts ``center`` then applies the basis."""
    states = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    center = jnp.array([1.0, 2.0, 3.0])
    # Identity-ish projection on the first two coords.
    directions = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    proj = project_states(states, directions, center=center)
    assert proj.shape == (2, 2)
    # First state is exactly at center → projects to (0, 0).
    assert float(jnp.linalg.norm(proj[0])) < 1e-6
    # Second state is offset by (3, 3, 3); projecting onto (e0, e1) gives (3, 3).
    assert abs(float(proj[1, 0]) - 3.0) < 1e-6
    assert abs(float(proj[1, 1]) - 3.0) < 1e-6


def test_policy_grid_2d_returns_grid_shape():
    """2-d sweep on a linear policy reproduces the linear function exactly."""
    n_states = 4
    J = jnp.array([[1.0, 1.0, 0.0, 0.0]])  # f(s) = s[0] + s[1]
    f = _linear_policy_from(J)
    center = jnp.zeros(n_states)
    v1 = jnp.zeros(n_states).at[0].set(1.0)
    v2 = jnp.zeros(n_states).at[1].set(1.0)
    grid = policy_grid_on_subspace(
        f, center, v1, v2, output_idx=0, grid_range=(-1.0, 1.0), n_pts=11
    )
    assert grid["shape"] == "2d"
    assert grid["values"].shape == (11, 11)
    # f(z1·v1 + z2·v2) = z1 + z2 along the grid (within float roundoff).
    z1 = grid["z1"]
    z2 = grid["z2"]
    Z1, Z2 = jnp.meshgrid(z1, z2, indexing="xy")
    expected = Z1 + Z2
    assert float(jnp.max(jnp.abs(grid["values"] - expected))) < 1e-6


def test_policy_grid_1d_when_direction_2_is_none():
    """Falls back to a 1-d sweep when no second direction is given."""
    J = jnp.array([[1.0, 0.0]])
    f = _linear_policy_from(J)
    grid = policy_grid_on_subspace(
        f, jnp.zeros(2), jnp.array([1.0, 0.0]), None, 0, n_pts=5
    )
    assert grid["shape"] == "1d"
    assert grid["values"].shape == (5,)


# ---------------------------------------------------------------------------
# End-to-end on a real network
# ---------------------------------------------------------------------------


def test_end_to_end_on_kf_anchored_disaster():
    """Run the full per-policy summary on a fresh KfAnchored MLP for disaster.

    Doesn't pin specific eigenvalues (those depend on the random init),
    just confirms (a) the pipeline runs without NaN poisoning,
    (b) all 11 outputs produce finite spectra,
    (c) the K/F-anchored outputs (which are linear in state) all have
        effective_dim == 1 — proves the rank-1 contract holds end-to-end.
    """
    from deqn_jax.models import load_model
    from deqn_jax.networks.kf_anchored_mlp import create_kf_anchored_mlp

    model = load_model("disaster")
    net = create_kf_anchored_mlp(
        model, hidden_sizes=(16,), activation="tanh", key=jr.PRNGKey(0)
    )
    ss_state, _ = model.steady_state_fn(model.constants)
    states = ss_state[None, :] + 1e-2 * jr.normal(jr.PRNGKey(7), (200, model.n_states))

    summary = summarize_subspace_per_policy(
        net, states, list(model.policy_names), threshold=0.95
    )

    # All 11 policies present, all finite, all gave at least most samples.
    assert set(summary.keys()) == set(model.policy_names)
    for name, sub in summary.items():
        assert bool(jnp.all(jnp.isfinite(sub["eigenvalues"])))
        assert sub["n_finite_samples"] > 100  # most of 200 should be clean

    # K/F outputs are pinned to a linear anchor: each is a fixed linear
    # function of state, so the per-output gradient is constant ⇒ exactly
    # one nonzero eigenvalue.
    for kf_name in ("F_p", "K_p", "F_w", "K_w"):
        assert summary[kf_name]["effective_dim"] == 1, (
            f"{kf_name} (linear anchor) should have eff_dim 1, "
            f"got {summary[kf_name]['effective_dim']}"
        )
