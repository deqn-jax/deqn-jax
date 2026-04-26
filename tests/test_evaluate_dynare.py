"""Tests for the Dynare-comparison evaluators in evaluate.py and dynare_io."""

from __future__ import annotations

import csv
import os
import tempfile

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from deqn_jax.dynare_io import (
    deqn_policy_to_dynare,
    deqn_state_col_to_dynare,
    load_dynare_irf,
    load_dynare_jacobian,
    load_dynare_moments,
    read_csv_matrix,
)

DYNARE_DIR = "dynare/results"


# ---------------------------------------------------------------------------
# dynare_io
# ---------------------------------------------------------------------------


def test_read_csv_matrix_basic():
    """read_csv_matrix returns ([col_names], {row_label: [floats]})."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tiny.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["variable", "a", "b"])
            w.writerow(["x", 1.0, 2.0])
            w.writerow(["y", 3.0, 4.0])
        cols, rows = read_csv_matrix(path)
        assert cols == ["a", "b"]
        assert rows == {"x": [1.0, 2.0], "y": [3.0, 4.0]}


def test_name_mapping_known_aliases():
    """The DEQN ↔ Dynare aliases are stable."""
    assert deqn_policy_to_dynare("i") == "i_var"
    # Non-aliased policies are identity.
    assert deqn_policy_to_dynare("c") == "c"
    assert deqn_policy_to_dynare("pi") == "pi"
    # State-column mapping handles the lag suffix.
    assert deqn_state_col_to_dynare("k_lag") == "k(-1)"
    assert deqn_state_col_to_dynare("eps") == "eps(-1)"
    # m_p has no ghx column (handled via ghu in load_dynare_jacobian).
    assert deqn_state_col_to_dynare("m_p") is None


def test_load_dynare_moments_disaster():
    """The disaster reference moments load and contain known variables."""
    if not os.path.isdir(DYNARE_DIR):
        pytest.skip("dynare/results not present in this repo")
    mom = load_dynare_moments(DYNARE_DIR)
    for var in ("c", "i_var", "pi", "h", "lambda_z", "w_tilda"):
        assert var in mom, f"missing {var!r} in dynare_moments"
        assert "mean" in mom[var] and "std" in mom[var]
        assert mom[var]["std"] > 0


def test_load_dynare_jacobian_disaster_shape():
    """ghx-derived Jacobian has shape [n_policies, n_states] for disaster."""
    from deqn_jax.models import load_model

    if not os.path.isdir(DYNARE_DIR):
        pytest.skip("dynare/results not present in this repo")
    model = load_model("disaster")
    J = load_dynare_jacobian(model, DYNARE_DIR)
    assert J.shape == (model.n_policies, model.n_states)
    # Some entry should be non-trivial.
    assert float(jnp.max(jnp.abs(J))) > 1e-3


def test_load_dynare_irf_disaster():
    """Per-shock IRF CSVs load with the right horizon length."""
    if not os.path.isdir(DYNARE_DIR):
        pytest.skip("dynare/results not present in this repo")
    irf = load_dynare_irf(DYNARE_DIR, "eps")
    for var in ("c", "i_var", "pi"):
        assert var in irf
    horizon = len(irf["c"])
    assert horizon >= 10  # Dynare IRFs go for 40 periods by default
    # All series share the same length.
    assert all(len(v) == horizon for v in irf.values())


# ---------------------------------------------------------------------------
# compare_to_dynare_ghx — sharpest contract test
# ---------------------------------------------------------------------------


class _LinearPolicy:
    """A minimal stand-in for a trained policy: π(s) = ss_policy + J @ (s - ss_state).

    Not an Equinox module, but jax.jacrev only needs a callable. Tests can
    construct one with J equal to the Dynare jacobian and verify zero diff.
    """

    def __init__(self, ss_state, ss_policy, J):
        self.ss_state = ss_state
        self.ss_policy = ss_policy
        self.J = J

    def __call__(self, state):
        return self.ss_policy + self.J @ (state - self.ss_state)


def test_ghx_diff_zero_when_policy_matches_dynare():
    """If the policy *is* the Dynare linearization, ghx diff should be exactly zero."""
    from deqn_jax.evaluate import compare_to_dynare_ghx
    from deqn_jax.models import load_model

    if not os.path.isdir(DYNARE_DIR):
        pytest.skip("dynare/results not present in this repo")
    model = load_model("disaster")
    assert model.steady_state_fn is not None
    ss_state, ss_policy = model.steady_state_fn(model.constants)
    J_dyn = load_dynare_jacobian(model, DYNARE_DIR)

    policy = _LinearPolicy(ss_state, ss_policy, J_dyn)
    diff = compare_to_dynare_ghx(policy, model, DYNARE_DIR)

    assert diff["frobenius"] < 1e-5, (
        f"linear-Dynare-policy should give ~0 ghx diff, got {diff['frobenius']:.2e}"
    )
    for pp in diff["per_policy"].values():
        assert pp["max_abs"] < 1e-5


def test_ghx_diff_nonzero_when_policy_perturbed():
    """Perturbing J by a small amount should produce a non-zero diff matching the perturbation norm."""
    from deqn_jax.evaluate import compare_to_dynare_ghx
    from deqn_jax.models import load_model

    if not os.path.isdir(DYNARE_DIR):
        pytest.skip("dynare/results not present in this repo")
    model = load_model("disaster")
    assert model.steady_state_fn is not None
    ss_state, ss_policy = model.steady_state_fn(model.constants)
    J_dyn = load_dynare_jacobian(model, DYNARE_DIR)

    eps = 0.05
    perturbation = eps * jr.normal(jr.PRNGKey(0), J_dyn.shape)
    policy = _LinearPolicy(ss_state, ss_policy, J_dyn + perturbation)
    diff = compare_to_dynare_ghx(policy, model, DYNARE_DIR)
    expected_fro = float(jnp.linalg.norm(perturbation))
    # jacrev is exact, so the recovered diff should match within float roundoff.
    assert abs(diff["frobenius"] - expected_fro) < 1e-3 * max(expected_fro, 1.0)


# ---------------------------------------------------------------------------
# compare_to_dynare_moments — schema test
# ---------------------------------------------------------------------------


def test_moments_diff_overlap_and_shape():
    """The moments diff returns per-policy entries for overlapping vars only."""

    from deqn_jax.evaluate import compare_to_dynare_moments
    from deqn_jax.models import load_model
    from deqn_jax.networks.mlp import MLP

    if not os.path.isdir(DYNARE_DIR):
        pytest.skip("dynare/results not present in this repo")
    model = load_model("disaster")

    # Cheap MLP — moments will be wildly off from Dynare, but the schema test
    # only cares about which keys are populated and that the values are floats.
    net = MLP(
        in_features=model.n_states,
        out_features=model.n_policies,
        hidden_sizes=(8,),
        activations=(jax.nn.tanh,),
        key=jr.PRNGKey(0),
    )
    out = compare_to_dynare_moments(net, model, DYNARE_DIR, n_periods=200, seed=0)
    n_compared = out["n_compared"]
    assert n_compared > 0
    for pname, row in out["per_var"].items():
        assert pname in model.policy_names
        for k in (
            "dynare_var",
            "dynare_mean",
            "net_mean",
            "mean_diff_pct",
            "dynare_std",
            "net_std",
            "std_diff_pct",
        ):
            assert k in row
        assert isinstance(row["mean_diff_pct"], float)
        assert isinstance(row["std_diff_pct"], float)
    # Summary scalars present + numeric.
    for k in (
        "median_abs_mean_diff_pct",
        "median_abs_std_diff_pct",
        "max_abs_mean_diff_pct",
        "max_abs_std_diff_pct",
    ):
        assert isinstance(out[k], float)


# ---------------------------------------------------------------------------
# compare_to_dynare_irfs — schema test
# ---------------------------------------------------------------------------


def test_irfs_diff_runs_and_skips_missing():
    """IRF diff returns per-shock results and a skipped-shocks list."""
    from deqn_jax.evaluate import compare_to_dynare_irfs
    from deqn_jax.models import load_model
    from deqn_jax.networks.mlp import MLP

    if not os.path.isdir(DYNARE_DIR):
        pytest.skip("dynare/results not present in this repo")
    model = load_model("disaster")
    net = MLP(
        in_features=model.n_states,
        out_features=model.n_policies,
        hidden_sizes=(8,),
        activations=(jax.nn.tanh,),
        key=jr.PRNGKey(0),
    )
    out = compare_to_dynare_irfs(net, model, DYNARE_DIR, horizon=10)
    per_shock = out["per_shock"]
    assert per_shock, "expected at least one shock with a Dynare IRF CSV"
    for shock, payload in per_shock.items():
        assert shock in model.shock_names
        assert payload["n_vars"] > 0
        for var, vrow in payload["per_var"].items():
            assert vrow["max_abs_diff"] >= 0
            assert vrow["l2_diff"] >= 0
            assert vrow["horizon"] > 0
