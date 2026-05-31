"""Regression test pinning the Brock-Mirman analytical steady state.

Audit bm-ss-01 / models-06 / tests-03: the SS was previously pinned by no
*value* test, so the recurring "three-way mismatch" confusion (sim 4.0 /
closed-form 0.18 / partial-delta 14) had no guard and kept getting re-filed
(task #110). The audit established the shipped formula is correct and
self-consistent; the mismatch was a calibration mixup. These tests lock the
canonical values so a future formula edit cannot silently shift the SS.
"""

import jax.numpy as jnp  # noqa: F401  (imported for parity with sibling tests)
import pytest

from deqn_jax.models import load_model

# Canonical constants: alpha=0.36, beta=0.99, delta=0.1.
CANON_K = 6.366837
CANON_SAV = 0.326972


def test_brock_mirman_ss_canonical_value():
    m = load_model("brock_mirman")
    ss_state, ss_policy = m.steady_state_fn(m.constants)
    k, z = float(ss_state[0]), float(ss_state[1])
    sav = float(ss_policy[0])
    assert k == pytest.approx(CANON_K, rel=1e-5)
    assert z == pytest.approx(0.0, abs=1e-12)
    assert sav == pytest.approx(CANON_SAV, rel=1e-5)


def test_brock_mirman_ss_savings_identity():
    """sav_rate* = delta * k* / y* with y* = k*^alpha."""
    m = load_model("brock_mirman")
    c = m.constants
    ss_state, ss_policy = m.steady_state_fn(c)
    k = float(ss_state[0])
    sav = float(ss_policy[0])
    y = k ** c["alpha"]
    # rel=1e-5: model SS is computed in float32 (JAX default); the recomputed
    # identity here is float64, so agreement is bounded by float32 resolution.
    assert sav == pytest.approx(c["delta"] * k / y, rel=1e-5)


def test_brock_mirman_full_depreciation_closed_form():
    """At delta=1 the SS collapses to log-utility (alpha*beta)^(1/(1-alpha)).

    This is the source of the historical "~0.18" figure -- a different
    calibration, not a bug.
    """
    m = load_model("brock_mirman")
    c = {**m.constants, "delta": 1.0}
    ss_state, _ = m.steady_state_fn(c)
    a, b = c["alpha"], c["beta"]
    expected = (a * b) ** (1.0 / (1.0 - a))
    # rel=1e-5: float32 (JAX) vs float64 (this recomputation) resolution.
    assert float(ss_state[0]) == pytest.approx(expected, rel=1e-5)


def test_brock_mirman_ss_is_euler_fixed_point():
    """Euler residual vanishes at the canonical SS (model-specific sanity)."""
    m = load_model("brock_mirman")
    ss_state, ss_policy = m.steady_state_fn(m.constants)
    s, p = ss_state[None, :], ss_policy[None, :]
    resid = m.equations_fn(s, p, s, p, m.constants)
    assert abs(float(resid["euler"][0])) < 1e-6
