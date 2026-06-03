"""Tests for ``olg_lifecycle`` (Geneva Day 2 Ex 4): 6-generation life-cycle OLG
with borrowing constraints, trained via the two-stage (expectation-inside-
residual) loss.

Guards:
  1. ModelSpec dims + two-stage hooks wired; equation_names == combine_fn keys.
  2. GOLDEN: the within-period block and the single-shock Euler residual match an
     independent numpy transcription of the notebook formulas (anti-"built from
     memory" guard — the Ex3 landing shipped a wrong FB form from memory once).
  3. inside_fn -> H cohort terms; the two-stage compute_loss -> H-1 finite euler
     losses keyed euler_0..euler_4.
  4. Smoke train: loss stays finite and decreases over a short run.
"""

import jax
import jax.numpy as jnp
import numpy as np

from deqn_jax.models import load_model
from deqn_jax.models.olg_lifecycle.equations import (
    EQUATION_NAMES,
    _cohort_block,
    equations,
    inside_fn,
)
from deqn_jax.models.olg_lifecycle.variables import CONSTANTS, L_CYCLE, H
from deqn_jax.training.loss import compute_loss

MODEL = load_model("olg_lifecycle")


# --------------------------------------------------------------------------- #
# Independent numpy reference (transcribed from 04_DEQN_Exercises_Solutions.ipynb)
# --------------------------------------------------------------------------- #
def _np_block(Z, k, s):
    alpha, delta = CONSTANTS["alpha"], CONSTANTS["delta"]
    l_cycle = np.asarray(L_CYCLE)
    L = l_cycle.sum()
    K = k.sum(axis=1, keepdims=True)
    Y = Z * K**alpha * L ** (1 - alpha)
    r = alpha * (Y / K)
    w = (1 - alpha) * (Y / L)
    cah = l_cycle[None, :] * w + k * (1 - delta + r)
    s_full = np.concatenate([s, np.zeros((s.shape[0], 1))], axis=1)
    sav = cah * s_full
    c = cah - sav
    return r, w, cah, sav, c


def _np_euler(Z, k, s, Zp, kp, sp):
    beta, delta = CONSTANTS["beta"], CONSTANTS["delta"]
    _, _, _, sav, c = _np_block(Z, k, s)
    rp, _, _, _, cp = _np_block(Zp, kp, sp)
    inside = (1 - delta + rp) / cp  # [b,H]
    out = []
    for h in range(H - 1):
        a = 1.0 / (c[:, h] * beta * inside[:, h + 1]) - 1.0
        b = sav[:, h] / c[:, h]
        out.append(np.sqrt(a * a + b * b + 1e-13) - a - b)
    return np.stack(out, axis=1)  # [b,H-1]


def _fixtures(seed=0):
    rng = np.random.default_rng(seed)
    Z = np.exp(rng.uniform(size=(4, 1))).astype(np.float64)
    k = np.exp(rng.uniform(size=(4, H))).astype(np.float64)
    s = rng.uniform(0.05, 0.6, size=(4, H - 1)).astype(np.float64)
    Zp = np.exp(rng.uniform(size=(4, 1))).astype(np.float64)
    kp = np.exp(rng.uniform(size=(4, H))).astype(np.float64)
    sp = rng.uniform(0.05, 0.6, size=(4, H - 1)).astype(np.float64)
    return Z, k, s, Zp, kp, sp


# --------------------------------------------------------------------------- #
# 1. Contract / wiring
# --------------------------------------------------------------------------- #
def test_dims_and_two_stage_hooks():
    assert MODEL.n_states == 1 + H == 7
    assert MODEL.n_policies == H - 1 == 5
    assert MODEL.n_shocks == 1
    assert len(MODEL.state_names) == MODEL.n_states
    assert len(MODEL.policy_names) == MODEL.n_policies
    assert MODEL.steady_state_fn is None  # no closed form -> trains from init
    assert MODEL.inside_fn is not None and MODEL.combine_fn is not None
    assert MODEL.policy_lower.shape == (MODEL.n_policies,)
    assert MODEL.policy_upper.shape == (MODEL.n_policies,)
    assert tuple(EQUATION_NAMES) == tuple(f"euler_{h}" for h in range(H - 1))


def test_combine_keys_match_equation_names():
    Z, k, s, Zp, kp, sp = _fixtures()
    resid = equations(
        jnp.asarray(np.concatenate([Z, k], axis=1)),
        jnp.asarray(s),
        jnp.asarray(np.concatenate([Zp, kp], axis=1)),
        jnp.asarray(sp),
        CONSTANTS,
    )
    assert tuple(resid.keys()) == tuple(MODEL.equation_names)


# --------------------------------------------------------------------------- #
# 2. Golden vs notebook transcription
# --------------------------------------------------------------------------- #
def test_block_matches_numpy_reference():
    Z, k, s, *_ = _fixtures()
    r, w, cah, sav, c = _np_block(Z, k, s)
    blk = _cohort_block(jnp.asarray(Z), jnp.asarray(k), jnp.asarray(s), CONSTANTS)
    np.testing.assert_allclose(np.asarray(blk["r"]), r, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(blk["w"]), w, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(blk["cah"]), cah, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(blk["sav"]), sav, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(blk["c"]), c, rtol=1e-5)


def test_single_shock_euler_matches_numpy_reference():
    Z, k, s, Zp, kp, sp = _fixtures()
    golden = _np_euler(Z, k, s, Zp, kp, sp)  # [b,H-1]
    resid = equations(
        jnp.asarray(np.concatenate([Z, k], axis=1)),
        jnp.asarray(s),
        jnp.asarray(np.concatenate([Zp, kp], axis=1)),
        jnp.asarray(sp),
        CONSTANTS,
    )
    got = np.stack([np.asarray(resid[f"euler_{h}"]) for h in range(H - 1)], axis=1)
    np.testing.assert_allclose(got, golden, rtol=1e-4, atol=1e-6)


# --------------------------------------------------------------------------- #
# 3. inside_fn shape + two-stage loss finiteness
# --------------------------------------------------------------------------- #
def test_inside_fn_returns_H_cohort_terms():
    Z, k, s, Zp, kp, sp = _fixtures()
    inside = inside_fn(
        jnp.asarray(np.concatenate([Z, k], axis=1)),
        jnp.asarray(s),
        jnp.asarray(np.concatenate([Zp, kp], axis=1)),
        jnp.asarray(sp),
        CONSTANTS,
    )
    assert set(inside.keys()) == {f"inside_{j}" for j in range(H)}
    for v in inside.values():
        assert v.shape == (4,)
        assert jnp.all(jnp.isfinite(v))


def test_two_stage_compute_loss_finite():
    key = jax.random.PRNGKey(0)
    init = MODEL.init_state_fn(key, 16, CONSTANTS)

    def policy_fn(states):
        return jnp.full((states.shape[0], MODEL.n_policies), 0.3)

    loss, eq_losses = compute_loss(MODEL, policy_fn, init, key, mc_samples=4)
    assert jnp.isfinite(loss)
    assert tuple(eq_losses.keys()) == tuple(f"euler_{h}" for h in range(H - 1))
    for v in eq_losses.values():
        assert jnp.isfinite(v)


# --------------------------------------------------------------------------- #
# 4. Smoke train
# --------------------------------------------------------------------------- #
def test_smoke_train_decreases():
    from deqn_jax.training.trainer import train

    _params, history = train(
        "olg_lifecycle",
        episodes=60,
        hidden_sizes=(32, 32),
        learning_rate=1e-3,
        batch_size=32,
        episode_length=20,
        mc_samples=4,
        seed=0,
        verbose=False,
    )
    losses = history["loss"]
    assert all(jnp.isfinite(jnp.asarray(l)) for l in losses), "non-finite loss"
    initial = losses[0]
    final = min(losses[-10:])
    assert final < initial, f"loss did not decrease: {initial:.3e} -> {final:.3e}"
