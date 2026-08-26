"""56-generation OLG: dimensions, machinery reuse, and economic invariants.

The heavy behavioral references live in test_olg_lifecycle.py (the shared
machinery is H-agnostic and numpy-pinned there); these tests check what is
specific to H=56: the layout, the interpolated age profile, market clearing,
and that the two-stage hooks produce consistently-keyed residuals.
"""

import jax
import jax.numpy as jnp
import pytest

from deqn_jax.models import load_model

MODEL = load_model("olg_lifecycle_56")
H = 56


@pytest.fixture(scope="module")
def batch():
    key = jax.random.PRNGKey(0)
    k_s, k_p, k_e = jax.random.split(key, 3)
    state = MODEL.init_state_fn(k_s, 8, MODEL.constants)
    policy = jax.random.uniform(k_p, (8, MODEL.n_policies), minval=0.05, maxval=0.6)
    shock = jax.random.normal(k_e, (8, MODEL.n_shocks))
    return state, policy, shock


def test_dims():
    assert MODEL.n_states == 1 + H
    assert MODEL.n_policies == H - 1
    assert len(MODEL.equation_names) == H - 1
    assert len(MODEL.constants["l_cycle"]) == H


def test_l_cycle_is_humped():
    lc = MODEL.constants["l_cycle"]
    peak = max(range(H), key=lambda i: lc[i])
    # Peak strictly interior (the hump), endpoints match the 6-gen anchors.
    assert 0 < peak < H - 1
    assert lc[0] == pytest.approx(1.0)
    assert lc[-1] == pytest.approx(1.25)


def test_step_shapes_and_newborn(batch):
    state, policy, shock = batch
    nxt = MODEL.step_fn(state, policy, shock, MODEL.constants)
    assert nxt.shape == state.shape
    # Newborn cohort enters with zero assets.
    assert jnp.all(nxt[:, 1] == 0.0)


def test_aging_ladder(batch):
    # Cohort h's savings become cohort h+1's capital: k'^{h+1} = cah^h * s^h.
    state, policy, shock = batch
    nxt = MODEL.step_fn(state, policy, shock, MODEL.constants)
    d = MODEL.definitions_fn(state, policy, MODEL.constants)
    lc = jnp.asarray(MODEL.constants["l_cycle"])
    cah = lc[None, :] * d["w"][:, None] + state[:, 1:] * (
        1.0 - MODEL.constants["delta"] + d["r"][:, None]
    )
    expected = cah[:, : H - 1] * policy
    assert jnp.allclose(nxt[:, 2:], expected, rtol=1e-12)


def test_market_clearing(batch):
    # sum(c) + sum(k') = Y + (1-delta) K  (goods market, Walras).
    state, policy, shock = batch
    c = MODEL.constants
    nxt = MODEL.step_fn(state, policy, shock, c)
    Z, k = state[:, :1], state[:, 1:]
    lc = jnp.asarray(c["l_cycle"])
    K = jnp.sum(k, axis=1)
    Y = (Z[:, 0]) * K ** c["alpha"] * jnp.sum(lc) ** (1.0 - c["alpha"])
    d = MODEL.definitions_fn(state, policy, c)
    cah = lc[None, :] * d["w"][:, None] + k * (1.0 - c["delta"] + d["r"][:, None])
    s_full = jnp.concatenate([policy, jnp.zeros((policy.shape[0], 1))], axis=1)
    consumption = jnp.sum(cah * (1.0 - s_full), axis=1)
    K_next = jnp.sum(nxt[:, 1:], axis=1)
    # fp32: the two sides aggregate 56 cohorts in different orders.
    assert jnp.allclose(consumption + K_next, Y + (1.0 - c["delta"]) * K, rtol=1e-5)


def test_two_stage_keys(batch):
    state, policy, shock = batch
    nxt = MODEL.step_fn(state, policy, shock, MODEL.constants)
    inside = MODEL.inside_fn(state, policy, nxt, policy, MODEL.constants)
    assert set(inside) == {f"inside_{j}" for j in range(H)}
    combined = MODEL.combine_fn(state, policy, inside, MODEL.constants)
    assert set(combined) == set(MODEL.equation_names)
    for v in combined.values():
        assert jnp.all(jnp.isfinite(v))
