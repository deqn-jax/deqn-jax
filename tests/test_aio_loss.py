"""Tests for the all-in-one (AiO) loss estimator (`loss_choice='aio'`).

Background. The default MC aggregation squares the sample mean of the
residual over shocks:

    L_mse = mean_b[ (1/N sum_i r(s_b, eps_i))^2 ]

whose expectation is (E[r])^2 + Var(rbar) -- biased upward by the MC
variance of the shock average. The bias is a real gradient force: it
rewards policies that make the residual *insensitive to the shock*
rather than zero in expectation (Maliar-Maliar-Winant 2021, JME).

The AiO estimator splits the N draws into two INDEPENDENT groups and
multiplies the two group means:

    L_aio = mean_b[ rbar_1(s_b) * rbar_2(s_b) ],   E[L_aio] = (E[r])^2

exactly (independence + each group mean unbiased for E[r]).

The tests verify the bias of the mse estimator and the unbiasedness of
the aio estimator statistically, against the Gauss-Hermite quadrature
path as the exact reference (brock_mirman has 1 shock, so 64 GH nodes
integrate the smooth residual essentially exactly).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deqn_jax.config import NetworkConfig, TrainConfig
from deqn_jax.models import load_model
from deqn_jax.training.loss import compute_loss, gauss_hermite_nd
from deqn_jax.types import ModelSpec

BATCH = 16
# Amplify the shock so the residual's curvature in eps (the source of
# Var(rbar) under antithetic sampling) is well above statistical noise.
SHOCK_SCALE = 5.0
N_KEYS = 4096
MC_SAMPLES = 4


def _bm_setup():
    model = load_model("brock_mirman")
    states = model.init_state_fn(jax.random.PRNGKey(0), BATCH, model.constants)

    def policy_fn(s):
        return jnp.full((s.shape[0], 1), 0.33)

    return model, states, policy_fn


def _exact_loss(model, states, policy_fn):
    """(E[r])^2 averaged over the batch, via 64-node Gauss-Hermite."""
    nodes, weights = gauss_hermite_nd(64, model.n_shocks)
    loss, _ = compute_loss(
        model,
        policy_fn,
        states,
        jax.random.PRNGKey(0),  # ignored on the quadrature path
        shock_scale=SHOCK_SCALE,
        quad_nodes=jnp.array(nodes),
        quad_weights=jnp.array(weights),
    )
    return float(loss)


def _mc_mean_and_se(model, states, policy_fn, loss_choice, mc_samples=MC_SAMPLES):
    """Mean and standard error of the MC loss estimator over N_KEYS keys."""
    keys = jax.random.split(jax.random.PRNGKey(1), N_KEYS)

    def one(k):
        loss, _ = compute_loss(
            model,
            policy_fn,
            states,
            k,
            mc_samples=mc_samples,
            shock_scale=SHOCK_SCALE,
            loss_choice=loss_choice,
        )
        return loss

    losses = np.asarray(jax.jit(jax.vmap(one))(keys))
    return float(losses.mean()), float(losses.std(ddof=1) / np.sqrt(N_KEYS))


# ---------------------------------------------------------------------------
# Statistical properties (standard path)
# ---------------------------------------------------------------------------


def test_mse_estimator_is_biased_upward():
    """E[(mean r)^2] = (E[r])^2 + Var(rbar) > (E[r])^2.

    Establishes the premise: the default estimator's bias is detectable,
    so the aio test below is a meaningful discrimination, not a vacuous
    pass on a bias too small to measure.
    """
    model, states, policy_fn = _bm_setup()
    exact = _exact_loss(model, states, policy_fn)
    mean_mse, se_mse = _mc_mean_and_se(model, states, policy_fn, "mse")

    bias = mean_mse - exact
    assert bias > 5 * se_mse, (
        f"mse bias {bias:.3e} not significantly positive (se={se_mse:.3e}); "
        "test has no discriminating power -- increase SHOCK_SCALE or N_KEYS"
    )


def test_aio_estimator_is_unbiased():
    """E[rbar_1 * rbar_2] = (E[r])^2 -- within statistical tolerance."""
    model, states, policy_fn = _bm_setup()
    exact = _exact_loss(model, states, policy_fn)
    mean_aio, se_aio = _mc_mean_and_se(model, states, policy_fn, "aio")

    assert abs(mean_aio - exact) < 4 * se_aio, (
        f"aio estimator mean {mean_aio:.6e} differs from exact {exact:.6e} "
        f"by {abs(mean_aio - exact):.3e} > 4*se ({se_aio:.3e})"
    )


def test_aio_estimator_beats_mse_bias():
    """The aio mean must sit strictly below the biased mse mean."""
    model, states, policy_fn = _bm_setup()
    mean_mse, se_mse = _mc_mean_and_se(model, states, policy_fn, "mse")
    mean_aio, se_aio = _mc_mean_and_se(model, states, policy_fn, "aio")
    se = float(np.hypot(se_mse, se_aio))

    assert mean_mse - mean_aio > 4 * se


# ---------------------------------------------------------------------------
# Statistical properties (two-stage combine_fn path)
# ---------------------------------------------------------------------------


def _two_stage_setup():
    """Synthetic two-stage model with a LINEAR combine_fn.

    inside g = p + x*eps + eps^2 with eps ~ N(0,1)  =>  E[g] = p + 1
    combine r = 2*E[g] + 1                          =>  r = 2p + 3

    Linear combine isolates the aio grouping/product wiring from the
    (separate, documented) Jensen bias of a nonlinear combine_fn. The
    eps^2 term gives Var(ghat) > 0 even under antithetic pairing, so
    the mse path is measurably biased here too.
    """

    def step_fn(state, policy, shock, constants):
        return shock  # next_state IS the shock

    def inside_fn(state, policy, next_state, next_policy, constants):
        x = state[:, 0]
        p = policy[:, 0]
        eps = next_state[:, 0]
        return {"eq": p + x * eps + eps**2}

    def combine_fn(state, policy, expectations, constants):
        return {"eq": 2.0 * expectations["eq"] + 1.0}

    model = ModelSpec(
        name="synthetic_two_stage",
        n_states=1,
        n_policies=1,
        n_shocks=1,
        equations_fn=inside_fn,  # unused on the two-stage path
        step_fn=step_fn,
        constants={},
        inside_fn=inside_fn,
        combine_fn=combine_fn,
    )
    states = jnp.linspace(0.5, 2.0, BATCH).reshape(-1, 1)

    def policy_fn(s):
        return jnp.full((s.shape[0], 1), 0.33)

    return model, states, policy_fn


def test_aio_two_stage_unbiased():
    model, states, policy_fn = _two_stage_setup()
    # r = 2p + 3 exactly (linear combine, E[g] = p + 1)
    exact = float(jnp.mean((2.0 * 0.33 + 3.0) ** 2))

    keys = jax.random.split(jax.random.PRNGKey(2), N_KEYS)

    def one(k, choice):
        loss, _ = compute_loss(
            model, policy_fn, states, k, mc_samples=MC_SAMPLES, loss_choice=choice
        )
        return loss

    aio = np.asarray(jax.jit(jax.vmap(lambda k: one(k, "aio")))(keys))
    mse = np.asarray(jax.jit(jax.vmap(lambda k: one(k, "mse")))(keys))

    se_aio = aio.std(ddof=1) / np.sqrt(N_KEYS)
    se_mse = mse.std(ddof=1) / np.sqrt(N_KEYS)

    assert mse.mean() - exact > 5 * se_mse, "two-stage mse bias not detectable"
    assert abs(aio.mean() - exact) < 4 * se_aio


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------


def test_aio_raises_on_quadrature():
    model, states, policy_fn = _bm_setup()
    nodes, weights = gauss_hermite_nd(8, model.n_shocks)
    with pytest.raises(ValueError, match="aio"):
        compute_loss(
            model,
            policy_fn,
            states,
            jax.random.PRNGKey(0),
            quad_nodes=jnp.array(nodes),
            quad_weights=jnp.array(weights),
            loss_choice="aio",
        )


def test_aio_requires_two_samples():
    model, states, policy_fn = _bm_setup()
    with pytest.raises(ValueError, match="aio"):
        compute_loss(
            model,
            policy_fn,
            states,
            jax.random.PRNGKey(0),
            mc_samples=1,
            loss_choice="aio",
        )


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def test_config_accepts_aio():
    cfg = TrainConfig(model="brock_mirman", loss_choice="aio")
    assert cfg.loss_choice == "aio"


def test_config_rejects_aio_with_quadrature():
    with pytest.raises(ValueError, match="aio"):
        TrainConfig(
            model="brock_mirman", loss_choice="aio", expectation_type="gauss_hermite"
        )


def test_config_rejects_aio_with_one_sample():
    with pytest.raises(ValueError, match="aio"):
        TrainConfig(model="brock_mirman", loss_choice="aio", mc_samples=1)


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


def test_train_smoke_aio():
    from deqn_jax.training.trainer import train_from_config

    config = TrainConfig(
        model="brock_mirman",
        episodes=3,
        batch_size=16,
        mc_samples=2,
        network=NetworkConfig(hidden_sizes=(16,)),
        loss_choice="aio",
        verbose=False,
    )
    params, history = train_from_config(config)
    assert np.isfinite(history["loss"][-1])
