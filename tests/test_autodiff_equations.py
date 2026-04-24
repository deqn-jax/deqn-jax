"""Tests for the autodiff-synthesized Euler in brock_mirman_autodiff.

The variant's equations.py computes

    euler = -(dPi/dK' + beta * dPi/dK_at_next)

via ``jax.grad`` on a pure per-period return function. It should agree
with the hand-derived ``u'(c) - beta * u'(c')(1 + r' - delta)`` from the
canonical brock_mirman to floating-point noise on any valid state.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def models():
    from deqn_jax.models.brock_mirman import MODEL as MODEL_HAND
    from deqn_jax.models.brock_mirman_autodiff import MODEL as MODEL_AD
    return MODEL_HAND, MODEL_AD


def test_ss_residuals_both_zero(models):
    """At steady state, both models should give Euler residual ~ 0."""
    MODEL_HAND, MODEL_AD = models
    ss_state, ss_policy = MODEL_HAND.steady_state_fn(MODEL_HAND.constants)
    state = ss_state[None, :]
    policy = ss_policy[None, :]

    resid_hand = MODEL_HAND.equations_fn(state, policy, state, policy, MODEL_HAND.constants)
    resid_ad = MODEL_AD.equations_fn(state, policy, state, policy, MODEL_AD.constants)

    assert abs(float(resid_hand["euler"][0])) < 1e-5
    assert abs(float(resid_ad["euler"][0])) < 1e-5


def test_match_on_random_batch(models):
    """Hand-derived == autodiff-synthesized, on 64 random states.

    Key constraint: next_state must be *policy-consistent* with (state, policy),
    i.e. produced by the model's step_fn. Without that, the budget identity
    C = Y - (K' - (1-delta)K) and the policy identity C = (1 - s)*Y
    disagree, and so do the two equation formulations. In real training
    this is automatic because the trainer always calls step_fn to produce
    next_state before equations_fn sees it.
    """
    MODEL_HAND, MODEL_AD = models
    constants = MODEL_HAND.constants

    key = jax.random.PRNGKey(11)
    k_key, z_key, sp_key, sn_key, eps_key, eps2_key = jax.random.split(key, 6)

    k = jax.random.uniform(k_key, (64,), minval=0.5, maxval=3.0)
    z = jax.random.uniform(z_key, (64,), minval=-0.3, maxval=0.3)
    state = jnp.stack([k, z], axis=1)
    policy = jax.random.uniform(sp_key, (64, 1), minval=0.1, maxval=0.6)

    # Consistent next_state from the model's step
    shock = jax.random.normal(eps_key, (64, 1))
    next_state = MODEL_HAND.step_fn(state, policy, shock, constants)
    next_policy = jax.random.uniform(sn_key, (64, 1), minval=0.1, maxval=0.6)

    resid_hand = np.asarray(
        MODEL_HAND.equations_fn(state, policy, next_state, next_policy, constants)["euler"]
    )
    resid_ad = np.asarray(
        MODEL_AD.equations_fn(state, policy, next_state, next_policy, constants)["euler"]
    )

    np.testing.assert_allclose(resid_ad, resid_hand, rtol=1e-4, atol=1e-5)


def test_autodiff_model_registers_and_loads():
    from deqn_jax.models import list_models, load_model

    model = load_model("brock_mirman_autodiff")
    assert model.name == "brock_mirman_autodiff"
    assert "brock_mirman_autodiff" in dict(list_models())


# ---------------------------------------------------------------------------
# bm_labor_autodiff: multi-policy + intratemporal FOC via the extended helper
# ---------------------------------------------------------------------------

@pytest.fixture
def labor_models():
    from deqn_jax.models.bm_labor import MODEL as MODEL_HAND
    from deqn_jax.models.bm_labor_autodiff import MODEL as MODEL_AD
    return MODEL_HAND, MODEL_AD


def test_bm_labor_ss_residuals_both_zero(labor_models):
    """At SS, both the hand-derived euler+labor_foc and the autodiff
    variant should give float32-zero residuals."""
    MODEL_HAND, MODEL_AD = labor_models
    ss_state, ss_policy = MODEL_HAND.steady_state_fn(MODEL_HAND.constants)
    state = ss_state[None, :]
    policy = ss_policy[None, :]

    resid_hand = MODEL_HAND.equations_fn(state, policy, state, policy, MODEL_HAND.constants)
    resid_ad = MODEL_AD.equations_fn(state, policy, state, policy, MODEL_AD.constants)

    for name in ("euler", "labor_foc"):
        assert abs(float(resid_hand[name][0])) < 1e-5, f"hand {name} not zero at SS"
        assert abs(float(resid_ad[name][0])) < 1e-5, f"autodiff {name} not zero at SS"


def test_bm_labor_autodiff_matches_hand_on_random_batch(labor_models):
    """Per-equation parity against hand-derived bm_labor on policy-consistent
    transitions (next_state computed via step_fn, matching production use)."""
    MODEL_HAND, MODEL_AD = labor_models
    constants = MODEL_HAND.constants

    key = jax.random.PRNGKey(17)
    k_key, z_key, sp_key, L_key, eps_key, sn_key, Ln_key = jax.random.split(key, 7)

    k = jax.random.uniform(k_key, (64,), minval=2.0, maxval=10.0)
    z = jax.random.uniform(z_key, (64,), minval=-0.25, maxval=0.25)
    state = jnp.stack([k, z], axis=1)
    sav = jax.random.uniform(sp_key, (64,), minval=0.1, maxval=0.5)
    L = jax.random.uniform(L_key, (64,), minval=0.6, maxval=1.4)
    policy = jnp.stack([sav, L], axis=1)

    shock = jax.random.normal(eps_key, (64, 1))
    next_state = MODEL_HAND.step_fn(state, policy, shock, constants)

    sav_n = jax.random.uniform(sn_key, (64,), minval=0.1, maxval=0.5)
    L_n = jax.random.uniform(Ln_key, (64,), minval=0.6, maxval=1.4)
    next_policy = jnp.stack([sav_n, L_n], axis=1)

    resid_hand = MODEL_HAND.equations_fn(state, policy, next_state, next_policy, constants)
    resid_ad = MODEL_AD.equations_fn(state, policy, next_state, next_policy, constants)

    for name in ("euler", "labor_foc"):
        np.testing.assert_allclose(
            np.asarray(resid_ad[name]),
            np.asarray(resid_hand[name]),
            rtol=1e-4, atol=1e-5,
            err_msg=f"autodiff != hand-derived on {name}",
        )


def test_bm_labor_autodiff_registers_and_loads():
    from deqn_jax.models import list_models, load_model

    model = load_model("bm_labor_autodiff")
    assert model.name == "bm_labor_autodiff"
    assert "bm_labor_autodiff" in dict(list_models())
    assert model.equation_names == ("euler", "labor_foc")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
