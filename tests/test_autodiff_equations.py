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

    resid_hand = MODEL_HAND.equations_fn(
        state, policy, state, policy, MODEL_HAND.constants
    )
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
        MODEL_HAND.equations_fn(state, policy, next_state, next_policy, constants)[
            "euler"
        ]
    )
    resid_ad = np.asarray(
        MODEL_AD.equations_fn(state, policy, next_state, next_policy, constants)[
            "euler"
        ]
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

    resid_hand = MODEL_HAND.equations_fn(
        state, policy, state, policy, MODEL_HAND.constants
    )
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

    resid_hand = MODEL_HAND.equations_fn(
        state, policy, next_state, next_policy, constants
    )
    resid_ad = MODEL_AD.equations_fn(state, policy, next_state, next_policy, constants)

    for name in ("euler", "labor_foc"):
        np.testing.assert_allclose(
            np.asarray(resid_ad[name]),
            np.asarray(resid_hand[name]),
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"autodiff != hand-derived on {name}",
        )


def test_bm_labor_autodiff_registers_and_loads():
    from deqn_jax.models import list_models, load_model

    model = load_model("bm_labor_autodiff")
    assert model.name == "bm_labor_autodiff"
    assert "bm_labor_autodiff" in dict(list_models())
    assert model.equation_names == ("euler", "labor_foc")


# ---------------------------------------------------------------------------
# Envelope theorem regression: the autodiff Euler residual must be
# insensitive (in the autograd sense) to next_policy. The forward value
# depends on next_policy's value (via K_{t+2} reconstruction and the policy
# slot of Π), but the gradient of the residual wrt next_policy must be
# exactly zero — otherwise backprop ‖residual‖² → θ would pick up the
# spurious ∂Π/∂c_{t+1} · ∂π/∂s_{t+1} term that envelope theorem zeros.
#
# These tests will FAIL if anyone removes the jax.lax.stop_gradient on
# next_policy in src/deqn_jax/training/autodiff.py. They are the regression
# guard the contract has been missing since the file was written.
# ---------------------------------------------------------------------------


def _envelope_grad_wrt_next_policy(model, state, policy, next_state, next_policy):
    """Helper: return ∂(euler residual at row 0)/∂next_policy.

    With the envelope freeze in place this is exactly zero. Without it,
    the gradient picks up at least the K_{t+2}-reconstruction leak and,
    for models with Π depending on policy directly, the slot-3 leak too.
    """

    def euler_scalar(np_arr):
        out = model.equations_fn(state, policy, next_state, np_arr, model.constants)
        return out["euler"][0]

    return jax.grad(euler_scalar)(next_policy)


def test_envelope_freeze_brock_mirman(models):
    """BM-style model: next_policy must not contribute to residual gradient.

    In Brock-Mirman (Path A), K_{t+2} = next_policy[..., 0]. Without
    stop_gradient, ∂euler/∂next_policy is nonzero through the K_{t+2} path.
    Π itself does not read policy, so this is the *only* leak — but it is
    enough on its own.
    """
    _MODEL_HAND, MODEL_AD = models
    # Off-SS state so the residual carries non-trivial second-order content.
    state = jnp.array([[1.5, 0.1]])
    policy = jnp.array([[0.3]])
    shock = jnp.zeros((1, 1))
    next_state = MODEL_AD.step_fn(state, policy, shock, MODEL_AD.constants)
    next_policy = jnp.array([[0.4]])

    grad_np = _envelope_grad_wrt_next_policy(
        MODEL_AD, state, policy, next_state, next_policy
    )
    max_abs = float(jnp.max(jnp.abs(grad_np)))
    assert max_abs < 1e-12, (
        f"Envelope freeze missing in brock_mirman_autodiff: "
        f"|∂euler/∂next_policy|_max = {max_abs:.3e} (should be 0). "
        f"stop_gradient on next_policy must be applied where it feeds "
        f"K_{{t+2}} reconstruction and dPi2."
    )


def test_envelope_freeze_bm_labor(labor_models):
    """bm_labor_autodiff: BOTH leak paths are live (Path A K_{t+2} chain
    AND direct slot-3 dependence — Π reads labor for the disutility term).
    Either gives a nonzero gradient if stop_gradient is removed.
    """
    _MODEL_HAND, MODEL_AD = labor_models
    ss_state, ss_policy = MODEL_AD.steady_state_fn(MODEL_AD.constants)
    # Off-SS state and a deliberately mismatched next_policy so Π's labor
    # slot meaningfully differs from SS.
    state = ss_state[None, :].at[0, 0].add(0.5)
    policy = ss_policy[None, :]
    shock = jnp.zeros((1, 1))
    next_state = MODEL_AD.step_fn(state, policy, shock, MODEL_AD.constants)
    next_policy = ss_policy[None, :].at[0, 1].set(float(ss_policy[1]) * 1.1)

    grad_np = _envelope_grad_wrt_next_policy(
        MODEL_AD, state, policy, next_state, next_policy
    )
    max_abs = float(jnp.max(jnp.abs(grad_np)))
    assert max_abs < 1e-12, (
        f"Envelope freeze missing in bm_labor_autodiff: "
        f"|∂euler/∂next_policy|_max = {max_abs:.3e} (should be 0). "
        f"Π reads policy directly here, so removing stop_gradient also "
        f"opens the slot-3 leak in addition to the K_{{t+2}} one."
    )


def test_envelope_freeze_does_not_zero_t_period_policy(models):
    """Sanity check: the envelope freeze is on next_policy, NOT on policy.
    The t-period control must remain in the autograd graph so backprop
    drives θ via the LHS ∂Π/∂c term. ∂euler/∂policy must be nonzero.
    """
    _MODEL_HAND, MODEL_AD = models
    state = jnp.array([[1.5, 0.1]])
    policy = jnp.array([[0.3]])
    shock = jnp.zeros((1, 1))
    next_policy = jnp.array([[0.4]])

    def euler_scalar(p_arr):
        # Recompute next_state from the test policy so backprop sees the
        # full chain through policy_t → next_state → euler.
        ns = MODEL_AD.step_fn(state, p_arr, shock, MODEL_AD.constants)
        out = MODEL_AD.equations_fn(state, p_arr, ns, next_policy, MODEL_AD.constants)
        return out["euler"][0]

    grad_p = jax.grad(euler_scalar)(policy)
    max_abs = float(jnp.max(jnp.abs(grad_p)))
    assert max_abs > 1e-6, (
        f"t-period policy gradient unexpectedly zero ({max_abs:.3e}); "
        f"the envelope freeze should NOT extend to policy at time t — "
        f"that would block backprop from driving θ via the LHS ∂Π/∂c term."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
