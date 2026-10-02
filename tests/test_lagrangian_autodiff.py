"""residuals_from_lagrangian: the period-return helper as its special case,
and the pieces the period-return helper never had (constraints, prices,
ratio form, time-varying discount).

The OLG application is tested in ``test_olg_analytic_6_autodiff.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deqn_jax.models import load_model
from deqn_jax.models._complementarity import fischer_burmeister
from deqn_jax.training.autodiff import euler_from_period_return
from deqn_jax.training.lagrangian import Constraint, residuals_from_lagrangian
from tests._frozen_period_return import frozen_euler_from_period_return


def _transitions(model, key, k_range, policy_ranges, n=64):
    """Random (state, policy, next_state, next_policy) with next_state from step_fn."""
    keys = jax.random.split(key, 4 + 2 * len(policy_ranges))
    k = jax.random.uniform(keys[0], (n,), minval=k_range[0], maxval=k_range[1])
    z = jax.random.uniform(keys[1], (n,), minval=-0.25, maxval=0.25)
    state = jnp.stack([k, z], axis=1)

    def draw(offset):
        return jnp.stack(
            [
                jax.random.uniform(keys[offset + i], (n,), minval=lo, maxval=hi)
                for i, (lo, hi) in enumerate(policy_ranges)
            ],
            axis=1,
        )

    policy = draw(4)
    next_policy = draw(4 + len(policy_ranges))
    shock = jax.random.normal(keys[2], (n, model.n_shocks))
    next_state = model.step_fn(state, policy, shock, model.constants)
    return state, policy, next_state, next_policy


CASES = {
    "brock_mirman_autodiff": dict(
        kwargs=dict(capital_idx=0, exog_idx=(1,), n_shocks=1),
        k_range=(0.5, 3.0),
        policy_ranges=[(0.1, 0.6)],
    ),
    "bm_labor_autodiff": dict(
        kwargs=dict(
            capital_idx=0,
            exog_idx=(1,),
            n_shocks=1,
            equation_name="euler",
            intratemporal_policy_idx=(1,),
            intratemporal_equation_names=("labor_foc",),
        ),
        k_range=(2.0, 10.0),
        policy_ranges=[(0.1, 0.5), (0.6, 1.4)],
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_shipped_autodiff_models_unchanged(name):
    """The re-expressed helper gives the residuals of the frozen original,
    and the same gradient with respect to the period-t policy."""
    case = CASES[name]
    model = load_model(name)
    pkg = __import__(f"deqn_jax.models.{name}.equations", fromlist=["x"])
    frozen = frozen_euler_from_period_return(
        pkg.period_return, model.step_fn, **case["kwargs"]
    )
    args = _transitions(
        model, jax.random.PRNGKey(3), case["k_range"], case["policy_ranges"]
    )
    new = model.equations_fn(*args, model.constants)
    old = frozen(*args, model.constants)
    assert list(new) == list(old) == list(model.equation_names)
    for eq in old:
        np.testing.assert_array_equal(new[eq], old[eq])

    def loss(fn, p):
        out = fn(args[0], p, args[2], args[3], model.constants)
        return sum(jnp.sum(v**2) for v in out.values())

    g_new = jax.grad(lambda p: loss(model.equations_fn, p))(args[1])
    g_old = jax.grad(lambda p: loss(frozen, p))(args[1])
    np.testing.assert_allclose(g_new, g_old, rtol=1e-12, atol=1e-12)


def _pi_two(K, K_next, z, policy, constants, *, agent_index):
    c = (z[0] + 0.1 * (agent_index + 1)) - K_next / jnp.maximum(K, 1e-3)
    return jnp.log(jnp.maximum(c, 1e-8))


def _step_two(state, policy, shock, constants):
    del shock, constants
    return jnp.stack(
        [policy[:, 0] * state[:, 0], policy[:, 0] * state[:, 1], state[:, 2]], axis=1
    )


def test_multi_agent_mode_unchanged():
    kwargs = dict(
        exog_idx=(2,),
        n_shocks=1,
        capital_indices=(0, 1),
        equation_names=("euler_a0", "euler_a1"),
    )
    state = jnp.array([[1.0, 1.0, 1.0], [0.5, 0.8, 1.05]])
    policy = jnp.full((2, 1), 0.3)
    next_state = _step_two(state, policy, None, None)
    constants = {"beta": 0.96}
    args = (state, policy, next_state, jnp.full((2, 1), 0.35), constants)
    new = euler_from_period_return(_pi_two, _step_two, **kwargs)(*args)
    old = frozen_euler_from_period_return(_pi_two, _step_two, **kwargs)(*args)
    for eq in old:
        np.testing.assert_array_equal(new[eq], old[eq])


# ---------------------------------------------------------------------------
# A two-period consumption-saving toy with every residual class
# ---------------------------------------------------------------------------
# state = (a, y): assets and income; policy = (c, lam, mu). The household
# chooses c and a' subject to the budget a' = R a + y - c (equality, lam)
# and a' >= 0 (inequality, mu). Hand-derived conditions:
#   u'(c) - lam = 0;  -lam + mu + beta R lam' = 0;  budget;  FB(mu, a').

R = 1.02
CONST = {"beta": 0.95}


def _obj(s, xn, p, prices, constants):
    return jnp.log(p[0])


def _budget(s, xn, p, prices, constants):
    return R * s[0] + s[1] - p[0] - xn[0]


def _borrow(s, xn, p, prices, constants):
    return xn[0]


def _step_toy(state, policy, shock, constants):
    a_next = R * state[:, 0] + state[:, 1] - policy[:, 0]
    return jnp.stack([a_next, state[:, 1] + 0.1 * shock[:, 0]], axis=1)


def _toy(**kw):
    return residuals_from_lagrangian(
        _obj,
        _step_toy,
        endogenous=(0,),
        euler_names=("euler",),
        n_shocks=1,
        static_controls=(0,),
        static_names=("foc_c",),
        equalities=(Constraint("budget", _budget, 1),),
        inequalities=(Constraint("borrowing", _borrow, 2),),
        **kw,
    )


def _toy_batch():
    state = jnp.array([[0.5, 1.0], [0.0, 0.8], [1.2, 1.1]])
    policy = jnp.array([[1.1, 0.9, 0.0], [0.7, 1.3, 0.2], [1.5, 0.6, 0.1]])
    next_state = _step_toy(state, policy, jnp.zeros((3, 1)), CONST)
    next_policy = jnp.array([[1.0, 1.0, 0.0], [0.9, 1.1, 0.0], [1.2, 0.8, 0.3]])
    return state, policy, next_state, next_policy


@pytest.mark.parametrize("form", ["raw", "ratio"])
def test_every_residual_class_matches_hand_derivation(form):
    state, policy, next_state, next_policy = _toy_batch()
    out = _toy(euler_form=form)(state, policy, next_state, next_policy, CONST)
    assert list(out) == ["euler", "foc_c", "budget", "borrowing"]
    c, lam, mu = policy.T
    lam_next = next_policy[:, 1]
    a_next = next_state[:, 0]
    euler_raw = lam - mu - CONST["beta"] * R * lam_next
    # Ratio scale: A = -d(F + lam*h)/da' = lam; mu sits in the numerator.
    euler = euler_raw if form == "raw" else euler_raw / lam
    np.testing.assert_allclose(out["euler"], euler, atol=1e-12)
    np.testing.assert_allclose(out["foc_c"], -(1.0 / c - lam), atol=1e-12)
    np.testing.assert_allclose(
        out["budget"], R * state[:, 0] + state[:, 1] - c - a_next, atol=1e-12
    )
    np.testing.assert_allclose(
        out["borrowing"], fischer_burmeister(mu, a_next), atol=1e-12
    )


def test_prices_are_taken_as_given():
    """Prices enter undifferentiated: an objective r*k' with r = r(k') via
    prices_fn has derivative r, not r + k' dr/dk'."""

    def obj(s, xn, p, prices, constants):
        return prices * xn[0] - 0.5 * xn[0] ** 2

    def prices_fn(s, p, constants):
        return 2.0 + p[0] ** 2

    def step(state, policy, shock, constants):
        return jnp.stack([policy[:, 0], state[:, 1]], axis=1)

    fn = residuals_from_lagrangian(
        obj,
        step,
        (0,),
        ("e",),
        n_shocks=1,
        prices_fn=prices_fn,
        discount=0.0,
        euler_form="raw",
    )
    state = jnp.array([[1.0, 0.0]])
    policy = jnp.array([[0.7]])
    out = fn(state, policy, step(state, policy, None, None), policy, {})
    np.testing.assert_allclose(out["e"], -(2.0 + 0.49 - 0.7), atol=1e-12)


def test_time_varying_discount():
    """A callable discount is evaluated at t, per sample."""

    def obj(s, xn, p, prices, constants):
        return -(xn[0] ** 2) + s[0]

    def step(state, policy, shock, constants):
        return jnp.stack([policy[:, 0], state[:, 1] + 1.0], axis=1)

    fn = residuals_from_lagrangian(
        obj,
        step,
        (0,),
        ("e",),
        n_shocks=1,
        discount=lambda s, constants: 0.9 ** s[1],
        euler_form="raw",
    )
    state = jnp.array([[1.0, 0.0], [1.0, 2.0]])
    policy = jnp.array([[0.5], [0.5]])
    out = fn(state, policy, step(state, policy, None, None), policy, {})
    np.testing.assert_allclose(out["e"], -(-1.0 + 0.9 ** state[:, 1]), atol=1e-12)


def test_next_policy_gradient_switch():
    state, policy, next_state, next_policy = _toy_batch()

    def g(stop):
        fn = _toy(stop_next_policy_gradient=stop)
        return jax.grad(
            lambda q: jnp.sum(fn(state, policy, next_state, q, CONST)["euler"])
        )(next_policy)

    assert float(jnp.max(jnp.abs(g(True)))) == 0.0
    assert float(jnp.max(jnp.abs(g(False)))) > 0.0


@pytest.mark.parametrize(
    "kw,match",
    [
        (dict(endogenous=(0, 0), euler_names=("a", "b")), "repeat"),
        (dict(euler_names=("a", "b")), "names"),
        (dict(euler_form="relative"), "euler_form"),
        (dict(static_controls=(1,)), "multiplier column"),
    ],
)
def test_validation(kw, match):
    base = dict(endogenous=(0,), euler_names=("euler",), n_shocks=1)
    base.update(kw)
    with pytest.raises(ValueError, match=match):
        residuals_from_lagrangian(
            _obj,
            _step_toy,
            equalities=(Constraint("budget", _budget, 1),),
            **base,
        )
