"""olg_analytic_6_autodiff: Euler equations derived from the Lagrangian.

Acceptance against the hand-written olg_analytic_6 and the Krueger-Kubler
closed form:

1. at the closed-form policy, on a grid of states and at every shock node,
   the derived residuals vanish to rounding;
2. at random states and random (also infeasible) policies the derived
   residuals equal the hand-written ones, so the zero sets coincide;
3. negative controls: a Lagrangian that differentiates through prices, and
   the closed form under a perturbed discount factor, do not vanish, so (1)
   discriminates.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deqn_jax.models import load_model
from deqn_jax.models.olg_analytic_6 import analytic_policy
from deqn_jax.models.olg_analytic_6_autodiff.equations import objective, prices
from deqn_jax.training.lagrangian import residuals_from_lagrangian

NODES = jnp.array(list(itertools.product([-1.0, 1.0], repeat=2)))  # GH-2 x GH-2


@pytest.fixture(scope="module")
def models():
    return load_model("olg_analytic_6"), load_model("olg_analytic_6_autodiff")


def _grid_states(model):
    """3^5 capital profiles around the zero-shock SS x 4 shock states."""
    ss, _ = model.steady_state_fn(model.constants)
    c = model.constants
    rows = []
    for scale in itertools.product([0.6, 1.0, 1.4], repeat=5):
        for e1, e2 in NODES.tolist():
            k = np.asarray(ss[:5]) * np.asarray(scale)
            eta = c["eta_mid"] + c["eta_half"] * e1
            delta = c["delta_mid"] + c["delta_half"] * e2
            rows.append(np.concatenate([k, [eta, delta]]))
    return jnp.asarray(np.array(rows))


def _residuals_per_node(eq_fn, model, state, policy, next_policy_fn):
    """Residuals at every shock node: dict name -> [n_nodes, batch]."""
    out = []
    for node in NODES:
        shock = jnp.broadcast_to(node, (state.shape[0], 2))
        next_state = model.step_fn(state, policy, shock, model.constants)
        next_policy = next_policy_fn(next_state)
        out.append(eq_fn(state, policy, next_state, next_policy, model.constants))
    return {k: jnp.stack([o[k] for o in out]) for k in out[0]}


def test_registered_with_hand_model_layout(models):
    hand, auto = models
    assert auto.equation_names == hand.equation_names
    assert auto.state_names == hand.state_names
    assert auto.policy_names == hand.policy_names


def test_closed_form_zeroes_derived_residuals(models):
    hand, auto = models
    state = _grid_states(auto)
    policy = analytic_policy(state, auto.constants)
    res = _residuals_per_node(
        auto.equations_fn,
        auto,
        state,
        policy,
        lambda s: analytic_policy(s, auto.constants),
    )
    worst = max(float(jnp.max(jnp.abs(v))) for v in res.values())
    assert worst < 1e-12, f"derived residual at the closed form: {worst:.3e}"


def test_derived_equals_hand_written_at_random_points(models):
    """Random states and random policies, including negative consumption
    (both variants cap u' at u'(1e-3) there)."""
    hand, auto = models
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    n = 512
    ss, _ = auto.steady_state_fn(auto.constants)
    c = auto.constants
    k = ss[:5] * jax.random.uniform(k1, (n, 5), minval=0.3, maxval=1.8)
    eps = jax.random.choice(k2, jnp.array([-1.0, 1.0]), (n, 2))
    state = jnp.concatenate(
        [
            k,
            (c["eta_mid"] + c["eta_half"] * eps[:, :1]),
            (c["delta_mid"] + c["delta_half"] * eps[:, 1:]),
        ],
        axis=1,
    )
    closed = analytic_policy(state, c)
    policy = closed * jax.random.uniform(k3, (n, 5), minval=0.0, maxval=2.5)

    def next_policy_fn(s):
        return analytic_policy(s, c) * jax.random.uniform(
            k4, (n, 5), minval=0.0, maxval=2.5
        )

    r_auto = _residuals_per_node(auto.equations_fn, auto, state, policy, next_policy_fn)
    r_hand = _residuals_per_node(hand.equations_fn, hand, state, policy, next_policy_fn)
    for name in hand.equation_names:
        np.testing.assert_allclose(r_auto[name], r_hand[name], rtol=1e-12, atol=1e-12)
    # Not a degenerate comparison: away from the closed form both are large.
    assert float(jnp.median(jnp.abs(r_hand["euler_h1"]))) > 0.1


def test_training_gradient_equals_hand_written(models):
    """Same gradient of the squared residuals with respect to the period-t
    and the next-period policy, so training sees the same signal."""
    hand, auto = models
    c = auto.constants
    state = _grid_states(auto)[::5]
    policy = analytic_policy(state, c) * 0.8
    shock = jnp.broadcast_to(NODES[1], (state.shape[0], 2))
    next_state = auto.step_fn(state, policy, shock, c)
    next_policy = analytic_policy(next_state, c) * 1.3

    def loss(model, p, q):
        ns = model.step_fn(state, p, shock, c)
        out = model.equations_fn(state, p, ns, q, c)
        return sum(jnp.sum(v**2) for v in out.values())

    for argnum in (0, 1):
        g_auto = jax.grad(lambda p, q: loss(auto, p, q), argnums=argnum)(
            policy, next_policy
        )
        g_hand = jax.grad(lambda p, q: loss(hand, p, q), argnums=argnum)(
            policy, next_policy
        )
        assert float(jnp.max(jnp.abs(g_hand))) > 1e-3
        np.testing.assert_allclose(g_auto, g_hand, rtol=1e-10, atol=1e-10)


def _closed_form_worst(eq_fn, model, constants):
    state = _grid_states(model)[::7]
    policy = analytic_policy(state, model.constants)
    res = _residuals_per_node(
        lambda *a: eq_fn(*a[:4], constants),
        model,
        state,
        policy,
        lambda s: analytic_policy(s, model.constants),
    )
    return max(float(jnp.max(jnp.abs(v))) for v in res.values())


def test_negative_control_prices_differentiated(models):
    """Computing prices inside the objective (a planner's derivative, not
    the cohorts') breaks the closed-form zero."""
    _, auto = models

    def planner_objective(state, k_next, policy, _prices, constants):
        return objective(
            state, k_next, policy, prices(state, policy, constants), constants
        )

    wrong = residuals_from_lagrangian(
        planner_objective,
        auto.step_fn,
        endogenous=range(5),
        euler_names=auto.equation_names,
        n_shocks=2,
    )
    assert _closed_form_worst(wrong, auto, auto.constants) > 1e-2


def test_negative_control_discount(models):
    _, auto = models
    perturbed = {**auto.constants, "beta": auto.constants["beta"] * 1.01}
    assert _closed_form_worst(auto.equations_fn, auto, perturbed) > 1e-3
