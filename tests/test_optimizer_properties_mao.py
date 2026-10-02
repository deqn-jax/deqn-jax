"""Defining-property tests for the Multi-Adaptive Optimizer (``mao``).

What the code computes (``optimizers/mao.py``):

* for each equation ``i`` with gradient ``J_i`` (row ``i`` of the per-equation
  Jacobian), Adam moments ``m_i <- b1 m_i + (1-b1) J_i`` and
  ``v_i <- b2 v_i + (1-b2) J_i**2`` (lines 84-93), bias-corrected with the
  shared step count (lines 97-98, 103-104);
* the update is ``-lr * mean_i m_hat_i / (sqrt(v_hat_i) + eps)``
  (lines 106-109) -- the mean of the per-equation Adam directions, i.e. the
  MultiAdam rule (Yao et al., 2023). The comment on line 100 says "sum"; the
  code averages.
* in the MAO step (``make_grad_step_mao``) the Jacobian is ``jacrev`` of the
  *unweighted* base per-equation losses (lines 179-191), and the update is
  multiplied by ``lr_scale`` (line 215).

Consequences tested here, each against an independent computation: with one
equation MAO is Adam (``optax.adam``); each equation's moments see only its
own gradient; the update is invariant to rescaling any one equation's loss
(the property that motivates per-equation moments); and the train step feeds
the right Jacobian.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from deqn_jax.optimizers.mao import MAOTransform, make_grad_step_mao

LR, B1, B2, EPS = 0.01, 0.8, 0.95, 1e-8


def _params():
    return {"w": jnp.array([[0.1, -0.2], [0.3, 0.4]]), "b": jnp.array([0.05, -0.1])}


def _random_jacobian(rng, n_eq):
    return {
        "w": jnp.asarray(rng.standard_normal((n_eq, 2, 2))),
        "b": jnp.asarray(rng.standard_normal((n_eq, 2))),
    }


def test_single_equation_mao_is_adam():
    """n_tasks=1: the update sequence equals optax.adam with the same b1, b2, eps."""
    mao = MAOTransform(learning_rate=LR, beta1=B1, beta2=B2, epsilon=EPS, n_tasks=1)
    adam = optax.adam(LR, b1=B1, b2=B2, eps=EPS)
    p = _params()
    s_mao, s_adam = mao.init(p), adam.init(p)
    rng = np.random.default_rng(0)
    for _ in range(6):
        jac = _random_jacobian(rng, 1)
        grad = jax.tree.map(lambda j: j[0], jac)
        u_mao, s_mao = mao.update(jac, s_mao, p)
        u_adam, s_adam = adam.update(grad, s_adam, p)
        for k in p:
            np.testing.assert_allclose(u_mao[k], u_adam[k], rtol=1e-10)


def test_each_equation_moments_track_only_its_own_gradient():
    n_eq = 3
    mao = MAOTransform(learning_rate=LR, beta1=B1, beta2=B2, epsilon=EPS, n_tasks=n_eq)
    p = _params()
    state = mao.init(p)
    rng = np.random.default_rng(1)
    m = {k: np.zeros((n_eq,) + v.shape) for k, v in p.items()}
    v = {k: np.zeros((n_eq,) + v.shape) for k, v in p.items()}
    for _ in range(4):
        jac = _random_jacobian(rng, n_eq)
        for k in p:
            for i in range(n_eq):
                g = np.asarray(jac[k][i])
                m[k][i] = B1 * m[k][i] + (1 - B1) * g
                v[k][i] = B2 * v[k][i] + (1 - B2) * g**2
        _, state = mao.update(jac, state, p)
        for k in p:
            np.testing.assert_allclose(state.m[k], m[k], rtol=1e-12)
            np.testing.assert_allclose(state.v[k], v[k], rtol=1e-12)


def test_update_is_the_mean_of_per_equation_adam_directions():
    """Independent per-equation optax.adam runs, averaged, equal the MAO update."""
    n_eq = 3
    mao = MAOTransform(learning_rate=LR, beta1=B1, beta2=B2, epsilon=EPS, n_tasks=n_eq)
    adams = [optax.adam(LR, b1=B1, b2=B2, eps=EPS) for _ in range(n_eq)]
    p = _params()
    s_mao = mao.init(p)
    s_adams = [a.init(p) for a in adams]
    rng = np.random.default_rng(2)
    for _ in range(5):
        jac = _random_jacobian(rng, n_eq)
        u_mao, s_mao = mao.update(jac, s_mao, p)
        per_eq = []
        for i, a in enumerate(adams):
            u_i, s_adams[i] = a.update(jax.tree.map(lambda j: j[i], jac), s_adams[i], p)
            per_eq.append(u_i)
        for k in p:
            expected = np.mean([np.asarray(u[k]) for u in per_eq], axis=0)
            np.testing.assert_allclose(u_mao[k], expected, rtol=1e-10)


def test_update_is_invariant_to_rescaling_one_equation():
    """Scaling equation i's loss by c_i > 0 scales J_i by c_i and leaves the step unchanged."""
    n_eq = 3
    c = np.array([1e-4, 1.0, 250.0])
    p = _params()
    rng = np.random.default_rng(3)
    jacs = [_random_jacobian(rng, n_eq) for _ in range(4)]
    eps = (
        1e-14  # negligible against every |J_i| here, so invariance is exact to rounding
    )

    def run(scale):
        mao = MAOTransform(
            learning_rate=LR, beta1=B1, beta2=B2, epsilon=eps, n_tasks=n_eq
        )
        s = mao.init(p)
        out = []
        for jac in jacs:
            scaled = jax.tree.map(
                lambda j: j * jnp.asarray(scale).reshape((-1,) + (1,) * (j.ndim - 1)),
                jac,
            )
            u, s = mao.update(scaled, s, p)
            out.append(u)
        return out

    for u_ref, u_scaled in zip(run(np.ones(n_eq)), run(c)):
        for k in p:
            np.testing.assert_allclose(u_scaled[k], u_ref[k], rtol=1e-8)


@pytest.mark.parametrize("lr_scale", [1.0, 0.25])
def test_train_step_feeds_per_equation_gradients_of_the_base_loss(lr_scale):
    """make_grad_step_mao: two steps equal per-equation Adam on jacrev of base per-eq losses.

    The reference recomputes the per-equation losses with ``compute_loss``
    (same loss key the step draws from ``state.key``), takes ``jax.jacrev``
    and runs per-equation Adam in NumPy -- no call into MAOTransform. Two
    steps, because after one step every Adam direction is sign(J) and a
    wrong Jacobian with the right signs would pass. Per-equation Adam cannot
    see a positive per-equation rescaling of a row (see the invariance test),
    so the property is "row i is the gradient of equation i, up to scale".
    """
    from deqn_jax.config import OptimizerConfig
    from deqn_jax.models import load_model
    from deqn_jax.training.episode import sample_initial_states
    from deqn_jax.training.loss import compute_loss, eq_losses_to_array
    from deqn_jax.training.trainer import create_train_state

    model = load_model("brock_mirman")
    n_eq = len(model.equation_names)
    state, opt, _ = create_train_state(
        model=model,
        key=jax.random.PRNGKey(0),
        hidden_sizes=(4,),
        batch_size=4,
        n_equations=n_eq,
        optimizer_config=OptimizerConfig(
            name="mao", learning_rate=LR, beta1=B1, beta2=B2, epsilon=EPS
        ),
    )
    batch = sample_initial_states(model, jax.random.PRNGKey(1), 4)
    mc = 2
    step = make_grad_step_mao(
        model, opt, mc, None, None, "none", 0.0, False, None, None
    )

    arrays, static = eqx.partition(state.params, eqx.is_array)
    theta = [np.asarray(a) for a in jax.tree.leaves(arrays)]
    m = [np.zeros((n_eq,) + t.shape) for t in theta]
    v = [np.zeros((n_eq,) + t.shape) for t in theta]
    treedef = jax.tree.structure(arrays)
    key = state.key
    for t in (1, 2):
        loss_key, key = jax.random.split(key)

        def per_eq(a, loss_key=loss_key):
            _, eq = compute_loss(model, eqx.combine(a, static), batch, loss_key, mc)
            return eq_losses_to_array(eq)

        jac = jax.jacrev(per_eq)(
            jax.tree.unflatten(treedef, [jnp.asarray(x) for x in theta])
        )
        jac = [np.asarray(j) for j in jax.tree.leaves(jac)]
        assert jac[0].shape[0] == n_eq
        for k, j in enumerate(jac):
            m[k] = B1 * m[k] + (1 - B1) * j
            v[k] = B2 * v[k] + (1 - B2) * j**2
            m_hat, v_hat = m[k] / (1 - B1**t), v[k] / (1 - B2**t)
            direction = np.mean(m_hat / (np.sqrt(v_hat) + EPS), axis=0)
            theta[k] = theta[k] - lr_scale * LR * direction
        state, _ = step(state, batch, jnp.asarray(lr_scale))
        got = jax.tree.leaves(eqx.filter(state.params, eqx.is_array))
        for e, g in zip(theta, got):
            np.testing.assert_allclose(np.asarray(g), e, rtol=1e-9, atol=1e-12)
