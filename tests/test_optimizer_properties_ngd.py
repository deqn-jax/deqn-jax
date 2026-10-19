"""Defining-property tests for the diagonal-Fisher "natural gradient" optimizer (``ngd``).

What the code computes (``optimizers/ngd.py``):

* "Fisher" is ``F <- decay * F + (1 - decay) * g**2`` (lines 51-55), where
  ``g`` is whatever gradient the transformation receives. In training that is
  the STANDARD step's gradient of the *minibatch-mean* loss
  (``optimizers/standard.py``), so ``F`` is an EMA of the squared mean
  gradient -- not the per-sample empirical Fisher ``mean_i (grad l_i)**2``.
  ``F`` starts at zero and is not bias-corrected.
* The update is ``-lr * g / (sqrt(F) + damping)`` (lines 57-61), with the new
  ``F``. The module docstring says the same.

That is RMSProp (``optax.rmsprop`` with ``eps=damping, eps_in_sqrt=False``;
optax 0.2.6 observed), which the first
tests confirm against optax and a hand computation. A diagonal *natural
gradient* step is ``-lr * F^{-1} g``; its defining property -- equivariance
under a reparametrisation ``theta = C phi`` -- fails for the ``sqrt(F)``
preconditioner, which the last test pins.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from deqn_jax.optimizers.ngd import ngd

X = np.array([[1.0, 0.5, -0.3], [0.2, -1.0, 0.8], [0.7, 0.1, 0.4], [-0.5, 0.9, 1.1]])
Y = np.array([0.3, -0.2, 1.0, 0.5])


def _loss(p):
    """Tiny linear model: mean squared error over a batch of four rows."""
    pred = jnp.asarray(X) @ p["w"] + p["b"]
    return jnp.mean((pred - jnp.asarray(Y)) ** 2)


def _hand_grad(w, b):
    """d/dw, d/db of mean((Xw + b - y)^2), by hand."""
    e = X @ w + b - Y
    return 2.0 * X.T @ e / len(Y), 2.0 * np.mean(e)


def test_fisher_is_ema_of_squared_minibatch_gradient_and_update_divides_by_its_root():
    lr, damping, decay = 0.01, 1e-4, 0.9
    opt = ngd(learning_rate=lr, damping=damping, decay=decay)
    w, b = np.array([0.1, -0.2, 0.3]), 0.05
    p = {"w": jnp.asarray(w), "b": jnp.asarray(b)}
    state = opt.init(p)
    Fw, Fb = np.zeros(3), 0.0
    for _ in range(4):
        gw, gb = _hand_grad(w, b)
        Fw = decay * Fw + (1 - decay) * gw**2
        Fb = decay * Fb + (1 - decay) * gb**2
        uw = -lr * gw / (np.sqrt(Fw) + damping)
        ub = -lr * gb / (np.sqrt(Fb) + damping)
        updates, state = opt.update(jax.grad(_loss)(p), state, p)
        np.testing.assert_allclose(state.fisher_diag["w"], Fw, rtol=1e-12)
        np.testing.assert_allclose(state.fisher_diag["b"], Fb, rtol=1e-12)
        np.testing.assert_allclose(updates["w"], uw, rtol=1e-12)
        np.testing.assert_allclose(updates["b"], ub, rtol=1e-12)
        p = optax.apply_updates(p, updates)
        w, b = w + uw, b + ub


def test_first_step_is_not_bias_corrected():
    """F_1 = (1 - decay) g^2, so step 1 has size lr / sqrt(1 - decay) per coordinate."""
    lr, decay = 1e-3, 0.999
    opt = ngd(learning_rate=lr, damping=0.0, decay=decay)
    g = jnp.array([3.0, -0.01, 50.0])
    updates, _ = opt.update(g, opt.init(g), g)
    np.testing.assert_allclose(
        updates, -lr / np.sqrt(1 - decay) * np.sign(g), rtol=1e-12
    )


def test_equals_optax_rmsprop():
    lr, damping, decay = 0.02, 1e-3, 0.95
    mine, ref = (
        ngd(lr, damping, decay),
        optax.rmsprop(lr, decay=decay, eps=damping, eps_in_sqrt=False),
    )
    rng = np.random.default_rng(0)
    p = jnp.zeros(5)
    s_mine, s_ref = mine.init(p), ref.init(p)
    for _ in range(6):
        g = jnp.asarray(rng.standard_normal(5))
        u_mine, s_mine = mine.update(g, s_mine, p)
        u_ref, s_ref = ref.update(g, s_ref, p)
        np.testing.assert_allclose(u_mine, u_ref, rtol=1e-12)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ngd preconditions by sqrt(F) (ngd.py:59), i.e. RMSProp, not F^{-1}: the step "
        "is not a natural-gradient step and is not reparametrisation-equivariant"
    ),
)
def test_step_is_reparametrisation_equivariant_like_a_natural_gradient():
    """Natural gradient: rescaling theta = C phi maps the step as dphi = C^{-1} dtheta.

    With g_phi = C g_theta and F_phi = C^2 F_theta, the step -lr F^{-1} g gives
    dphi = C^{-1} dtheta. The sqrt(F) preconditioner gives dphi = dtheta instead.
    Damping is set negligible so only the preconditioner's exponent matters.
    """
    C = np.array([2.0, 0.5, 3.0])
    w0 = np.array([0.1, -0.2, 0.3])

    def f_theta(w):
        return _loss({"w": w, "b": jnp.asarray(0.0)})

    def f_phi(v):
        return f_theta(jnp.asarray(C) * v)

    def run(f, x0):
        opt = ngd(learning_rate=0.01, damping=1e-14, decay=0.9)
        x = jnp.asarray(x0)
        state = opt.init(x)
        for _ in range(3):
            u, state = opt.update(jax.grad(f)(x), state, x)
            x = x + u
        return np.asarray(x) - x0

    d_theta = run(f_theta, w0)
    d_phi = run(f_phi, w0 / C)
    np.testing.assert_allclose(d_phi, d_theta / C, rtol=1e-6)
