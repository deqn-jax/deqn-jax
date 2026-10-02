"""Defining-property tests for the in-house Shampoo optimizer (``shampoo``).

Reference algorithm: Gupta, Koren & Singer (2018), Algorithm 1, for a matrix
parameter with gradient ``G_t``::

    L_t = L_{t-1} + G_t G_t^T,   R_t = R_{t-1} + G_t^T G_t,   L_0 = R_0 = eps I
    W_{t+1} = W_t - lr * L_t^{-1/4} G_t R_t^{-1/4}

(the statistics enter at *every* step; practical versions replace the sum by
an EMA and recompute the inverse roots only every few steps, using stale roots
in between).

What the code does (``optimizers/shampoo.py``):

* statistics are an EMA, ``L <- beta L + (1 - beta) G G^T`` and
  ``R <- beta R + (1 - beta) G^T G`` (lines 88, 100), started at the identity
  (lines 57-66), and refreshed only when ``count % precond_update_freq == 0``
  (lines 78-79, 86-91, 98-103);
* the inverse fourth roots are recomputed from the current statistics at every
  step by an eigendecomposition with eigenvalues clipped from below at
  ``epsilon`` (lines 30-36, 112-113); the update is
  ``-lr L^{-1/4} G R^{-1/4}`` (lines 112-118);
* a 1-D parameter is treated as a ``[1, n]`` matrix (lines 82-83, 107-108).

Independent reference: NumPy ``eigh``/``svd``. The passing tests check the
recurrence and the root against the state the optimizer returns, so they do
not depend on the initialisation; the two pinned failures are the deviations
from Algorithm 1.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from deqn_jax.optimizers.shampoo import shampoo


def _inv_quarter(M):
    w, V = np.linalg.eigh((M + M.T) / 2)
    return V @ np.diag(w**-0.25) @ V.T


def _as_2d(g):
    return g.reshape(1, -1) if g.ndim < 2 else g


@pytest.mark.parametrize("shape", [(3, 2), (4,)], ids=["matrix", "vector"])
def test_statistics_follow_the_ema_recurrence_and_update_uses_their_roots(shape):
    """With freq=1: L_t = beta L_{t-1} + (1-beta) G G^T, same for R; u = -lr L^-1/4 G R^-1/4."""
    lr, beta = 0.05, 0.7
    opt = shampoo(learning_rate=lr, beta=beta, precond_update_freq=1)
    rng = np.random.default_rng(0)
    p = jnp.zeros(shape)
    state = opt.init(p)
    for _ in range(4):
        G = rng.standard_normal(shape)
        G2 = _as_2d(G)
        L_prev, R_prev = np.asarray(state.L), np.asarray(state.R)
        u, state = opt.update(jnp.asarray(G), state, p)
        L, R = np.asarray(state.L), np.asarray(state.R)
        np.testing.assert_allclose(
            L, beta * L_prev + (1 - beta) * G2 @ G2.T, rtol=1e-12
        )
        np.testing.assert_allclose(
            R, beta * R_prev + (1 - beta) * G2.T @ G2, rtol=1e-12
        )
        expected = -lr * _inv_quarter(L) @ G2 @ _inv_quarter(R)
        np.testing.assert_allclose(np.asarray(u), expected.reshape(shape), rtol=1e-9)


def test_pure_gradient_statistics_give_the_polar_factor():
    """beta=0: L = G G^T, R = G^T G, so L^-1/4 G R^-1/4 = U V^T for G = U S V^T."""
    lr = 0.1
    rng = np.random.default_rng(1)
    G = rng.standard_normal((3, 3))
    U, _, Vt = np.linalg.svd(G)
    opt = shampoo(learning_rate=lr, beta=0.0, precond_update_freq=1)
    p = jnp.zeros((3, 3))
    u, _ = opt.update(jnp.asarray(G), opt.init(p), p)
    np.testing.assert_allclose(np.asarray(u), -lr * U @ Vt, rtol=1e-8, atol=1e-10)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "shampoo adds G G^T to the statistics only on refresh steps "
        "(shampoo.py:86-91, 98-103): with precond_update_freq=k, k-1 of every k "
        "gradients never enter L, R"
    ),
)
def test_every_gradient_enters_the_statistics_between_refreshes():
    """freq=3: after steps 1..3, L = beta^3 L_0 + (1-beta) sum_s beta^(3-s) G_s G_s^T."""
    beta = 0.7
    opt = shampoo(learning_rate=0.05, beta=beta, precond_update_freq=3)
    rng = np.random.default_rng(2)
    p = jnp.zeros((3, 2))
    state = opt.init(p)
    L = np.asarray(state.L)
    for _ in range(3):
        G = rng.standard_normal((3, 2))
        L = beta * L + (1 - beta) * G @ G.T
        _, state = opt.update(jnp.asarray(G), state, p)
    np.testing.assert_allclose(np.asarray(state.L), L, rtol=1e-10)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "shampoo starts L, R at the identity (shampoo.py:57-66), not eps*I: while "
        "||G||^2 << 1 the identity dominates and the step is ~ -lr*G, so the update "
        "is not invariant to the gradient's scale"
    ),
)
def test_update_is_invariant_to_gradient_scale():
    """With L_0 = eps I (eps -> 0), L^-1/4 (cG) R^-1/4 = L^-1/4 G R^-1/4 for c > 0.

    The exponents -1/4, -1/4 make the step homogeneous of degree 0 in G; this is
    what lets Shampoo take sensible steps on losses of any magnitude (the
    DEQN residual losses are ~1e-6).
    """
    rng = np.random.default_rng(3)
    G = rng.standard_normal((3, 3))
    p = jnp.zeros((3, 3))
    steps = []
    for c in (1.0, 1e-3):
        opt = shampoo(learning_rate=0.1, beta=0.9, precond_update_freq=1)
        u, _ = opt.update(jnp.asarray(c * G), opt.init(p), p)
        steps.append(np.asarray(u))
    np.testing.assert_allclose(steps[1], steps[0], rtol=1e-3)
