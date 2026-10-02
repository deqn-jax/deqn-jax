"""Defining-property tests for the matrix-free Gauss-Newton optimizer (``ign``).

Convention derived from ``optimizers/gauss_newton.py`` (``ImplicitGaussNewton``):

* it solves ``(J^T J + lam I) delta = -J^T r`` by conjugate gradients started
  at ``delta = 0`` (lines 217-230, 261), with ``lam = max(damping, 1e-12)``
  (line 218 -- a different floor from ``gn``/``lm``, which use 1e-6);
* the operator is applied as ``v -> VJP(JVP(v)) + lam v`` (lines 221-223);
* CG stops after ``cg_iters`` iterations or once the recursive residual
  satisfies ``||rhs - A x|| <= cg_tol * ||rhs||`` (line 265, 269);
* the parameters move by ``learning_rate * lr_scale * delta`` (232-233), and
  ``last_cg_residual`` is the final recursive residual norm (line 285).

Independent references: a NumPy dense solve, and the Krylov characterisation
of CG -- the k-th CG iterate from zero is the minimiser of the A-norm error
over ``span{b, A b, ..., A^{k-1} b}``, computed here by a projected solve on
an orthonormal Krylov basis. That characterisation pins the operator too: a
wrong product (``J J^T``, a missing ``lam``, a sign) changes the Krylov space
and the iterate.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deqn_jax.optimizers.gauss_newton import (
    _conjugate_gradient,
    implicit_gauss_newton,
)


def _residual(t):
    return jnp.stack(
        [
            t[0] ** 2 + t[1] - 1.0,
            jnp.sin(t[0]) - t[1] * t[2],
            jnp.exp(0.3 * t[2]) - t[3],
            t[1] ** 3 - 0.5 * t[3],
            t[0] * t[3] - 0.2,
            jnp.tanh(t[2] + t[3]),
        ]
    )


THETA0 = np.array([0.4, -0.7, 0.9, 0.2])


def _system(residual, theta, lam):
    J = np.asarray(jax.jacfwd(residual)(jnp.asarray(theta)))
    r = np.asarray(residual(jnp.asarray(theta)))
    return J.T @ J + lam * np.eye(len(theta)), -J.T @ r


def _krylov_iterate(A, b, k):
    """argmin over x in K_k(A, b) of ||x - A^{-1} b||_A, via a projected solve."""
    basis = [b]
    for _ in range(k - 1):
        basis.append(A @ basis[-1])
    Q, _ = np.linalg.qr(np.stack(basis, axis=1))
    return Q @ np.linalg.solve(Q.T @ A @ Q, Q.T @ b)


def _ign_step(theta, residual, damping, cg_iters, cg_tol=1e-14, lr=1.0, lr_scale=1.0):
    opt = implicit_gauss_newton(
        learning_rate=lr, damping=damping, cg_iters=cg_iters, cg_tol=cg_tol
    )
    p = jnp.asarray(theta)
    new_p, state = opt.update(residual, p, opt.init(p), lr_scale)
    return np.asarray(new_p), state


@pytest.mark.parametrize("damping", [1e-3, 0.3])
def test_converged_cg_matches_dense_damped_gn_solve(damping):
    A, b = _system(_residual, THETA0, damping)
    expected = THETA0 + np.linalg.solve(A, b)
    theta1, state = _ign_step(THETA0, _residual, damping, cg_iters=50, cg_tol=1e-12)
    np.testing.assert_allclose(theta1, expected, rtol=1e-8, atol=1e-10)
    true_res = np.linalg.norm(b - A @ (theta1 - THETA0))
    assert float(state.last_cg_residual) <= 1e-10 * np.linalg.norm(b)
    assert true_res <= 1e-8 * np.linalg.norm(b)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_truncated_cg_iterate_is_the_krylov_minimiser(k):
    """k CG iterations from zero give the A-norm-optimal point of K_k(A, -J^T r)."""
    damping = 0.05
    A, b = _system(_residual, THETA0, damping)
    expected = THETA0 + _krylov_iterate(A, b, k)
    theta1, state = _ign_step(THETA0, _residual, damping, cg_iters=k)
    assert int(state.last_cg_iters) == k
    np.testing.assert_allclose(theta1, expected, rtol=1e-8, atol=1e-11)


def test_one_cg_iteration_is_exact_line_search_along_the_gradient():
    """x_1 = (g^T g / g^T A g) g with g = -J^T r."""
    damping = 0.05
    A, g = _system(_residual, THETA0, damping)
    expected = THETA0 + (g @ g) / (g @ A @ g) * g
    theta1, _ = _ign_step(THETA0, _residual, damping, cg_iters=1)
    np.testing.assert_allclose(theta1, expected, rtol=1e-10)


def test_underdetermined_residual_matches_dense_solve():
    """Fewer residuals than parameters: J^T J is singular, lam I makes it SPD."""

    def residual(t):
        return jnp.stack([t[0] * t[1] - 1.0, jnp.sin(t[2]) + t[3] ** 2 - 0.3])

    damping = 1e-2
    A, b = _system(residual, THETA0, damping)
    theta1, _ = _ign_step(THETA0, residual, damping, cg_iters=50, cg_tol=1e-13)
    np.testing.assert_allclose(theta1, THETA0 + np.linalg.solve(A, b), rtol=1e-8)


def test_learning_rate_and_lr_scale_scale_the_step():
    full, _ = _ign_step(THETA0, _residual, 0.05, cg_iters=50)
    scaled, _ = _ign_step(THETA0, _residual, 0.05, cg_iters=50, lr=0.5, lr_scale=0.3)
    np.testing.assert_allclose(scaled - THETA0, 0.15 * (full - THETA0), rtol=1e-9)


class TestConjugateGradientKernel:
    """``_conjugate_gradient`` on a dense SPD matrix, against NumPy."""

    def _spd(self, n=6, cond=50.0, seed=0):
        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        A = Q @ np.diag(np.geomspace(1.0, cond, n)) @ Q.T
        return A, rng.standard_normal(n)

    def test_solves_to_tolerance(self):
        A, b = self._spd()
        x, res, iters = _conjugate_gradient(
            lambda v: jnp.asarray(A) @ v, jnp.asarray(b), max_iters=100, tol=1e-12
        )
        np.testing.assert_allclose(np.asarray(x), np.linalg.solve(A, b), rtol=1e-9)
        assert int(iters) <= 6 + 2  # n steps in exact arithmetic

    def test_stops_at_the_first_iterate_meeting_relative_tolerance(self):
        """Stops at the first k with ||b - A x_k|| <= tol ||b||, not before or after."""
        A, b = self._spd(n=20, cond=10.0, seed=1)
        tol = 1e-3
        _, res, iters = _conjugate_gradient(
            lambda v: jnp.asarray(A) @ v, jnp.asarray(b), max_iters=100, tol=tol
        )
        k = int(iters)
        assert 1 < k < 20  # stopped by the tolerance, not by n or max_iters
        bnorm = np.linalg.norm(b)
        x_k = _krylov_iterate(A, b, k)
        x_prev = _krylov_iterate(A, b, k - 1)
        assert np.linalg.norm(b - A @ x_k) <= tol * bnorm * (1 + 1e-6)
        assert np.linalg.norm(b - A @ x_prev) > tol * bnorm
        assert float(res) == pytest.approx(np.linalg.norm(b - A @ x_k), rel=1e-6)
