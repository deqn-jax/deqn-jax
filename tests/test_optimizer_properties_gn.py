"""Defining-property tests for the dense Gauss-Newton optimizer (``gn``).

Convention derived from ``optimizers/gauss_newton.py`` (``GaussNewton.update``):

* the objective is ``||r(theta)||^2`` with ``r`` the flattened residual vector;
* the step is ``delta = -(J^T J + lam I)^{-1} J^T r`` with ``J = dr/dtheta``
  (lines 141-147; the dual form ``-J^T (J J^T + lam I)^{-1} r`` is used when
  there are fewer residuals than parameters, which is the same vector);
* ``lam = max(damping, 1e-6)`` (line 135): even ``damping=0`` solves with
  ``lam = 1e-6``, so "pure GN" is GN with a 1e-6 ridge;
* the parameters move by ``learning_rate * lr_scale * delta`` (lines 152-153),
  so ``learning_rate=1, lr_scale=1`` is the full GN step.

The independent computations below use NumPy least squares / pseudo-inverse
and a Jacobian from ``jax.jacfwd``; none of them call into the optimizer.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deqn_jax.optimizers.gauss_newton import gauss_newton

FLOOR = 1e-6  # gauss_newton.py:135


def _split(theta):
    """A two-leaf pytree, so the ravel/unravel path is exercised."""
    return {"a": jnp.asarray(theta[:2]), "b": jnp.asarray(theta[2:])}


def _join(p):
    return jnp.concatenate([p["a"], p["b"]])


def _well_conditioned(rng, n_rows, n_cols):
    """A matrix with singular values in [1, 3]."""
    u, _ = np.linalg.qr(rng.standard_normal((n_rows, n_rows)))
    v, _ = np.linalg.qr(rng.standard_normal((n_cols, n_cols)))
    k = min(n_rows, n_cols)
    s = np.zeros((n_rows, n_cols))
    s[np.arange(k), np.arange(k)] = np.linspace(1.0, 3.0, k)
    return u @ s @ v.T


def _gn_step(params, residual_fn, damping=0.0, lr=1.0, lr_scale=1.0):
    opt = gauss_newton(learning_rate=lr, damping=damping)
    new_params, state = opt.update(residual_fn, params, opt.init(params), lr_scale)
    return np.asarray(_join(new_params)), state


@pytest.mark.parametrize("n_res", [6, 4], ids=["overdetermined", "square"])
def test_one_step_lands_on_least_squares_solution(n_res):
    """r = A theta - b is linear, so one full GN step from any theta0 is exact."""
    rng = np.random.default_rng(0)
    A = _well_conditioned(rng, n_res, 4)
    b = rng.standard_normal(n_res)
    theta_star = np.linalg.lstsq(A, b, rcond=None)[0]

    def residual_fn(p):
        return jnp.asarray(A) @ _join(p) - jnp.asarray(b)

    for seed in range(3):
        theta0 = 5.0 * np.random.default_rng(seed + 10).standard_normal(4)
        theta1, state = _gn_step(_split(theta0), residual_fn)
        # With sigma_min(A) = 1 the 1e-6 ridge moves the answer by <= 1e-6 |delta|.
        np.testing.assert_allclose(
            theta1, theta_star, atol=1e-5 * (1 + np.abs(theta0).max())
        )
        loss_star = np.sum((A @ theta_star - b) ** 2)
        assert float(state.last_loss) == pytest.approx(loss_star, abs=1e-8)


def test_ridge_floor_is_exactly_one_in_a_million():
    """damping=0 solves (J^T J + 1e-6 I) delta = -J^T r, not the bare normal equations.

    The test uses a nearly rank-deficient A so the floor is visible.
    """
    A = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0 + 1e-4, 0.0, 0.0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
        ]
    )
    b = np.array([1.0, 2.0, 0.5, -0.5, 0.25])
    theta0 = np.array([0.3, -0.2, 0.1, 0.4])

    def residual_fn(p):
        return jnp.asarray(A) @ _join(p) - jnp.asarray(b)

    r0 = A @ theta0 - b
    for damping, lam in [(0.0, FLOOR), (1e-3, 1e-3)]:
        expected = theta0 - np.linalg.solve(A.T @ A + lam * np.eye(4), A.T @ r0)
        theta1, _ = _gn_step(_split(theta0), residual_fn, damping=damping)
        np.testing.assert_allclose(theta1, expected, rtol=1e-8, atol=1e-10)


def test_underdetermined_step_is_minimum_norm_correction():
    """Fewer residuals than parameters (dual branch): theta1 = theta0 - A^+ r0.

    The full GN step solves A theta = b exactly and moves the least distance
    from theta0 to the solution set; A^+ is NumPy's pseudo-inverse.
    """
    rng = np.random.default_rng(1)
    A = _well_conditioned(rng, 2, 4)
    b = rng.standard_normal(2)
    theta0 = rng.standard_normal(4)

    def residual_fn(p):
        return jnp.asarray(A) @ _join(p) - jnp.asarray(b)

    theta1, state = _gn_step(_split(theta0), residual_fn)
    expected = theta0 - np.linalg.pinv(A) @ (A @ theta0 - b)
    np.testing.assert_allclose(theta1, expected, atol=1e-5)
    assert float(state.last_loss) < 1e-10


def _nonlinear_residual(p):
    t = _join(p)
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


@pytest.mark.parametrize("damping", [0.0, 0.05])
def test_nonlinear_step_matches_independent_normal_equations(damping):
    """delta = -(J^T J + lam I)^{-1} J^T r with J from jax.jacfwd, lam = max(damping, 1e-6)."""
    theta0 = np.array([0.4, -0.7, 0.9, 0.2])
    J = np.asarray(
        jax.jacfwd(lambda t: _nonlinear_residual(_split(t)))(jnp.asarray(theta0))
    )
    r = np.asarray(_nonlinear_residual(_split(theta0)))
    lam = max(damping, FLOOR)
    expected = theta0 - np.linalg.solve(J.T @ J + lam * np.eye(4), J.T @ r)
    theta1, _ = _gn_step(_split(theta0), _nonlinear_residual, damping=damping)
    np.testing.assert_allclose(theta1, expected, rtol=1e-9, atol=1e-11)
    if damping == 0.0:
        # And it is the undamped GN step -(J^T J)^{-1} J^T r up to the floor.
        pure = theta0 - np.linalg.lstsq(J, r, rcond=None)[0]
        np.testing.assert_allclose(theta1, pure, rtol=1e-4)


def test_learning_rate_and_lr_scale_scale_the_gn_step():
    """theta1 - theta0 = learning_rate * lr_scale * delta_GN."""
    theta0 = np.array([0.4, -0.7, 0.9, 0.2])
    full, _ = _gn_step(_split(theta0), _nonlinear_residual)
    scaled, _ = _gn_step(_split(theta0), _nonlinear_residual, lr=0.5, lr_scale=0.3)
    np.testing.assert_allclose(scaled - theta0, 0.15 * (full - theta0), rtol=1e-9)
