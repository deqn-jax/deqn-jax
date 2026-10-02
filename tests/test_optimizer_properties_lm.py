"""Defining-property tests for Levenberg-Marquardt (``lm``).

Convention derived from ``optimizers/gauss_newton.py`` (``LevenbergMarquardt``):

* objective ``L(theta) = ||r(theta)||^2`` (line 365);
* trial step ``delta = -(J^T J + lam D)^{-1} J^T r`` with ``D = I`` -- the
  Levenberg form, not Marquardt's ``diag(J^T J)`` (lines 374-380);
  ``lam = max(state.damping, 1e-6)`` (line 373);
* trial point ``theta + s * delta``, ``s = learning_rate * lr_scale``
  (lines 385-386);
* gain ratio ``rho = actual / predicted`` with ``actual = L(theta) -
  L(theta + s delta)`` and ``predicted = ||r||^2 - ||r + s J delta||^2``, the
  reduction the linearised model promises for the step actually taken
  (lines 397-400);
* the step is accepted iff ``actual > 0`` (line 414); on acceptance the
  damping becomes ``max(min_damping, 0.1 lam)`` if ``rho > 0.75``,
  ``min(max_damping, 10 lam)`` if ``rho < 0.25``, unchanged otherwise
  (lines 403-411); on rejection the parameters stay and the damping becomes
  ``min(max_damping, 10 lam)`` (lines 417-420). Here ``lam`` in the update
  rule is the *stored* ``state.damping``, without the 1e-6 floor.

Expected values are computed with NumPy from a Jacobian taken by
``jax.jacfwd``; the optimizer is only ever the system under test.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deqn_jax.optimizers.gauss_newton import levenberg_marquardt


def _residual(t):
    return jnp.stack(
        [
            t[0] ** 2 + t[1] - 1.0,
            jnp.sin(t[0]) - t[1] * t[2],
            jnp.exp(0.3 * t[2]) - 1.0,
            t[1] ** 3 - 0.5 * t[0],
            t[0] * t[2] - 0.2,
        ]
    )


THETA0 = np.array([0.6, 0.5, 0.3])


def _jr(theta):
    t = jnp.asarray(theta)
    return np.asarray(jax.jacfwd(_residual)(t)), np.asarray(_residual(t))


def _loss(theta):
    return float(np.sum(np.asarray(_residual(jnp.asarray(theta))) ** 2))


def _lm_step(theta, damping, lr=1.0, **kw):
    opt = levenberg_marquardt(learning_rate=lr, initial_damping=damping, **kw)
    p = jnp.asarray(theta)
    new_p, state = opt.update(_residual, p, opt.init(p))
    return np.asarray(new_p), state


@pytest.mark.parametrize("lam", [1e-3, 0.1, 3.0])
def test_step_is_damped_normal_equations_with_identity_damping(lam):
    """theta1 = theta0 - (J^T J + lam I)^{-1} J^T r when the step is accepted."""
    J, r = _jr(THETA0)
    expected = THETA0 - np.linalg.solve(J.T @ J + lam * np.eye(3), J.T @ r)
    assert _loss(expected) < _loss(THETA0), "test premise: the trial step descends"
    theta1, _ = _lm_step(THETA0, lam)
    np.testing.assert_allclose(theta1, expected, rtol=1e-10, atol=1e-12)


def test_damping_is_identity_not_marquardt_diagonal():
    """With a badly scaled column, D = I and D = diag(J^T J) give different steps.

    Pins which one the code uses (D = I) so a silent switch is caught.
    """
    scale = np.array([1.0, 30.0, 1.0])

    def scaled(t):
        return _residual(t * jnp.asarray(scale))

    t0 = THETA0 / scale
    J = np.asarray(jax.jacfwd(scaled)(jnp.asarray(t0)))
    r = np.asarray(scaled(jnp.asarray(t0)))
    lam = 0.5
    levenberg = t0 - np.linalg.solve(J.T @ J + lam * np.eye(3), J.T @ r)
    marquardt = t0 - np.linalg.solve(J.T @ J + lam * np.diag(np.diag(J.T @ J)), J.T @ r)
    assert np.max(np.abs(levenberg - marquardt)) > 1e-3
    opt = levenberg_marquardt(learning_rate=1.0, initial_damping=lam)
    p = jnp.asarray(t0)
    theta1 = np.asarray(opt.update(scaled, p, opt.init(p))[0])
    np.testing.assert_allclose(theta1, levenberg, rtol=1e-10)


def test_small_damping_limit_is_the_gauss_newton_step():
    J, r = _jr(THETA0)
    gn = THETA0 - np.linalg.lstsq(J, r, rcond=None)[0]
    theta1, _ = _lm_step(THETA0, 1e-6)
    np.testing.assert_allclose(theta1, gn, rtol=1e-4)


def test_large_damping_limit_is_scaled_gradient_descent():
    """lam -> inf: lam * delta -> -J^T r, i.e. a step along -grad(L)/2."""
    J, r = _jr(THETA0)
    lam = 1e6
    theta1, _ = _lm_step(THETA0, lam)
    np.testing.assert_allclose(lam * (theta1 - THETA0), -J.T @ r, rtol=1e-4)


def _expected_rule(theta, lam, lr, inc=10.0, dec=0.1, lo=1e-8, hi=1e8):
    """Independent accept/reject + damping update from the gain-ratio definition."""
    J, r = _jr(theta)
    lam_solve = max(lam, 1e-6)
    delta = -np.linalg.solve(J.T @ J + lam_solve * np.eye(len(theta)), J.T @ r)
    trial = theta + lr * delta
    actual = _loss(theta) - _loss(trial)
    predicted = np.sum(r**2) - np.sum((r + lr * J @ delta) ** 2)
    rho = actual / predicted
    if actual <= 0:
        return theta, min(hi, lam * inc), "reject"
    if rho > 0.75:
        return trial, max(lo, lam * dec), "good"
    if rho < 0.25:
        return trial, min(hi, lam * inc), "poor"
    return trial, lam, "fair"


def test_adaptive_damping_follows_the_gain_ratio_rule():
    """Accept iff the loss falls; shrink/grow/keep damping by rho; reject keeps theta.

    A grid over starting points, damping and step size is swept; the test
    asserts that every one of the four branches (good / fair / poor / reject)
    is visited, so a rule that never leaves one branch cannot pass.
    """
    rng = np.random.default_rng(3)
    seen = set()
    for _ in range(60):
        theta = rng.uniform(-1.5, 1.5, size=3)
        lam = float(10.0 ** rng.uniform(-3, 1))
        lr = float(rng.choice([0.5, 1.0, 1.6, 2.2]))
        exp_theta, exp_lam, branch = _expected_rule(theta, lam, lr)
        seen.add(branch)
        theta1, state = _lm_step(theta, lam, lr=lr)
        np.testing.assert_allclose(theta1, exp_theta, rtol=1e-9, atol=1e-12)
        assert float(state.damping) == pytest.approx(exp_lam, rel=1e-12), branch
        assert float(state.last_loss) == pytest.approx(_loss(exp_theta), rel=1e-9)
    assert seen == {"good", "fair", "poor", "reject"}, seen


def test_linear_residual_full_step_is_exact_and_shrinks_damping():
    """For r = A theta - b the linear model is exact, so rho = 1 at any step size."""
    rng = np.random.default_rng(4)
    A = rng.standard_normal((5, 3))
    b = rng.standard_normal(5)
    lam = 1e-2

    def res(p):
        return jnp.asarray(A) @ p - jnp.asarray(b)

    theta0 = rng.standard_normal(3)
    opt = levenberg_marquardt(learning_rate=1.0, initial_damping=lam)
    p = jnp.asarray(theta0)
    theta1, state = opt.update(res, p, opt.init(p))
    expected = theta0 - np.linalg.solve(
        A.T @ A + lam * np.eye(3), A.T @ (A @ theta0 - b)
    )
    np.testing.assert_allclose(np.asarray(theta1), expected, rtol=1e-10)
    assert float(state.damping) == pytest.approx(0.1 * lam)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "lm solves with max(damping, 1e-6) (gauss_newton.py:373) while the stored "
        "damping may fall to min_damping=1e-8 (gauss_newton.py:405): below 1e-6 the "
        "reported damping is not the one the step used"
    ),
)
def test_stored_damping_is_the_damping_the_step_uses():
    """A state carrying lam = 1e-8 should take the 1e-8-damped step."""
    # Nearly rank-deficient J so 1e-8 and 1e-6 ridges give different steps.
    A = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-5], [0.0, 0.0]])
    b = np.array([1.0, 0.0, 0.0])

    def res(p):
        return jnp.asarray(A) @ p - jnp.asarray(b)

    theta0 = np.array([0.0, 0.0])
    lam = 1e-8
    expected = theta0 - np.linalg.solve(
        A.T @ A + lam * np.eye(2), A.T @ (A @ theta0 - b)
    )
    opt = levenberg_marquardt(learning_rate=1.0, initial_damping=lam, min_damping=1e-8)
    p = jnp.asarray(theta0)
    theta1, _ = opt.update(res, p, opt.init(p))
    np.testing.assert_allclose(np.asarray(theta1), expected, rtol=1e-6)
