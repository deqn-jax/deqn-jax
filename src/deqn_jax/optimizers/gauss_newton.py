"""Gauss-Newton and Levenberg-Marquardt optimizers for JAX.

For DEQN, we minimize L = ||r(θ)||² where r are equilibrium residuals.
Gauss-Newton approximates Hessian as H ≈ J^T J where J = ∂r/∂θ.

Key advantage over first-order methods: quadratic convergence near solution.

This implementation uses JAX autodiff (jacrev/jacfwd) for efficient Jacobian
computation - much faster than finite differences.

Usage:
    opt = gauss_newton(learning_rate=1.0)
    state = opt.init(params)

    def residual_fn(p):
        return model.equations(states, p(states), ...)

    params, state = opt.update(residual_fn, params, state)
"""

from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
import jax.flatten_util


class GaussNewtonState(NamedTuple):
    """State for Gauss-Newton optimizer."""

    count: int  # Iteration count
    damping: float  # Current LM damping
    last_loss: float  # Previous loss for adaptive damping


class GaussNewton:
    """Gauss-Newton optimizer for nonlinear least squares."""

    def __init__(
        self,
        learning_rate: float = 1.0,
        damping: float = 0.0,
        solve_method: str = "lstsq",
    ):
        self.learning_rate = learning_rate
        self.damping = damping
        self.solve_method = solve_method

    def init(self, params) -> GaussNewtonState:
        return GaussNewtonState(count=0, damping=self.damping, last_loss=jnp.inf)

    def update(
        self,
        residual_fn: Callable,
        params: Any,
        state: GaussNewtonState,
    ) -> Tuple[Any, GaussNewtonState]:
        """Perform one GN step.

        Args:
            residual_fn: Function params -> flat residuals [n_residuals]
            params: Current parameters (pytree)
            state: Optimizer state

        Returns:
            Tuple of (new_params, new_state)
        """
        # Flatten params for linear algebra
        flat_params, unflatten = jax.flatten_util.ravel_pytree(params)
        n_params = flat_params.shape[0]

        # Wrap residual_fn to work with flat params
        def flat_residual_fn(flat_p):
            p = unflatten(flat_p)
            r = residual_fn(p)
            return jnp.ravel(r)

        # Compute residuals
        r = flat_residual_fn(flat_params)
        n_residuals = r.shape[0]

        # Compute Jacobian using autodiff
        # Choose forward or reverse mode based on dimensions
        if n_residuals <= n_params:
            J = jax.jacrev(flat_residual_fn)(flat_params)
        else:
            J = jax.jacfwd(flat_residual_fn)(flat_params)

        # Ensure minimum damping for numerical stability
        damping = jnp.maximum(state.damping, 1e-6)

        # Solve (J^T J + λI) δ = -J^T r
        # When n_residuals < n_params, use dual formulation (Woodbury):
        #   δ = -J^T (J J^T + λI)^{-1} r
        # Solves (n_res, n_res) system instead of (n_params, n_params).
        if n_residuals < n_params:
            G = J @ J.T + damping * jnp.eye(n_residuals)
            v = jnp.linalg.solve(G, r)
            delta = -J.T @ v
        else:
            JtJ = J.T @ J + damping * jnp.eye(n_params)
            delta = jnp.linalg.solve(JtJ, -(J.T @ r))

        # Apply update with learning rate
        new_flat_params = flat_params + self.learning_rate * delta
        new_params = unflatten(new_flat_params)

        # Compute new loss
        new_loss = jnp.sum(r ** 2)

        new_state = GaussNewtonState(
            count=state.count + 1,
            damping=state.damping,
            last_loss=new_loss,
        )

        return new_params, new_state


def gauss_newton(
    learning_rate: float = 1.0,
    damping: float = 0.0,
    solve_method: str = "lstsq",
) -> GaussNewton:
    """Create Gauss-Newton optimizer.

    Args:
        learning_rate: Step size multiplier (1.0 = full GN step)
        damping: Fixed damping (0 = pure GN, >0 = LM-style regularization)
        solve_method: How to solve normal equations ("lstsq", "cholesky", "svd")

    Returns:
        GaussNewton optimizer instance
    """
    return GaussNewton(learning_rate, damping, solve_method)


class LevenbergMarquardt:
    """Levenberg-Marquardt optimizer (adaptive damped Gauss-Newton)."""

    def __init__(
        self,
        learning_rate: float = 1.0,
        initial_damping: float = 1e-3,
        damping_increase: float = 10.0,
        damping_decrease: float = 0.1,
        min_damping: float = 1e-8,
        max_damping: float = 1e8,
    ):
        self.learning_rate = learning_rate
        self.initial_damping = initial_damping
        self.damping_increase = damping_increase
        self.damping_decrease = damping_decrease
        self.min_damping = min_damping
        self.max_damping = max_damping

    def init(self, params) -> GaussNewtonState:
        return GaussNewtonState(count=0, damping=self.initial_damping, last_loss=jnp.inf)

    def update(
        self,
        residual_fn: Callable,
        params: Any,
        state: GaussNewtonState,
    ) -> Tuple[Any, GaussNewtonState]:
        """Perform one LM step with adaptive damping."""
        # Flatten params
        flat_params, unflatten = jax.flatten_util.ravel_pytree(params)
        n_params = flat_params.shape[0]

        def flat_residual_fn(flat_p):
            p = unflatten(flat_p)
            return jnp.ravel(residual_fn(p))

        # Compute residuals and Jacobian
        r = flat_residual_fn(flat_params)
        n_residuals = r.shape[0]
        current_loss = jnp.sum(r ** 2)

        if n_residuals <= n_params:
            J = jax.jacrev(flat_residual_fn)(flat_params)
        else:
            J = jax.jacfwd(flat_residual_fn)(flat_params)

        # Solve with current damping (dual formulation when underdetermined)
        damping = jnp.maximum(state.damping, 1e-6)
        if n_residuals < n_params:
            G = J @ J.T + damping * jnp.eye(n_residuals)
            v = jnp.linalg.solve(G, r)
            delta = -J.T @ v
        else:
            JtJ = J.T @ J + damping * jnp.eye(n_params)
            delta = jnp.linalg.solve(JtJ, -(J.T @ r))

        # Tentative update
        new_flat_params = flat_params + self.learning_rate * delta
        new_r = flat_residual_fn(new_flat_params)
        new_loss = jnp.sum(new_r ** 2)

        # Gain ratio for damping adaptation
        Jdelta = J @ delta
        predicted = -2 * Jdelta.T @ r - Jdelta.T @ Jdelta
        actual = current_loss - new_loss
        rho = jnp.where(jnp.abs(predicted) > 1e-10, actual / predicted, 1.0)

        # Adapt damping
        new_damping = jnp.where(
            rho > 0.75,
            jnp.maximum(self.min_damping, state.damping * self.damping_decrease),
            jnp.where(
                rho < 0.25,
                jnp.minimum(self.max_damping, state.damping * self.damping_increase),
                state.damping,
            ),
        )

        # Accept or reject step (broadcast for pytree leaves)
        accept = new_loss < current_loss * 1.1
        final_params = jnp.where(accept, new_flat_params, flat_params)
        final_loss = jnp.where(accept, new_loss, current_loss)
        new_damping = jnp.where(accept, new_damping, state.damping)

        new_state = GaussNewtonState(
            count=state.count + 1,
            damping=new_damping,
            last_loss=final_loss,
        )

        return unflatten(final_params), new_state


def levenberg_marquardt(
    learning_rate: float = 1.0,
    initial_damping: float = 1e-3,
    damping_increase: float = 10.0,
    damping_decrease: float = 0.1,
    min_damping: float = 1e-8,
    max_damping: float = 1e8,
) -> LevenbergMarquardt:
    """Create Levenberg-Marquardt optimizer (adaptive damped GN).

    Args:
        learning_rate: Step size multiplier
        initial_damping: Starting damping value
        damping_increase: Factor when step is bad
        damping_decrease: Factor when step is good
        min_damping: Minimum damping
        max_damping: Maximum damping

    Returns:
        LevenbergMarquardt optimizer instance
    """
    return LevenbergMarquardt(
        learning_rate, initial_damping, damping_increase,
        damping_decrease, min_damping, max_damping
    )
