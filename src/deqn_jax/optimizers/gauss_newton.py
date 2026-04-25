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

from typing import Any, Callable, NamedTuple, Tuple

import jax
import jax.flatten_util
import jax.numpy as jnp
from jax import Array


class GaussNewtonState(NamedTuple):
    """State for Gauss-Newton optimizer.

    ``last_loss`` is a JAX scalar Array at runtime (sum of squared
    residuals from inside the JIT'd update step). It was annotated as
    Python ``float`` originally, which produced spurious
    ``invalid-argument-type`` errors at every constructor call — same
    pattern as the ``Metrics`` annotation lie cleared in commit
    ``3ae741f``. ``damping`` stays ``float`` because the LM update
    keeps it on the Python side.
    """

    count: int  # Iteration count
    damping: float  # Current LM damping
    last_loss: Array  # Previous loss for adaptive damping


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
        return GaussNewtonState(
            count=0, damping=self.damping, last_loss=jnp.asarray(jnp.inf)
        )

    def update(
        self,
        residual_fn: Callable,
        params: Any,
        state: GaussNewtonState,
        lr_scale: Any = 1.0,
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

        # Match the train_step contract used by the other optimizers: when a
        # schedule is active, self.learning_rate is 1.0 and lr_scale carries
        # the per-step learning rate.
        step_size = self.learning_rate * lr_scale
        new_flat_params = flat_params + step_size * delta
        new_params = unflatten(new_flat_params)

        # Compute new loss (from updated params, not old residuals)
        new_r = flat_residual_fn(new_flat_params)
        new_loss = jnp.sum(new_r**2)

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
        return GaussNewtonState(
            count=0, damping=self.initial_damping, last_loss=jnp.asarray(jnp.inf)
        )

    def update(
        self,
        residual_fn: Callable,
        params: Any,
        state: GaussNewtonState,
        lr_scale: Any = 1.0,
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
        current_loss = jnp.sum(r**2)

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

        # Match the train_step contract used by the other optimizers: when a
        # schedule is active, self.learning_rate is 1.0 and lr_scale carries
        # the per-step learning rate.
        step_size = self.learning_rate * lr_scale
        new_flat_params = flat_params + step_size * delta
        new_r = flat_residual_fn(new_flat_params)
        new_loss = jnp.sum(new_r**2)

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

        # LM should only accept steps that improve the actual objective.
        accept = actual > 0.0
        final_params = jnp.where(accept, new_flat_params, flat_params)
        final_loss = jnp.where(accept, new_loss, current_loss)
        reject_damping = jnp.minimum(
            self.max_damping, state.damping * self.damping_increase
        )
        final_damping = jnp.where(accept, new_damping, reject_damping)

        new_state = GaussNewtonState(
            count=state.count + 1,
            damping=final_damping,
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
        learning_rate,
        initial_damping,
        damping_increase,
        damping_decrease,
        min_damping,
        max_damping,
    )


def make_grad_step_gn(
    model,
    opt: Any,
    mc_samples: int,
    batch_size: int,
    quad_nodes,
    quad_weights,
    loss_reweight: str,
    reweight_alpha: float,
    use_target_network: bool,
    compute_loss_fn,
):
    """JIT'd: one Gauss-Newton / Levenberg-Marquardt update on a minibatch."""
    import equinox as eqx
    import optax

    from deqn_jax.training.loss import (
        compute_loss,
        compute_residuals,
        sample_antithetic_shocks,
    )
    from deqn_jax.training.reweighting import update_reweighting
    from deqn_jax.types import Metrics, TrainState

    n_eq = len(model.equation_names) if model.equation_names else 1
    _compute_loss_log = compute_loss_fn or compute_loss
    use_quadrature = quad_nodes is not None and quad_weights is not None

    @jax.jit
    def grad_step(
        state,
        batch,
        lr_scale,
        shock_scale=jnp.array(1.0),
    ):
        loss_key, new_key = jax.random.split(state.key)
        target_fn = state.target_params if use_target_network else None

        def residual_fn(params):
            if use_quadrature:
                n_nodes = quad_nodes.shape[0]
                shocks = (
                    jnp.broadcast_to(
                        quad_nodes[:, None, :],
                        (n_nodes, batch_size, model.n_shocks),
                    )
                    * shock_scale
                )
                sample_weights = quad_weights
            else:
                shocks = sample_antithetic_shocks(
                    loss_key,
                    mc_samples,
                    batch_size,
                    model.n_shocks,
                    shock_scale,
                )
                n_samples = shocks.shape[0]
                sample_weights = jnp.ones(n_samples) / n_samples

            def sample_residuals(shock):
                return compute_residuals(
                    model, params, batch, shock, target_policy_fn=target_fn
                )

            all_residuals = jax.vmap(sample_residuals)(shocks)
            per_eq = []
            for r in all_residuals.values():
                mean_r = jnp.einsum("s,sb->b", sample_weights, r)
                per_eq.append(mean_r)
            return jnp.concatenate(per_eq)

        loss, eq_losses = _compute_loss_log(
            model,
            state.params,
            batch,
            loss_key,
            mc_samples,
            weights=state.loss_weights,
            shock_scale=shock_scale,
            quad_nodes=quad_nodes,
            quad_weights=quad_weights,
            target_policy_fn=target_fn,
        )

        new_params, new_opt_state = opt.update(
            residual_fn, state.params, state.opt_state, lr_scale=lr_scale
        )

        def scalar_loss(p):
            r = residual_fn(p)
            return jnp.sum(r**2)

        grad_norm = optax.global_norm(
            eqx.filter(jax.grad(scalar_loss)(state.params), eqx.is_array)
        )

        new_weights, new_rw = update_reweighting(
            eq_losses,
            state,
            loss_reweight,
            reweight_alpha,
            n_eq,
        )
        new_state = TrainState(
            params=new_params,
            opt_state=new_opt_state,
            episode_state=state.episode_state,
            key=new_key,
            step=state.step + 1,
            episode=state.episode,
            loss_weights=new_weights,
            reweight_state=new_rw,
            target_params=state.target_params,
        )
        return new_state, Metrics(loss=loss, residuals=eq_losses, grad_norm=grad_norm)

    return grad_step
