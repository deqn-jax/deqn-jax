"""Multi-Adaptive Optimizer (MAO) for per-equation optimization.

MAO maintains separate Adam-style moment estimates for each equation,
then combines updates via per-task adaptive learning rates.

This is NOT an optax.GradientTransformation -- it has a custom interface
because it receives per-equation Jacobians instead of standard gradients.

Usage in training:
    eq_jac = jax.jacrev(per_eq_loss_fn)(params)  # pytree, each leaf [n_eq, *shape]
    updates, new_state = mao.update(eq_jac, state, params)
    new_params = optax.apply_updates(params, updates)
"""

from typing import Any, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.optimizers.registry import OptimizerKind, register_optimizer


class MAOState(NamedTuple):
    """State for MAO optimizer."""

    count: Array  # scalar step count
    m: Any  # first moment pytree, each leaf [n_eq, *param_shape]
    v: Any  # second moment pytree, each leaf [n_eq, *param_shape]


class MAOTransform:
    """Multi-Adaptive Optimizer.

    Maintains per-equation moment estimates and combines updates.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        n_tasks: int = 1,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.n_tasks = n_tasks

    def init(self, params: Any) -> MAOState:
        """Initialize MAO state with per-equation moments."""
        m = jax.tree.map(
            lambda p: jnp.zeros((self.n_tasks,) + p.shape, dtype=p.dtype),
            params,
        )
        v = jax.tree.map(
            lambda p: jnp.zeros((self.n_tasks,) + p.shape, dtype=p.dtype),
            params,
        )
        return MAOState(count=jnp.zeros([], dtype=jnp.int32), m=m, v=v)

    def update(
        self,
        eq_jacobian: Any,
        state: MAOState,
        params: Any,
    ) -> Tuple[Any, MAOState]:
        """Compute MAO update from per-equation Jacobians.

        Args:
            eq_jacobian: Pytree matching params, each leaf [n_eq, *param_shape]
                        (output of jax.jacrev over per-equation losses)
            state: Current MAO state
            params: Current parameters (unused but kept for API consistency)

        Returns:
            Tuple of (updates pytree, new_state)
        """
        count = state.count + 1
        b1, b2, eps = self.beta1, self.beta2, self.epsilon

        # Update per-equation moments
        new_m = jax.tree.map(
            lambda m, j: b1 * m + (1.0 - b1) * j,
            state.m,
            eq_jacobian,
        )
        new_v = jax.tree.map(
            lambda v, j: b2 * v + (1.0 - b2) * j**2,
            state.v,
            eq_jacobian,
        )

        # Bias correction
        bc1 = 1.0 - b1**count
        bc2 = 1.0 - b2**count

        # Per-equation Adam updates, then sum across equations
        def compute_update(m_leaf, v_leaf):
            # m_leaf: [n_eq, *shape], v_leaf: [n_eq, *shape]
            m_hat = m_leaf / bc1
            v_hat = v_leaf / bc2
            # Per-equation update: [n_eq, *shape]
            per_eq = m_hat / (jnp.sqrt(v_hat) + eps)
            # Average across equations → [*shape]
            return -self.learning_rate * jnp.mean(per_eq, axis=0)

        updates = jax.tree.map(compute_update, new_m, new_v)

        new_state = MAOState(count=count, m=new_m, v=new_v)
        return updates, new_state


class _MAOFactory:
    """Deferred MAO construction -- resolves n_tasks at create_train_state time."""

    def __init__(self, config):
        self.learning_rate = config.learning_rate
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.epsilon = config.epsilon

    def with_num_tasks(self, n_tasks: int) -> MAOTransform:
        return MAOTransform(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            n_tasks=n_tasks,
        )


@register_optimizer("mao", kind=OptimizerKind.MAO)
def _mao(config):
    return _MAOFactory(config)


def make_grad_step_mao(
    model,
    mao_opt: Any,
    mc_samples: int,
    quad_nodes,
    quad_weights,
    loss_reweight: str,
    reweight_alpha: float,
    use_target_network: bool,
    compute_loss_fn,
    grad_clip,
):
    """JIT'd: one MAO (per-equation Jacobian) gradient update on a minibatch."""
    import equinox as eqx
    import optax

    from deqn_jax.training.loss import compute_loss, eq_losses_to_array
    from deqn_jax.training.reweighting import update_reweighting
    from deqn_jax.types import Metrics, TrainState

    n_eq = len(model.equation_names) if model.equation_names else 1
    _compute_loss_total = compute_loss_fn or compute_loss

    @jax.jit
    def grad_step(
        state: TrainState,
        batch: Array,
        lr_scale: Array,
        shock_scale: Array = jnp.array(1.0),
    ) -> Tuple[TrainState, Metrics]:
        loss_key, new_key = jax.random.split(state.key)
        target_fn = state.target_params if use_target_network else None

        params_arrays = eqx.filter(state.params, eqx.is_array)
        params_static = eqx.filter(state.params, lambda x: not eqx.is_array(x))

        def per_eq_loss_fn(p_arrays):
            full_params = eqx.combine(p_arrays, params_static)
            _, eq_losses = compute_loss(
                model,
                full_params,
                batch,
                loss_key,
                mc_samples,
                weights=None,
                shock_scale=shock_scale,
                quad_nodes=quad_nodes,
                quad_weights=quad_weights,
                target_policy_fn=target_fn,
            )
            return eq_losses_to_array(eq_losses)

        eq_jac = jax.jacrev(per_eq_loss_fn)(params_arrays)

        def total_loss_fn(params):
            loss, eq_losses = _compute_loss_total(
                model,
                params,
                batch,
                loss_key,
                mc_samples,
                weights=state.loss_weights,
                shock_scale=shock_scale,
                quad_nodes=quad_nodes,
                quad_weights=quad_weights,
                target_policy_fn=target_fn,
            )
            return loss, eq_losses

        (loss, eq_losses), grads = eqx.filter_value_and_grad(
            total_loss_fn, has_aux=True
        )(state.params)
        grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_array))

        updates, new_opt_state = mao_opt.update(eq_jac, state.opt_state, params_arrays)
        if grad_clip is not None:
            update_norm = optax.global_norm(updates)
            clip_scale = jnp.minimum(1.0, grad_clip / (update_norm + 1e-8))
            updates = jax.tree.map(lambda u: clip_scale * u, updates)
        updates = jax.tree.map(lambda u: lr_scale * u, updates)
        new_params_arrays = optax.apply_updates(params_arrays, updates)
        new_params = eqx.combine(new_params_arrays, state.params)

        new_weights, new_rw = update_reweighting(
            eq_losses,
            state,
            loss_reweight,
            reweight_alpha,
            n_eq,
        )
        new_state = state._replace(
            params=new_params,
            opt_state=new_opt_state,
            key=new_key,
            step=state.step + 1,
            loss_weights=new_weights,
            reweight_state=new_rw,
        )
        return new_state, Metrics(loss=loss, residuals=eq_losses, grad_norm=grad_norm)

    return grad_step
