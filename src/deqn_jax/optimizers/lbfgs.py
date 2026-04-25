"""L-BFGS optimizer via optax.

Thin wrapper around optax.lbfgs() which is a GradientTransformationExtraArgs --
it needs ``value`` and ``value_fn`` passed to update() for line search.
"""

from typing import Any, Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from deqn_jax.optimizers.registry import OptimizerKind, register_optimizer
from deqn_jax.training.loss import compute_loss
from deqn_jax.training.reweighting import update_reweighting
from deqn_jax.types import Metrics, ModelSpec, TrainState


@register_optimizer("lbfgs", kind=OptimizerKind.LBFGS)
def _lbfgs(config):
    return optax.lbfgs(
        learning_rate=config.learning_rate,
        memory_size=config.memory_size,
    )


def make_grad_step_lbfgs(
    model: ModelSpec,
    opt: Any,
    mc_samples: int,
    quad_nodes: Optional[Array],
    quad_weights: Optional[Array],
    loss_reweight: str,
    reweight_alpha: float,
    use_target_network: bool,
    compute_loss_fn: Optional[Callable],
):
    """JIT'd: one L-BFGS gradient update on a minibatch (with line search)."""
    n_eq = len(model.equation_names) if model.equation_names else 1
    _compute_loss = compute_loss_fn or compute_loss

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

        def loss_fn(params):
            loss, eq_losses = _compute_loss(
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

        (loss, eq_losses), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
        grads_arrays = eqx.filter(grads, eqx.is_array)
        grad_norm = optax.global_norm(grads_arrays)

        def value_fn(p_arrays):
            full_params = eqx.combine(p_arrays, params_static)
            v, _ = _compute_loss(
                model,
                full_params,
                batch,
                loss_key,
                mc_samples,
                weights=state.loss_weights,
                shock_scale=shock_scale,
                quad_nodes=quad_nodes,
                quad_weights=quad_weights,
                target_policy_fn=target_fn,
            )
            return v

        updates, new_opt_state = opt.update(
            grads_arrays,
            state.opt_state,
            params_arrays,
            value=loss,
            grad=grads_arrays,
            value_fn=value_fn,
        )
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
