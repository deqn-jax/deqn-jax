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

from deqn_jax.optimizers._step_common import finalize_step, make_loss_call
from deqn_jax.optimizers.registry import OptimizerKind, register_optimizer
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
    loss_call = make_loss_call(
        model, mc_samples, quad_nodes, quad_weights, compute_loss_fn
    )

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
            loss, eq_losses = loss_call(
                params,
                batch,
                loss_key,
                weights=state.loss_weights,
                shock_scale=shock_scale,
                target_policy_fn=target_fn,
                aux_params=state.aux_params,
            )
            return loss, eq_losses

        (loss, eq_losses), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
        grads_arrays = eqx.filter(grads, eqx.is_array)
        grad_norm = optax.global_norm(grads_arrays)

        def value_fn(p_arrays):
            full_params = eqx.combine(p_arrays, params_static)
            v, _ = loss_call(
                full_params,
                batch,
                loss_key,
                weights=state.loss_weights,
                shock_scale=shock_scale,
                target_policy_fn=target_fn,
                aux_params=state.aux_params,
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

        return finalize_step(
            state,
            params=new_params,
            opt_state=new_opt_state,
            key=new_key,
            loss=loss,
            eq_losses=eq_losses,
            grad_norm=grad_norm,
            loss_reweight=loss_reweight,
            reweight_alpha=reweight_alpha,
            n_eq=n_eq,
        )

    return grad_step
