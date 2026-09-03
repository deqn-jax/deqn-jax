"""Standard grad-step factory.

Used for any optimizer registered with ``OptimizerKind.STANDARD``
(adam, sgd, adamw, lion, muon, ngd, shampoo, kfac, ...). Builds a
JIT'd ``grad_step(state, batch, lr_scale, shock_scale)`` that applies
one optax-style update on an explicit minibatch.

Lives in ``optimizers/`` (rather than ``training/``) so each optimizer
family owns its own grad-step variant — paired with ``mao.py``,
``lbfgs.py``, ``gauss_newton.py``, and ``pcgrad.py``.
"""

from typing import Any, Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from deqn_jax.optimizers._step_common import finalize_step, make_loss_call
from deqn_jax.types import Metrics, ModelSpec, TrainState


def make_grad_step_standard(
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
    """JIT'd: one STANDARD gradient update on an explicit minibatch.

    The minibatch replaces what was ``ctx.train_states`` in the legacy
    single-batch train step; no rollout is run. Optimizer state, loss
    reweighting, and Metrics construction are identical to the legacy
    path so consumers see no difference per step.
    """
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

        params_arrays = eqx.filter(state.params, eqx.is_array)
        grads_arrays = eqx.filter(grads, eqx.is_array)

        updates, new_opt_state = opt.update(
            grads_arrays, state.opt_state, params_arrays
        )
        updates = jax.tree.map(lambda u: lr_scale * u, updates)
        new_params_arrays = optax.apply_updates(params_arrays, updates)
        new_params = eqx.combine(new_params_arrays, state.params)
        grad_norm = optax.global_norm(grads_arrays)

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
