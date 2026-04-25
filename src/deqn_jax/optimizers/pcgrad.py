"""PCGrad grad-step factory.

PCGrad (Yu et al. 2020) projects per-equation gradients onto each other
to remove conflicting components before summing. It's a wrapper around
any STANDARD-kind optimizer; selected via ``config.pcgrad_enabled``,
not by optimizer name.
"""

from typing import Any, Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from deqn_jax.training.loss import compute_loss, eq_losses_to_array
from deqn_jax.training.reweighting import update_reweighting
from deqn_jax.types import Metrics, ModelSpec, TrainState


def make_grad_step_pcgrad(
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
    """JIT'd: one PCGrad gradient update on an explicit minibatch."""
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

        def eq_loss_vector(params):
            _, eq_losses = compute_loss(
                model, params, batch, loss_key, mc_samples,
                weights=state.loss_weights, shock_scale=shock_scale,
                quad_nodes=quad_nodes, quad_weights=quad_weights,
                target_policy_fn=target_fn,
            )
            return eq_losses_to_array(eq_losses)

        def total_loss_fn(params):
            loss, eq_losses = _compute_loss_total(
                model, params, batch, loss_key, mc_samples,
                weights=state.loss_weights, shock_scale=shock_scale,
                quad_nodes=quad_nodes, quad_weights=quad_weights,
                target_policy_fn=target_fn,
            )
            return loss, eq_losses

        eq_jac = jax.jacrev(eq_loss_vector)(state.params)
        params_arrays = eqx.filter(state.params, eqx.is_array)
        flat_params, unflatten_fn = jax.flatten_util.ravel_pytree(params_arrays)
        eq_jac_arrays = eqx.filter(eq_jac, eqx.is_array)
        flat_eq_grads = jnp.stack([
            jax.flatten_util.ravel_pytree(jax.tree.map(lambda x: x[i], eq_jac_arrays))[0]
            for i in range(n_eq)
        ])
        gram = flat_eq_grads @ flat_eq_grads.T
        norms_sq = jnp.diag(gram)
        coeffs = jnp.where(gram < 0, gram / (norms_sq[None, :] + 1e-8), 0.0)
        coeffs = coeffs.at[jnp.diag_indices(n_eq)].set(0.0)
        projected = flat_eq_grads - coeffs @ flat_eq_grads
        final_flat_grad = jnp.sum(projected, axis=0)
        grads_arrays = unflatten_fn(final_flat_grad)

        updates, new_opt_state = opt.update(grads_arrays, state.opt_state, params_arrays)
        updates = jax.tree.map(lambda u: lr_scale * u, updates)
        new_params_arrays = optax.apply_updates(params_arrays, updates)
        new_params = eqx.combine(new_params_arrays, state.params)

        loss, eq_losses = total_loss_fn(state.params)
        grad_norm = jnp.sqrt(jnp.sum(final_flat_grad ** 2))

        new_weights, new_rw = update_reweighting(
            eq_losses, state, loss_reweight, reweight_alpha, n_eq,
        )
        new_state = TrainState(
            params=new_params, opt_state=new_opt_state,
            episode_state=state.episode_state, key=new_key,
            step=state.step + 1, episode=state.episode,
            loss_weights=new_weights, reweight_state=new_rw,
            target_params=state.target_params,
        )
        return new_state, Metrics(loss=loss, residuals=eq_losses, grad_norm=grad_norm)
    return grad_step
