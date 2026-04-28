"""Standard grad-step factory.

Used for any optimizer registered with ``OptimizerKind.STANDARD``
(adam, sgd, adamw, lion, muon, ngd, shampoo, kfac, ...). Builds a
JIT'd ``grad_step(state, batch, lr_scale, shock_scale)`` that applies
one optax-style update on an explicit minibatch.

Lives in ``optimizers/`` (rather than ``training/``) so each optimizer
family owns its own grad-step variant — paired with ``mao.py``,
``lbfgs.py``, ``gauss_newton.py``, and ``pcgrad.py``.

Actor-critic note: the loss is always differentiated w.r.t. the tuple
``(state.params, state.aux_params)``. When ``aux_params`` is ``None``
(no critic, or shared-trunk AC where the value head lives inside
``params``), ``filter_value_and_grad`` returns ``(policy_grads, None)``
and the critic-side update is skipped. The ``critic_opt`` plumbing
runs only when ``aux_params`` and ``critic_opt`` are both populated
(separate-mode AC). One JIT'd function handles all three modes.

Architectural caveat (revisit when lifting the STANDARD-only restriction
on AC): the actor-critic logic lives *inside* this STANDARD grad-step
factory rather than as an orthogonal layer over any optimizer family.
The TrainConfig validator currently gates ``actor_critic.mode != None``
to STANDARD optimizers (adam/sgd/adamw/lion); the moment we want AC +
MAO / PCGrad / GN / LBFGS, this approach forces us to duplicate the
``(params, aux_params)``-tuple grad pattern + critic-update block
inside each of those grad-step files. The right refactor at that point
is to extract AC into a wrapper:

    ac_step = wrap_with_critic(any_grad_step, critic_opt)

so each optimizer family stays AC-agnostic. v1 sidesteps this because
all four supported AC optimizers are STANDARD-family, but the
restriction is documented in the validator for that reason.
"""

from typing import Any, Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from deqn_jax.training.loss import compute_loss
from deqn_jax.training.reweighting import update_reweighting
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
    critic_opt: Optional[Any] = None,
):
    """JIT'd: one STANDARD gradient update on an explicit minibatch.

    Differentiates the loss w.r.t. ``(state.params, state.aux_params)``
    and applies ``opt`` to the policy grads. If ``critic_opt`` is set
    AND ``state.aux_params`` is non-None (separate-mode actor-critic),
    also applies ``critic_opt`` to the critic grads using
    ``state.aux_opt_state``. Otherwise the critic-side path is a
    compile-time no-op.
    """
    n_eq = len(model.equation_names) if model.equation_names else 1
    _compute_loss = compute_loss_fn or compute_loss
    has_critic_opt = critic_opt is not None  # Python-level constant

    @jax.jit
    def grad_step(
        state: TrainState,
        batch: Array,
        lr_scale: Array,
        shock_scale: Array = jnp.array(1.0),
    ) -> Tuple[TrainState, Metrics]:
        loss_key, new_key = jax.random.split(state.key)
        target_fn = state.target_params if use_target_network else None

        def loss_fn(params_tuple):
            policy_params, critic_params = params_tuple
            loss, eq_losses = _compute_loss(
                model,
                policy_params,
                batch,
                loss_key,
                mc_samples,
                weights=state.loss_weights,
                shock_scale=shock_scale,
                quad_nodes=quad_nodes,
                quad_weights=quad_weights,
                target_policy_fn=target_fn,
                aux_params=critic_params,
            )
            return loss, eq_losses

        (loss, eq_losses), (policy_grads, critic_grads) = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )((state.params, state.aux_params))

        # --- Policy update (always) ---
        policy_arrays = eqx.filter(state.params, eqx.is_array)
        policy_grads_a = eqx.filter(policy_grads, eqx.is_array)
        p_updates, new_opt_state = opt.update(
            policy_grads_a, state.opt_state, policy_arrays
        )
        p_updates = jax.tree.map(lambda u: lr_scale * u, p_updates)
        new_policy_arrays = optax.apply_updates(policy_arrays, p_updates)
        new_params = eqx.combine(new_policy_arrays, state.params)

        # --- Critic update (separate-mode AC only) ---
        # Compile-time branch: when has_critic_opt is False or
        # state.aux_params is None, this entire block is skipped at
        # construction time.
        if has_critic_opt and state.aux_params is not None:
            critic_arrays = eqx.filter(state.aux_params, eqx.is_array)
            critic_grads_a = eqx.filter(critic_grads, eqx.is_array)
            c_updates, new_aux_opt_state = critic_opt.update(
                critic_grads_a, state.aux_opt_state, critic_arrays
            )
            c_updates = jax.tree.map(lambda u: lr_scale * u, c_updates)
            new_critic_arrays = optax.apply_updates(critic_arrays, c_updates)
            new_aux_params = eqx.combine(new_critic_arrays, state.aux_params)
            grad_norm = optax.global_norm((policy_grads_a, critic_grads_a))
        else:
            new_aux_params = state.aux_params
            new_aux_opt_state = state.aux_opt_state
            grad_norm = optax.global_norm(policy_grads_a)

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
            aux_params=new_aux_params,
            aux_opt_state=new_aux_opt_state,
            key=new_key,
            step=state.step + 1,
            loss_weights=new_weights,
            reweight_state=new_rw,
        )
        return new_state, Metrics(loss=loss, residuals=eq_losses, grad_norm=grad_norm)

    return grad_step
