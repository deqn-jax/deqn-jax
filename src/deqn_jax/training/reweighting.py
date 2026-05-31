"""Adaptive per-equation loss reweighting strategies.

Pure functions used inside JIT'd train steps to update ``loss_weights``
on the ``TrainState`` based on running per-equation loss statistics.

Strategies:
- ``lr_annealing``: inverse-EMA weighting normalized to sum=n_eq.
- ``relobralo``: relative balancing with softmax of loss ratios.
- anything else (e.g. ``"none"``): pass-through, weights unchanged.

Lives here, not in ``trainer.py``, so the per-optimizer grad-step
modules in ``deqn_jax.optimizers`` can use it without circular imports
through trainer.py.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.training.loss import eq_losses_to_array
from deqn_jax.types import ReweightState


def _update_weights_lr_annealing(
    eq_loss_arr: Array,
    reweight_state: ReweightState,
    alpha: float,
    n_eq: int,
) -> Tuple[Array, ReweightState]:
    """LR annealing: inverse EMA weighting, normalized to sum=n_eq."""
    new_running = alpha * reweight_state.running_ema + (1.0 - alpha) * eq_loss_arr
    raw = 1.0 / (new_running + 1e-8)
    weights = raw / jnp.sum(raw) * n_eq
    new_rw = reweight_state._replace(running_ema=new_running, prev_losses=eq_loss_arr)
    return weights, new_rw


def _update_weights_relobralo(
    eq_loss_arr: Array,
    reweight_state: ReweightState,
    alpha: float,
    n_eq: int,
) -> Tuple[Array, ReweightState]:
    """ReLoBRaLo: relative balancing with softmax of loss ratios."""
    init = jnp.where(
        reweight_state.initialized, reweight_state.init_losses, eq_loss_arr
    )
    prev = jnp.where(
        reweight_state.initialized, reweight_state.prev_losses, eq_loss_arr
    )

    eps = 1e-8
    w_t = jax.nn.softmax(eq_loss_arr / (prev + eps)) * n_eq
    w_0 = jax.nn.softmax(eq_loss_arr / (init + eps)) * n_eq
    weights = alpha * w_t + (1.0 - alpha) * w_0

    new_rw = reweight_state._replace(
        prev_losses=eq_loss_arr,
        init_losses=init,
        initialized=jnp.array(True),
    )
    return weights, new_rw


def update_reweighting(eq_losses, state, loss_reweight, reweight_alpha, n_eq):
    """Apply adaptive loss reweighting and return ``(new_weights, new_rw)``."""
    eq_loss_arr = eq_losses_to_array(eq_losses)

    if loss_reweight == "lr_annealing":
        new_weights, new_rw = _update_weights_lr_annealing(
            eq_loss_arr,
            state.reweight_state,
            reweight_alpha,
            n_eq,
        )
    elif loss_reweight == "relobralo":
        new_weights, new_rw = _update_weights_relobralo(
            eq_loss_arr,
            state.reweight_state,
            reweight_alpha,
            n_eq,
        )
    else:
        new_weights = state.loss_weights
        new_rw = state.reweight_state

    return new_weights, new_rw
