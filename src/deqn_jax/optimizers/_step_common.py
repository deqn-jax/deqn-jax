"""Shared pieces of the five train-step variants.

Every ``make_grad_step_*`` factory builds the same two things: a call into
``compute_loss`` with the construction-time constants (model, mc_samples,
quadrature rule) already bound, and the tail that turns a finished update
into ``(TrainState, Metrics)``. They are written once here and used from
``standard.py``, ``pcgrad.py``, ``mao.py``, ``lbfgs.py`` and
``gauss_newton.py``.

Both helpers are traced inside the variants' ``jax.jit``; they add no
operations of their own, so the XLA graph is unchanged.
"""

import inspect
from typing import Any, Callable, Optional, Tuple

from jax import Array

from deqn_jax.training.loss import compute_loss
from deqn_jax.training.reweighting import update_reweighting
from deqn_jax.types import Metrics, ModelSpec, TrainState


def make_loss_call(
    model: ModelSpec,
    mc_samples: int,
    quad_nodes: Optional[Array],
    quad_weights: Optional[Array],
    compute_loss_fn: Optional[Callable] = None,
) -> Callable:
    """Bind the construction-time constants of a ``compute_loss`` call.

    Returns ``loss_call(params, batch, loss_key, *, weights, shock_scale,
    target_policy_fn, aux_params=None) -> (loss, eq_losses)``. Pass
    ``compute_loss_fn=None`` for the base (per-equation MSE) loss and the
    configured composite loss otherwise.

    ``aux_params`` is forwarded only when the wrapped loss declares it, so a
    loss builder that does not know about a second trainable module keeps its
    exact signature. A wrapper that swallows everything into ``**kwargs``
    without naming ``aux_params`` is *opaque*: we cannot tell whether the loss
    it wraps wants the argument, so forwarding is refused loudly rather than
    silently dropping the aux module. (Wrappers should carry
    ``functools.wraps`` so the inner signature shows through and this never
    trips.)
    """
    fn = compute_loss_fn or compute_loss
    sig_params = inspect.signature(fn).parameters
    wants_aux = "aux_params" in sig_params
    opaque = not wants_aux and any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig_params.values()
    )

    def loss_call(
        params: Any,
        batch: Array,
        loss_key: Array,
        *,
        weights: Optional[Array],
        shock_scale: Array,
        target_policy_fn: Optional[Callable],
        aux_params: Any = None,
    ):
        if opaque and aux_params is not None:
            raise ValueError(
                f"Loss function {getattr(fn, '__qualname__', fn)!r} takes "
                "**kwargs and does not name 'aux_params', so it cannot be "
                "told whether the wrapped loss uses it. Give the wrapper a "
                "functools.wraps of the loss it wraps, or an explicit "
                "aux_params parameter."
            )
        return fn(
            model,
            params,
            batch,
            loss_key,
            mc_samples,
            weights=weights,
            shock_scale=shock_scale,
            quad_nodes=quad_nodes,
            quad_weights=quad_weights,
            target_policy_fn=target_policy_fn,
            **({"aux_params": aux_params} if wants_aux else {}),
        )

    return loss_call


def finalize_step(
    state: TrainState,
    *,
    params: Any,
    opt_state: Any,
    key: Array,
    loss: Array,
    eq_losses: Any,
    grad_norm: Array,
    loss_reweight: str,
    reweight_alpha: float,
    n_eq: int,
) -> Tuple[TrainState, Metrics]:
    """Advance the loss reweighting, rebuild the state, and build Metrics."""
    new_weights, new_rw = update_reweighting(
        eq_losses,
        state,
        loss_reweight,
        reweight_alpha,
        n_eq,
    )
    new_state = state._replace(
        params=params,
        opt_state=opt_state,
        key=key,
        step=state.step + 1,
        loss_weights=new_weights,
        reweight_state=new_rw,
    )
    return new_state, Metrics(loss=loss, residuals=eq_losses, grad_norm=grad_norm)
