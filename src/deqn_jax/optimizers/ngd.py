"""The optimizer registered as ``ngd``: an RMSProp-style update.

Despite the name, this is not a natural-gradient step. It divides the
gradient by the square root of a running average of squared mini-batch
gradients, with no bias correction:

    v ← decay * v + (1 - decay) * g²
    θ ← θ - lr * g / (sqrt(v) + damping)

``g`` is the minibatch-mean gradient the optimizer receives, not per-sample
gradients, so ``v`` is not an empirical Fisher. ``v`` starts at 0, so the
first step is about ``lr / sqrt(1 - decay)`` per coordinate. The update
equals ``optax.rmsprop(eps=damping, eps_in_sqrt=False)``.
"""

from typing import Any, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from jax import Array

from deqn_jax.optimizers.registry import OptimizerKind, register_optimizer


class NGDState(NamedTuple):
    """State for ``ngd``: step count and the running average of g²."""

    count: Array
    fisher_diag: Any  # pytree matching params, EMA of g²


def ngd(
    learning_rate: float = 1e-3,
    damping: float = 1e-4,
    decay: float = 0.999,
) -> optax.GradientTransformation:
    """RMSProp-style update registered as ``ngd`` (see the module docstring).

    Args:
        learning_rate: Step size
        damping: Added to sqrt(v) in the denominator
        decay: EMA decay of the running average v of g²

    Returns:
        optax.GradientTransformation
    """

    def init_fn(params) -> NGDState:
        fisher_diag = jax.tree.map(jnp.zeros_like, params)
        return NGDState(count=jnp.zeros([], dtype=jnp.int32), fisher_diag=fisher_diag)

    def update_fn(
        updates: Any,
        state: NGDState,
        params: Optional[Any] = None,
    ) -> Tuple[Any, NGDState]:
        # Update Fisher diagonal: F ← decay * F + (1-decay) * g²
        new_fisher = jax.tree.map(
            lambda f, g: decay * f + (1.0 - decay) * g**2,
            state.fisher_diag,
            updates,
        )
        # Preconditioned step: -lr * g / (sqrt(F) + damping)
        preconditioned = jax.tree.map(
            lambda g, f: -learning_rate * g / (jnp.sqrt(f) + damping),
            updates,
            new_fisher,
        )
        return preconditioned, NGDState(count=state.count + 1, fisher_diag=new_fisher)

    return optax.GradientTransformation(init_fn, update_fn)


@register_optimizer("ngd", kind=OptimizerKind.STANDARD)
def _ngd(config):
    return ngd(
        learning_rate=config.learning_rate,
        damping=config.damping,
        decay=config.decay,
    )
