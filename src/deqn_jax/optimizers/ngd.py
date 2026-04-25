"""Natural Gradient Descent (diagonal Fisher approximation).

Running diagonal Fisher via EMA of g², preconditioned step:
    θ ← θ - lr * g / (sqrt(F) + damping)

Cheap and effective for PINN-style losses.
"""

from typing import Any, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from jax import Array

from deqn_jax.optimizers.registry import OptimizerKind, register_optimizer


class NGDState(NamedTuple):
    """State for diagonal Fisher NGD."""

    count: Array
    fisher_diag: Any  # pytree matching params, EMA of g²


def ngd(
    learning_rate: float = 1e-3,
    damping: float = 1e-4,
    decay: float = 0.999,
) -> optax.GradientTransformation:
    """Diagonal Fisher Natural Gradient Descent.

    Args:
        learning_rate: Step size
        damping: Regularization added to sqrt(Fisher)
        decay: EMA decay for Fisher diagonal estimate

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
