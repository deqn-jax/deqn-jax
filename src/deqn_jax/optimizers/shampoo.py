"""In-house Kronecker-factored Shampoo optimizer.

For 2D parameters (weight matrices), maintains Kronecker factors:
    L = EMA(G @ G^T),  R = EMA(G^T @ G)
    update = L^{-1/4} @ G @ R^{-1/4}

For 1D parameters (biases), reshapes to [1, n] to avoid data-dependent branching.

Preconditioners are updated every ``precond_update_freq`` steps.
"""

from typing import Any, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from jax import Array

from deqn_jax.optimizers.registry import OptimizerKind, register_optimizer


class ShampooState(NamedTuple):
    """Shampoo optimizer state."""

    count: Array
    L: Any  # pytree of left preconditioners
    R: Any  # pytree of right preconditioners


def _matrix_power_neg_quarter(M: Array, ridge: float = 1e-6) -> Array:
    """Compute M^{-1/4} via eigendecomposition."""
    M = (M + M.T) / 2.0
    eigvals, eigvecs = jnp.linalg.eigh(M)
    eigvals = jnp.maximum(eigvals, ridge)
    inv_quarter = eigvecs @ jnp.diag(eigvals ** (-0.25)) @ eigvecs.T
    return inv_quarter


def shampoo(
    learning_rate: float = 1e-3,
    beta: float = 0.9,
    precond_update_freq: int = 10,
    epsilon: float = 1e-12,
) -> optax.GradientTransformation:
    """Kronecker-factored Shampoo optimizer.

    Args:
        learning_rate: Step size
        beta: EMA decay for preconditioner statistics
        precond_update_freq: Steps between preconditioner updates
        epsilon: Ridge for numerical stability

    Returns:
        optax.GradientTransformation
    """

    def init_fn(params) -> ShampooState:
        def make_L(p):
            if p.ndim < 2:
                return jnp.eye(1, dtype=p.dtype)
            return jnp.eye(p.shape[0], dtype=p.dtype)

        def make_R(p):
            if p.ndim < 2:
                n = p.shape[0] if p.ndim == 1 else 1
                return jnp.eye(n, dtype=p.dtype)
            return jnp.eye(p.shape[1], dtype=p.dtype)

        L = jax.tree.map(make_L, params)
        R = jax.tree.map(make_R, params)
        return ShampooState(count=jnp.zeros([], dtype=jnp.int32), L=L, R=R)

    def update_fn(
        updates: Any,
        state: ShampooState,
        params: Optional[Any] = None,
    ) -> Tuple[Any, ShampooState]:
        count = state.count + 1
        do_update = (count % precond_update_freq) == 0

        def update_L(g, L_old):
            if g.ndim < 2:
                g_2d = g.reshape(1, -1)
            else:
                g_2d = g
            return jax.lax.cond(
                do_update,
                lambda _: beta * L_old + (1.0 - beta) * (g_2d @ g_2d.T),
                lambda _: L_old,
                None,
            )

        def update_R(g, R_old):
            if g.ndim < 2:
                g_2d = g.reshape(1, -1)
            else:
                g_2d = g
            return jax.lax.cond(
                do_update,
                lambda _: beta * R_old + (1.0 - beta) * (g_2d.T @ g_2d),
                lambda _: R_old,
                None,
            )

        def precondition(g, L_new, R_new):
            original_shape = g.shape
            if g.ndim < 2:
                g_2d = g.reshape(1, -1)
            else:
                g_2d = g

            L_inv4 = _matrix_power_neg_quarter(L_new, ridge=epsilon)
            R_inv4 = _matrix_power_neg_quarter(R_new, ridge=epsilon)
            precond = L_inv4 @ g_2d @ R_inv4

            if g.ndim < 2:
                return -learning_rate * precond.reshape(original_shape)
            return -learning_rate * precond

        new_L = jax.tree.map(update_L, updates, state.L)
        new_R = jax.tree.map(update_R, updates, state.R)
        new_updates = jax.tree.map(precondition, updates, new_L, new_R)

        return new_updates, ShampooState(count=count, L=new_L, R=new_R)

    return optax.GradientTransformation(init_fn, update_fn)


@register_optimizer("shampoo", kind=OptimizerKind.STANDARD)
def _shampoo(config):
    return shampoo(
        learning_rate=config.learning_rate,
        precond_update_freq=config.precond_update_freq,
    )
