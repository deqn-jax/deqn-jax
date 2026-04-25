"""MAO-KFAC: Multi-Adaptive Optimizer with Kronecker-Factored Natural Gradient.

Combines per-equation Kronecker-factored preconditioning (K-FAC/Shampoo-style)
with MAO-style adaptive task balancing.

Key insight: since all equations share the same network forward pass, the
input-side Kronecker factor (R) is shared across equations. Only the
output-side factor (L) is per-equation. This cuts factor storage nearly in half.

For each equation n and layer l:
    L_n = EMA(J_n @ J_n^T)           per-equation [out, out]
    R   = EMA(mean_n(J_n^T @ J_n))   shared [in, in]
    natural_grad_n = L_n^{-1/4} @ J_n @ R^{-1/4}

Then MAO normalization (Option A -- NGD for curvature, MAO for balance):
    v_n = EMA(||natural_grad_n||^2)   per-equation scalar
    update = -lr * mean_n[ natural_grad_n / (sqrt(v_n) + eps) ]

This is NOT an optax.GradientTransformation -- it has the same custom interface
as MAO, receiving per-equation Jacobians instead of standard gradients.

Config field mapping:
    beta1             -> beta_kfac (Kronecker factor EMA, like Shampoo beta)
    beta2             -> beta_mao  (per-equation second moment EMA, like Adam beta2)
    damping           -> eigendecomposition ridge
    precond_update_freq -> inverse recomputation frequency
    epsilon           -> normalization epsilon
"""

from typing import Any, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.optimizers.registry import OptimizerKind, register_optimizer


class MAOKFACState(NamedTuple):
    """State for MAO-KFAC optimizer."""

    count: Array  # scalar step count
    shared_R: Any  # pytree, each leaf [in_dim, in_dim]
    per_eq_L: Any  # pytree, each leaf [n_eq, out_dim, out_dim]
    per_eq_v: Array  # [n_eq] second moment of natural grad norms
    R_inv4: Any  # cached R^{-1/4}, pytree, each leaf [in_dim, in_dim]
    L_inv4: Any  # cached per-eq L^{-1/4}, pytree, each leaf [n_eq, out_dim, out_dim]


def _matrix_power_neg_quarter(M: Array, ridge: float = 1e-6) -> Array:
    """Compute M^{-1/4} via eigendecomposition."""
    M = (M + M.T) / 2.0
    eigvals, eigvecs = jnp.linalg.eigh(M)
    eigvals = jnp.maximum(eigvals, ridge)
    return eigvecs @ jnp.diag(eigvals ** (-0.25)) @ eigvecs.T


class MAOKFACTransform:
    """MAO with Kronecker-Factored Natural Gradient preconditioning.

    Maintains shared input-side (R) and per-equation output-side (L) Kronecker
    factors, applies K-FAC preconditioning to each equation's gradient, then
    uses MAO-style adaptive normalization to balance equation contributions.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta_kfac: float = 0.9,
        beta_mao: float = 0.999,
        epsilon: float = 1e-8,
        damping: float = 1e-4,
        n_tasks: int = 1,
        precond_update_freq: int = 10,
    ):
        self.learning_rate = learning_rate
        self.beta_kfac = beta_kfac
        self.beta_mao = beta_mao
        self.epsilon = epsilon
        self.damping = damping
        self.n_tasks = n_tasks
        self.precond_update_freq = precond_update_freq

    def init(self, params: Any) -> MAOKFACState:
        """Initialize state with identity Kronecker factors."""
        n = self.n_tasks

        def make_R(p):
            """Right (input-side) factor: [in_dim, in_dim]."""
            in_dim = p.shape[-1] if p.ndim >= 2 else 1
            return jnp.eye(in_dim, dtype=p.dtype)

        def make_L(p):
            """Left (output-side) factor: [n_eq, out_dim, out_dim]."""
            out_dim = p.shape[0] if p.ndim >= 1 else 1
            return jnp.repeat(jnp.eye(out_dim, dtype=p.dtype)[None], n, axis=0)

        shared_R = jax.tree.map(make_R, params)
        per_eq_L = jax.tree.map(make_L, params)
        R_inv4 = jax.tree.map(make_R, params)
        L_inv4 = jax.tree.map(make_L, params)

        return MAOKFACState(
            count=jnp.zeros([], dtype=jnp.int32),
            shared_R=shared_R,
            per_eq_L=per_eq_L,
            per_eq_v=jnp.zeros(n),
            R_inv4=R_inv4,
            L_inv4=L_inv4,
        )

    def update(
        self,
        eq_jacobian: Any,
        state: MAOKFACState,
        params: Any,
    ) -> Tuple[Any, MAOKFACState]:
        """Compute MAO-KFAC update from per-equation Jacobians.

        Args:
            eq_jacobian: Pytree matching params, each leaf [n_eq, *param_shape]
            state: Current optimizer state
            params: Current parameters (unused, kept for API consistency)

        Returns:
            Tuple of (updates pytree, new_state)
        """
        count = state.count + 1
        bk = self.beta_kfac
        bm = self.beta_mao
        eps = self.epsilon
        damping = self.damping
        n_tasks = self.n_tasks
        do_update = (count % self.precond_update_freq) == 0

        # --- 1. Update Kronecker factors ---

        def _to_3d(j_leaf):
            """Reshape 1D param Jacobians [n_eq, dim] -> [n_eq, dim, 1]."""
            if j_leaf.ndim == 2:
                return j_leaf[..., None]
            return j_leaf

        def update_R(j_leaf, R_old):
            j_3d = _to_3d(j_leaf)  # [n_eq, out, in]
            JtJ = jnp.mean(jnp.einsum("noi,noj->nij", j_3d, j_3d), axis=0)
            return bk * R_old + (1.0 - bk) * JtJ

        def update_L(j_leaf, L_old):
            j_3d = _to_3d(j_leaf)  # [n_eq, out, in]
            JJt = jnp.einsum("noi,npi->nop", j_3d, j_3d)  # [n_eq, out, out]
            return bk * L_old + (1.0 - bk) * JJt

        new_R = jax.tree.map(update_R, eq_jacobian, state.shared_R)
        new_L = jax.tree.map(update_L, eq_jacobian, state.per_eq_L)

        # --- 2. Periodically recompute inverse fourth roots ---

        def maybe_update_R_inv(R_new, R_inv_old):
            new_inv = _matrix_power_neg_quarter(R_new, ridge=damping)
            return jax.lax.cond(
                do_update,
                lambda _: new_inv,
                lambda _: R_inv_old,
                None,
            )

        def maybe_update_L_inv(L_new, L_inv_old):
            # vmap eigendecomposition across equations
            new_inv = jax.vmap(lambda M: _matrix_power_neg_quarter(M, ridge=damping))(
                L_new
            )  # [n_eq, out, out]
            return jax.lax.cond(
                do_update,
                lambda _: new_inv,
                lambda _: L_inv_old,
                None,
            )

        new_R_inv4 = jax.tree.map(maybe_update_R_inv, new_R, state.R_inv4)
        new_L_inv4 = jax.tree.map(maybe_update_L_inv, new_L, state.L_inv4)

        # --- 3. Precondition: natural_grad_n = L_n^{-1/4} @ J_n @ R^{-1/4} ---

        def precondition(j_leaf, L_inv, R_inv):
            is_1d = j_leaf.ndim == 2
            j_3d = _to_3d(j_leaf)  # [n_eq, out, in]
            # L_inv @ J: [n_eq,out,out] @ [n_eq,out,in] -> [n_eq,out,in]
            p = jnp.einsum("nop,npi->noi", L_inv, j_3d)
            # @ R_inv: [n_eq,out,in] @ [in,in] -> [n_eq,out,in]
            p = jnp.einsum("noi,ij->noj", p, R_inv)
            if is_1d:
                return p.squeeze(-1)  # [n_eq, out]
            return p

        natural_grads = jax.tree.map(
            precondition,
            eq_jacobian,
            new_L_inv4,
            new_R_inv4,
        )

        # --- 4. MAO normalization: balance equation contributions ---

        per_eq_norms_sq = jnp.zeros(n_tasks)
        for leaf in jax.tree.leaves(natural_grads):
            per_eq_norms_sq = per_eq_norms_sq + jnp.sum(
                leaf.reshape(n_tasks, -1) ** 2,
                axis=1,
            )

        new_v = bm * state.per_eq_v + (1.0 - bm) * per_eq_norms_sq
        v_hat = new_v / (1.0 - bm**count)  # bias correction

        def normalize_and_average(ng_leaf):
            scale = 1.0 / (jnp.sqrt(v_hat) + eps)  # [n_eq]
            # Broadcast [n_eq] -> [n_eq, 1, 1, ...]
            for _ in range(ng_leaf.ndim - 1):
                scale = scale[..., None]
            return jnp.mean(ng_leaf * scale, axis=0)

        updates = jax.tree.map(normalize_and_average, natural_grads)
        updates = jax.tree.map(lambda u: -self.learning_rate * u, updates)

        new_state = MAOKFACState(
            count=count,
            shared_R=new_R,
            per_eq_L=new_L,
            per_eq_v=new_v,
            R_inv4=new_R_inv4,
            L_inv4=new_L_inv4,
        )
        return updates, new_state


class _MAOKFACFactory:
    """Deferred MAO-KFAC construction -- resolves n_tasks at create_train_state time."""

    def __init__(self, config):
        self.learning_rate = config.learning_rate
        self.beta_kfac = config.beta1
        self.beta_mao = config.beta2
        self.epsilon = config.epsilon
        self.damping = config.damping
        self.precond_update_freq = config.precond_update_freq

    def with_num_tasks(self, n_tasks: int) -> MAOKFACTransform:
        return MAOKFACTransform(
            learning_rate=self.learning_rate,
            beta_kfac=self.beta_kfac,
            beta_mao=self.beta_mao,
            epsilon=self.epsilon,
            damping=self.damping,
            n_tasks=n_tasks,
            precond_update_freq=self.precond_update_freq,
        )


@register_optimizer("mao_kfac", kind=OptimizerKind.MAO)
def _mao_kfac(config):
    return _MAOKFACFactory(config)
