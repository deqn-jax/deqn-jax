"""Composite loss for disaster model: anchor + Jacobian + barrier + Newton terms.

Drop-in replacement for compute_loss() — returns the same (total_loss, eq_losses_dict)
signature, with auxiliary losses keyed with "aux_" prefix so adaptive reweighting
and per-equation gradient surgery only see the base equilibrium residuals.

Usage:
    data = prepare_composite_data(model, P, Q)
    loss_fn = make_composite_loss(model, data, config.composite_loss)
    # loss_fn has the same signature as compute_loss
"""

from typing import Callable, Dict, Optional, Tuple, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.types import ModelSpec
from deqn_jax.training.loss import compute_loss


class CompositeData(NamedTuple):
    """Pre-computed linearization data for composite loss terms.

    Attributes:
        P: Policy rule matrix [n_policies, n_states] from Blanchard-Kahn
        ss_state: Steady state [n_states]
        ss_policy: Steady state policy [n_policies]
        ergodic_cov_chol: Cholesky of ergodic covariance [n_states, n_states]
        ss_leverage: Steady-state leverage scalar
    """
    P: Array
    ss_state: Array
    ss_policy: Array
    ergodic_cov_chol: Array
    ss_leverage: float


def prepare_composite_data(
    model: ModelSpec,
    P: Array,
    Q: Array,
    verbose: bool = True,
) -> CompositeData:
    """Build CompositeData from linearization results.

    Args:
        model: Model specification
        P: Policy rule matrix from linearize_model
        Q: Transition matrix from linearize_model
        verbose: Print diagnostic info
    """
    from deqn_jax.training.linearize import compute_ergodic_covariance

    ss_state, ss_policy = model.steady_state_fn(model.constants)
    ergodic_cov = compute_ergodic_covariance(Q, model, verbose=verbose)

    # Cholesky with regularization for numerical stability
    n = ergodic_cov.shape[0]
    ergodic_cov_chol = jnp.linalg.cholesky(ergodic_cov + 1e-8 * jnp.eye(n))

    # Compute SS leverage
    ss_defs = model.definitions_fn(ss_state, ss_policy, model.constants)
    ss_leverage = float(ss_defs["L"])

    if verbose:
        print(f"  Composite loss: SS leverage = {ss_leverage:.4f}")

    return CompositeData(
        P=P,
        ss_state=ss_state,
        ss_policy=ss_policy,
        ergodic_cov_chol=ergodic_cov_chol,
        ss_leverage=ss_leverage,
    )


def _anchor_loss(
    policy_fn: Callable[[Array], Array],
    key: Array,
    data: CompositeData,
    n_points: int,
    sigma: float,
) -> Array:
    """Anchor loss: ||f_net(x) - f_lin(x)||^2 for x ~ N(ss, sigma^2 * Sigma).

    Penalizes deviation of the neural network policy from the linearized
    (Blanchard-Kahn) policy near the steady state. This anchors the net
    to the correct local behavior.
    """
    # Sample points near SS: x = ss + sigma * L @ z, z ~ N(0, I)
    z = jax.random.normal(key, (n_points, data.ss_state.shape[0]))
    deviations = sigma * z @ data.ergodic_cov_chol.T  # [n_points, n_states]
    sample_states = data.ss_state + deviations

    # Neural network policy at sampled points
    net_policy = jax.vmap(policy_fn)(sample_states)  # [n_points, n_policies]

    # Linear policy: p = ss_policy + P @ (s - ss_state)
    lin_policy = data.ss_policy + deviations @ data.P.T  # [n_points, n_policies]

    return jnp.mean((net_policy - lin_policy) ** 2)


def _jac_loss(
    policy_fn: Callable[[Array], Array],
    data: CompositeData,
) -> Array:
    """Jacobian loss: ||J_net(ss) - P||^2_F.

    Penalizes deviation of the neural network Jacobian at the steady state
    from the linearized policy rule matrix P. This ensures the net has
    the correct first-order response to state perturbations.
    """
    # Jacobian of net at SS: [n_policies, n_states]
    J_net = jax.jacfwd(policy_fn)(data.ss_state)
    return jnp.mean((J_net - data.P) ** 2)


def _barrier_losses(
    defs: Dict[str, Array],
    data: CompositeData,
    leverage_mult: float,
) -> Dict[str, Array]:
    """Log-barrier and penalty losses for economic feasibility.

    Prevents the optimizer from driving the model into pathological regions
    (negative net worth, collapsed consumption, divergent leverage).
    """
    losses = {}

    # Net worth barrier: -log(n) penalizes n approaching zero
    n = defs["n"]
    losses["aux_barrier_n"] = jnp.mean(-jnp.log(jnp.maximum(n, 1e-8)))

    # Leverage penalty: (L - L_ss)^2 / L_ss^2 when L > leverage_mult * L_ss
    L = defs["L"]
    L_threshold = leverage_mult * data.ss_leverage
    excess = jnp.maximum(L - L_threshold, 0.0)
    losses["aux_barrier_L"] = jnp.mean((excess / data.ss_leverage) ** 2)

    # Consumption floor: -log(c) penalizes c approaching zero
    c = defs["c"]
    losses["aux_barrier_c"] = jnp.mean(-jnp.log(jnp.maximum(c, 1e-8)))

    return losses


def _newton_losses(
    defs: Dict[str, Array],
) -> Dict[str, Array]:
    """Newton solver diagnostic losses.

    Penalizes regions where the Newton solver for omega_bar is ill-conditioned
    (h'(omega) near zero) or has high residual.
    """
    losses = {}

    # Newton condition: penalize h'(omega) < 0.1 (ill-conditioned solver)
    h_prime = defs["newton_h_prime"]
    # Soft penalty: (max(0.1 - h', 0))^2
    deficit = jnp.maximum(0.1 - h_prime, 0.0)
    losses["aux_newton_cond"] = jnp.mean(deficit ** 2)

    # Newton residual: should be near zero if solver converged
    newton_resid = defs["newton_residual"]
    losses["aux_newton_resid"] = jnp.mean(newton_resid ** 2)

    return losses


def make_composite_loss(
    model: ModelSpec,
    data: CompositeData,
    anchor_weight: float = 1.0,
    jac_weight: float = 0.1,
    barrier_weight: float = 0.01,
    newton_weight: float = 0.01,
    n_anchor_points: int = 64,
    anchor_sigma: float = 1.0,
    leverage_mult: float = 5.0,
) -> Callable:
    """Create composite loss function as drop-in replacement for compute_loss.

    Returns a function with the same signature as compute_loss():
        (model, policy_fn, states, key, mc_samples, weights, shock_scale,
         quad_nodes, quad_weights) -> (total_loss, eq_losses_dict)

    Auxiliary loss entries are keyed with "aux_" prefix.
    """

    def composite_loss_fn(
        model_: ModelSpec,
        policy_fn: Callable[[Array], Array],
        states: Array,
        key: Array,
        mc_samples: int = 5,
        weights: Optional[Array] = None,
        shock_scale: float = 1.0,
        quad_nodes: Optional[Array] = None,
        quad_weights: Optional[Array] = None,
    ) -> Tuple[Array, Dict[str, Array]]:
        # 1. Base MSE loss (standard Euler residuals)
        base_loss, eq_losses = compute_loss(
            model_, policy_fn, states, key, mc_samples,
            weights=weights, shock_scale=shock_scale,
            quad_nodes=quad_nodes, quad_weights=quad_weights,
        )

        # Split key for anchor sampling
        key, anchor_key = jax.random.split(key)

        # 2. Anchor loss: net should match linearized policy near SS
        anchor = _anchor_loss(policy_fn, anchor_key, data, n_anchor_points, anchor_sigma)
        eq_losses["aux_anchor"] = anchor

        # 3. Jacobian loss: net Jacobian at SS should match P
        jac = _jac_loss(policy_fn, data)
        eq_losses["aux_jac"] = jac

        # 4. Barrier + Newton losses from training batch definitions
        defs = jax.vmap(
            lambda s: model_.definitions_fn(s, policy_fn(s), model_.constants)
        )(states)

        barriers = _barrier_losses(defs, data, leverage_mult)
        eq_losses.update(barriers)

        newtons = _newton_losses(defs)
        eq_losses.update(newtons)

        # 5. Weighted total
        total = base_loss
        total = total + anchor_weight * anchor
        total = total + jac_weight * jac
        for k, v in barriers.items():
            total = total + barrier_weight * v
        for k, v in newtons.items():
            total = total + newton_weight * v

        return total, eq_losses

    return composite_loss_fn
