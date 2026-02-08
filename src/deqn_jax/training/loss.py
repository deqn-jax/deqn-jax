"""Loss computation with Monte Carlo or Gauss-Hermite quadrature expectations.

The DEQN loss is the mean squared residual of equilibrium equations:

    L = E_s[ E_ε[ r(s, π(s), s', π(s'))² ] ]

where the expectation is over:
1. States s drawn from episode trajectories
2. Shocks ε determining next state s' = step(s, π(s), ε)

Expectation methods:
- **MC**: Antithetic variates (pair each ε with -ε for variance reduction)
- **Quadrature**: Gauss-Hermite tensor-product nodes (exact for polynomial integrands)

Residual aggregation uses (E[r])² (average THEN square):
- Correct loss for E[r]=0 equilibrium conditions
- Robust to outlier residuals (averages first, tames singularities)
- With quadrature weights: weighted mean then square
"""

import math
from functools import lru_cache
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from deqn_jax.types import ModelSpec


# ---------------------------------------------------------------------------
# Shock sampling: Monte Carlo
# ---------------------------------------------------------------------------

def sample_antithetic_shocks(
    key: Array,
    n_samples: int,
    batch_size: int,
    shock_dim: int,
    shock_scale: float = 1.0,
) -> Array:
    """Generate Monte Carlo shocks with antithetic variates.

    Antithetic sampling pairs each shock ε with -ε, reducing variance
    for symmetric distributions (like standard normal).

    Args:
        key: JAX PRNG key
        n_samples: Number of MC samples (will be rounded to even)
        batch_size: Batch size
        shock_dim: Dimension of shock vector
        shock_scale: Curriculum scaling for shocks (0→1 ramp)

    Returns:
        Shocks array [n_samples, batch_size, shock_dim]
    """
    if n_samples <= 0 or shock_dim <= 0:
        return jnp.zeros((1, batch_size, shock_dim))

    half = n_samples // 2
    base = jax.random.normal(key, (half, batch_size, shock_dim))

    # Pair each shock with its antithetic twin
    shocks = jnp.concatenate([base, -base], axis=0)

    # Handle odd n_samples
    if n_samples % 2 == 1:
        key, subkey = jax.random.split(key)
        extra = jax.random.normal(subkey, (1, batch_size, shock_dim))
        shocks = jnp.concatenate([shocks, extra], axis=0)

    return shocks * shock_scale


# ---------------------------------------------------------------------------
# Shock sampling: Gauss-Hermite quadrature
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _hermgauss_1d(n_points: int):
    """Cached 1D Gauss-Hermite nodes/weights for exp(-x²)."""
    return np.polynomial.hermite.hermgauss(n_points)


def gauss_hermite_nd(
    n_points: int,
    dim: int,
    max_points: int = 4096,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Tensor-product Gauss-Hermite nodes/weights for standard normal.

    Transforms from Hermite basis (weight exp(-x²)) to standard normal:
    - Nodes: x' = sqrt(2) * x
    - Weights: w' = w / sqrt(π)

    Args:
        n_points: Quadrature points per dimension
        dim: Number of shock dimensions
        max_points: Safety cap on total grid points

    Returns:
        Tuple of (nodes [n_nodes, dim], weights [n_nodes]), or None if too many.
    """
    if dim <= 0 or n_points <= 0:
        return None

    n_nodes = n_points ** dim
    if n_nodes > max_points:
        return None

    x, w = _hermgauss_1d(n_points)
    # Convert to standard normal: x' = sqrt(2)*x, w' = w/sqrt(pi)
    x = x * math.sqrt(2.0)
    w = w / math.sqrt(math.pi)

    if dim == 1:
        return x.reshape(-1, 1), w

    # Tensor product grid
    grids = np.array(np.meshgrid(*([x] * dim), indexing="ij"))
    nodes = grids.reshape(dim, -1).T  # [n_nodes, dim]
    w_grids = np.array(np.meshgrid(*([w] * dim), indexing="ij"))
    weights = np.prod(w_grids, axis=0).reshape(-1)  # [n_nodes]

    return nodes, weights


# ---------------------------------------------------------------------------
# Residual computation
# ---------------------------------------------------------------------------

def compute_residuals(
    model: ModelSpec,
    policy_fn: Callable[[Array], Array],
    train_batch: Array,
    shock: Array,
) -> Dict[str, Array]:
    """Compute equilibrium equation residuals for a single shock realization.

    Handles both MLP [B, D] and sequence [B, H, D] inputs:
    - For [B, D]: standard MLP path, policy_fn(states)
    - For [B, H, D]: extract current state from last timestep,
      compute next_state, shift history window for next_policy

    The ndim check resolves at JAX trace time (no runtime branching).

    Args:
        model: Model specification
        policy_fn: Policy network (states -> policies) or (history -> policies)
        train_batch: Current states [batch, n_states] or history windows [batch, H, n_states]
        shock: Shock realization [batch, n_shocks]

    Returns:
        Dict mapping equation names to residuals [batch]
    """
    if train_batch.ndim == 3:
        # Sequence path: [B, H, D]
        states = train_batch[:, -1, :]  # current state from last timestep
        policy = policy_fn(train_batch)
        next_state = model.step_fn(states, policy, shock, model.constants)
        # Shift history: drop oldest, append next_state
        next_batch = jnp.concatenate(
            [train_batch[:, 1:, :], next_state[:, None, :]], axis=1
        )
        next_policy = policy_fn(next_batch)
    else:
        # MLP path: [B, D]
        states = train_batch
        policy = policy_fn(states)
        next_state = model.step_fn(states, policy, shock, model.constants)
        next_policy = policy_fn(next_state)

    return model.equations_fn(
        states, policy, next_state, next_policy, model.constants
    )


# ---------------------------------------------------------------------------
# Loss computation (unified MC + quadrature)
# ---------------------------------------------------------------------------

def compute_loss(
    model: ModelSpec,
    policy_fn: Callable[[Array], Array],
    states: Array,
    key: Array,
    mc_samples: int = 5,
    weights: Optional[Array] = None,
    shock_scale: float = 1.0,
    quad_nodes: Optional[Array] = None,
    quad_weights: Optional[Array] = None,
) -> Tuple[Array, Dict[str, Array]]:
    """Compute DEQN loss with MC or quadrature expectations.

    Aggregation: (E[r])² — square the weighted mean residual per batch element.
    This is the correct loss for E[r]=0 equilibrium conditions and is robust
    to outlier residuals (averages first, then squares).

    For MC:        shocks ~ N(0, shock_scale²), uniform weights 1/N
    For quadrature: shocks = nodes * shock_scale, Gauss-Hermite weights

    Handles both MLP [batch, n_states] and sequence [batch, H, n_states] inputs
    transparently (dispatched inside compute_residuals via ndim check).

    Args:
        model: Model specification
        policy_fn: Policy network (states -> policies) or (history -> policies)
        states: State batch [batch, n_states] or history windows [batch, H, n_states]
        key: PRNG key for MC shock sampling (ignored for quadrature)
        mc_samples: Number of Monte Carlo samples (ignored for quadrature)
        weights: Per-equation loss weights [n_eq] (default: uniform)
        shock_scale: Curriculum scaling for shocks (0→1 ramp)
        quad_nodes: Quadrature nodes [n_nodes, shock_dim] (None -> use MC)
        quad_weights: Quadrature weights [n_nodes] (None -> use MC)

    Returns:
        Tuple of (scalar loss, dict of per-equation losses)
    """
    batch_size = states.shape[0]
    use_quadrature = quad_nodes is not None and quad_weights is not None

    if use_quadrature:
        n_nodes = quad_nodes.shape[0]
        # Broadcast nodes to [n_nodes, batch_size, shock_dim] and apply curriculum
        shocks = jnp.broadcast_to(
            quad_nodes[:, None, :],
            (n_nodes, batch_size, model.n_shocks),
        ) * shock_scale
        sample_weights = quad_weights  # [n_nodes]
    else:
        shocks = sample_antithetic_shocks(
            key, mc_samples, batch_size, model.n_shocks, shock_scale,
        )
        n_samples = shocks.shape[0]
        sample_weights = jnp.ones(n_samples) / n_samples  # uniform

    # Compute residuals for each shock/node
    def compute_sample_residuals(shock):
        return compute_residuals(model, policy_fn, states, shock)

    # vmap over samples/nodes: Dict[str, [n_samples, batch]]
    all_residuals = jax.vmap(compute_sample_residuals)(shocks)

    # (E[r])² aggregation: weighted mean over samples, then square
    eq_losses = {}
    total_loss = 0.0

    for i, (eq_name, residuals) in enumerate(all_residuals.items()):
        # residuals: [n_samples, batch]
        # Weighted mean over samples: E[r] for each batch element
        mean_residual = jnp.einsum('s,sb->b', sample_weights, residuals)  # [batch]
        # Square then average over batch: E_batch[(E_shock[r])²]
        eq_loss = jnp.mean(mean_residual ** 2)
        eq_losses[eq_name] = eq_loss
        w = 1.0 if weights is None else weights[i]
        total_loss = total_loss + w * eq_loss

    return total_loss, eq_losses


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def eq_losses_to_array(eq_losses: Dict[str, Array]) -> Array:
    """Convert per-equation loss dict to stacked array [n_eq]."""
    return jnp.stack(list(eq_losses.values()))


def compute_loss_for_grad(
    params,
    model: ModelSpec,
    states: Array,
    key: Array,
    mc_samples: int = 5,
) -> Array:
    """Loss function signature suitable for jax.grad."""
    loss, _ = compute_loss(model, params, states, key, mc_samples)
    return loss


def make_loss_fn(
    model: ModelSpec,
    mc_samples: int = 5,
) -> Callable:
    """Create a loss function closed over model spec.

    Returns a function (params, states, key) -> (loss, eq_losses)
    suitable for use with jax.value_and_grad.
    """
    def loss_fn(params, states: Array, key: Array):
        return compute_loss(model, params, states, key, mc_samples)

    return loss_fn
