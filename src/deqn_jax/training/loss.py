"""Loss computation with Monte Carlo expectations.

The DEQN loss is the mean squared residual of equilibrium equations:

    L = E[||r(s, π(s), s', π(s'))||²]

where the expectation is over:
1. States s drawn from episode trajectories
2. Shocks ε determining next state s' = step(s, π(s), ε)

We approximate E[...] using Monte Carlo with antithetic variates.
"""

from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.types import ModelSpec


def sample_antithetic_shocks(
    key: Array,
    n_samples: int,
    batch_size: int,
    shock_dim: int,
) -> Array:
    """Generate Monte Carlo shocks with antithetic variates.

    Antithetic sampling pairs each shock ε with -ε, reducing variance
    for symmetric distributions (like standard normal).

    Args:
        key: JAX PRNG key
        n_samples: Number of MC samples (will be rounded to even)
        batch_size: Batch size
        shock_dim: Dimension of shock vector

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

    return shocks


def compute_residuals(
    model: ModelSpec,
    policy_fn: Callable[[Array], Array],
    states: Array,
    shock: Array,
) -> Dict[str, Array]:
    """Compute equilibrium equation residuals for a single shock realization.

    Args:
        model: Model specification
        policy_fn: Policy network (states -> policies)
        states: Current states [batch, n_states]
        shock: Shock realization [batch, n_shocks]

    Returns:
        Dict mapping equation names to residuals [batch]
    """
    # Current policy
    policy = policy_fn(states)

    # Next state given shock
    next_state = model.step_fn(states, policy, shock, model.constants)

    # Next policy
    next_policy = policy_fn(next_state)

    # Equilibrium residuals
    residuals = model.equations_fn(
        states, policy, next_state, next_policy, model.constants
    )

    return residuals


def compute_loss(
    model: ModelSpec,
    policy_fn: Callable[[Array], Array],
    states: Array,
    key: Array,
    mc_samples: int = 5,
    weights: Optional[Array] = None,
) -> Tuple[Array, Dict[str, Array]]:
    """Compute DEQN loss with Monte Carlo expectations.

    The loss is:
        L = mean over samples [ mean over batch [ sum over equations [ w_i * r_i² ] ] ]

    Args:
        model: Model specification
        policy_fn: Policy network (states -> policies)
        states: State batch [batch, n_states]
        key: PRNG key for shock sampling
        mc_samples: Number of Monte Carlo samples
        weights: Per-equation weights [n_eq] (default: uniform)

    Returns:
        Tuple of (scalar loss, dict of per-equation losses)
    """
    batch_size = states.shape[0]

    # Sample shocks with antithetic variates
    shocks = sample_antithetic_shocks(key, mc_samples, batch_size, model.n_shocks)

    # Compute residuals for each shock sample
    def compute_sample_residuals(shock):
        return compute_residuals(model, policy_fn, states, shock)

    # vmap over MC samples: [n_samples] -> Dict[str, [n_samples, batch]]
    all_residuals = jax.vmap(compute_sample_residuals)(shocks)

    # Average over MC samples, then compute MSE over batch
    eq_losses = {}
    total_loss = 0.0

    for i, (eq_name, residuals) in enumerate(all_residuals.items()):
        # residuals: [n_samples, batch]
        # Average over samples first (MC expectation)
        mean_residual = jnp.mean(residuals, axis=0)  # [batch]
        # MSE over batch
        eq_loss = jnp.mean(mean_residual ** 2)
        eq_losses[eq_name] = eq_loss
        w = 1.0 if weights is None else weights[i]
        total_loss = total_loss + w * eq_loss

    return total_loss, eq_losses


def eq_losses_to_array(eq_losses: Dict[str, Array]) -> Array:
    """Convert per-equation loss dict to stacked array.

    Args:
        eq_losses: Dict mapping equation names to scalar losses

    Returns:
        Array of shape [n_eq] with losses in dict iteration order
    """
    return jnp.stack(list(eq_losses.values()))


def compute_loss_for_grad(
    params,
    model: ModelSpec,
    states: Array,
    key: Array,
    mc_samples: int = 5,
) -> Array:
    """Loss function signature suitable for jax.grad.

    Args:
        params: Equinox model (policy network)
        model: Model specification
        states: State batch [batch, n_states]
        key: PRNG key
        mc_samples: Number of MC samples

    Returns:
        Scalar loss value
    """
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
