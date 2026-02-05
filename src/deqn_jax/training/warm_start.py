"""Warm start: L-BFGS initialization from steady state.

Fits the policy network to match steady state policy before training.
This gives a much better starting point than random initialization.

L-BFGS is ideal here because:
1. Fitting to steady state is a deterministic supervised problem
2. Second-order methods excel at this (converges in ~10-50 steps vs 500+ Adam)
3. No mini-batching needed (full batch fits in memory)
"""

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
import equinox as eqx
import jaxopt

from deqn_jax.types import ModelSpec


def warm_start_network(
    policy_net: eqx.Module,
    model: ModelSpec,
    n_points: int = 256,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = True,
    key: Optional[Array] = None,
) -> eqx.Module:
    """Warm-start policy network to match steady state.

    Samples points around steady state and fits network to output
    the steady state policy at each point. Uses L-BFGS for fast convergence.

    Args:
        policy_net: Equinox policy network to initialize
        model: Model specification (must have steady_state_fn)
        n_points: Number of fitting points
        max_iter: Maximum L-BFGS iterations
        tol: Convergence tolerance
        verbose: Print progress
        key: PRNG key (uses default if None)

    Returns:
        Warm-started policy network
    """
    if model.steady_state_fn is None:
        if verbose:
            print("No steady_state_fn, skipping warm start")
        return policy_net

    if key is None:
        key = jax.random.PRNGKey(0)

    # Get steady state
    ss_state, ss_policy = model.steady_state_fn(model.constants)

    if verbose:
        print(f"Warm starting from steady state...")
        print(f"  SS state: {ss_state}")
        print(f"  SS policy: {ss_policy}")

    # Sample points around steady state
    noise = jax.random.uniform(key, (n_points, model.n_states), minval=-0.2, maxval=0.2)
    states = ss_state * (1 + noise)

    # Target: steady state policy at all points
    targets = jnp.tile(ss_policy, (n_points, 1))

    # Loss function: MSE between network output and steady state policy
    def loss_fn(params):
        pred = jax.vmap(params)(states)
        return jnp.mean((pred - targets) ** 2)

    # L-BFGS optimizer
    solver = jaxopt.LBFGS(
        fun=loss_fn,
        maxiter=max_iter,
        tol=tol,
    )

    # Run optimization
    result = solver.run(policy_net)
    final_params = result.params
    final_state = result.state

    if verbose:
        final_loss = loss_fn(final_params)
        print(f"  Warm start complete: loss={float(final_loss):.2e}, iters={final_state.iter_num}")

    return final_params


def warm_start_to_function(
    policy_net: eqx.Module,
    target_fn: Callable[[Array], Array],
    sample_states: Array,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = True,
) -> eqx.Module:
    """Warm-start policy network to match an arbitrary target function.

    More general version - fits network to match target_fn(state) for each state.

    Args:
        policy_net: Network to initialize
        target_fn: Target function (state -> policy)
        sample_states: States to fit on [n_points, n_states]
        max_iter: Maximum L-BFGS iterations
        tol: Convergence tolerance
        verbose: Print progress

    Returns:
        Warm-started network
    """
    # Compute targets
    targets = jax.vmap(target_fn)(sample_states)

    def loss_fn(params):
        pred = jax.vmap(params)(sample_states)
        return jnp.mean((pred - targets) ** 2)

    solver = jaxopt.LBFGS(fun=loss_fn, maxiter=max_iter, tol=tol)
    result = solver.run(policy_net)

    if verbose:
        final_loss = loss_fn(result.params)
        print(f"Warm start: loss={float(final_loss):.2e}, iters={result.state.iter_num}")

    return result.params
