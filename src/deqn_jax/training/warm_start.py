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
import jax.flatten_util
from jax import Array
import equinox as eqx
import optax

from deqn_jax.types import ModelSpec


def _lbfgs_minimize(
    loss_fn: Callable,
    init_params,
    max_iter: int = 100,
    tol: float = 1e-6,
    memory_size: int = 10,
) -> Tuple:
    """Run L-BFGS optimization on a pytree of parameters.

    Uses optax.lbfgs with a flat-parameter loop.

    Args:
        loss_fn: Scalar loss function taking the pytree params
        init_params: Initial parameter pytree
        max_iter: Maximum iterations
        tol: Convergence tolerance (on loss value)
        memory_size: L-BFGS history size

    Returns:
        Tuple of (optimized_params, n_iters, final_loss)
    """
    flat, unravel = jax.flatten_util.ravel_pytree(init_params)

    def flat_loss(x):
        return loss_fn(unravel(x))

    opt = optax.lbfgs(memory_size=memory_size)
    opt_state = opt.init(flat)

    @jax.jit
    def step(x, opt_state):
        val, g = jax.value_and_grad(flat_loss)(x)
        updates, new_opt_state = opt.update(
            g, opt_state, x, value=val, grad=g, value_fn=flat_loss,
        )
        new_x = optax.apply_updates(x, updates)
        return new_x, new_opt_state, val

    n_iters = 0
    for i in range(max_iter):
        flat, opt_state, val = step(flat, opt_state)
        n_iters = i + 1
        if float(val) < tol:
            break

    return unravel(flat), n_iters, float(val)


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

    # Run L-BFGS optimization
    final_params, n_iters, final_loss = _lbfgs_minimize(
        loss_fn, policy_net, max_iter=max_iter, tol=tol,
    )

    if verbose:
        print(f"  Warm start complete: loss={final_loss:.2e}, iters={n_iters}")

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

    final_params, n_iters, final_loss = _lbfgs_minimize(
        loss_fn, policy_net, max_iter=max_iter, tol=tol,
    )

    if verbose:
        print(f"Warm start: loss={final_loss:.2e}, iters={n_iters}")

    return final_params
