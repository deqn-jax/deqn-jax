"""Steady state computation utilities.

For models without analytical steady states, we can solve numerically
using the equilibrium equations with zero shocks.
"""

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
import jaxopt

from deqn_jax.types import ModelSpec


def solve_steady_state(
    model: ModelSpec,
    init_state: Optional[Array] = None,
    init_policy: Optional[Array] = None,
    max_iter: int = 1000,
    tol: float = 1e-8,
    verbose: bool = True,
    force_numerical: bool = False,
) -> Tuple[Array, Array]:
    """Solve for deterministic steady state.

    If model has analytical steady_state_fn, uses that (recommended).
    Otherwise solves numerically using L-BFGS.

    At steady state:
    - State doesn't change: s' = s
    - Policy doesn't change: π' = π
    - Shocks are zero: ε = 0
    - Equilibrium equations hold: r(s, π, s, π) = 0

    Args:
        model: Model specification
        init_state: Initial guess for state (uses ones if None)
        init_policy: Initial guess for policy (uses 0.5 if None)
        max_iter: Maximum iterations
        tol: Convergence tolerance
        verbose: Print progress
        force_numerical: Use numerical solver even if analytical available

    Returns:
        Tuple of (ss_state, ss_policy)
    """
    # Use analytical if available
    if model.steady_state_fn is not None and not force_numerical:
        ss_state, ss_policy = model.steady_state_fn(model.constants)
        if verbose:
            print(f"Using analytical steady state:")
            print(f"  State: {ss_state}")
            print(f"  Policy: {ss_policy}")
        return ss_state, ss_policy
    n_states = model.n_states
    n_policies = model.n_policies

    # Initial guesses
    if init_state is None:
        init_state = jnp.ones(n_states)
    if init_policy is None:
        init_policy = jnp.ones(n_policies) * 0.5

    # Pack state and policy into single vector for optimization
    init_x = jnp.concatenate([init_state, init_policy])

    def residual_fn(x):
        """Compute sum of squared residuals at candidate steady state."""
        state = x[:n_states][None, :]  # Add batch dim
        policy = x[n_states:][None, :]

        # At steady state: next_state = state, next_policy = policy
        residuals = model.equations_fn(state, policy, state, policy, model.constants)

        # Sum of squared residuals
        total = 0.0
        for name, r in residuals.items():
            total = total + jnp.sum(r ** 2)
        return total

    # Solve
    solver = jaxopt.LBFGS(fun=residual_fn, maxiter=max_iter, tol=tol)
    result = solver.run(init_x)

    ss_x = result.params
    ss_state = ss_x[:n_states]
    ss_policy = ss_x[n_states:]

    if verbose:
        final_residual = residual_fn(ss_x)
        print(f"Steady state solved: residual={float(final_residual):.2e}, iters={result.state.iter_num}")
        print(f"  State: {ss_state}")
        print(f"  Policy: {ss_policy}")

    return ss_state, ss_policy


def verify_steady_state(
    model: ModelSpec,
    ss_state: Array,
    ss_policy: Array,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """Verify that a candidate steady state satisfies equilibrium conditions.

    Args:
        model: Model specification
        ss_state: Candidate steady state
        ss_policy: Candidate steady state policy
        tol: Tolerance for residuals

    Returns:
        Dict of equation residuals
    """
    state = ss_state[None, :]
    policy = ss_policy[None, :]

    residuals = model.equations_fn(state, policy, state, policy, model.constants)

    result = {}
    all_ok = True
    for name, r in residuals.items():
        val = float(r[0])
        result[name] = val
        if abs(val) > tol:
            all_ok = False

    if not all_ok:
        print("WARNING: Some residuals exceed tolerance!")
        for name, val in result.items():
            status = "OK" if abs(val) < tol else "FAIL"
            print(f"  {name}: {val:.2e} [{status}]")

    return result
