"""Linearized policy rule via Blanchard-Kahn method.

Computes the first-order approximation to the policy function by:
1. Taking Jacobians of equilibrium equations and state transition at SS
2. Forming the generalized eigenvalue problem A E[z'] = B z
3. QZ decomposition to separate stable/unstable eigenvalues
4. Extracting the linear policy rule p̂ = P ŝ

This gives a STATE-DEPENDENT warm start (unlike constant-SS warm start).
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.linalg import ordqz, solve_discrete_lyapunov

from deqn_jax.types import ModelSpec


def linearize_model(
    model: ModelSpec,
    verbose: bool = True,
) -> Tuple[Array, Array]:
    """Compute linearized policy rule P and transition matrix Q.

    Solves the linear rational expectations system via QZ decomposition.

    The equilibrium: F(s, p, s', p') = 0
    State transition: s' = G(s, p) + H ε

    Linearized: p̂ = P ŝ  (policy rule)
                ŝ' = Q ŝ  (deterministic transition)

    Args:
        model: Model specification with equations_fn, step_fn, steady_state_fn
        verbose: Print diagnostic info

    Returns:
        Tuple of:
            P: Policy rule matrix [n_policies, n_states]
            Q: Transition matrix [n_states, n_states]
    """
    constants = model.constants
    ss_state, ss_policy = model.steady_state_fn(constants)

    n_s = model.n_states
    n_p = model.n_policies

    if verbose:
        print(f"Linearizing: {n_s} states, {n_p} policies, {n_s + n_p} total")

    # --- Compute Jacobians at steady state ---

    # Equations wrapper: unbatched, returns stacked residual vector
    def F_vec(s, p, s_next, p_next):
        res = model.equations_fn(
            s[None, :], p[None, :], s_next[None, :], p_next[None, :], constants
        )
        return jnp.stack([v[0] for v in res.values()])

    # State transition wrapper: unbatched, zero shock
    def G_vec(s, p):
        shock = jnp.zeros((1, model.n_shocks))
        ns = model.step_fn(s[None, :], p[None, :], shock, constants)
        return ns[0]

    # F Jacobians: ∂F/∂(s, p, s', p') at SS
    F_s = jax.jacobian(F_vec, argnums=0)(ss_state, ss_policy, ss_state, ss_policy)
    F_p = jax.jacobian(F_vec, argnums=1)(ss_state, ss_policy, ss_state, ss_policy)
    F_sn = jax.jacobian(F_vec, argnums=2)(ss_state, ss_policy, ss_state, ss_policy)
    F_pn = jax.jacobian(F_vec, argnums=3)(ss_state, ss_policy, ss_state, ss_policy)

    # G Jacobians: ∂step/∂(s, p) at SS with shock=0
    G_s = jax.jacobian(G_vec, argnums=0)(ss_state, ss_policy)
    G_p = jax.jacobian(G_vec, argnums=1)(ss_state, ss_policy)

    if verbose:
        print(f"  F_s: {F_s.shape}, F_p: {F_p.shape}")
        print(f"  F_sn: {F_sn.shape}, F_pn: {F_pn.shape}")
        print(f"  G_s: {G_s.shape}, G_p: {G_p.shape}")

    # --- Form generalized eigenvalue problem ---
    # System: A E[z'] = B z  where z = (s, p)
    #
    # A = [[I,    0  ],    B = [[G_s,  G_p ],
    #      [F_sn, F_pn]]        [-F_s, -F_p]]

    A = np.block([
        [np.eye(n_s),       np.zeros((n_s, n_p))],
        [np.array(F_sn),    np.array(F_pn)],
    ])
    B = np.block([
        [np.array(G_s),     np.array(G_p)],
        [-np.array(F_s),    -np.array(F_p)],
    ])

    # --- QZ decomposition ---
    # ordqz(B, A) gives eigenvalues of A^{-1}B, sorted stable first
    AA, BB, alpha, beta, Q, Z = ordqz(B, A, output='complex', sort='iuc')

    # Generalized eigenvalues
    with np.errstate(divide='ignore', invalid='ignore'):
        eigenvalues = np.where(np.abs(beta) > 1e-15, np.abs(alpha / beta), np.inf)

    n_stable = np.sum(eigenvalues < 1.0)

    if verbose:
        print(f"  Eigenvalues (modulus): {np.sort(eigenvalues)}")
        print(f"  Stable: {n_stable}, Unstable: {len(eigenvalues) - n_stable}")
        print(f"  Expected: {n_s} stable, {n_p} unstable")

    if n_stable != n_s:
        msg = (
            f"Blanchard-Kahn violation: {n_stable} stable eigenvalues, "
            f"expected {n_s}. The model may not have a unique rational "
            f"expectations equilibrium."
        )
        if verbose:
            print(f"  WARNING: {msg}")
        # Continue anyway — we'll get the best approximation we can

    # --- Extract policy rule ---
    # Z partitioned: Z = [[Z_11, Z_12], [Z_21, Z_22]]
    # Policy rule: p̂ = Z_21 @ Z_11^{-1} @ ŝ
    Z_11 = Z[:n_s, :n_s]
    Z_21 = Z[n_s:, :n_s]

    P = np.real(Z_21 @ np.linalg.inv(Z_11))

    # Transition: ŝ' = (G_s + G_p @ P) ŝ
    Q_mat = np.array(G_s) + np.array(G_p) @ P

    if verbose:
        # Check: eigenvalues of Q should all be inside unit circle
        Q_eigs = np.abs(np.linalg.eigvals(Q_mat))
        print(f"  Transition eigenvalues: {np.sort(Q_eigs)}")
        print(f"  Max |eig(Q)|: {Q_eigs.max():.6f} (should be < 1)")

        # Check: P @ 0 = 0 (policy deviation is zero at SS)
        print(f"  Policy rule shape: P [{P.shape[0]}x{P.shape[1]}]")
        print(f"  Max |P|: {np.abs(P).max():.4f}")

    return jnp.array(P), jnp.array(Q_mat)


def compute_ergodic_covariance(
    Q: Array,
    model: ModelSpec,
    verbose: bool = False,
) -> Array:
    """Compute ergodic state covariance via discrete Lyapunov equation.

    The linearized dynamics: s' = Q s + B epsilon, epsilon ~ N(0, I).
    The ergodic covariance satisfies: Sigma = Q Sigma Q' + B B'.

    B is the shock loading matrix: d(step)/d(shock) at steady state with zero shock.

    Args:
        Q: Transition matrix [n_states, n_states] from linearize_model
        model: Model specification with step_fn, steady_state_fn
        verbose: Print diagnostic info

    Returns:
        Sigma: Ergodic covariance matrix [n_states, n_states]
    """
    constants = model.constants
    ss_state, ss_policy = model.steady_state_fn(constants)

    # Compute B = d(step)/d(shock) at SS with zero shock
    def step_wrt_shock(shock):
        return model.step_fn(
            ss_state[None, :], ss_policy[None, :], shock[None, :], constants
        )[0]

    zero_shock = jnp.zeros(model.n_shocks)
    B = jax.jacobian(step_wrt_shock)(zero_shock)  # [n_states, n_shocks]

    B_np = np.array(B)
    Q_np = np.array(Q)

    if verbose:
        print(f"  B (shock loading): {B_np.shape}, max |B|: {np.abs(B_np).max():.6f}")

    # Solve Sigma = Q Sigma Q' + B B'
    Sigma = solve_discrete_lyapunov(Q_np, B_np @ B_np.T)

    if verbose:
        print(f"  Ergodic covariance: diag = {np.diag(Sigma)[:5]}...")
        print(f"  Max |Sigma|: {np.abs(Sigma).max():.6f}")

    return jnp.array(Sigma)


def linear_policy_fn(
    P: Array,
    ss_state: Array,
    ss_policy: Array,
) -> callable:
    """Create a linear policy function from the policy rule matrix.

    Args:
        P: Policy rule [n_policies, n_states]
        ss_state: Steady state [n_states]
        ss_policy: Steady state policy [n_policies]

    Returns:
        Function: state [n_states] -> policy [n_policies]
    """
    def policy_fn(state: Array) -> Array:
        deviation = state - ss_state
        return ss_policy + P @ deviation

    return policy_fn
