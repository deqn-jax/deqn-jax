"""Steady state computation for Disaster (NK-DSGE) model."""

from typing import Dict, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from scipy.optimize import root

from deqn_jax.models.disaster.variables import SPEC, STEADY_STATE, CONSTANTS, OMEGA_BAR_SS
from deqn_jax.models.disaster.equations import equations, Gamma, G_omega, solve_omega_bar


def _solve_steady_state(constants: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Numerically solve for the deterministic steady state.

    At SS: state = next_state, policy = next_policy, shocks = 0.
    Unknowns: 11 policy variables (s, L, omega_bar computed analytically).
    State constructed from them. Uses JAX autodiff Jacobian.
    """
    c = constants

    # Initial guess from hardcoded values
    x0 = np.array([STEADY_STATE[n] for n in SPEC.policy_names])

    def _build_state(x):
        """Construct SS state from 11 policy variables.

        s (marginal cost), L (leverage), and omega_bar (default threshold)
        are computed analytically.
        """
        lambda_z, i, pi, cc, w_tilda, h, F_w, F_p, q, K_p, K_w = x
        mu_z = c["mu_z_ss"]
        k = i / (1.0 - (1.0 - c["delta"]) / mu_z)

        # Marginal cost (analytical, satisfies eq10 exactly)
        # At SS: eps = 1.0
        s = (mu_z * h / k) ** c["alpha"] * w_tilda / (1 - c["alpha"])

        # R_k does NOT depend on omega_bar
        r_k = c["alpha"] * (mu_z * h / k) ** (1 - c["alpha"]) * s
        R_k = ((1 - c["tau_k"]) * r_k + (1 - c["delta"]) * q) / q * pi + c["tau_k"] * c["delta"]

        # At SS: R_lag = R, L_lag = L, but L is unknown.
        # Use the Taylor rule to get R first:
        # We need y_gdp for R, but y_gdp needs c which needs omega_bar.
        # However, at SS we can compute R from the Euler equation:
        # From bond Euler: R = pi * mu_z / beta (at SS, lambda_z = lambda_z_next)
        R = pi * mu_z / c["beta"]

        # omega_bar (analytical, satisfies eq8 bank participation exactly)
        # At SS: L_lag = L, R_lag = R. From eq8: target = (L-1)/(L*R_k/R)
        # But L depends on omega_bar... use definitions() approach:
        # L = q*k/n, n depends on omega_bar. So solve iteratively:
        # Start from OMEGA_BAR_SS and do Newton iterations.
        # Actually, at SS we can use the same solve_omega_bar function.
        # We need L_lag (= L at SS). Bootstrap from a guess:
        omega_bar_init = jnp.array(OMEGA_BAR_SS)
        Gamma_val_init = Gamma(omega_bar_init, c["sigma_omega"])
        n_init = (c["gamma_e"] / (pi * mu_z)) * (1.0 - Gamma_val_init) * R_k * q * k + c["w_e"]
        L_init = q * k / (n_init + 1e-8)
        target = (L_init - 1.0) / (L_init * R_k / R + 1e-10)
        omega_bar = solve_omega_bar(target, c["sigma_omega"], c["mu_mon"])

        # Now recompute with solved omega_bar
        Gamma_val = Gamma(omega_bar, c["sigma_omega"])
        G_val = G_omega(omega_bar, c["sigma_omega"])
        n = (c["gamma_e"] / (pi * mu_z)) * (1.0 - Gamma_val) * R_k * q * k + c["w_e"]
        L = q * k / (n + 1e-8)

        return jnp.array([pi, k, cc, q, i, R, w_tilda, L,
                          1.0, 1.0, c["g_ss"], mu_z, 0.0])

    def _resid_jax(x):
        state = _build_state(x)
        r = equations(state, x, state, x, constants)
        return jnp.stack(list(r.values()))

    _jac_fn = jax.jacobian(_resid_jax)

    def residuals(x_np):
        x = jnp.array(x_np)
        return np.array(_resid_jax(x))

    def jacobian(x_np):
        x = jnp.array(x_np)
        return np.array(_jac_fn(x))

    sol = root(residuals, x0, jac=jacobian, method='hybr', tol=1e-14)
    max_resid = np.max(np.abs(sol.fun))
    if not sol.success and max_resid > 1e-6:
        print(f"WARNING: SS solver did not converge: {sol.message}")
        print(f"  Max |residual|: {max_resid:.2e}")

    x = sol.x
    ss_state = np.array(_build_state(jnp.array(x)))
    return ss_state, x


# Cache the numerical solution at module load
_SS_STATE, _SS_POLICY = _solve_steady_state(CONSTANTS)


def steady_state(constants: Dict) -> Tuple[Array, Array]:
    return jnp.array(_SS_STATE), jnp.array(_SS_POLICY)


def init_state(key: Array, batch_size: int, constants: Dict) -> Array:
    ss_state, _ = steady_state(constants)
    noise = jax.random.uniform(key, (batch_size, 13), minval=-0.05, maxval=0.05)
    return ss_state * (1 + noise)
