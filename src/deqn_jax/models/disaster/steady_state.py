"""Steady state computation for Disaster (NK-DSGE) model."""

from typing import Dict, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from scipy.optimize import root

from deqn_jax.models.disaster.variables import SPEC, STEADY_STATE, CONSTANTS
from deqn_jax.models.disaster.equations import equations, Gamma


def _solve_steady_state(constants: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Numerically solve for the deterministic steady state.

    At SS: state = next_state, policy = next_policy, shocks = 0.
    Unknowns: 10 policy variables (s and L computed analytically).
    State constructed from them. Uses JAX autodiff Jacobian.
    """
    c = constants

    # Initial guess from hardcoded values
    x0 = np.array([STEADY_STATE[n] for n in SPEC.policy_names])

    def _build_state(x):
        """Construct SS state from 10 policy variables.

        s (marginal cost) and L (leverage) are computed analytically
        to satisfy eq10 and eq12 by construction.
        """
        lambda_z, i, pi, cc, w_tilda, omega_bar, h, F_w, F_p, q = x
        mu_z = c["mu_z_ss"]
        k = i / (1.0 - (1.0 - c["delta"]) / mu_z)

        # Marginal cost (analytical, satisfies eq10 exactly)
        # At SS: eps = 1.0
        s = (mu_z * h / k) ** c["alpha"] * w_tilda / (1 - c["alpha"])

        # Leverage (analytical, satisfies eq12 exactly)
        r_k = c["alpha"] * (mu_z * h / k) ** (1 - c["alpha"]) * s
        R_k = ((1 - c["tau_k"]) * r_k + (1 - c["delta"]) * q) / q * pi + c["tau_k"] * c["delta"]
        Gamma_val = Gamma(omega_bar, c["sigma_omega"])
        n = (c["gamma_e"] / (pi * mu_z)) * (1.0 - Gamma_val) * R_k * q * k + c["w_e"]
        L = q * k / (n + 1e-8)

        y_gdp = c["g_ss"] + cc + i
        R = c["R_ss"] * (pi / c["pi_ss"]) ** c["alpha_pi"] * (y_gdp / c["y_ss"]) ** c["alpha_y"]
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
