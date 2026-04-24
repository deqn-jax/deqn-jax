"""Steady state computation for Disaster (NK-DSGE) model."""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.optimize import root

from deqn_jax.models.disaster.equations import G_omega, Gamma, equations, solve_omega_bar
from deqn_jax.models.disaster.variables import CONSTANTS, OMEGA_BAR_SS, SPEC, STEADY_STATE


def _solve_steady_state(constants: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Numerically solve for the deterministic steady state.

    At SS: state = next_state, policy = next_policy, shocks = 0.
    Unknowns: 11 policy variables (s, L, omega_bar computed analytically).
    11 equations.
    """
    c = constants

    # Initial guess from hardcoded values
    x0 = np.array([STEADY_STATE[n] for n in SPEC.policy_names])

    def _build_state(x):
        """Construct SS state from 11 policy variables.

        s (marginal cost), L (leverage), omega_bar (default threshold)
        are computed analytically. K_p, K_w are direct policy outputs.
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


# Cache of solved steady states, keyed by frozenset(constants.items()).
# Prevents stale results when the caller passes modified constants (e.g.
# disaster calibration with different p_disaster / theta_disaster).
#
# NOTE: this solver computes the DETERMINISTIC steady state (no disaster
# realizations). For the RISKY steady state under disaster risk, the Euler
# equations need mixture expectations over (1-p) * no-disaster + p * disaster.
# That solver is pending — see session_log.md Part 9.
_ss_cache: dict = {}


def _constants_key(constants: Dict) -> tuple:
    """Hashable key for a constants dict. Sorted for determinism."""
    return tuple(sorted(constants.items()))


def steady_state(constants: Dict) -> Tuple[Array, Array]:
    """Return (deterministic) steady state for the given calibration.

    Caches solutions keyed on constants so repeat calls with the same
    calibration are fast, but a different calibration triggers a re-solve.
    """
    key = _constants_key(constants)
    if key not in _ss_cache:
        _ss_cache[key] = _solve_steady_state(constants)
    ss_state_np, ss_policy_np = _ss_cache[key]
    return jnp.array(ss_state_np), jnp.array(ss_policy_np)


# Pre-populate cache with the module-default calibration so first call is fast.
steady_state(CONSTANTS)


def init_state(key: Array, batch_size: int, constants: Dict) -> Array:
    ss_state, _ = steady_state(constants)
    noise = jax.random.uniform(key, (batch_size, 13), minval=-0.05, maxval=0.05)
    return ss_state * (1 + noise)


# ---------------------------------------------------------------------------
# Risky steady state
# ---------------------------------------------------------------------------

def _solve_risky_steady_state(constants: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Numerically solve for the RISKY steady state under disaster risk.

    At the risky SS, agents' expected-period equations hold under the
    disaster mixture:

        (1-p) · eq(s*, π*, step(s*, π*, d=0), π*)
          + p · eq(s*, π*, step(s*, π*, d=1), π*) = 0

    Both next-period policies are taken at π* (locally-flat policy
    approximation). Capital in the disaster branch is scaled by
    exp(-theta_disaster).

    Falls back to the deterministic SS when p_disaster = 0.
    """
    from deqn_jax.models.disaster.dynamics import step as step_fn
    c = constants
    p_disaster = float(c.get("p_disaster", 0.0))
    theta = float(c.get("theta_disaster", 0.0))

    if p_disaster <= 0.0:
        return _solve_steady_state(constants)

    # Initial guess from deterministic SS
    det_state, det_policy = _solve_steady_state(constants)
    x0 = np.array(det_policy)

    def _build_state(x):
        """Same helper as deterministic SS — constructs state vector from 11 policies."""
        lambda_z, i, pi, cc, w_tilda, h, F_w, F_p, q, K_p, K_w = x
        mu_z = c["mu_z_ss"]
        k = i / (1.0 - (1.0 - c["delta"]) / mu_z)
        s = (mu_z * h / k) ** c["alpha"] * w_tilda / (1 - c["alpha"])
        r_k = c["alpha"] * (mu_z * h / k) ** (1 - c["alpha"]) * s
        R_k = ((1 - c["tau_k"]) * r_k + (1 - c["delta"]) * q) / q * pi + c["tau_k"] * c["delta"]
        R = pi * mu_z / c["beta"]

        omega_bar_init = jnp.array(OMEGA_BAR_SS)
        Gamma_val_init = Gamma(omega_bar_init, c["sigma_omega"])
        n_init = (c["gamma_e"] / (pi * mu_z)) * (1.0 - Gamma_val_init) * R_k * q * k + c["w_e"]
        L_init = q * k / (n_init + 1e-8)
        target = (L_init - 1.0) / (L_init * R_k / R + 1e-10)
        omega_bar = solve_omega_bar(target, c["sigma_omega"], c["mu_mon"])

        Gamma_val = Gamma(omega_bar, c["sigma_omega"])
        n = (c["gamma_e"] / (pi * mu_z)) * (1.0 - Gamma_val) * R_k * q * k + c["w_e"]
        L = q * k / (n + 1e-8)

        return jnp.array([pi, k, cc, q, i, R, w_tilda, L,
                          1.0, 1.0, c["g_ss"], mu_z, 0.0])

    def _resid_jax(x):
        """Mixture-residual across disaster realizations at the candidate SS."""
        state = _build_state(x)
        state_b = state[None, :]
        policy_b = jnp.array(x)[None, :]
        zero_shock = jnp.zeros((1, 5))

        # No-disaster branch: use regular step_fn call (d_disaster defaults to 0)
        next_n = step_fn(state_b, policy_b, zero_shock, c)
        # Disaster branch: d_disaster=1
        next_d = step_fn(state_b, policy_b, zero_shock, c, d_disaster=jnp.array(1.0))

        # Policies at next states: locally-flat approximation → same π*
        r_n = equations(state_b, policy_b, next_n, policy_b, c)
        r_d = equations(state_b, policy_b, next_d, policy_b, c)

        r_n_stacked = jnp.stack([v[0] for v in r_n.values()])
        r_d_stacked = jnp.stack([v[0] for v in r_d.values()])
        return (1.0 - p_disaster) * r_n_stacked + p_disaster * r_d_stacked

    _jac_fn = jax.jacobian(_resid_jax)

    def residuals(x_np):
        return np.array(_resid_jax(jnp.array(x_np)))

    def jacobian(x_np):
        return np.array(_jac_fn(jnp.array(x_np)))

    sol = root(residuals, x0, jac=jacobian, method='hybr', tol=1e-12)
    max_resid = np.max(np.abs(sol.fun))
    if not sol.success and max_resid > 1e-5:
        print(f"WARNING: Risky SS solver did not fully converge: {sol.message}")
        print(f"  Max |residual|: {max_resid:.2e}")

    x = sol.x
    ss_state = np.array(_build_state(jnp.array(x)))
    return ss_state, x


_risky_ss_cache: dict = {}


def risky_steady_state(constants: Dict) -> Tuple[Array, Array]:
    """Return risky steady state for the given calibration.

    Agents price in disaster probability p > 0 and the disaster magnitude.
    Reduces to deterministic SS when p_disaster = 0.

    Uses locally-flat policy approximation: next-period policies in both
    disaster realizations are taken to equal the risky-SS policy. This is
    first-order correct and sufficient for warm-starting / anchoring.
    """
    key = _constants_key(constants)
    if key not in _risky_ss_cache:
        _risky_ss_cache[key] = _solve_risky_steady_state(constants)
    ss_state_np, ss_policy_np = _risky_ss_cache[key]
    return jnp.array(ss_state_np), jnp.array(ss_policy_np)
