"""Steady state and init sampler for 6-agent OLG.

The model has i.i.d. shocks, so there is no deterministic stationary
distribution -- but there IS a zero-shock steady state: apply the
analytic policy at shock realisation (eta_mid, delta_mid) and find the
fixed point in capital holdings. That's the reference point for warm
start and IRFs.

Derivation of the zero-shock SS:

    k^{h+1} = beta_h * inc^h,         inc^h = r * k^h + w * labor^h,
    K = sum_h k^h,
    r = alpha * eta_mid * K^(alpha-1) + (1 - delta_mid),
    w = (1 - alpha) * eta_mid * K^alpha.

Labor is 1 for agent 1, 0 otherwise, so agent 1's income is wage only;
agent 2 onward carry financial income r * k^h. Substituting the aging
rule k^{h+1} = beta_h inc^h iteratively gives a 1-D fixed-point equation
in K alone, solved here by damped iteration.

Init sampler: uniform rect around the zero-shock SS, plus binary eta /
delta drawn uniformly from the four shock states.
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.models.olg_analytic_6.variables import A


def analytic_beta_h(beta: float) -> list:
    """Krueger-Kubler saving fractions beta_h for h = 1..A-1."""
    return [
        beta * (1.0 - beta ** (A - 1 - h)) / (1.0 - beta ** (A - h))
        for h in range(A - 1)
    ]


def steady_state(constants: Dict) -> Tuple[Array, Array]:
    alpha = constants["alpha"]
    beta = constants["beta"]
    eta = constants["eta_mid"]
    delta = constants["delta_mid"]
    labor_1 = constants["labor_1"]

    b = analytic_beta_h(beta)

    def K_implied(K):
        L = labor_1
        r = alpha * eta * K ** (alpha - 1.0) * L ** (1.0 - alpha) + (1.0 - delta)
        w = (1.0 - alpha) * eta * K**alpha * L ** (-alpha)
        k2 = b[0] * w
        k3 = b[1] * r * k2
        k4 = b[2] * r * k3
        k5 = b[3] * r * k4
        k6 = b[4] * r * k5
        return k2 + k3 + k4 + k5 + k6, (k2, k3, k4, k5, k6, r, w)

    # Damped fixed-point iteration
    K = 1.0
    for _ in range(500):
        K_new, pieces = K_implied(K)
        if abs(K_new - K) < 1e-12:
            K = K_new
            break
        K = 0.5 * K + 0.5 * K_new

    k2, k3, k4, k5, k6, r, w = pieces

    ss_state = jnp.array([k2, k3, k4, k5, k6, eta, delta])

    # Optimal savings at SS: s^h = k^{h+1} (i.e. what agent h saves becomes agent h+1's capital)
    # s^1 = k^2_SS, s^2 = k^3_SS, ..., s^5 = k^6_SS.
    ss_policy = jnp.array([k2, k3, k4, k5, k6])

    return ss_state, ss_policy


def init_state(key: Array, batch_size: int, constants: Dict) -> Array:
    """Sample initial states.

    - k^h ~ Uniform[0.5 * SS, 1.5 * SS] around the zero-shock SS
    - eta, delta ~ uniformly from {eta_mid +/- eta_half} x {delta_mid +/- delta_half}
    """
    ss, _ = steady_state(constants)
    k_ss = ss[:5]  # k2..k6 SS values

    k_key, eta_key, delta_key = jax.random.split(key, 3)
    # Jitter each k-dimension by +/- 50%
    k_samples = k_ss[None, :] * (
        1.0 + 0.5 * jax.random.uniform(k_key, (batch_size, 5), minval=-1.0, maxval=1.0)
    )

    eta_half = constants["eta_half"]
    delta_half = constants["delta_half"]
    eta_mid = constants["eta_mid"]
    delta_mid = constants["delta_mid"]

    eta_signs = jax.random.choice(eta_key, jnp.array([-1.0, 1.0]), shape=(batch_size,))
    delta_signs = jax.random.choice(
        delta_key, jnp.array([-1.0, 1.0]), shape=(batch_size,)
    )
    eta_samples = eta_mid + eta_half * eta_signs
    delta_samples = delta_mid + delta_half * delta_signs

    return jnp.concatenate(
        [k_samples, eta_samples[:, None], delta_samples[:, None]], axis=1
    )


def analytic_policy(state: Array, constants: Dict) -> Array:
    """Krueger-Kubler closed form: k'^h = beta_h * inc^h for h=1..A-1.

    Useful as the oracle that the DEQN should recover. This function is
    a pure function of the state and constants -- no network involved.
    Returns [batch, A-1] savings.
    """
    from deqn_jax.models.olg_analytic_6.variables import SPEC

    s = SPEC.unpack_state(state)
    alpha = constants["alpha"]
    labor_1 = constants["labor_1"]
    b = analytic_beta_h(constants["beta"])

    batch_size = state.shape[0]
    k = jnp.stack([jnp.zeros((batch_size,)), s.k2, s.k3, s.k4, s.k5, s.k6], axis=1)
    K = jnp.sum(k, axis=1)
    L = jnp.full((batch_size,), labor_1)

    r = alpha * s.eta * jnp.power(K, alpha - 1.0) * jnp.power(L, 1.0 - alpha) + (
        1.0 - s.delta
    )
    w = (1.0 - alpha) * s.eta * jnp.power(K, alpha) * jnp.power(L, -alpha)

    fin = k * r[:, None]
    lab = jnp.zeros_like(k).at[:, 0].set(labor_1 * w)
    inc = fin + lab  # [batch, A]

    # Savings for agents 1..A-1: s^h = b[h-1] * inc^h
    b_arr = jnp.array(b)  # [A-1]
    return inc[:, : A - 1] * b_arr[None, :]
