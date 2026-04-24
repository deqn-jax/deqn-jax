"""Equilibrium equations for 2-country IRBC.

Five residuals for the N=2 model:

    euler_j     (2x)  intertemporal, one per country
    arc         (1x)  intratemporal aggregate resource constraint
    fb_j        (2x)  Fischer-Burmeister complementarity on (mu_j, i_j)

Economic structure:

- Each country produces y_j = A_tfp * exp(z_j) * k_j^zeta.
- Investment: i_j = k_j_next - (1-delta) k_j + adj_cost_j, where
  adj_cost_j = (kappa/2) * (k_j_next/k_j - 1)^2 * k_j. Zero at SS.
- Consumption is NOT a policy output. It's pinned by the Pareto-
  weighted FOC c_j = (lambda / tau_j)^(-1/gamma_j): marginal utility
  equalization under complete markets.
- lambda is the shared aggregate-resource shadow price; mu_j is the
  KKT multiplier on the irreversibility constraint i_j >= 0.

Residual forms (all raw; no LHS/RHS ratios, so MC-safe):

    euler_j = mu_j + beta * E[lambda' * mpk_j' - (1 - delta) * mu_j']
              - lambda * (1 + d_adj_cost_dk_next_j)
    arc     = sum_j (y_j + (1-delta) k_j - k_j_next - adj_cost_j - c_j)
    fb_j    = mu_j + i_j - sqrt(mu_j^2 + i_j^2 + fb_eps)

At equilibrium:
- euler_j = 0 (capital FOC for country j; expectation taken by the
  framework via MC or quadrature over shocks).
- arc = 0 (market clearing).
- fb_j = 0 (complementary slackness on irreversibility). See
  Fischer & Burmeister (1997): fb(a,b)=0 iff a>=0, b>=0, ab=0.
"""

from typing import Dict

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.irbc.dynamics import step as irbc_step
from deqn_jax.models.irbc.variables import N_COUNTRIES, N_SHOCKS, SPEC

EQUATION_NAMES = tuple(
    ["euler_0", "euler_1", "arc", "fb_0", "fb_1"]
)


def _adj_cost(k: Array, k_next: Array, kappa: float) -> Array:
    """Quadratic adjustment cost: kappa/2 * (k'/k - 1)^2 * k."""
    ratio = k_next / k - 1.0
    return 0.5 * kappa * ratio * ratio * k


def _d_adj_cost_dk_next(k: Array, k_next: Array, kappa: float) -> Array:
    """d adj_cost / d k_next = kappa * (k'/k - 1)."""
    return kappa * (k_next / k - 1.0)


def _d_adj_cost_dk(k: Array, k_next: Array, kappa: float) -> Array:
    """d adj_cost / d k = -(kappa/2) * ((k'/k)^2 - 1)."""
    # adj = (kappa/2)(k'/k-1)^2 * k. Compute the partial w.r.t. k directly.
    ratio = k_next / k
    return -0.5 * kappa * (ratio * ratio - 1.0)


def definitions(state: Array, policy: Array, constants: Dict) -> Dict[str, Array]:
    """Derived per-country and aggregate quantities.

    Returns a dict with entries for each country (suffixed by index) plus
    aggregates. Shape-agnostic: works for unbatched [n_states] and
    batched [batch, n_states] inputs.
    """
    s = SPEC.unpack_state(state)
    p = SPEC.unpack_policy(policy)

    beta = constants["beta"]                    # noqa: F841 (kept for readability)
    delta = constants["delta"]
    zeta = constants["zeta"]
    kappa = constants["kappa"]
    A_tfp = constants["A_tfp"]
    gamma = [constants["gamma_0"], constants["gamma_1"]]
    tau = [constants["tau_0"], constants["tau_1"]]

    lam = p.lam
    mus = [p.mu_0, p.mu_1]
    ks = [s.k_0, s.k_1]
    zs = [s.z_0, s.z_1]
    ks_next = [p.k_0_next, p.k_1_next]

    # Consumption pinned by marginal-utility equalization under Pareto
    # weights: u'(c_j) = lambda / tau_j, with CRRA(gamma_j).
    c = [jnp.power(lam / tau[j], -1.0 / gamma[j]) for j in range(N_COUNTRIES)]

    # Output, marginal product of capital, adjustment cost, investment.
    Z = [jnp.exp(zs[j]) for j in range(N_COUNTRIES)]
    y = [A_tfp * Z[j] * jnp.power(ks[j], zeta) for j in range(N_COUNTRIES)]
    mpk = [zeta * A_tfp * Z[j] * jnp.power(ks[j], zeta - 1.0)
           for j in range(N_COUNTRIES)]
    adj_cost = [_adj_cost(ks[j], ks_next[j], kappa) for j in range(N_COUNTRIES)]
    i = [ks_next[j] - (1.0 - delta) * ks[j] + adj_cost[j]
         for j in range(N_COUNTRIES)]

    defs = {"lam": lam}
    for j in range(N_COUNTRIES):
        defs[f"Z_{j}"] = Z[j]
        defs[f"y_{j}"] = y[j]
        defs[f"mpk_{j}"] = mpk[j]
        defs[f"c_{j}"] = c[j]
        defs[f"i_{j}"] = i[j]
        defs[f"adj_cost_{j}"] = adj_cost[j]
        defs[f"mu_{j}"] = mus[j]
    return defs


def equations(
    state: Array,
    policy: Array,
    next_state: Array,
    next_policy: Array,
    constants: Dict,
) -> Dict[str, Array]:
    """Five equilibrium residuals.

    Uses definitions() at both t and t+1 to get per-country derived
    quantities. The framework's loss module averages per-shock residuals
    into the expectation, so the raw form here is MC-safe.
    """
    beta = constants["beta"]
    delta = constants["delta"]
    kappa = constants["kappa"]
    fb_eps = constants["fb_eps"]

    s = SPEC.unpack_state(state)
    p = SPEC.unpack_policy(policy)
    p_next = SPEC.unpack_policy(next_policy)

    defs = definitions(state, policy, constants)
    defs_next = definitions(next_state, next_policy, constants)

    ks = [s.k_0, s.k_1]
    ks_next = [p.k_0_next, p.k_1_next]
    mus = [p.mu_0, p.mu_1]
    lam = p.lam
    lam_next = p_next.lam
    mus_next = [p_next.mu_0, p_next.mu_1]
    mpk_next = [defs_next[f"mpk_{j}"] for j in range(N_COUNTRIES)]

    # Reconstruct k_{t+2} via a zero-shock step so the Euler's future-period
    # adjustment-cost derivative can be evaluated at (k_{t+1}, k_{t+2}).
    # We use step_fn(next_state, next_policy, zero_shock) for this, mirroring
    # the pattern in training/autodiff.euler_from_period_return.
    zero_shock = jnp.zeros((state.shape[0], N_SHOCKS))
    next_next_state = irbc_step(next_state, next_policy, zero_shock, constants)
    ks_nextnext = [next_next_state[:, 0], next_next_state[:, 1]]

    out: Dict[str, Array] = {}

    # Capital Euler, per country. Gross return to capital (Simon's "MPK") is
    #     M_j' = (1 - delta) + mpk_j' - dGamma/dk_{t+1} at (k_{t+1}, k_{t+2})
    # so that the FOC reads
    #     mu_j + beta * E[lambda' * M_j' - (1 - delta) * mu_j']
    #       = lambda * (1 + dGamma/dk_{t+1} at (k_t, k_{t+1}))
    # Residual = LHS - RHS.
    for j in range(N_COUNTRIES):
        d_adj_today = _d_adj_cost_dk_next(ks[j], ks_next[j], kappa)
        d_adj_next = _d_adj_cost_dk(ks_next[j], ks_nextnext[j], kappa)
        M_next_j = (1.0 - delta) + mpk_next[j] - d_adj_next
        rhs_continuation = lam_next * M_next_j - (1.0 - delta) * mus_next[j]
        out[f"euler_{j}"] = (
            mus[j] + beta * rhs_continuation - lam * (1.0 + d_adj_today)
        )

    # Aggregate resource constraint: sum_j (y_j + (1-delta) k_j - k_j' - adj_j - c_j) = 0.
    arc = jnp.zeros_like(ks[0])
    for j in range(N_COUNTRIES):
        arc = arc + (
            defs[f"y_{j}"] + (1.0 - delta) * ks[j]
            - ks_next[j] - defs[f"adj_cost_{j}"] - defs[f"c_{j}"]
        )
    out["arc"] = arc

    # Fischer-Burmeister complementarity: fb(a,b) = a + b - sqrt(a^2 + b^2 + eps).
    # fb(mu_j, i_j) = 0 iff mu_j >= 0, i_j >= 0, mu_j * i_j = 0. The eps
    # keeps the sqrt smooth at (0,0); 1e-8 is a standard choice that leaves
    # the equilibrium zero in place.
    for j in range(N_COUNTRIES):
        i_j = defs[f"i_{j}"]
        out[f"fb_{j}"] = (
            mus[j] + i_j
            - jnp.sqrt(mus[j] * mus[j] + i_j * i_j + fb_eps)
        )

    return out
