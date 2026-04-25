"""6-agent OLG with analytic closed-form solution (Krueger & Kubler 2004).

Six overlapping generations, log utility, Cobb-Douglas production, i.i.d.
shocks on TFP (eta) and depreciation (delta). Only agent 1 works. Agents
1..A-1 save; agent A consumes all income. Borrowing constraint k' >= 0
enforced via softplus output on the policy network.

Closed form:
    k'^h = beta_h * inc^h,
    beta_h = beta * (1 - beta^(A-h)) / (1 - beta^(A-h+1))

independent of shock realisation. Used as the oracle for validating the
trained DEQN.

State representation (minimal): [k^2, k^3, k^4, k^5, k^6, eta, delta].
Agent 1 is newborn with k^1 = 0, so k^1 is implicit. (eta, delta) carry
the current-period shock realisation.

Shock structure: 2 independent binary shocks. Framework-side, we use
Gauss-Hermite with n=2 points per shock (nodes at +-1 with weight 0.5).
The 2D tensor product gives 4 nodes at the corners of +-1 x +-1, with
uniform weights 0.25 -- exactly Simon's 4-state uniform expectation.
"""

import jax.numpy as jnp

from deqn_jax.models.variable_spec import VariableSpec

A = 6  # number of generations


SPEC = VariableSpec(
    state_names=("k2", "k3", "k4", "k5", "k6", "eta", "delta"),
    policy_names=("s1", "s2", "s3", "s4", "s5"),
)

CONSTANTS = {
    "alpha": 0.3,
    "beta": 0.7,  # low beta because each period is a generation
    "gamma": 1.0,  # log utility
    # Shock parameterisation: eta = eta_mid + eta_half * eps1
    #                         delta = delta_mid + delta_half * eps2
    # with eps1, eps2 in {-1, +1} via GH-2 quadrature.
    "eta_mid": 1.0,
    "eta_half": 0.05,  # eta in {0.95, 1.05}
    "delta_mid": 0.7,
    "delta_half": 0.2,  # delta in {0.5, 0.9}
    "labor_1": 1.0,  # agent 1 labor endowment; others 0
}

# Savings are non-negative; softplus output via inf upper bound.
POLICY_LOWER = jnp.zeros(A - 1)
POLICY_UPPER = jnp.full(A - 1, jnp.inf)

N_SHOCKS = 2

DESCRIPTION = "6-agent OLG with analytic closed-form solution (Krueger-Kubler 2004)"
