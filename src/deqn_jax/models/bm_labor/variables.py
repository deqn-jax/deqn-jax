"""Variable specification and constants for Brock-Mirman with endogenous labor.

Extension of the stochastic Brock-Mirman model with a labor-supply choice.
Representative agent chooses (s, L) to maximize

    E sum_t beta^t [ ln C_t - psi * L_t^(1+theta) / (1+theta) ]

s.t.  K_{t+1} = (1 - delta) K_t + Y_t - C_t
      Y_t = Z_t * L_t^(1-alpha) * K_t^alpha
      ln Z_{t+1} = rho_z * ln Z_t + sigma_z * eps

Two FOCs (capital Euler + intratemporal labor). Utility is log in C and
convex-in-effort in L with Frisch elasticity 1/theta.
"""

import jax.numpy as jnp

from deqn_jax.models.variable_spec import VariableSpec

SPEC = VariableSpec(
    state_names=("k", "z"),
    policy_names=("sav_rate", "L"),
)

CONSTANTS = {
    "alpha": 0.36,
    "beta": 0.99,
    "gamma": 1.0,        # CRRA exponent on consumption; gamma=1 => log utility
    "delta": 0.1,
    "psi": 1.0,          # labor disutility weight
    "theta": 1.0,        # inverse Frisch elasticity (1 = quadratic effort cost)
    "rho_z": 0.9,
    "sigma_z": 0.04,
}

# sav_rate in (0, 1); L in (0, inf). `inf` on upper bound makes the output
# layer use softplus for that dimension (see networks/common._apply_bounds).
POLICY_LOWER = jnp.array([1e-6, 1e-6])
POLICY_UPPER = jnp.array([1 - 1e-6, jnp.inf])

N_SHOCKS = 1

DESCRIPTION = "Brock-Mirman (1972) with endogenous labor supply"
