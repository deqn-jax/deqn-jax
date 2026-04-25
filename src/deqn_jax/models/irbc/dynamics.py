"""State transition for 2-country IRBC.

Capital evolves directly from the policy's k_next outputs (investment
is absorbed into the policy, not an ARG). TFP follows AR(1) per country
with a shared aggregate innovation plus a country-specific idiosyncratic
innovation.
"""

from typing import Dict

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.irbc.variables import N_COUNTRIES, SPEC


def step(state: Array, policy: Array, shock: Array, constants: Dict) -> Array:
    """Next state.

        k_j'    = policy.k_j_next
        ln z_j' = rho_z ln z_j + sigma_eps (eps_j + eps_agg)

    Shock convention: shock[:, 0..N-1] are the country-specific innovations;
    shock[:, N] is the aggregate innovation hitting all countries.
    """
    s = SPEC.unpack_state(state)
    p = SPEC.unpack_policy(policy)
    rho_z = constants["rho_z"]
    sigma_eps = constants["sigma_eps"]

    # Pull per-country and aggregate shocks. Tolerate 1-D shock for
    # unbatched trace-time usage.
    if shock.ndim > 1:
        eps = [shock[:, j] for j in range(N_COUNTRIES)]
        eps_agg = shock[:, N_COUNTRIES]
    else:
        eps = [jnp.ones_like(s.k_0) * shock[j] for j in range(N_COUNTRIES)]
        eps_agg = jnp.ones_like(s.k_0) * shock[N_COUNTRIES]

    zs = [s.z_0, s.z_1]
    ks_next = [p.k_0_next, p.k_1_next]
    zs_next = [
        rho_z * zs[j] + sigma_eps * (eps[j] + eps_agg) for j in range(N_COUNTRIES)
    ]

    # Pack in SPEC order: k_0, k_1, z_0, z_1
    return jnp.stack([ks_next[0], ks_next[1], zs_next[0], zs_next[1]], axis=-1)
