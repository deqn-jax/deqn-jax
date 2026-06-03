"""State transition for the 6-generation life-cycle OLG (Geneva Day 2 Ex 4)."""

from typing import Dict

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.olg_lifecycle.equations import _cohort_block
from deqn_jax.models.olg_lifecycle.variables import H


def step(state: Array, policy: Array, shock: Array, constants: Dict) -> Array:
    """Next state.

    Z'        = exp(rho_z * log Z + sigma_z * eps)
    k'^0      = 0                          (newborn enters with no assets)
    k'^{h+1}  = sav^h  for h = 0..H-2      (this period's savings become next
                                            period's capital, aged one cohort)
    """
    rho_z = constants["rho_z"]
    sigma_z = constants["sigma_z"]
    Z = state[:, :1]
    k = state[:, 1 : 1 + H]
    blk = _cohort_block(Z, k, policy, constants)
    sav = blk["sav"]  # [b,H]; last column is 0

    eps = shock[:, :1]  # [b,1]
    Z_next = jnp.exp(rho_z * jnp.log(Z) + sigma_z * eps)
    newborn = jnp.zeros((Z.shape[0], 1))
    # k'^0 = 0; k'^{1..H-1} = savings of cohorts 0..H-2.
    return jnp.concatenate([Z_next, newborn, sav[:, : H - 1]], axis=1)
