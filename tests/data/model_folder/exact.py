"""Known solution of Brock–Mirman with log utility and full depreciation: s = alpha * beta."""

import jax.numpy as jnp


def exact_policy(state, constants):
    return jnp.full((state.shape[0], 1), constants["alpha"] * constants["beta"])
