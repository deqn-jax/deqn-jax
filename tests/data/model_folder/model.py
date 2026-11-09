"""Brock–Mirman with full depreciation, written as a Lagrangian (see model.md).

The economist's model enters in three places: the period objective, the law of
motion, and the calibration. The Euler equation is derived by deqn-jax.
"""

import jax
import jax.numpy as jnp

from deqn_jax.api import ModelSpec, residuals_from_lagrangian

CONSTANTS = {"alpha": 0.36, "beta": 0.96, "rho": 0.9, "sigma": 0.04}

STATES = ("k", "z")  # capital, log productivity
POLICIES = ("s",)  # savings rate k'/y


def output(k, z, c):
    return jnp.exp(z) * k ** c["alpha"]


def objective(state, k_next, policy, prices, c):
    """Period utility log(c_t), with c_t = y_t - k_{t+1}. One sample."""
    k, z = state[0], state[1]
    consumption = output(k, z, c) - k_next[0]
    return jnp.log(jnp.maximum(consumption, 1e-8))


def step(state, policy, shock, c):
    """Law of motion for a batch: k' = s y,  z' = rho z + sigma eps."""
    k, z = state[:, 0], state[:, 1]
    k_next = policy[:, 0] * output(k, z, c)
    z_next = c["rho"] * z + c["sigma"] * shock[:, 0]
    return jnp.stack([k_next, z_next], axis=1)


def steady_state(c):
    s = c["alpha"] * c["beta"]
    k = s ** (1.0 / (1.0 - c["alpha"]))
    return jnp.array([k, 0.0]), jnp.array([s])


def init_state(key, batch_size, c):
    """Training draws: k within [0.5, 2] x steady state, z within +-4 std."""
    k_ss = steady_state(c)[0][0]
    z_sd = c["sigma"] / jnp.sqrt(1.0 - c["rho"] ** 2)
    kk, kz = jax.random.split(key)
    k = k_ss * jax.random.uniform(kk, (batch_size,), minval=0.5, maxval=2.0)
    z = jax.random.uniform(kz, (batch_size,), minval=-4 * z_sd, maxval=4 * z_sd)
    return jnp.stack([k, z], axis=1)


equations = residuals_from_lagrangian(
    objective,
    step,
    endogenous=[0],  # capital is chosen one period ahead
    euler_names=["euler_k"],
    n_shocks=1,
)

MODEL = ModelSpec(
    name="brock_mirman",
    n_states=2,
    n_policies=1,
    n_shocks=1,
    state_names=STATES,
    policy_names=POLICIES,
    equation_names=("euler_k",),
    shock_names=("eps",),
    constants=CONSTANTS,
    equations_fn=equations,
    step_fn=step,
    steady_state_fn=steady_state,
    init_state_fn=init_state,
    policy_lower=jnp.array([0.0]),
    policy_upper=jnp.array([1.0]),
)
