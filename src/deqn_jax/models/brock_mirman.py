"""Brock-Mirman (1972) optimal growth model.

A simple RBC model with:
- State: (k, z) = (capital, log TFP)
- Policy: sav_rate (savings rate)
- One Euler equation

This is the canonical test case for DEQN methods.
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.types import ModelSpec
from deqn_jax.models.variables import BROCK_MIRMAN_SPEC as SPEC


# Model constants (calibration)
CONSTANTS = {
    "alpha": 1 / 3,      # Capital share
    "beta": 0.95,        # Discount factor
    "gamma": 2.0,        # Risk aversion (CRRA)
    "delta": 0.1,        # Depreciation rate
    "rho_z": 0.8,        # TFP persistence
    "sigma_z": 0.03,     # TFP shock std
}


def definitions(state: Array, policy: Array, constants: Dict) -> Dict[str, Array]:
    """Compute derived quantities from state and policy."""
    s = SPEC.unpack_state(state)
    p = SPEC.unpack_policy(policy)

    alpha = constants["alpha"]
    gamma = constants["gamma"]

    # TFP level
    Z = jnp.exp(s.z)

    # Production: y = Z * k^alpha
    y = Z * jnp.power(s.k, alpha)

    # Marginal product of capital
    mpk = alpha * Z * jnp.power(s.k, alpha - 1)

    # Consumption and savings
    c = (1 - p.sav_rate) * y
    sav = p.sav_rate * y

    # Marginal utility (CRRA)
    u_c = jnp.power(c, -gamma)

    return {"Z": Z, "y": y, "mpk": mpk, "c": c, "s": sav, "u_c": u_c}


def equations(
    state: Array,
    policy: Array,
    next_state: Array,
    next_policy: Array,
    constants: Dict,
) -> Dict[str, Array]:
    """Compute equilibrium equation residuals.

    Euler equation:
        u'(c) = beta * E[u'(c') * (1 + mpk' - delta)]
    """
    beta = constants["beta"]
    delta = constants["delta"]

    defs = definitions(state, policy, constants)
    next_defs = definitions(next_state, next_policy, constants)

    # Euler equation residual
    euler = defs["u_c"] - beta * next_defs["u_c"] * (1 + next_defs["mpk"] - delta)

    return {"euler": euler}


def step(
    state: Array,
    policy: Array,
    shock: Array,
    constants: Dict,
) -> Array:
    """Transition to next state.

    Capital: k' = (1 - delta) * k + s
    TFP:     z' = rho * z + sigma * eps
    """
    s = SPEC.unpack_state(state)
    defs = definitions(state, policy, constants)

    delta = constants["delta"]
    rho_z = constants["rho_z"]
    sigma_z = constants["sigma_z"]

    # Capital accumulation
    k_next = (1 - delta) * s.k + defs["s"]

    # TFP shock
    eps = shock[:, 0] if shock.ndim > 1 else shock
    z_next = rho_z * s.z + sigma_z * eps

    return jnp.stack([k_next, z_next], axis=1)


def steady_state(constants: Dict) -> Tuple[Array, Array]:
    """Compute deterministic steady state.

    At steady state with z=0:
        1 = beta * (1 + alpha * k^(alpha-1) - delta)
    """
    alpha = constants["alpha"]
    beta = constants["beta"]
    delta = constants["delta"]

    # Steady state capital
    k_ss = ((1 / beta - 1 + delta) / alpha) ** (1 / (alpha - 1))
    z_ss = 0.0

    # Steady state output and savings rate
    y_ss = k_ss ** alpha
    s_ss = delta * k_ss
    sav_rate_ss = s_ss / y_ss

    return jnp.array([k_ss, z_ss]), jnp.array([sav_rate_ss])


def init_state(key: Array, batch_size: int, constants: Dict) -> Array:
    """Sample initial states around steady state."""
    ss_state, _ = steady_state(constants)

    k_key, z_key = jax.random.split(key)
    k_init = ss_state[0] * (1 + 0.2 * jax.random.uniform(k_key, (batch_size,), minval=-1, maxval=1))
    z_init = jax.random.normal(z_key, (batch_size,)) * constants["sigma_z"] * 2

    return jnp.stack([k_init, z_init], axis=1)


# Model specification
MODEL = ModelSpec(
    name="brock_mirman",
    n_states=SPEC.n_states,
    n_policies=SPEC.n_policies,
    n_shocks=1,
    state_names=SPEC.state_names,
    policy_names=SPEC.policy_names,
    equation_names=("euler",),
    constants=CONSTANTS,
    equations_fn=equations,
    step_fn=step,
    steady_state_fn=steady_state,
    init_state_fn=init_state,
    policy_lower=jnp.array([1e-6]),
    policy_upper=jnp.array([1 - 1e-6]),
)
