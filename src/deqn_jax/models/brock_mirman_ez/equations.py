"""Equilibrium equations for Brock-Mirman + Epstein-Zin (actor-critic demo).

Two residuals: Euler (CRRA-style with γ_ez) + Bellman (EZ recursion).

Calibration assumption: ``psi ≠ 1``. The ψ → 1 limit is multiplicative
Cobb-Douglas-like and would require a separate equation form; not
supported. Defaults are ψ=1.5, γ_ez=5.0 (genuine recursive EZ regime).

Bellman MC-safety. The recursion ``V = ((1-β)·c^{1-1/ψ} + β·V'^{1-1/ψ})^{1/(1-1/ψ)}``
is nonlinear in V', so per-shock residuals are biased under E[r] (Jensen).
For low TFP volatility (σ_z = 0.04) and bounded V the bias is small;
we accept it for the demo. A rigorous fix would compute the certainty
equivalent ``CE = E[V'^{1-γ}]^{1/(1-γ)}`` outside the per-shock loop
and form residuals as ``V - aggregator(c, CE)`` — would require either
a quadrature scheme or a two-pass MC (mean of V'^{1-γ} first, then
residual). Out of scope here.

Euler MC-safety. Standard CRRA-style FOC with risk aversion γ_ez:

    0 = u_c(c) - β · u_c(c') · (1 + r' - δ)

linear in the shock-dependent quantity ``u_c(c') (1 + r' - δ)``, so
``E[r] = 0`` holds exactly (no Jensen bias).
"""

from typing import Any, Dict

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.brock_mirman_ez.variables import SPEC

EQUATION_NAMES = ("euler", "bellman")


def definitions(state: Array, policy: Array, constants: Dict) -> Dict[str, Array]:
    """Compute derived quantities from state and policy."""
    s = SPEC.unpack_state(state)
    p = SPEC.unpack_policy(policy)

    alpha = constants["alpha"]
    gamma_ez = constants["gamma_ez"]

    Z = jnp.exp(s.z)
    y = Z * jnp.power(s.k, alpha)
    mpk = alpha * Z * jnp.power(s.k, alpha - 1)

    c = (1 - p.sav_rate) * y
    sav = p.sav_rate * y

    # Marginal utility of consumption with EZ risk aversion γ_ez.
    u_c = jnp.power(c, -gamma_ez)

    return {"Z": Z, "y": y, "mpk": mpk, "c": c, "s": sav, "u_c": u_c}


def equations(
    state: Array,
    policy: Array,
    next_state: Array,
    next_policy: Array,
    constants: Dict,
    *,
    value_now: Any,
    value_next: Any,
) -> Dict[str, Array]:
    """Two-equation residual: CRRA-style Euler + Epstein-Zin Bellman.

    The framework auto-supplies ``value_now`` and ``value_next`` from
    the critic head when ``actor_critic.mode`` is set; declaring them
    in the signature is the contract that flips the AC value-passing
    on. ``value_grad_next`` could be added here too — not used by
    this demo but available.
    """
    beta = constants["beta"]
    delta = constants["delta"]
    psi = constants["psi"]

    defs = definitions(state, policy, constants)
    next_defs = definitions(next_state, next_policy, constants)

    # ---- Euler (CRRA-style) ----
    u_c = defs["u_c"]
    u_c_next = next_defs["u_c"]
    mpk_next = next_defs["mpk"]
    euler = u_c - beta * u_c_next * (1.0 + mpk_next - delta)

    # ---- Bellman (Epstein-Zin recursion) ----
    # V_t = ((1-β)·c_t^{1-1/ψ} + β·V_{t+1}^{1-1/ψ})^{1/(1-1/ψ)}
    # Note: per-shock form (Jensen bias for ψ ≠ 1; see module docstring).
    rho = 1.0 - 1.0 / psi  # exponent
    c = defs["c"]
    # Clip to keep the powers well-defined under transient infeasible policies.
    c_safe = jnp.clip(c, 1e-6, None)
    v_next_safe = jnp.clip(value_next, 1e-6, None)
    aggregator = jnp.power(
        (1.0 - beta) * jnp.power(c_safe, rho) + beta * jnp.power(v_next_safe, rho),
        1.0 / rho,
    )
    bellman = value_now - aggregator

    return {"euler": euler, "bellman": bellman}
