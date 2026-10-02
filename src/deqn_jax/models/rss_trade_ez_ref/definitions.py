"""Derived quantities of the RSS trade DSGE (writeup §2.2 and the budget block).

Everything the equations share is computed once here, vectorized over
countries: arrays are ``[b, n]`` per country and ``[b, n, n]`` per
importer/exporter pair. The homotopy weights read from the state interpolate
the country-specific calibration against its cross-country average
(``homo``) and the iceberg trade costs against a near-autarky anchor in log
space (``homo_1``); at the converged values (both 1) the true calibration is
in force.

The economics (:func:`economy`, :func:`interpolate_calibration`,
:func:`capital_identity`) is layout-free and shared with the Phase-1 variant
``rss_trade_ez``; only :func:`core` reads the replica's state and policy
layout.
"""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.rss_trade_ez_ref.variables import (
    FLOOR_1E3_BLOCKS,
    UNIT_INTERVAL_BLOCKS,
    Layout,
)


def _c(constants, key):
    return jnp.asarray(constants[key])


def clip_policy(policy: Array, layout: Layout) -> Array:
    """The reference accessor layer's hard clips: shares and the bond return in
    [0, 1]; the value function and certainty equivalent floored at 1e-3.
    Applied wherever a policy enters a definition or residual."""
    lower = jnp.full(layout.n_policies, -jnp.inf)
    upper = jnp.full(layout.n_policies, jnp.inf)
    for block in UNIT_INTERVAL_BLOCKS:
        lower = lower.at[layout.blocks[block]].set(0.0)
        upper = upper.at[layout.blocks[block]].set(1.0)
    for block in FLOOR_1E3_BLOCKS:
        lower = lower.at[layout.blocks[block]].set(1e-3)
    return jnp.clip(policy, lower, upper)


def unpack_policy(policy: Array, layout: Layout) -> Dict[str, Array]:
    """Per-block views ``[b, n]`` (``q`` is ``[b, 1]``) of a clipped policy."""
    return {block: policy[:, idx] for block, idx in layout.blocks.items()}


def interpolate_calibration(h: Array, h1: Array, constants, n: int) -> Dict[str, Array]:
    """Homotopy-interpolated calibration: ``h * true + (1 - h) * average`` for
    the country arrays, and log-space interpolation of the trade-cost matrix
    against the near-autarky anchor (diagonal 1, off-diagonal
    ``d_average_off_diag``). ``h`` broadcasts against ``[b, n]``, ``h1``
    against ``[b, n, n]``; at ``h = h1 = 1`` the true calibration is in force."""
    out = {}
    for key in ("L", "nu_c", "nu_m", "nu_x", "A_c", "A_x", "T_m"):
        out[key] = h * _c(constants, key)[None, :] + (1.0 - h) * float(
            constants["base_" + key]
        )
    d = _c(constants, "d")
    anchor = jnp.where(
        jnp.eye(n, dtype=bool), 0.0, jnp.log(float(constants["d_average_off_diag"]))
    )
    out["d"] = jnp.exp(h1 * jnp.log(d)[None] + (1.0 - h1) * anchor[None])
    return out


def effective_calibration(state: Array, constants, layout: Layout) -> Dict[str, Array]:
    """The replica reads its homotopy weights from the state columns
    ``homo`` (calibration) and ``homo_1`` (trade costs)."""
    h = state[:, layout.homo][:, None]
    h1 = state[:, layout.homo_1][:, None, None]
    return interpolate_calibration(h, h1, constants, layout.n)


def capital_identity(K_state: Array, X: Array, constants) -> Array:
    """Accumulation identity (writeup (25)):
    ``K' = (1-delta) K + delta^(1-lambda) X^lambda max(K, min_K)^(1-lambda)``."""
    delta = float(constants["delta"])
    lam = float(constants["lambda_"])
    K_floor = jnp.maximum(K_state, float(constants["min_K"]))
    return (1.0 - delta) * K_state + delta ** (1.0 - lam) * X**lam * K_floor ** (
        1.0 - lam
    )


def economy(
    p: Dict[str, Array],
    K_state: Array,
    A_state: Array,
    tau: Array,
    cal: Dict[str, Array],
    constants,
) -> Dict[str, Array]:
    """The model's derived quantities, shared by the replica and the Phase-1
    variant: ``p`` holds the per-block policy views (``p["A"]`` is the
    next-period per-capita bond position, already cleared), ``tau`` the
    ``[b, n, n]`` tariff matrix (zero diagonal), ``cal`` the effective
    calibration.

    Budget pieces (per capita positions ``A`` scaled by effective labor;
    ``q`` is the bond's net return in the reference's accounting -- holdings
    pay ``1 + q`` next period -- not a discount price):
      B      = (A_policy - A_state) * L_eff                 (net bond purchase)
      wealth = max(-B + r K + w L_eff + q A_state L_eff, 1e-4)
    Tariff revenue is not rebated to the household here, as in the reference
    (graph @aleph/deqn: the accounting question is recorded on the port).
      C      = (1 - s) wealth / P_C,   X = s wealth / P_X   (savings share s)
    Capital adjustment terms Phi_1, Phi_2 (writeup (31)–(32)) from the
    accumulation identity K' = (1-delta) K + delta^(1-lambda) X^lambda K^(1-lambda);
    unit cost of the intermediate variety (30); marginal utility
    muc = (max(C, 1e-5) / L_eff)^(-1/psi); payment fraction omega = 1/(1+tau)
    (33); Eaton–Kortum trade shares pi_ij (19) in closed form.
    """
    alpha = float(constants["alpha"])
    delta = float(constants["delta"])
    lam = float(constants["lambda_"])
    psi = float(constants["psi"])
    theta = float(constants["theta"])
    min_K = float(constants["min_K"])

    L_eff = cal["L"]
    q = p["q"]  # [b, 1]

    B = (p["A"] - A_state) * L_eff
    wealth = jnp.maximum(
        -B + p["r"] * K_state + p["w"] * L_eff + q * A_state * L_eff, 1e-4
    )
    C = (1.0 - p["s"]) * wealth / p["P_C"]
    X = p["s"] * wealth / p["P_X"]

    K_floor = jnp.maximum(K_state, min_K)
    g = delta ** (1.0 - lam) * (X / K_floor) ** lam
    Phi_1 = delta ** ((lam - 1.0) / lam) * (1.0 / lam) * g ** ((1.0 - lam) / lam)
    Phi_2 = Phi_1 * ((lam - 1.0) * g - (1.0 - delta))

    def unit_cost(nu):
        return (
            (p["r"] / (alpha * nu)) ** (alpha * nu)
            * (p["w"] / ((1.0 - alpha) * nu)) ** ((1.0 - alpha) * nu)
            * (p["P_M"] / (1.0 - nu)) ** (1.0 - nu)
        )

    u_M = unit_cost(cal["nu_m"])
    uc_C = unit_cost(cal["nu_c"]) / cal["A_c"]
    uc_X = unit_cost(cal["nu_x"]) / cal["A_x"]
    muc = (jnp.maximum(C, 1e-5) / L_eff) ** (-1.0 / psi)

    omega = 1.0 / (1.0 + tau)
    # Fréchet term T_j (u_M_j d_ij / omega_ij)^(-theta), i importer, j exporter
    frechet = cal["T_m"][:, None, :] * (u_M[:, None, :] * cal["d"] / omega) ** (-theta)
    pi = frechet / jnp.sum(frechet, axis=2, keepdims=True)
    P_M_index = float(constants["gam"]) * jnp.sum(frechet, axis=2) ** (-1.0 / theta)

    return dict(
        L_eff=L_eff,
        nu_c=cal["nu_c"],
        nu_m=cal["nu_m"],
        nu_x=cal["nu_x"],
        K_state=K_state,
        A_state=A_state,
        B=B,
        wealth=wealth,
        C=C,
        X=X,
        Phi_1=Phi_1,
        Phi_2=Phi_2,
        u_M=u_M,
        uc_C=uc_C,
        uc_X=uc_X,
        muc=muc,
        tau=tau,
        omega=omega,
        pi=pi,
        P_M_index=P_M_index,
    )


def core(state: Array, policy: Array, constants, layout: Layout) -> Dict[str, Array]:
    """All replica definitions as ``[b, n]`` / ``[b, n, n]`` arrays: the
    clipped policy blocks, :func:`economy` at the state's homotopy weights,
    and the scaffolding columns the replica's residuals read."""
    p = unpack_policy(clip_policy(policy, layout), layout)
    cal = effective_calibration(state, constants, layout)
    tau = state[:, layout.tau.reshape(-1)].reshape(-1, layout.n, layout.n)
    out = dict(p)
    out.update(economy(p, state[:, layout.K], state[:, layout.A], tau, cal, constants))
    out.update(
        U_store_state=state[:, layout.U_store],
        A_min=state[:, layout.A_min],
        a_mask=state[:, layout.a_mask],
    )
    return out


FLAT_DEFINITION_KEYS = ("C", "X", "B", "wealth", "muc", "Phi_1", "Phi_2", "u_M")


def flat_definitions(d: Dict[str, Array], n: int) -> Dict[str, Array]:
    """Per-name ``[b]`` views of the shared definitions (the logger
    histograms them): per-country blocks, the pair matrices, and the world
    diagnostics (net bond supply, which must clear, and minimum C / K)."""
    out: Dict[str, Array] = {}
    for key in FLAT_DEFINITION_KEYS:
        for i in range(n):
            out[f"{key}_{i + 1}"] = d[key][:, i]
    for i in range(n):
        for j in range(n):
            out[f"pi_{i + 1}{j + 1}"] = d["pi"][:, i, j]
            out[f"omega_{i + 1}{j + 1}"] = d["omega"][:, i, j]
    out["world_bond_supply"] = jnp.sum(d["A"] * d["L_eff"], axis=1)
    out["min_C"] = jnp.min(d["C"], axis=1)
    out["min_K"] = jnp.min(d["K_state"], axis=1)
    return out


def make_definitions(layout: Layout):
    """``definitions_fn`` for the ModelSpec: flat per-name ``[b]`` arrays
    (the logger histograms them); accepts a single state too."""

    def definitions(state: Array, policy: Array, constants) -> Dict[str, Array]:
        single = state.ndim == 1
        if single:
            state, policy = state[None, :], policy[None, :]
        out = flat_definitions(core(state, policy, constants, layout), layout.n)
        return {k: v[0] for k, v in out.items()} if single else out

    return definitions
