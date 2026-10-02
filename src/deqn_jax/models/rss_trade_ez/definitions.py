"""Derived quantities of the Phase-1 variant ``rss_trade_ez``.

The economics is the replica's (:func:`~deqn_jax.models.rss_trade_ez_ref.
definitions.economy`, shared, not copied); what differs is where the inputs
come from:

- the homotopy weights are the constants ``homotopy_asym`` (calibration)
  and ``homotopy_trade`` (trade costs) instead of state columns;
- the tariff matrix is assembled from the six off-diagonal state columns
  with a zero diagonal;
- the world bond market is cleared here, not inside a network: the policy
  carries raw positions ``a_i`` and every equation uses the projection
  ``A_i = a_i - (sum_j a_j L_j / sum_j L_j^2) L_i``, so ``sum_i A_i L_i = 0``
  for any network architecture (``L`` the effective labor endowments);
- ``bonds_active = 0`` shuts the bond (``A = 0``).
"""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.rss_trade_ez.variables import Layout
from deqn_jax.models.rss_trade_ez_ref.definitions import (
    capital_identity,
    economy,
    flat_definitions,
    interpolate_calibration,
)


def bonds_active(constants) -> bool:
    """The ``bonds_active`` switch, validated: the curriculum uses 0 and 1
    only (a fractional value has no economic reading)."""
    v = float(constants["bonds_active"])
    if v not in (0.0, 1.0):
        raise ValueError(f"constants['bonds_active'] must be 0 or 1, got {v}")
    return v == 1.0


def bond_projection(a: Array, L: Array) -> Array:
    """Clear the world bond market: remove from ``a`` its component along
    ``L`` (orthogonal projection onto ``sum_i A_i L_i = 0``)."""
    coef = jnp.sum(a * L, axis=1, keepdims=True) / jnp.sum(L * L, axis=1, keepdims=True)
    return a - coef * L


def calibration(state: Array, constants, layout: Layout) -> Dict[str, Array]:
    """Effective calibration at the constants' homotopy weights, ``[b, ...]``."""
    b = state.shape[0]
    h = jnp.full((b, 1), float(constants["homotopy_asym"]), dtype=state.dtype)
    h1 = jnp.full((b, 1, 1), float(constants["homotopy_trade"]), dtype=state.dtype)
    return interpolate_calibration(h, h1, constants, layout.n)


def tariff_matrix(state: Array, layout: Layout) -> Array:
    """``[b, n, n]`` tariffs, importer-major, zero on the diagonal."""
    tau = jnp.zeros((state.shape[0], layout.n, layout.n), dtype=state.dtype)
    return tau.at[:, layout.pair_i, layout.pair_j].set(state[:, layout.tau])


def policy_blocks(
    state: Array, policy: Array, cal: Dict[str, Array], constants, layout: Layout
) -> Dict[str, Array]:
    """Per-block ``[b, n]`` policy views (``q`` is ``[b, 1]``), plus the two
    names the shared economics reads: ``A`` (cleared next-period bond) and
    ``K`` (next-capital policy)."""
    p = {block: policy[:, idx] for block, idx in layout.blocks.items()}
    if bonds_active(constants):
        p["A"] = bond_projection(p["a"], cal["L"])
    else:
        p["A"] = jnp.zeros_like(p["a"])
    p["K"] = p["K_prime"]
    return p


def core(state: Array, policy: Array, constants, layout: Layout) -> Dict[str, Array]:
    """All definitions as ``[b, n]`` / ``[b, n, n]`` arrays."""
    cal = calibration(state, constants, layout)
    p = policy_blocks(state, policy, cal, constants, layout)
    out = dict(p)
    out.update(
        economy(
            p,
            state[:, layout.K],
            state[:, layout.A],
            tariff_matrix(state, layout),
            cal,
            constants,
        )
    )
    out["K_next"] = capital_identity(out["K_state"], out["X"], constants)
    # Tariff revenue P_M M sum_j pi_ij (1 - omega_ij): collected on imports
    # but not rebated to the household (as in the reference), so it is the
    # gap between the household budget and the balance of payments.
    out["tariff_revenue"] = (
        out["P_M"] * out["M"] * jnp.sum(out["pi"] * (1.0 - out["omega"]), axis=2)
    )
    return out


def next_endogenous(state: Array, policy: Array, constants, layout: Layout):
    """``(K', A')``: the next-capital policy ``K_prime`` (the
    ``law_of_motion_K`` residual ties it to the accumulation identity) and
    the cleared bond.

    Not the identity itself: it compounds the investment policy's error
    every period, and under an untrained network capital grows ~30% a
    period and overflows within a 256-step episode, while the policy output
    is a bounded function of the state."""
    cal = calibration(state, constants, layout)
    p = policy_blocks(state, policy, cal, constants, layout)
    return p["K_prime"], p["A"]


def make_definitions(layout: Layout):
    """``definitions_fn``: the replica's flat definitions plus the cleared
    bond, the identity next capital, tariff revenue and world net exports."""
    n = layout.n

    def definitions(state: Array, policy: Array, constants) -> Dict[str, Array]:
        single = state.ndim == 1
        if single:
            state, policy = state[None, :], policy[None, :]
        d = core(state, policy, constants, layout)
        out = flat_definitions(d, n)
        for key in ("A", "K_next", "tariff_revenue"):
            for i in range(n):
                out[f"{key}_{i + 1}"] = d[key][:, i]
        imports_share = jnp.sum(d["pi"] * d["omega"], axis=2)
        out["world_net_exports"] = jnp.sum(
            d["P_M"] * (d["Y_M"] - d["M"] * imports_share), axis=1
        )
        return {k: v[0] for k, v in out.items()} if single else out

    return definitions
