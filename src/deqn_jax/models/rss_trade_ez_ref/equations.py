"""Equilibrium conditions of the RSS trade DSGE (Epstein–Zin), reference layout.

82 residuals in the reference solution's order and naming, so that the
checkpoint-parity protocol can compare them one by one. Writeup equation
numbers are cited per block; the last two blocks (``SDF``, ``Transversality``,
``Wealth``) are the reference's training scaffolding — they are not in the
economics writeup and exist only because the checkpoint was trained against
them.

Expectations: the certainty equivalent (29), the bond Euler (26) and the
capital Euler (27) are linear in one next-period expectation each, so the
two-stage hooks factor them as ``E[inside]`` (``inside_fn``, 3n columns in the
order CE / EB / EC) and current-period factors (``combine_fn``). The
single-stage ``equations_fn`` composes the two on a single draw and is what
the standard loss path integrates.
"""

from __future__ import annotations

from typing import Dict, Tuple

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.rss_trade_ez_ref.definitions import clip_policy, core
from deqn_jax.models.rss_trade_ez_ref.variables import Layout


def equation_names(n: int) -> Tuple[str, ...]:
    """The reference's 82 residual keys (n = 3), in its order."""
    names = []
    for sector in ("C", "M", "X"):
        names += [f"capital_income_{sector}{i}" for i in range(1, n + 1)]
    for sector in ("C", "M", "X"):
        names += [f"labor_income_{sector}{i}" for i in range(1, n + 1)]
    for sector in ("C", "M", "X"):
        names += [f"input_demand_{sector}{i}" for i in range(1, n + 1)]
    for block in (
        "capital_allocation",
        "labor_allocation",
        "intermediate_goods_allocation",
    ):
        names += [f"{block}{i}" for i in range(1, n + 1)]
    for sector in ("C", "M", "X"):
        names += [f"market_clearing_{sector}{i}" for i in range(1, n + 1)]
    for sector in ("C", "M", "X"):
        names += [f"price_index_{sector}{i}" for i in range(1, n + 1)]
    names += [f"BoP_{i}" for i in range(1, n + 1)]
    names.append("Good_Market_Clearing_condition")
    names += [f"law_of_motion_K_{i}" for i in range(1, n + 1)]
    names += [f"certainty_equivalent_{i}" for i in range(1, n + 1)]
    names += [f"value_function_{i}" for i in range(1, n + 1)]
    names += [f"EE_bond_{i}" for i in range(1, n + 1)]
    names += [f"EE_capital_{i}" for i in range(1, n + 1)]
    names += [f"SDF_{i}" for i in range(1, n + 1)]
    for i in range(1, n + 1):
        names += [f"Transversality_{i}", f"Wealth_{i}"]
    return tuple(names)


def inside_keys(n: int) -> Tuple[str, ...]:
    """Continuation columns, reference layout: CE = 0..n-1, EB = n..2n-1, EC = 2n..3n-1."""
    return tuple(
        [f"ce_{i}" for i in range(1, n + 1)]
        + [f"eb_{i}" for i in range(1, n + 1)]
        + [f"ec_{i}" for i in range(1, n + 1)]
    )


def make_equations(layout: Layout):
    n = layout.n
    names = equation_names(n)
    keys = inside_keys(n)

    def _spread(
        out: Dict[str, Array], prefix: str, values: Array, sep: str = ""
    ) -> None:
        for i in range(n):
            out[f"{prefix}{sep}{i + 1}"] = values[:, i]

    def inside_fn(
        state, policy, next_state, next_policy, constants
    ) -> Dict[str, Array]:
        """Pure forward parts of the three expectation-bearing blocks, at a
        next-period draw. ``mu`` (current certainty equivalent) enters the
        EZ kernel and is the only current-period quantity used here."""
        gama = float(constants["gama"])
        psi = float(constants["psi"])
        p_exp = -gama + 1.0 / psi
        cur = core(state, policy, constants, layout)
        nxt = core(next_state, next_policy, constants, layout)
        kernel = (nxt["U"] / cur["mu"] + 1e-8) ** p_exp * nxt["muc"]
        ce = nxt["U"] ** (1.0 - gama)
        eb = kernel * (1.0 + nxt["q"]) / nxt["P_C"]
        ec = kernel * nxt["P_X"] * (nxt["r"] / nxt["P_X"] - nxt["Phi_2"]) / nxt["P_C"]
        out: Dict[str, Array] = {}
        _spread(out, "ce_", ce)
        _spread(out, "eb_", eb)
        _spread(out, "ec_", ec)
        return out

    def combine_fn(
        state, policy, expectations: Dict[str, Array], constants
    ) -> Dict[str, Array]:
        """All 82 residuals from the current state/policy and E[inside]."""
        c = constants
        alpha = float(c["alpha"])
        beta = float(c["beta"])
        delta = float(c["delta"])
        lam = float(c["lambda_"])
        gama = float(c["gama"])
        psi = float(c["psi"])
        min_K = float(c["min_K"])
        p_exp = -gama + 1.0 / psi

        d = core(state, policy, c, layout)
        K_floor = jnp.maximum(d["K_state"], min_K)
        E_ce = jnp.stack([expectations[k] for k in keys[:n]], axis=1)
        E_eb = jnp.stack([expectations[k] for k in keys[n : 2 * n]], axis=1)
        E_ec = jnp.stack([expectations[k] for k in keys[2 * n :]], axis=1)

        r: Dict[str, Array] = {}
        sectors = {
            "C": (d["nu_c"], d["P_C"], d["Y_C"], d["K_C"], d["L_C"], d["M_C"]),
            "M": (d["nu_m"], d["P_M"], d["Y_M"], d["K_M"], d["L_M"], d["M_M"]),
            "X": (d["nu_x"], d["P_X"], d["Y_X"], d["K_X"], d["L_X"], d["M_X"]),
        }
        # (1)-(3) capital income, ratio form: 1 - alpha nu P Y / (r K_share K)
        for s, (nu, P, Y, Ks, Ls, Ms) in sectors.items():
            _spread(
                r,
                f"capital_income_{s}",
                1.0 - alpha * nu * P * Y / (d["r"] * Ks * K_floor),
            )
        # (4)-(6) labor income: ((1-alpha) nu P Y - w L_share L_eff) / ((1-alpha) nu P Y)
        for s, (nu, P, Y, Ks, Ls, Ms) in sectors.items():
            lhs = (1.0 - alpha) * nu * P * Y
            _spread(r, f"labor_income_{s}", (lhs - d["w"] * Ls * d["L_eff"]) / lhs)
        # (7)-(9) intermediate input demand: ((1-nu) P Y - P_M M_share M) / ((1-nu) P Y)
        for s, (nu, P, Y, Ks, Ls, Ms) in sectors.items():
            lhs = (1.0 - nu) * P * Y
            _spread(r, f"input_demand_{s}", (lhs - d["P_M"] * Ms * d["M"]) / lhs)
        # (10)-(12) allocation constraints as shares summing to one
        _spread(r, "capital_allocation", 1.0 - (d["K_C"] + d["K_M"] + d["K_X"]))
        _spread(r, "labor_allocation", 1.0 - (d["L_C"] + d["L_M"] + d["L_X"]))
        _spread(
            r, "intermediate_goods_allocation", 1.0 - (d["M_C"] + d["M_M"] + d["M_X"])
        )
        # (13) C = Y_C ; (14) exports = output of M ; (15) X = Y_X
        _spread(r, "market_clearing_C", (d["C"] - d["Y_C"]) / d["Y_C"])
        # importer j pays P_M_j M_j pi_ji omega_ji to exporter i
        exports = jnp.einsum("bj,bji->bi", d["P_M"] * d["M"], d["pi"] * d["omega"])
        pmy = d["P_M"] * d["Y_M"]
        _spread(r, "market_clearing_M", (exports - pmy) / (-pmy))
        _spread(r, "market_clearing_X", (d["X"] - d["Y_X"]) / d["Y_X"])
        # (16)-(18) price indices: unit cost = price
        _spread(r, "price_index_C", (d["uc_C"] - d["P_C"]) / d["P_C"])
        _spread(r, "price_index_M", (d["P_M_index"] - d["P_M"]) / d["P_M"])
        _spread(r, "price_index_X", (d["uc_X"] - d["P_X"]) / d["P_X"])
        # (22) balance of payments per country, normalized by P_M Y_M; (24) world
        # clearing as the sum of net exports plus interest on outstanding bonds
        imports_share = jnp.sum(d["pi"] * d["omega"], axis=2)
        trade_balance = d["P_M"] * (d["Y_M"] - d["M"] * imports_share)
        bop = (
            -d["A"] * d["L_eff"]
            + trade_balance
            + (1.0 + d["q"]) * d["A_state"] * d["L_eff"]
        ) / pmy
        _spread(r, "BoP_", bop)
        r["Good_Market_Clearing_condition"] = jnp.sum(
            trade_balance + d["q"] * d["A_state"] * d["L_eff"], axis=1
        )
        # (25) law of motion of capital against the K policy, ratio form
        K_next_identity = (1.0 - delta) * d["K_state"] + delta ** (1.0 - lam) * d[
            "X"
        ] ** lam * K_floor ** (1.0 - lam)
        _spread(
            r,
            "law_of_motion_K_",
            (K_next_identity - d["K"]) / (-jnp.maximum(d["K"], min_K)),
        )
        # (29) certainty equivalent: mu^(1-gama) = E[U'^(1-gama)]
        mu_pow = d["mu"] ** (1.0 - gama)
        _spread(r, "certainty_equivalent_", (mu_pow - E_ce) / (mu_pow + 1e-8))
        # (28) value function aggregator
        rho = 1.0 - 1.0 / psi
        U_pow = (d["U"] + 1e-8) ** rho
        vf = (
            U_pow
            - (
                (1.0 - beta) * (d["C"] / d["L_eff"] + 1e-8) ** rho
                + beta * (d["mu"] + 1e-8) ** rho
            )
        ) / U_pow
        _spread(r, "value_function_", vf)
        # (26) bond Euler: 1 = beta E[kernel (1+q') P_C/P_C'] with kernel = (U'/mu)^p muc'/muc
        _spread(r, "EE_bond_", 1.0 - beta * (d["P_C"] / d["muc"]) * E_eb)
        # (27) capital Euler
        _spread(
            r,
            "EE_capital_",
            1.0
            - beta
            * d["P_C"]
            / (d["muc"] * d["P_X"] * jnp.maximum(d["Phi_1"], 1e-5))
            * E_ec,
        )
        # --- reference training scaffolding (not in the writeup) ---
        # SDF accumulator: U_store' = U U_store / max(mu, 1e-5)
        sdf = (
            d["U_store"] - d["U"] * d["U_store_state"] / jnp.maximum(d["mu"], 1e-5)
        ) / jnp.maximum(d["U_store"], 1e-5)
        _spread(r, "SDF_", sdf)
        # Transversality: A_min * batch-mean of beta^t muc (U_store U)^p / P_C * A^2.
        # The reference reduces over the batch (one scalar per country,
        # broadcast); replicated here, so this residual couples the batch.
        disc = beta ** (float(c["tv_progress"]) * float(c["tv_episode_length"]))
        terminal = (
            disc
            * d["muc"]
            * jnp.maximum(d["U_store_state"] * d["U"], 1e-5) ** p_exp
            / d["P_C"]
        )
        tv = d["A_min"][:, None] * jnp.mean(
            terminal * d["A"] ** 2, axis=0, keepdims=True
        )
        wealth_pen = 100.0 * d["a_mask"][:, None] * d["A"]
        for i in range(n):
            r[f"Transversality_{i + 1}"] = tv[:, i]
            r[f"Wealth_{i + 1}"] = wealth_pen[:, i]
        assert tuple(r.keys()) == names, "residual order drifted from EQUATION_NAMES"
        return r

    def equations(
        state, policy, next_state, next_policy, constants
    ) -> Dict[str, Array]:
        """Single-draw residuals (the standard loss path averages them over
        shocks; every expectation-bearing block is linear in E[inside], so the
        average of these equals combine_fn(E[inside]))."""
        inside = inside_fn(state, policy, next_state, next_policy, constants)
        return combine_fn(state, policy, inside, constants)

    return equations, inside_fn, combine_fn, names


def ez_kernel(
    U_next: Array, mu: Array, muc_next: Array, muc: Array, gama: float, psi: float
) -> Array:
    """Epstein–Zin stochastic discount factor kernel ``(U'/mu)^(1/psi - gama)
    muc'/muc`` (writeup (26)); reduces to ``muc'/muc`` when ``gama = 1/psi``.
    Exposed for the kernel sanity test."""
    return (U_next / mu + 1e-8) ** (-gama + 1.0 / psi) * muc_next / muc


__all__ = [
    "make_equations",
    "equation_names",
    "inside_keys",
    "ez_kernel",
    "clip_policy",
]
