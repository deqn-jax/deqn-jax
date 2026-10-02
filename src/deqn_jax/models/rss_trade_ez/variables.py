"""Variables and calibration of ``rss_trade_ez``, the Phase-1 variant of the
RSS three-country trade DSGE with Epstein–Zin households.

Same economy and calibration as the faithful replica ``rss_trade_ez_ref``
(Ravikumar, Santacreu & Sposi 2019; numbers in
:mod:`deqn_jax.models.rss_trade_ez_ref.variables`), in a cleaned layout:

- **State (18)**: ``K_i, A_i`` interleaved per country, then the six
  off-diagonal tariffs ``tau_ij`` and their log-volatilities ``sig_ij``
  (importer-major). Own-country tariffs do not exist and are not carried;
  the reference's training scaffolding (homotopy weights, transversality
  gate, bond mask, EZ discount accumulators) is gone from the state.
- **Policy (70)**: prices, the bond return ``q``, savings shares, next
  capital ``K_prime_i``, factor prices, sector outputs and allocation shares,
  the raw bond positions ``a_i`` (cleared in ``definitions_fn``), and the EZ
  value ``U_i`` and certainty equivalent ``mu_i``.
- **Shocks (12)**: the off-diagonal log-volatility innovations, then the
  off-diagonal tariff innovations.

The curriculum scalars become model constants (``homotopy_asym``,
``homotopy_trade``, ``bonds_active``), stepped by resumed runs with
``--set constants.<name>=<value>``; ``w_tv`` weighs the transversality term.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from deqn_jax.models.rss_trade_ez_ref.variables import (
    K_SS_REFERENCE,
    N_COUNTRIES,
)
from deqn_jax.models.rss_trade_ez_ref.variables import (
    build_constants as build_replica_constants,
)

# Constants the variant adds to the replica calibration (defaults: the
# converged stage, i.e. the true calibration with bonds traded).
VARIANT_CONSTANTS = {
    # calibration homotopy: h * country value + (1 - h) * cross-country mean
    "homotopy_asym": 1.0,
    # trade-cost homotopy in log space against the near-autarky anchor
    "homotopy_trade": 1.0,
    # 0: bonds shut (A = 0, the bond Euler replaced by a_i = 0); 1: traded
    "bonds_active": 1.0,
    # weight of aux_transversality_i (its loss is w_tv * mean(muc/P_C A^2))
    "w_tv": 1.0,
}
# Replica-only constants (the reference's terminal discount for its
# batch-reduced transversality residual), dropped with that residual.
_REPLICA_ONLY = ("tv_progress", "tv_episode_length")


def build_constants(**calibration) -> Dict[str, object]:
    """Variant constants for ``n = len(L)`` countries: the replica's
    calibration (same keyword arguments) plus :data:`VARIANT_CONSTANTS`."""
    c = build_replica_constants(**calibration)
    for key in _REPLICA_ONLY:
        c.pop(key)
    c.update(VARIANT_CONSTANTS)
    return c


CONSTANTS = build_constants()


def offdiag_pairs(n: int) -> Tuple[Tuple[int, int], ...]:
    """Importer-major ``(i, j)`` pairs with ``i != j`` (0-based)."""
    return tuple((i, j) for i in range(n) for j in range(n) if i != j)


def state_names(n: int) -> Tuple[str, ...]:
    names = []
    for i in range(1, n + 1):
        names += [f"K_{i}", f"A_{i}"]
    names += [f"tau_{i + 1}{j + 1}" for i, j in offdiag_pairs(n)]
    names += [f"sig_{i + 1}{j + 1}" for i, j in offdiag_pairs(n)]
    return tuple(names)


# Policy blocks, writeup order; "q" is the single world bond return.
POLICY_BLOCKS = (
    "P_C",
    "q",
    "s",
    "K_prime",
    "P_X",
    "r",
    "w",
    "P_M",
    "M",
    "Y_M",
    "Y_C",
    "Y_X",
    "K_C",
    "K_M",
    "K_X",
    "L_C",
    "L_M",
    "L_X",
    "M_C",
    "M_M",
    "M_X",
    "a",
    "U",
    "mu",
)

# Blocks bounded to [SHARE_LO, SHARE_HI] by a sigmoid head.
SHARE_BLOCKS = ("q", "s", "K_C", "K_M", "K_X", "L_C", "L_M", "L_X", "M_C", "M_M", "M_X")
SHARE_LO, SHARE_HI = 1e-4, 1.0 - 1e-4
# Everything else except the raw bond is a positive level: lower bound 1e-3
# through a softplus head.
POSITIVE_LO = 1e-3
# The raw bond position goes through the sigmoid head on [-A_BOUND, A_BOUND]:
# a = -b + 2b sigmoid(raw) = b tanh(raw / 2). The framework's bounded output
# head has no unbounded column, and the scale matters at initialization: a
# fresh MLP's O(1) raw outputs taken as per-capita bond positions push the
# small-labor country's next-period wealth to its floor and its marginal
# utility up by orders of magnitude. b = 0.5 (slope 1/4 at zero) starts
# positions an order below per-capita income and still reaches the edge of
# the trade-war stress range; the reference's network scales its bond
# columns by 0.01 for the same reason.
A_BOUND = 0.5


def policy_names(n: int) -> Tuple[str, ...]:
    names = []
    for block in POLICY_BLOCKS:
        if block == "q":
            names.append("q")
        else:
            names += [f"{block}_{i}" for i in range(1, n + 1)]
    return tuple(names)


def shock_names(n: int) -> Tuple[str, ...]:
    pairs = offdiag_pairs(n)
    return tuple(
        [f"eps_sigma_{i + 1}{j + 1}" for i, j in pairs]
        + [f"eps_tau_{i + 1}{j + 1}" for i, j in pairs]
    )


class Layout:
    """Index tables of the variant layout for a given country count."""

    def __init__(self, n: int):
        self.n = n
        self.pairs = offdiag_pairs(n)
        self.n_pairs = len(self.pairs)
        self.states = state_names(n)
        self.policies = policy_names(n)
        s = {name: k for k, name in enumerate(self.states)}
        p = {name: k for k, name in enumerate(self.policies)}
        self.s_idx, self.p_idx = s, p
        self.K = np.array([s[f"K_{i}"] for i in range(1, n + 1)])
        self.A = np.array([s[f"A_{i}"] for i in range(1, n + 1)])
        self.tau = np.array([s[f"tau_{i + 1}{j + 1}"] for i, j in self.pairs])
        self.sig = np.array([s[f"sig_{i + 1}{j + 1}"] for i, j in self.pairs])
        # flat indices of the pairs into an [n, n] matrix
        self.pair_flat = np.array([i * n + j for i, j in self.pairs])
        self.pair_i = np.array([i for i, _ in self.pairs])
        self.pair_j = np.array([j for _, j in self.pairs])
        self.blocks = {
            block: (
                np.array([p["q"]])
                if block == "q"
                else np.array([p[f"{block}_{i}"] for i in range(1, n + 1)])
            )
            for block in POLICY_BLOCKS
        }
        self.n_states = len(self.states)
        self.n_policies = len(self.policies)
        self.n_shocks = 2 * self.n_pairs

    def policy_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(lower, upper)`` for the framework's bounded output head:
        sigmoid on shares and ``q``, softplus on levels, a symmetric sigmoid
        on the raw bond."""
        lower = np.full(self.n_policies, POSITIVE_LO)
        upper = np.full(self.n_policies, np.inf)
        for block in SHARE_BLOCKS:
            lower[self.blocks[block]] = SHARE_LO
            upper[self.blocks[block]] = SHARE_HI
        lower[self.blocks["a"]] = -A_BOUND
        upper[self.blocks["a"]] = A_BOUND
        return lower, upper


DESCRIPTION = (
    "RSS-2019 three-country trade DSGE, Epstein-Zin, Phase-1 variant "
    "(18 states / 70 policies / 76 residuals; scaffolding as constants, "
    "12 off-diagonal tariff shocks, bond clearing in definitions)"
)

__all__ = [
    "N_COUNTRIES",
    "CONSTANTS",
    "VARIANT_CONSTANTS",
    "DESCRIPTION",
    "K_SS_REFERENCE",
    "Layout",
    "POLICY_BLOCKS",
    "build_constants",
    "offdiag_pairs",
    "state_names",
    "policy_names",
    "shock_names",
]
