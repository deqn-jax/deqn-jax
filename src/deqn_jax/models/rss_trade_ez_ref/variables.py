"""Variables and calibration for the RSS three-country trade DSGE (Epstein–Zin).

Ravikumar, Santacreu & Sposi (2019), *Understanding the aggregate effects of
trade policy uncertainty*-style dynamic Eaton–Kortum economy with capital: three
countries, three sectors per country (consumption C, intermediate M, investment
X), trade in the intermediate good only, one world bond, tariff level and
tariff-volatility shocks, and Epstein–Zin households. Equation numbers cited
in ``equations.py`` refer to the maintainer's model writeup (kept in the
private notes repo).

Phase-0 **faithful replica layout** (``rss_trade_ez_ref``): the state carries,
besides the economic states, the reference solution's training scaffolding
columns — the homotopy weights ``homo``/``homo_1`` (calibration and trade-cost
continuation), the transversality gate ``A_min``, the bond mask ``a_mask`` and
the EZ discount accumulators ``U_store_i`` — because the reference checkpoint's
policy is a function of all 31 columns. They are held at their converged
values here (``homo = homo_1 = A_min = U_store = 1``, ``a_mask = 0``) and are
never advanced by the dynamics. The Phase-1 variant model drops them.

Names are the reference's without its ``_x``/``_y`` suffixes; the parity map
(private repo) matches by name. Every function in this package indexes by
``n = len(constants["L"])`` so a 2-country constants dict runs end to end.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.special import gamma as _gamma_fn

N_COUNTRIES = 3

# --------------------------------------------------------------------------
# Calibration (RSS 2019, three countries). Country order 1, 2, 3.
# --------------------------------------------------------------------------
_L = (1.484633941650391e-01, 7.983677978515625e-01, 2.237836154457182e00)
_NU_C = (0.620416139427803, 0.502346836928490, 0.575983255667071)
_NU_M = (0.372052751869639, 0.269693771251962, 0.346613943540632)
_NU_X = (0.457865020022781, 0.200131207990859, 0.313938706355752)
_A_C = (1.0, 0.767501533715523, 0.646386064808545)
_A_X = (1.0, 1.249132559225356, 0.950602408828203)
_T_M = (1.0, 0.209313433600579, 0.206053277688308)
# Iceberg trade costs d[importer][exporter]; diagonal 1.
_D = (
    (1.0, 2.552341461203714, 1.657660262697154),
    (3.860028133522097, 1.0, 2.040511654087028),
    (2.239917228103284, 2.258215578859456, 1.0),
)

# Tariff rule estimates (writeup §3, medians): rho_tau, sigma-bar, rho_sigma, eta.
RHO_TAU = 0.99
BAR_SIGMA_TAU = -6.14
RHO_SIGMA_TAU = 0.96
SIGMA_SIGMA_TAU = 0.37


def _offdiag_matrix(n: int, value: float) -> Tuple[Tuple[float, ...], ...]:
    """``value`` off the diagonal, 0 on it (own-country tariffs do not exist)."""
    return tuple(tuple(0.0 if i == j else value for j in range(n)) for i in range(n))


def build_constants(
    L=_L,
    nu_c=_NU_C,
    nu_m=_NU_M,
    nu_x=_NU_X,
    A_c=_A_C,
    A_x=_A_X,
    T_m=_T_M,
    d=_D,
) -> Dict[str, object]:
    """Constants dict for ``n = len(L)`` countries (arrays as numpy)."""
    n = len(L)
    for name, v in (
        ("nu_c", nu_c),
        ("nu_m", nu_m),
        ("nu_x", nu_x),
        ("A_c", A_c),
        ("A_x", A_x),
        ("T_m", T_m),
    ):
        if len(v) != n:
            raise ValueError(f"{name} has {len(v)} entries, expected {n}")
    d = np.asarray(d, dtype=float)
    if d.shape != (n, n):
        raise ValueError(f"d must be {n}x{n}, got {d.shape}")
    theta, eta = 4.0, 2.0
    c: Dict[str, object] = {
        "n_countries": n,
        # preferences (Epstein–Zin: risk aversion gama, IES psi), discounting
        "gama": 5.0,
        "psi": 0.5,
        "beta": 0.96,
        # technology
        "delta": 0.06,
        "lambda_": 0.76,
        "alpha": 0.33,
        "theta": theta,
        "eta": eta,
        # Eaton–Kortum price-index constant gam = Gamma(1 + (1-eta)/theta)^(1/(1-eta))
        "gam": float(_gamma_fn(1.0 + (1.0 - eta) / theta) ** (1.0 / (1.0 - eta))),
        # country arrays (index 0..n-1 = countries 1..n)
        "L": np.asarray(L, dtype=float),
        "nu_c": np.asarray(nu_c, dtype=float),
        "nu_m": np.asarray(nu_m, dtype=float),
        "nu_x": np.asarray(nu_x, dtype=float),
        "A_c": np.asarray(A_c, dtype=float),
        "A_x": np.asarray(A_x, dtype=float),
        "T_m": np.asarray(T_m, dtype=float),
        "d": d,
        # homotopy anchor for off-diagonal trade costs (near-autarky start)
        "d_average_off_diag": 100.0,
        # tariff processes, [importer, exporter] matrices; diagonal dead
        "rho_tau": np.asarray(_offdiag_matrix(n, RHO_TAU)),
        "bar_tau": np.zeros((n, n)),
        "bar_sigma_tau": np.asarray(_offdiag_matrix(n, BAR_SIGMA_TAU)),
        "rho_sigma_tau": np.asarray(_offdiag_matrix(n, RHO_SIGMA_TAU)),
        "sigma_sigma_tau": np.asarray(_offdiag_matrix(n, SIGMA_SIGMA_TAU)),
        # guards (load-bearing for residual values; mirror the reference)
        "min_K": 0.005,
        "tau_cap": 2.5,
        # transversality discount beta^(progress * episode_length): the
        # replica keeps the reference's terminal weight at episode progress 1
        "tv_progress": 1.0,
        "tv_episode_length": 256.0,
    }
    # symmetric "average" calibration the homotopy interpolates from
    for key in ("nu_c", "nu_m", "nu_x", "A_c", "A_x", "T_m", "L"):
        c["base_" + key] = float(np.mean(c[key]))
    return c


CONSTANTS = build_constants()

# --------------------------------------------------------------------------
# State / policy layout (reference order)
# --------------------------------------------------------------------------


def state_names(n: int) -> Tuple[str, ...]:
    names = []
    for i in range(1, n + 1):
        names += [f"K_{i}", f"A_{i}"]
    names.append("homo")
    names += [f"tau_{i}{j}" for i in range(1, n + 1) for j in range(1, n + 1)]
    names += [f"sigma_tau_{i}{j}" for i in range(1, n + 1) for j in range(1, n + 1)]
    names.append("A_min")
    names += [f"U_store_{i}" for i in range(1, n + 1)]
    names.append("a_mask")
    names.append("homo_1")
    return tuple(names)


# Policy blocks in the reference's order; "q" is the single world bond price.
POLICY_BLOCKS = (
    "P_C",
    "q",
    "s",
    "K",
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
    "L_C",
    "L_X",
    "L_M",
    "M_C",
    "M_M",
    "A",
    "U",
    "U_store",
    "M_X",
    "K_X",
    "mu",
)


def policy_names(n: int) -> Tuple[str, ...]:
    names = []
    for block in POLICY_BLOCKS:
        if block == "q":
            names.append("q")
        else:
            names += [f"{block}_{i}" for i in range(1, n + 1)]
    return tuple(names)


def shock_names(n: int) -> Tuple[str, ...]:
    """18 standard normals: the log-volatility innovations (i-major, all
    pairs), then the tariff-level innovations. Diagonal entries are dead
    (zero loadings) but kept so the layout matches the reference's 36-node
    monomial rule."""
    return tuple(
        [f"eps_sigma_{i}{j}" for i in range(1, n + 1) for j in range(1, n + 1)]
        + [f"eps_tau_{i}{j}" for i in range(1, n + 1) for j in range(1, n + 1)]
    )


class Layout:
    """Index tables for a given country count (built once per model)."""

    def __init__(self, n: int):
        self.n = n
        self.states = state_names(n)
        self.policies = policy_names(n)
        s = {name: k for k, name in enumerate(self.states)}
        p = {name: k for k, name in enumerate(self.policies)}
        self.s_idx, self.p_idx = s, p
        self.K = np.array([s[f"K_{i}"] for i in range(1, n + 1)])
        self.A = np.array([s[f"A_{i}"] for i in range(1, n + 1)])
        self.homo = s["homo"]
        self.homo_1 = s["homo_1"]
        self.A_min = s["A_min"]
        self.a_mask = s["a_mask"]
        self.U_store = np.array([s[f"U_store_{i}"] for i in range(1, n + 1)])
        self.tau = np.array(
            [[s[f"tau_{i}{j}"] for j in range(1, n + 1)] for i in range(1, n + 1)]
        )
        self.sigma_tau = np.array(
            [[s[f"sigma_tau_{i}{j}"] for j in range(1, n + 1)] for i in range(1, n + 1)]
        )
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
        self.n_shocks = 2 * n * n


# Hard policy clips of the reference accessor layer (applied in
# definitions/equations, not in the network): shares and q in [0, 1]; the EZ
# value and certainty equivalent bounded away from zero.
UNIT_INTERVAL_BLOCKS = (
    "s",
    "q",
    "K_C",
    "K_M",
    "K_X",
    "L_C",
    "L_M",
    "L_X",
    "M_C",
    "M_M",
    "M_X",
)
FLOOR_1E3_BLOCKS = ("U", "mu")

# Deterministic steady-state capital stocks of the calibrated three-country
# economy (per capita; used to center the initial sampler).
K_SS_REFERENCE = (0.103010021388303, 0.092479877721915, 0.349742620334530)

DESCRIPTION = (
    "RSS-2019 three-country trade DSGE, Epstein-Zin, faithful reference layout "
    "(31 states / 73 policies / 82 residuals; two-stage loss)"
)

__all__ = [
    "N_COUNTRIES",
    "CONSTANTS",
    "DESCRIPTION",
    "Layout",
    "POLICY_BLOCKS",
    "UNIT_INTERVAL_BLOCKS",
    "FLOOR_1E3_BLOCKS",
    "K_SS_REFERENCE",
    "build_constants",
    "state_names",
    "policy_names",
    "shock_names",
]
