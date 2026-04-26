"""I/O for Dynare reference artifacts (steady state, moments, IRFs, perturbation).

Dynare ships a `.mod` model + a perturbation solver; running it produces a small
set of CSV files that we treat as ground truth. This module loads those files
and applies the DEQN ↔ Dynare name mapping so downstream eval / warm-start code
can talk to either side without re-doing the parsing each time.

Files we read:
    dynare_ss.csv       — steady-state values (variable, value)
    dynare_moments.csv  — ergodic mean & std (variable, mean, std)
    dynare_ghx.csv      — perturbation policy Jacobian on lagged states
                          (rows = variables, cols = `R(-1)`, …, `mu_z(-1)`)
    dynare_ghu.csv      — perturbation policy Jacobian on shocks (rows = vars,
                          cols = e_<shock>)
    irf_e_<shock>.csv   — period × variable IRF to a 1σ shock impulse

Two name conventions:
    Dynare uses current-period names without `_lag` (e.g. `c`, `pi`, `i_var`).
    DEQN policies use shorter aliases (`i` → Dynare `i_var`); states are
    `_lag`-suffixed (`c_lag` ↔ Dynare `c(-1)` column / current `c` row).

Public:
    read_csv_matrix(path) → (col_names, rows_dict)
    load_dynare_moments(dynare_dir) → Dict[var, {mean, std}]
    load_dynare_jacobian(model, dynare_dir) → Array [n_policies, n_states]
    load_dynare_irf(dynare_dir, shock_name) → Dict[var, List[float]]
    deqn_policy_to_dynare(policy_name) → str
    deqn_state_col_to_dynare(state_name) → str | None  (m_p has no ghx column)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import jax.numpy as jnp
from jax import Array

# ---------------------------------------------------------------------------
# Name mapping
# ---------------------------------------------------------------------------

# DEQN policy name → Dynare variable name. Identity when not listed.
_POLICY_ALIASES: Dict[str, str] = {
    "i": "i_var",
}

# DEQN state name → Dynare ghx column name (lagged endogenous label).
# m_p is intentionally absent: it's an i.i.d. shock, has no ghx column, and
# enters Dynare via ghu's e_mp column instead. Callers must handle m_p
# specially (see load_dynare_jacobian).
_STATE_COL_MAP: Dict[str, str] = {
    "pi_lag": "pi(-1)",
    "k_lag": "k(-1)",
    "c_lag": "c(-1)",
    "q_lag": "q(-1)",
    "i_lag": "i_var(-1)",
    "R_lag": "R(-1)",
    "w_tilda_lag": "w_tilda(-1)",
    "L_lag": "L(-1)",
    "eps": "eps(-1)",
    "mu_ups": "mu_ups(-1)",
    "g": "g(-1)",
    "mu_z": "mu_z(-1)",
}


def deqn_policy_to_dynare(policy_name: str) -> str:
    """Map a DEQN policy name to its Dynare variable name (identity by default)."""
    return _POLICY_ALIASES.get(policy_name, policy_name)


def deqn_state_col_to_dynare(state_name: str) -> str | None:
    """Map a DEQN state name to a Dynare ghx column. None for `m_p` (handled via ghu)."""
    return _STATE_COL_MAP.get(state_name)


# ---------------------------------------------------------------------------
# Raw CSV readers
# ---------------------------------------------------------------------------


def read_csv_matrix(path: str | Path) -> Tuple[List[str], Dict[str, List[float]]]:
    """Read a Dynare-style CSV with one row per variable.

    Returns:
        ``(col_names, rows)`` where ``col_names`` is the header excluding the
        leading variable column, and ``rows`` maps each row's variable name to
        a list of floats (one per column).
    """
    path = Path(path)
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        col_names = header[1:]
        rows: Dict[str, List[float]] = {}
        for row in reader:
            if not row or not row[0]:
                continue
            rows[row[0]] = [float(x) for x in row[1:]]
    return col_names, rows


# ---------------------------------------------------------------------------
# High-level loaders
# ---------------------------------------------------------------------------


def load_dynare_moments(dynare_dir: str | Path) -> Dict[str, Dict[str, float]]:
    """Read ``dynare_moments.csv`` into ``{variable: {mean, std}}``.

    Variable names are Dynare's (e.g. ``i_var``, ``c``, ``pi``).
    """
    path = Path(dynare_dir) / "dynare_moments.csv"
    cols, rows = read_csv_matrix(path)
    if cols != ["mean", "std"]:
        raise ValueError(
            f"dynare_moments.csv has unexpected columns {cols!r}; expected ['mean', 'std']"
        )
    return {var: {"mean": vals[0], "std": vals[1]} for var, vals in rows.items()}


def load_dynare_jacobian(model, dynare_dir: str | Path) -> Array:
    """Build a ``[n_policies × n_states]`` Jacobian from Dynare's ghx + ghu.

    Mirrors the construction inside ``warm_start_from_dynare``. Each row is the
    linearized response of one DEQN policy (mapped via ``_POLICY_ALIASES``) to
    one DEQN state (mapped via ``_STATE_COL_MAP``). The exception is the
    monetary-policy shock state ``m_p`` which is i.i.d.; its column is filled
    from ``ghu``'s ``e_mp`` divided by ``model.constants["sigma_mp"]``.
    """
    dynare_path = Path(dynare_dir)
    ghx_cols, ghx_rows = read_csv_matrix(dynare_path / "dynare_ghx.csv")
    ghu_cols, ghu_rows = read_csv_matrix(dynare_path / "dynare_ghu.csv")

    n_policies = model.n_policies
    n_states = model.n_states
    J = jnp.zeros((n_policies, n_states))

    deqn_state_names = list(model.state_names)
    deqn_policy_names = list(model.policy_names)
    sigma_mp = model.constants.get("sigma_mp", None)

    for pi, pname in enumerate(deqn_policy_names):
        dvar = deqn_policy_to_dynare(pname)
        if dvar not in ghx_rows or dvar not in ghu_rows:
            available = sorted(set(ghx_rows.keys()) & set(ghu_rows.keys()))
            raise KeyError(
                f"Policy '{pname}' maps to Dynare variable '{dvar}' but that row "
                f"is absent from dynare_ghx/ghu.csv. Available rows: {available}"
            )
        ghx_row = ghx_rows[dvar]
        ghu_row = ghu_rows[dvar]

        for si, sname in enumerate(deqn_state_names):
            if sname == "m_p":
                if sigma_mp is None:
                    raise KeyError(
                        "Model constants missing 'sigma_mp' but DEQN state list "
                        "includes 'm_p'; cannot map to Dynare ghu."
                    )
                mp_col = ghu_cols.index("e_mp")
                J = J.at[pi, si].set(ghu_row[mp_col] / sigma_mp)
            else:
                col_name = deqn_state_col_to_dynare(sname)
                if col_name is None:
                    # Unknown state; leave row at 0.
                    continue
                col_idx = ghx_cols.index(col_name)
                J = J.at[pi, si].set(ghx_row[col_idx])
    return J


def load_dynare_irf(dynare_dir: str | Path, shock_name: str) -> Dict[str, List[float]]:
    """Load Dynare's IRF for a single shock.

    Looks for ``irf_e_<shock_name>.csv`` (the ``e_`` prefix is Dynare's). The
    file is period-major (rows = periods, columns = variables). Returns a dict
    keyed by Dynare variable name with a list of floats per variable.
    """
    fname = f"irf_e_{shock_name}.csv"
    path = Path(dynare_dir) / fname
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        var_names = header[1:]  # skip the period column
        series: Dict[str, List[float]] = {v: [] for v in var_names}
        for row in reader:
            if not row:
                continue
            for j, v in enumerate(var_names):
                series[v].append(float(row[j + 1]))
    return series
