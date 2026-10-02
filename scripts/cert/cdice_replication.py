"""CDICE business-as-usual: compare a trained checkpoint with the reference solution.

Loads a ``cdice_bau`` checkpoint, simulates the deterministic path from the
2015 state, converts it to the units of the reference post-processing
(``post_process_baseline.py`` of the replication package of Folini, Friedl,
Kübler & Scheidegger 2024) and compares it with the stored reference solution
(``states.csv``, ``ps.csv``, ``defs.csv``, ``exoparams.csv``):

  - max relative deviation per series over the comparison window
    (default 2015-2100), with the year where it occurs;
  - Euler-equation error statistics in the form of the reference's
    ``simulated_euler_discrepancies_describe_2015-2100.csv``, side by side.
    Note the reference file is computed over its whole 500-year simulated
    path (20 identical deterministic copies) despite its name; both that
    window and 2015-2100 are reported;
  - a figure with both paths.

Units (reference post-processing): capital, output, consumption, investment
and damages in trillions of 2010 USD; carbon in GtC; emissions in GtCO2 per
year; the social cost of carbon in USD per tCO2; shadow prices normalized.

Usage:
  uv run python scripts/cert/cdice_replication.py <run>/checkpoint_000150.eqx \\
      --reference-dir <clone>/DEQN_for_IAMs/gdice_baseline/bau_results/BAU_cdice \\
      --out-dir cdice_replication
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from deqn_jax.models.cdice_bau import exogenous as ex
from deqn_jax.models.cdice_bau.initial_state import initial_state
from deqn_jax.training.checkpointing import load_policy_from_checkpoint

REF_EQUATIONS = {  # ours -> reference column
    "foc_k": "foc_kplus",
    "budget": "foc_lambd",
    "foc_tat": "foc_TATplus",
    "foc_mat": "foc_MATplus",
    "foc_muo": "foc_MUOplus",
    "foc_mlo": "foc_MLOplus",
    "foc_toc": "foc_TOCplus",
}
PERCENTILES = (0.1, 25.0, 50.0, 75.0, 99.9)
FIRST_YEAR = 2015


def simulate(net, model, n_years: int):
    """Deterministic path from 2015: states, policies, next states, next policies."""
    c = model.constants
    zero = jnp.zeros((1, 0))
    s = initial_state(c)[None, :]
    rows = []
    for _ in range(n_years):
        rows.append(s[0])
        s = model.step_fn(s, net(s), zero, c)
    states = jnp.stack(rows)
    policies = net(states)
    next_states = model.step_fn(states, policies, jnp.zeros((n_years, 0)), c)
    return states, policies, next_states, net(next_states)


def reference_units(states, policies, model) -> dict:
    """Our path in the reference post-processing units, reference column names."""
    c = model.constants
    d = model.definitions_fn(states, policies, c)
    scale = d["effective_labor"]
    co2 = 1000.0 * c["c2co2"]  # 1000 GtC -> GtCO2
    g = ex.growth_factor(d["t"], c)
    out = {
        # states.csv
        "kx": states[:, 0] * scale,
        "MATx": states[:, 1] * 1000.0,
        "MUOx": states[:, 2] * 1000.0,
        "MLOx": states[:, 3] * 1000.0,
        "TATx": states[:, 4],
        "TOCx": states[:, 5],
        # ps.csv
        "kplusy": policies[:, 0] * g * scale,
        "lambd_haty": policies[:, 1],
        "nuAT_haty": policies[:, 2],
        "nuUO_haty": policies[:, 3],
        "nuLO_haty": policies[:, 4],
        "etaAT_haty": policies[:, 5],
        "etaOC_haty": policies[:, 6],
        # defs.csv
        "con": d["con"] * scale,
        "Omega": d["omega"],
        "ygross": d["ygross"] * scale,
        "ynet": d["ynet"] * scale,
        "inv": d["inv"] * scale,
        "savings_rate": d["savings_rate"],
        "Eind": d["e_ind"] * co2,
        "Emissions": d["emissions"] * co2,
        "scc": d["scc"] / c["c2co2"],
        "Dam": d["damages"] * scale,
        # exoparams.csv
        "tfp": (1000.0 * d["tfp"]) ** (1.0 - c["alpha"]),
        "lab": d["lab"],
        "sigma": d["sigma"] * co2,
        "Eland": d["eland"] * co2,
        "Fex": d["fex"],
        "beta_hat": d["beta_hat"],
    }
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def load_reference(ref_dir: Path) -> dict:
    cols = {}
    for name in ("states", "ps", "defs", "exoparams", "time"):
        tab = np.genfromtxt(ref_dir / f"{name}.csv", delimiter=",", names=True)
        for col in tab.dtype.names:
            cols.setdefault(col, np.asarray(tab[col], dtype=np.float64))
    cols["savings_rate"] = cols["inv"] / cols["ynet"]
    years = cols["time"]
    if np.max(np.abs(years - np.arange(len(years)))) > 0.01:
        raise ValueError("reference time.csv is not one row per year from 2015")
    return cols


def describe(x: np.ndarray) -> dict:
    """pandas-style describe (count, mean, std ddof=1, min, percentiles, max)."""
    out = {
        "count": float(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)),
    }
    out["min"] = float(np.min(x))
    for q in PERCENTILES:
        out[f"{q:g}%"] = float(np.percentile(x, q))
    out["max"] = float(np.max(x))
    return out


def load_reference_euler(ref_dir: Path, n_path: int, n_window: int) -> dict:
    tab = np.genfromtxt(
        ref_dir / "simulated_euler_discrepancies_2015-2100.csv",
        delimiter=",",
        names=True,
    )
    full = {c: np.asarray(tab[c], dtype=np.float64) for c in tab.dtype.names}
    first = {c: v[:n_path] for c, v in full.items()}  # one copy of the path
    return {
        "file_all_rows": {c: describe(v) for c, v in full.items()},
        "window": {c: describe(v[:n_window]) for c, v in first.items()},
    }


SERIES = [  # (column, label) — the key series first
    ("kx", "capital K [tn USD]"),
    ("MATx", "carbon, atmosphere [GtC]"),
    ("MUOx", "carbon, upper ocean [GtC]"),
    ("MLOx", "carbon, lower ocean [GtC]"),
    ("TATx", "temperature, atmosphere [C]"),
    ("TOCx", "temperature, ocean [C]"),
    ("con", "consumption [tn USD]"),
    ("savings_rate", "savings rate I/Y_net"),
    ("Emissions", "emissions [GtCO2/yr]"),
    ("scc", "social cost of carbon [USD/tCO2]"),
    ("ygross", "gross output [tn USD]"),
    ("Omega", "damage share"),
    ("kplusy", "next capital [tn USD]"),
    ("inv", "investment [tn USD]"),
    ("Dam", "damages [tn USD]"),
    ("Eind", "industrial emissions [GtCO2/yr]"),
    ("lambd_haty", "shadow price: consumption"),
    ("nuAT_haty", "shadow price: -M_AT"),
    ("nuUO_haty", "shadow price: M_UO"),
    ("nuLO_haty", "shadow price: M_LO"),
    ("etaAT_haty", "shadow price: T_AT"),
    ("etaOC_haty", "shadow price: T_OC"),
    ("tfp", "TFP (DICE units)"),
    ("lab", "population [mn]"),
    ("sigma", "carbon intensity [tCO2/1000 USD]"),
    ("Eland", "land emissions [GtCO2/yr]"),
    ("Fex", "non-CO2 forcing [W/m2]"),
    ("beta_hat", "effective discount factor"),
]
PLOTTED = SERIES[:12]


def deviations(ours: dict, ref: dict, n: int) -> list:
    rows = []
    for col, label in SERIES:
        a, b = ours[col][:n], ref[col][:n]
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-12)
        i = int(np.argmax(rel))
        rows.append(
            {"series": col, "label": label, "max_rel_dev": float(rel[i]),
             "year": FIRST_YEAR + i, "ours": float(a[i]), "reference": float(b[i])}
        )  # fmt: skip
    return rows


def plot(ours: dict, ref: dict, n: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years = FIRST_YEAR + np.arange(n)
    fig, axes = plt.subplots(3, 4, figsize=(13, 8.2), sharex=True)
    for ax, (col, label) in zip(axes.ravel(), PLOTTED):
        ax.plot(
            years, ref[col][:n], color="#2a78d6", lw=2, ls="--", label="reference (TF)"
        )
        ax.plot(years, ours[col][:n], color="#eb6834", lw=2, label="deqn-jax")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "CDICE business as usual (multi-model mean), 2015-2100: deqn-jax vs reference",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "checkpoint", help="cdice_bau checkpoint (.eqx) with config.yaml beside it"
    )
    ap.add_argument(
        "--reference-dir",
        required=True,
        help="BAU_cdice directory of the cloned reference (states.csv, ps.csv, ...)",
    )
    ap.add_argument("--out-dir", default="cdice_replication")
    ap.add_argument("--last-year", type=int, default=2100)
    ap.add_argument(
        "--fp64",
        action="store_true",
        help="evaluate in float64 (default: the checkpoint's precision, as the reference)",
    )
    args = ap.parse_args()

    ref_dir = Path(args.reference_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The checkpoint loads in its training precision (fp32 for the shipped
    # config, as the reference). --fp64 then evaluates the same weights in fp64.
    net, model = load_policy_from_checkpoint(args.checkpoint)
    if args.fp64:
        jax.config.update("jax_enable_x64", True)
        net = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, net
        )
    if model.name != "cdice_bau":
        raise ValueError(f"checkpoint is for {model.name!r}, not cdice_bau")
    ref = load_reference(ref_dir)
    n_path = len(ref["time"])  # the reference simulates 500 years
    n_window = args.last_year - FIRST_YEAR + 1

    states, policies, next_states, next_policies = simulate(net, model, n_path)
    ours = reference_units(states, policies, model)
    resid = model.equations_fn(
        states, policies, next_states, next_policies, model.constants
    )
    resid = {k: np.abs(np.asarray(v, dtype=np.float64)) for k, v in resid.items()}

    dev = deviations(ours, ref, n_window)
    ref_euler = load_reference_euler(ref_dir, n_path, n_window)
    euler = {
        "path": {REF_EQUATIONS[k]: describe(v) for k, v in resid.items()},
        "window": {REF_EQUATIONS[k]: describe(v[:n_window]) for k, v in resid.items()},
    }

    print(
        f"\nMax relative deviation, {FIRST_YEAR}-{args.last_year} (deqn-jax vs reference)"
    )
    print("| series | max rel. dev. | year | deqn-jax | reference |")
    print("|---|---|---|---|---|")
    for r in dev:
        print(
            f"| {r['label']} | {r['max_rel_dev']:.2e} | {r['year']} | "
            f"{r['ours']:.6g} | {r['reference']:.6g} |"
        )

    for window, title, ref_key in (
        ("window", f"{FIRST_YEAR}-{args.last_year}", "window"),
        ("path", f"{FIRST_YEAR}-{FIRST_YEAR + n_path - 1} (the reference file's sample)",
         "file_all_rows"),
    ):  # fmt: skip
        print(f"\n|Euler error|, {title}: deqn-jax / reference")
        stats = ["mean", "std", "min", "0.1%", "25%", "50%", "75%", "99.9%", "max"]
        print("| equation | " + " | ".join(stats) + " |")
        print("|---|" + "---|" * len(stats))
        for col in REF_EQUATIONS.values():
            o, r = euler[window][col], ref_euler[ref_key][col]
            cells = [f"{o[s]:.2e} / {r[s]:.2e}" for s in stats]
            print(f"| {col} | " + " | ".join(cells) + " |")

    plot(ours, ref, n_window, out_dir / "cdice_bau_replication.png")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {"checkpoint": str(args.checkpoint), "window": [FIRST_YEAR, args.last_year],
             "deviations": dev, "euler": euler, "reference_euler": ref_euler},
            f, indent=2,
        )  # fmt: skip
    header = ",".join(c for c, _ in SERIES)
    table = np.column_stack([ours[c] for c, _ in SERIES])
    np.savetxt(out_dir / "path.csv", table, delimiter=",", header=header, comments="")
    print(f"\nwrote {out_dir}/cdice_bau_replication.png, summary.json, path.csv")


if __name__ == "__main__":
    main()
