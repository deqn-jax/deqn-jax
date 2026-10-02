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
    Despite its name, the reference file covers its whole 500-year path,
    repeated 20 times (identical deterministic copies). Statistics here are
    computed on one copy of each path, for 2015-2100 and for all 500 years
    (repetition shifts the interpolated extreme percentiles and the sample
    std slightly, so the raw describe file is not used);
  - an independent check of both: the perfect-foresight path of the same
    equations, solved by Newton on the whole simulated horizon (all 7 x 500
    equilibrium conditions at once, the network's policy used only as the
    continuation after the last year). The model is deterministic, so this
    path is exact up to that terminal condition, whose effect on 2015-2100 is
    reported by re-solving on a shorter horizon;
  - a figure with both paths.

Evaluation runs in float64 with the checkpoint's weights (the reference
post-processing ran in float32; the difference is far below the residuals).

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


def perfect_foresight(net, model, policies0, horizon: int, tol: float = 1e-12):
    """Newton solve of the deterministic path: policies for years 0..horizon-1.

    Unknowns are the policies of every year; states follow from the 2015 state
    by the model's transition; residuals are all equilibrium conditions, the
    last year's continuation policy coming from ``net``. Started from the
    network's own path, Newton converges in a few steps.
    """
    c = model.constants
    s0 = initial_state(c)
    zero = jnp.zeros((0,))

    def states_of(x):
        def f(s, p):
            return model.step_fn(s, p, zero, c), s

        s_end, states = jax.lax.scan(f, s0, x)
        return states, s_end

    def residuals(flat):
        x = flat.reshape(horizon, model.n_policies)
        states, s_end = states_of(x)
        nxt_s = jnp.concatenate([states[1:], s_end[None]])
        nxt_p = jnp.concatenate([x[1:], net(s_end)[None]])
        r = model.equations_fn(states, x, nxt_s, nxt_p, c)
        return jnp.stack([r[k] for k in model.equation_names], axis=1).ravel()

    res_fn, jac_fn = jax.jit(residuals), jax.jit(jax.jacfwd(residuals))
    flat = jnp.asarray(policies0[:horizon]).ravel()
    for _ in range(20):
        r = res_fn(flat)
        if float(jnp.max(jnp.abs(r))) < tol:
            break
        flat = flat - jnp.linalg.solve(jac_fn(flat), r)
    else:
        raise RuntimeError(
            f"perfect-foresight Newton did not converge: {float(jnp.max(jnp.abs(r))):.2e}"
        )
    x = flat.reshape(horizon, model.n_policies)
    return states_of(x)[0], x


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
    first = {c: np.asarray(tab[c][:n_path], dtype=np.float64) for c in tab.dtype.names}
    return {
        "path": {c: describe(v) for c, v in first.items()},
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


def rel_dev(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-12)


def max_dev(a: np.ndarray, b: np.ndarray) -> tuple:
    """(max relative deviation of a from b, year of the max)."""
    rel = rel_dev(a, b)
    i = int(np.argmax(rel))
    return float(rel[i]), FIRST_YEAR + i


def _style(ax, title: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot(ours: dict, ref: dict, pf: dict, n: int, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years = FIRST_YEAR + np.arange(n)
    ours_c, ref_c = "#eb6834", "#2a78d6"  # categorical slots 2 and 1

    fig, axes = plt.subplots(3, 4, figsize=(13, 8.2), sharex=True)
    for ax, (col, label) in zip(axes.ravel(), PLOTTED):
        ax.plot(years, ours[col][:n], color=ours_c, lw=2.5, label="deqn-jax")
        ax.plot(years, ref[col][:n], color=ref_c, lw=1.5, ls="--", label="reference")
        _style(ax, label)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "CDICE business as usual (multi-model mean), 2015-2100: deqn-jax vs reference",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "cdice_bau_replication.png", dpi=90)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharex=True)
    for ax, col in zip(axes, ("kx", "TATx", "savings_rate", "scc")):
        ax.plot(
            years,
            rel_dev(ours[col][:n], pf[col][:n]),
            color=ours_c,
            lw=2,
            label="deqn-jax",
        )
        ax.plot(years, rel_dev(ref[col][:n], pf[col][:n]), color=ref_c, lw=1.5,
                ls="--", label="reference")  # fmt: skip
        ax.set_yscale("log")
        _style(ax, dict(SERIES)[col])
    axes[0].set_ylabel("relative deviation from\nperfect-foresight path")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "cdice_bau_vs_perfect_foresight.png", dpi=90)
    plt.close(fig)


def euler_tables(resid: dict, ref_dir: Path, n_path: int, n_window: int) -> dict:
    ref_euler = load_reference_euler(ref_dir, n_path, n_window)
    ours = {
        "window": {REF_EQUATIONS[k]: describe(v[:n_window]) for k, v in resid.items()},
        "path": {REF_EQUATIONS[k]: describe(v) for k, v in resid.items()},
    }
    return {"deqn_jax": ours, "reference": ref_euler}


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
        "--short-horizon",
        type=int,
        default=300,
        help="horizon of the second perfect-foresight solve (terminal-condition check)",
    )
    args = ap.parse_args()

    ref_dir = Path(args.reference_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    net, model = load_policy_from_checkpoint(args.checkpoint)
    if model.name != "cdice_bau":
        raise ValueError(f"checkpoint is for {model.name!r}, not cdice_bau")
    # Weights as trained (fp32 for the shipped config), evaluated in fp64.
    jax.config.update("jax_enable_x64", True)
    net = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, net
    )

    ref = load_reference(ref_dir)
    n_path = len(ref["time"])  # the reference simulates 500 years
    n_window = args.last_year - FIRST_YEAR + 1
    if not 1 <= n_window <= min(n_path, args.short_horizon):
        raise ValueError(
            f"--last-year {args.last_year} needs {n_window} years; the reference "
            f"path has {n_path} and --short-horizon is {args.short_horizon}"
        )

    states, policies, next_states, next_policies = simulate(net, model, n_path)
    ours = reference_units(states, policies, model)
    resid = model.equations_fn(
        states, policies, next_states, next_policies, model.constants
    )
    resid = {k: np.abs(np.asarray(v, dtype=np.float64)) for k, v in resid.items()}

    pf_states, pf_policies = perfect_foresight(net, model, policies, n_path)
    pf = reference_units(pf_states, pf_policies, model)
    short_states, short_policies = perfect_foresight(
        net, model, policies, args.short_horizon
    )
    pf_short = reference_units(short_states, short_policies, model)

    n = n_window
    rows = []
    for col, label in SERIES:
        dev = max_dev(ours[col][:n], ref[col][:n])
        rows.append(
            {"series": col, "label": label,
             "ours_vs_ref": dev[0], "year": dev[1],
             "ours_vs_pf": max_dev(ours[col][:n], pf[col][:n])[0],
             "ref_vs_pf": max_dev(ref[col][:n], pf[col][:n])[0],
             "pf_terminal_sensitivity": max_dev(pf_short[col][:n], pf[col][:n])[0],
             "ours_last": float(ours[col][n - 1]), "ref_last": float(ref[col][n - 1])}
        )  # fmt: skip
    euler = euler_tables(resid, ref_dir, n_path, n_window)

    print(f"\nMax relative deviation, {FIRST_YEAR}-{args.last_year}")
    print(
        "PF = perfect-foresight path (Newton, all conditions on "
        f"{FIRST_YEAR}-{FIRST_YEAR + n_path - 1}); PF terminal = PF on "
        f"{args.short_horizon} years vs {n_path} years"
    )
    print(
        "| series | deqn-jax vs reference | (year) | deqn-jax vs PF | reference vs PF "
        f"| PF terminal | deqn-jax {args.last_year} | reference {args.last_year} |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['label']} | {r['ours_vs_ref']:.2e} | {r['year']} | "
            f"{r['ours_vs_pf']:.2e} | {r['ref_vs_pf']:.2e} | "
            f"{r['pf_terminal_sensitivity']:.1e} | {r['ours_last']:.6g} | "
            f"{r['ref_last']:.6g} |"
        )

    stats = ["mean", "std", "min", "0.1%", "25%", "50%", "75%", "99.9%", "max"]
    for window, title in (
        ("window", f"{FIRST_YEAR}-{args.last_year}"),
        (
            "path",
            f"{FIRST_YEAR}-{FIRST_YEAR + n_path - 1} (one copy of the reference file's path)",
        ),
    ):
        print(f"\n|Euler error|, {title}: deqn-jax / reference")
        print("| equation | " + " | ".join(stats) + " |")
        print("|---|" + "---|" * len(stats))
        for col in REF_EQUATIONS.values():
            o = euler["deqn_jax"][window][col]
            r = euler["reference"][window][col]
            cells = [f"{o[s]:.2e} / {r[s]:.2e}" for s in stats]
            print(f"| {col} | " + " | ".join(cells) + " |")

    plot(ours, ref, pf, n_window, out_dir)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {"checkpoint": str(args.checkpoint), "window": [FIRST_YEAR, args.last_year],
             "deviations": rows, "euler": euler},
            f, indent=2,
        )  # fmt: skip
    header = ",".join(c for c, _ in SERIES)
    table = np.column_stack([ours[c] for c, _ in SERIES])
    np.savetxt(out_dir / "path.csv", table, delimiter=",", header=header, comments="")
    print(f"\nwrote {out_dir}/: two figures, summary.json, path.csv")


if __name__ == "__main__":
    main()
