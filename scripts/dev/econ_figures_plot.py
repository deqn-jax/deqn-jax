"""Drawing for scripts/dev/econ_figures.py: one style, two themes.

Each figure is drawn twice, ``<name>.svg`` on the light page and
``<name>-dark.svg`` on the dark one; docs/econ/style.css uses the same tokens.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

THEMES = {
    "": {
        "ink": "#1d1c1a",
        "muted": "#6b665e",
        "grid": "#e4dfd6",
        "accent": "#1f5fa6",
        "neutral": "#8a857c",
        "cloud": "#1f5fa6",
    },
    "-dark": {
        "ink": "#e8e4dc",
        "muted": "#a19b91",
        "grid": "#33343a",
        "accent": "#4f95e6",
        "neutral": "#7d786f",
        "cloud": "#4f95e6",
    },
}
WIDE = (6.8, 3.4)


def _style(t):
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 10.5,
            "axes.edgecolor": t["muted"],
            "axes.labelcolor": t["ink"],
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": t["grid"],
            "grid.linewidth": 0.6,
            "xtick.color": t["muted"],
            "ytick.color": t["muted"],
            "xtick.labelcolor": t["ink"],
            "ytick.labelcolor": t["ink"],
            "text.color": t["ink"],
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "svg.hashsalt": "deqn-econ",
            "savefig.transparent": True,
        }
    )


def _cap_line(ax, t, L_max, label=True, x=0.99):
    ax.axhline(L_max, color=t["ink"], lw=0.8, ls=(0, (1, 2)))
    if label:
        ax.text(
            x,
            L_max,
            " labor cap",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            color=t["muted"],
            fontsize=9.5,
        )


def hero(d, t):
    fig, ax = plt.subplots(figsize=WIDE)
    ax.scatter(d["Z"], d["L"], s=3, color=t["cloud"], alpha=0.12, lw=0, rasterized=True)
    ax.plot(d["hero_Z"], d["hero_linear"], color=t["neutral"], ls=(0, (5, 3)), lw=1.8)
    ax.plot(d["hero_Z"], d["hero_global"], color=t["accent"])
    _cap_line(ax, t, d["L_max"], x=0.3)
    i = int(0.93 * len(d["hero_Z"]))
    ax.annotate(
        "first-order perturbation\n(same economy, no cap)",
        (d["hero_Z"][i], d["hero_linear"][i]),
        xytext=(-12, 8),
        textcoords="offset points",
        ha="right",
        color=t["muted"],
        fontsize=9.5,
    )
    j = int(0.3 * len(d["hero_Z"]))
    ax.annotate(
        "global solution",
        (d["hero_Z"][j], d["hero_global"][j]),
        xytext=(8, -18),
        textcoords="offset points",
        color=t["accent"],
        fontsize=9.5,
    )
    ax.set_xlabel("productivity level $Z$")
    ax.set_ylabel("hours worked $L$")
    return fig


def states(d, t):
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    k_lo, k_hi, z_lo, z_hi = d["box"]
    gk, gz = np.meshgrid(np.linspace(k_lo, k_hi, 10), np.linspace(z_lo, z_hi, 10))
    ax.scatter(gk, gz, s=10, facecolor="none", edgecolor=t["neutral"], lw=0.8)
    ax.scatter(d["k"], d["Z"], s=3, color=t["cloud"], alpha=0.25, lw=0, rasterized=True)
    ax.text(
        k_hi,
        z_hi + 0.03,
        "a 10 × 10 tensor grid on the same box",
        ha="right",
        va="bottom",
        color=t["muted"],
        fontsize=9.5,
    )
    ax.text(
        d["k"].mean(),
        d["Z"].min() - 0.06,
        "states the economy visits",
        ha="center",
        va="top",
        color=t["accent"],
        fontsize=9.5,
    )
    ax.set_xlabel("capital $K$")
    ax.set_ylabel("productivity level $Z$")
    ax.set_ylim(z_lo - 0.15, z_hi + 0.12)
    ax.grid(False)
    return fig


def labor_rule(d, t):
    fig, ax = plt.subplots(figsize=WIDE)
    names = ("low", "median", "high")
    for zl, rule, name, a in zip(d["Z_levels"], d["rule_by_z"], names, (0.45, 0.7, 1)):
        ax.plot(d["k_grid"], rule, color=t["accent"], alpha=a)
        ax.text(
            d["k_grid"][-1],
            rule[-1],
            f"  $Z={zl:.2f}$ ({name})",
            va="center",
            color=t["ink"],
            fontsize=9.5,
        )
    _cap_line(ax, t, d["L_max"], x=0.99)
    ax.set_xlim(d["k_grid"][0], d["k_grid"][-1] + 0.3 * np.ptp(d["k_grid"]))
    ax.set_xlabel("capital $K$")
    ax.set_ylabel("hours worked $L$")
    return fig


def path(d, t):
    n = 160
    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(6.8, 3.8), sharex=True, gridspec_kw={"height_ratios": [1, 1.6]}
    )
    x = np.arange(n)
    a1.plot(x, d["Z"][:n], color=t["muted"], lw=1.4)
    a1.set_ylabel("$Z_t$")
    a2.plot(
        x,
        d["L_lin_path"][:n],
        color=t["neutral"],
        ls=(0, (5, 3)),
        lw=1.4,
        label="first-order rule, no cap",
    )
    a2.plot(x, d["L"][:n], color=t["accent"], label="global solution")
    _cap_line(a2, t, d["L_max"], label=False)
    a2.set_ylabel("$L_t$")
    a2.set_xlabel("period")
    lo = min(d["L"][:n].min(), d["L_lin_path"][:n].min())
    a2.set_ylim(lo - 0.05, None)
    a2.legend(loc="lower right", fontsize=9.5, ncol=2)
    return fig


def irf(d, t):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    for sign, ls, lab in ((1, "-", "+2 s.d."), (-1, (0, (2, 1.5)), "−2 s.d.")):
        r = d["irf"][sign]
        a1.plot(r["period"], r["L"], color=t["accent"], ls=ls, label=lab)
        y = np.asarray(r["y"])
        a2.plot(r["period"], 100 * (y / y[0] - 1), color=t["accent"], ls=ls)
    rl = d["irf_lin"]
    a1.plot(
        rl["period"],
        rl["L"],
        color=t["neutral"],
        lw=1.4,
        ls=(0, (5, 3)),
        label="+2 s.d., first-order",
    )
    _cap_line(a1, t, d["L_max"], label=False)
    a1.set_title("hours worked $L$", fontsize=10.5)
    a2.set_title("output, % from steady state", fontsize=10.5)
    a2.axhline(0, color=t["muted"], lw=0.6)
    for a in (a1, a2):
        a.set_xlabel("periods after the shock")
    a1.legend(fontsize=9, loc="lower right")
    fig.tight_layout(w_pad=2)
    return fig


def euler_errors(d, t):
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    le = d["log_err"]
    ax.hist(le, bins=70, color=t["accent"], alpha=0.85, lw=0)
    for p, lab in ((0.5, "median"), (0.99, "99th pct.")):
        v = np.quantile(le, p)
        ax.axvline(v, color=t["ink"], lw=0.8, ls=(0, (1, 2)))
        ax.text(v, ax.get_ylim()[1], f" {lab} {v:.1f}", va="top", fontsize=9.5)
    ax.set_xlabel(r"$\log_{10}$ of the Euler-equation error (fraction of consumption)")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    return fig


def closed_form(cf, t):
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.semilogy(cf["episodes"], cf["max"], color=t["accent"], label="largest")
    ax.semilogy(
        cf["episodes"], cf["median"], color=t["accent"], alpha=0.5, label="median"
    )
    ax.set_xlabel("training rounds")
    ax.set_ylabel(r"$|s(K,Z) - \alpha\beta| \,/\, \alpha\beta$")
    ax.legend(fontsize=9.5)
    return fig


def plot_all(cap, cf, out):
    figs = {
        "hero": lambda t: hero(cap, t),
        "states": lambda t: states(cap, t),
        "labor_rule": lambda t: labor_rule(cap, t),
        "path": lambda t: path(cap, t),
        "irf": lambda t: irf(cap, t),
        "euler_errors": lambda t: euler_errors(cap, t),
        "closed_form": lambda t: closed_form(cf, t),
    }
    for suffix, t in THEMES.items():
        _style(t)
        for name, draw in figs.items():
            fig = draw(t)
            fig.savefig(
                out / f"{name}{suffix}.svg",
                bbox_inches="tight",
                dpi=160,
                metadata={"Date": None},
            )
            plt.close(fig)
