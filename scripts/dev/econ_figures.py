"""Figures and numbers for the economist pages (docs/econ/).

Reads trained checkpoints, trains nothing, and writes every result figure of
docs/econ/index.html in a light and a dark variant (``<name>.svg`` and
``<name>-dark.svg``), plus ``numbers.json`` with every number the page quotes.

Reproduce (from the repository root):

    uv run deqn-jax train --config configs/bm_labor_cap.yaml
    uv run deqn-jax train --config configs/brock_mirman_closed_form.yaml
    uv run python scripts/dev/econ_figures.py \\
        --cap-checkpoint runs/econ/bm_labor_cap/checkpoint_020000.eqx \\
        --closed-form-dir runs/econ/brock_mirman_closed_form

Figures, and the run each one reads:

    hero, states, labor_rule, path, irf, euler_errors   bm_labor_cap (final)
    closed_form                                          brock_mirman_closed_form
                                                         (every saved checkpoint)

The first-order rule drawn next to the global one is the in-framework
Blanchard-Kahn linearization of ``bm_labor`` (the same economy without the cap).
"""

import argparse
import json
import re
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yaml
from econ_figures_plot import (  # noqa: I001  (sibling module in scripts/dev)
    plot_all,
)

from deqn_jax.api import (
    euler_equation_errors,
    load_model,
    load_policy_from_checkpoint,
    run_irf,
)
from deqn_jax.training.linearize import linearize_model

N_PERIODS = 10_500
BURN_IN = 500
SHOCK_SD = 2.0


def load_run(checkpoint):
    """Policy and model of a run, with the run's calibration overrides applied.

    ``load_policy_from_checkpoint`` rebuilds the model from its registered
    calibration only, so the run's ``constants:`` block is merged here.
    """
    net, model = load_policy_from_checkpoint(str(checkpoint))
    cfg = yaml.safe_load((Path(checkpoint).parent / "config.yaml").read_text())
    model = model._replace(
        constants={**model.constants, **(cfg.get("constants") or {})}
    )
    return net, model


def linear_labor_rule():
    """First-order rule of ``bm_labor`` (no cap), as a batched policy function."""
    model = load_model("bm_labor")
    P, _ = linearize_model(model, verbose=False)
    ss_s, ss_p = model.steady_state_fn(model.constants)

    def policy(states):
        return ss_p[None, :] + (jnp.atleast_2d(states) - ss_s[None, :]) @ P.T

    return policy


def cap_results(checkpoint):
    net, model = load_run(checkpoint)
    c = model.constants
    lin = linear_labor_rule()
    ee = euler_equation_errors(
        net, model, n_periods=N_PERIODS, burn_in=BURN_IN, n_quadrature_points=16
    )
    states = ee["states"]
    pol = net(states)
    defs = model.definitions_fn(states, pol, c)
    e = ee["residuals"][:, 0]
    # Consumption-equivalent Euler error: u'(c) - e = beta E[u'(c') R'], so the
    # consumption that would satisfy the Euler equation exactly is 1/(u'(c) - e).
    rel = np.asarray(e / (defs["u_c"] - e))
    log_err = np.log10(np.maximum(np.abs(rel), 1e-12))
    L = np.asarray(pol[:, 1])
    at_cap = L >= c["L_max"] - 1e-3
    k, z = np.asarray(states[:, 0]), np.asarray(states[:, 1])

    k_bar = float(k.mean())
    z_grid = np.linspace(z.min(), z.max(), 200)
    grid = jnp.stack([jnp.full_like(z_grid, k_bar), z_grid], axis=1)
    k_grid = np.linspace(np.quantile(k, 0.005), np.quantile(k, 0.995), 200)
    z_levels = np.quantile(z, [0.1, 0.5, 0.9])
    rule_by_z = [
        np.asarray(net(jnp.stack([k_grid, np.full_like(k_grid, zl)], axis=1))[:, 1])
        for zl in z_levels
    ]

    irf = {}
    for sign in (1, -1):
        irf[sign] = run_irf(net, model, "eps_z", sign * SHOCK_SD, horizon=30)
    irf_lin = run_irf(lin, model, "eps_z", SHOCK_SD, horizon=30)
    labor_ss = float(model.steady_state_fn(c)[1][1])

    q = lambda p: float(np.quantile(log_err, p))  # noqa: E731
    numbers = {
        "L_max": c["L_max"],
        "L_ss": labor_ss,
        "share_at_cap": float(at_cap.mean()),
        "n_periods": int(len(rel)),
        "euler_log10_median": q(0.5),
        "euler_log10_p99": q(0.99),
        "euler_log10_max": float(log_err.max()),
        "euler_log10_median_at_cap": float(np.median(log_err[at_cap])),
        "euler_log10_median_interior": float(np.median(log_err[~at_cap])),
        "labor_residual_abs_max": float(
            np.abs(np.asarray(ee["residuals"][:, 1])).max()
        ),
        "linear_labor_max_on_path": float(np.asarray(lin(states)[:, 1]).max()),
        "linear_share_above_cap": float(
            (np.asarray(lin(states)[:, 1]) > c["L_max"]).mean()
        ),
        "k_bar": k_bar,
        "shock_sd": SHOCK_SD,
    }
    data = {
        "k": k,
        "Z": np.exp(z),
        "L": L,
        "L_lin_path": np.asarray(lin(states)[:, 1]),
        "L_max": c["L_max"],
        "hero_Z": np.exp(z_grid),
        "hero_global": np.asarray(net(grid)[:, 1]),
        "hero_linear": np.asarray(lin(grid)[:, 1]),
        "k_grid": k_grid,
        "Z_levels": np.exp(z_levels),
        "rule_by_z": rule_by_z,
        "irf": irf,
        "irf_lin": irf_lin,
        "log_err": log_err,
        "box": (0.9, 12.0, 0.7, 1.3),  # bm_labor init_state_fn sampling box
    }
    return data, numbers


def closed_form_results(run_dir):
    """Distance of each saved rule from alpha*beta on the exact ergodic path."""
    ckpts = sorted(Path(run_dir).glob("checkpoint_[0-9]*.eqx"))
    _, model = load_run(ckpts[-1])
    c = model.constants
    s_star = c["alpha"] * c["beta"]
    rng = np.random.default_rng(0)
    states, s = [], model.steady_state_fn(c)[0][None, :]
    exact = jnp.full((1, 1), s_star)
    for t in range(5_500):
        s = model.step_fn(s, exact, jnp.asarray(rng.standard_normal((1, 1))), c)
        if t >= 500:
            states.append(s)
    states = jnp.concatenate(states)
    episodes, med, worst = [], [], []
    for ck in ckpts:
        net, _ = load_run(ck)
        err = np.abs(np.asarray(net(states)[:, 0]) - s_star) / s_star
        episodes.append(int(re.findall(r"\d+", ck.stem)[-1]))
        med.append(float(np.median(err)))
        worst.append(float(err.max()))
    numbers = {
        "s_star": s_star,
        "final_episode": episodes[-1],
        "final_rel_err_median": med[-1],
        "final_rel_err_max": worst[-1],
        "first_episode": episodes[0],
        "first_rel_err_max": worst[0],
    }
    return {"episodes": episodes, "median": med, "max": worst}, numbers


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cap-checkpoint", required=True)
    ap.add_argument("--closed-form-dir", required=True)
    ap.add_argument("--out", default="docs/econ/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cap, cap_numbers = cap_results(args.cap_checkpoint)
    cf, cf_numbers = closed_form_results(args.closed_form_dir)
    plot_all(cap, cf, out)
    numbers = {"bm_labor_cap": cap_numbers, "brock_mirman_closed_form": cf_numbers}
    (out / "numbers.json").write_text(json.dumps(numbers, indent=2) + "\n")
    print(json.dumps(numbers, indent=2))


if __name__ == "__main__":
    main()
