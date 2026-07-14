"""EWM sweep evaluation table: rho(SS) + held-out stress-grid residuals.

The reproducible evaluation path for the EWM coverage experiments
(docs/superpowers/specs/2026-06-29-ewm-coverage-sampling-design.md,
"Stress-region metric"). For every (arm, seed) checkpoint under --runs-dir
it reports:

  - rho(SS): spectral radius of the one-step closed-loop state map
    linearized at the deterministic steady state (zero shock). The
    closed-loop-dynamics stability read; <1 = locally stable.
  - per-equation mean (E[r])^2 on a FIXED held-out stress grid: the
    stress box (read from --stress-config's coverage.stress_ranges)
    sampled once with a pinned seed, evaluated directly (raw box states;
    policy-independent, identical across arms). Expectations use
    Gauss-Hermite quadrature (same operator as training).
  - the same on a fixed base grid drawn from the model's init rect
    (the on-measure cost check).

Headline stress number = max over {fb_0, fb_1, arc}. When comparing to
the paper, convert to residual (unsquared) units: factor F in (E[r])^2
is sqrt(F) in |E[r]|.

Usage:
  uv run python scripts/ewm_stress_table.py \
      --runs-dir runs/ewm_sweep \
      --arms irbc_plain,irbc_ewm,irbc,irbc_ewm_anchor \
      --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from deqn_jax.irf import load_policy_from_checkpoint
from deqn_jax.training.loss import compute_loss, gauss_hermite_nd

HEADLINE_EQS = ("fb_0", "fb_1", "arc")


def rho_ss(policy_net, model) -> float:
    """Spectral radius of the closed-loop state map s -> s' at the SS."""
    constants = model.constants
    ss_state, _ = model.steady_state_fn(constants)
    ss_state = jnp.asarray(ss_state)
    zero = jnp.zeros((1, model.n_shocks))

    def closed_loop(s):
        sb = s[None, :]
        pi = policy_net(sb)
        if pi.ndim == 1:
            pi = pi[None, :]
        return model.step_fn(sb, pi, zero, constants)[0]

    A = jax.jacobian(closed_loop)(ss_state)
    eig = jnp.linalg.eigvals(A)
    return float(jnp.max(jnp.abs(eig)))


def sample_box_grid(key, n: int, model, ranges: dict) -> jnp.ndarray:
    """Uniform draw in a per-state-name box; unlisted dims filled from SS."""
    ss_state, _ = model.steady_state_fn(model.constants)
    name_to_idx = {nm: i for i, nm in enumerate(model.state_names)}
    grid = jnp.broadcast_to(jnp.asarray(ss_state), (n, model.n_states))
    names = list(ranges.keys())
    idx = jnp.array([name_to_idx[nm] for nm in names], dtype=jnp.int32)
    lows = jnp.array([ranges[nm][0] for nm in names])
    highs = jnp.array([ranges[nm][1] for nm in names])
    u = jax.random.uniform(key, (n, len(names)), minval=lows, maxval=highs)
    return grid.at[:, idx].set(u)


def eq_table(policy_net, model, states, quad) -> dict:
    """Per-equation mean (E[r])^2 on a fixed state grid under GH quadrature."""
    qn, qw = quad
    _, eq_losses = compute_loss(
        model,
        policy_net,
        states,
        jax.random.PRNGKey(0),  # unused under quadrature
        quad_nodes=qn,
        quad_weights=qw,
    )
    return {k: float(v) for k, v in eq_losses.items() if not k.startswith("aux_")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", default="runs/ewm_sweep")
    ap.add_argument(
        "--arms",
        default="irbc_plain,irbc_ewm,irbc,irbc_ewm_anchor",
        help="comma-separated config tags; checkpoints at <runs-dir>/<arm>_s<seed>/",
    )
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--ckpt-name", default="checkpoint_004000.eqx")
    ap.add_argument("--n-grid", type=int, default=512)
    ap.add_argument(
        "--grid-seed",
        type=int,
        default=1234,
        help="pinned PRNG seed for BOTH held-out grids (shared across arms)",
    )
    ap.add_argument(
        "--stress-config",
        default="configs/irbc_ewm.yaml",
        help="YAML whose coverage.stress_ranges defines the held-out stress box",
    )
    ap.add_argument("--n-quadrature-points", type=int, default=3)
    ap.add_argument("--json-out", default=None, help="also dump rows as JSON")
    args = ap.parse_args()

    with open(args.stress_config) as f:
        stress_ranges = yaml.safe_load(f)["coverage"]["stress_ranges"]

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    runs_dir = Path(args.runs_dir)

    rows = []
    grids = None  # built once from the first loadable model (shared across arms)
    for arm in arms:
        for seed in seeds:
            ckpt = runs_dir / f"{arm}_s{seed}" / args.ckpt_name
            if not ckpt.exists():
                print(f"[skip] {ckpt} missing")
                continue
            policy_net, model = load_policy_from_checkpoint(str(ckpt))
            if grids is None:
                k_stress, k_base = jax.random.split(jax.random.PRNGKey(args.grid_seed))
                stress_grid = sample_box_grid(
                    k_stress, args.n_grid, model, stress_ranges
                )
                base_grid = model.init_state_fn(k_base, args.n_grid, model.constants)
                quad = gauss_hermite_nd(args.n_quadrature_points, model.n_shocks)
                quad = (jnp.array(quad[0]), jnp.array(quad[1]))
                grids = (stress_grid, base_grid, quad)
            stress_grid, base_grid, quad = grids

            stress_eq = eq_table(policy_net, model, stress_grid, quad)
            base_eq = eq_table(policy_net, model, base_grid, quad)
            headline = [k for k in HEADLINE_EQS if k in stress_eq] or sorted(stress_eq)
            row = {
                "arm": arm,
                "seed": seed,
                "rho_ss": rho_ss(policy_net, model),
                "stress_max_fb_arc": max(stress_eq[k] for k in headline),
                "stress_eq": stress_eq,
                "base_total": sum(base_eq.values()),
                "base_eq": base_eq,
            }
            rows.append(row)
            print(
                f"{arm:>18} s{seed}  rho(SS)={row['rho_ss']:.4f}  "
                f"stress max(fb,arc)={row['stress_max_fb_arc']:.3e}  "
                f"base total={row['base_total']:.3e}"
            )

    if not rows:
        print("No checkpoints found.")
        return

    print(
        "\n| arm | median rho(SS) | pass rho<1 | median stress max(fb_0,fb_1,arc) | median base total |"
    )
    print("|---|---|---|---|---|")
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        if not arm_rows:
            continue
        rho = np.array([r["rho_ss"] for r in arm_rows])
        stress = np.array([r["stress_max_fb_arc"] for r in arm_rows])
        base = np.array([r["base_total"] for r in arm_rows])
        print(
            f"| {arm} | {np.median(rho):.4f} [{rho.min():.3f}, {rho.max():.3f}] "
            f"| {(rho < 1).sum()}/{len(rho)} "
            f"| {np.median(stress):.3e} | {np.median(base):.3e} |"
        )

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
