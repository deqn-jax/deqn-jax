"""Disaster SS-consistency probe: the certification table for spec-let 1.

For every (arm, seed) checkpoint under --runs-dir it reports the three
selection certificates the 2026-07 audits defined:

  - max |policy(SS) - policy*| / |policy*|  (level consistency at the SS)
  - zero-shock closed-loop drift from the SS (max relative state deviation
    at horizons 5/20/50/100)
  - rho(SS): spectral radius of the closed-loop state map at the SS

Plus the ELB context: the linearized Taylor-rate gap statistics, so the
gate's premise (how much of the anchor cloud sits past the floor) is
visible next to the outcome.

Usage (DGX):
  JAX_ENABLE_X64=1 uv run python scripts/disaster_ss_probe.py \
      --runs-dir runs/disaster_cert \
      --arms disaster,disaster_gated,disaster_elbcov,disaster_gated_elbcov \
      --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from deqn_jax.irf import load_policy_from_checkpoint  # noqa: E402


def probe(ckpt: str) -> dict:
    net, model = load_policy_from_checkpoint(ckpt)
    ss_state, ss_policy = model.steady_state_fn(model.constants)
    ss_state = jnp.asarray(ss_state)
    ss_policy = jnp.asarray(ss_policy)

    def pol(sb):
        p = net(sb)
        return p if p.ndim > 1 else p[None, :]

    # 1. levels at the exact SS
    rel = (pol(ss_state[None, :])[0] - ss_policy) / jnp.abs(ss_policy)
    per_policy = {n: float(r) for n, r in zip(model.policy_names, rel)}

    # 2. zero-shock closed-loop drift
    zero = jnp.zeros((1, model.n_shocks))
    s = ss_state[None, :]
    drift = {}
    for t in range(1, 101):
        s = model.step_fn(s, pol(s), zero, model.constants)
        if t in (5, 20, 50, 100):
            drift[t] = float(
                jnp.max(jnp.abs(s[0] - ss_state) / (jnp.abs(ss_state) + 1e-12))
            )

    # 3. rho(SS)
    def closed_loop(sv):
        return model.step_fn(sv[None, :], pol(sv[None, :]), zero, model.constants)[0]

    rho = float(
        jnp.max(jnp.abs(jnp.linalg.eigvals(jax.jacobian(closed_loop)(ss_state))))
    )

    return {
        "max_ss_rel_err": float(jnp.max(jnp.abs(rel))),
        "per_policy_ss_rel_err": per_policy,
        "drift": drift,
        "rho_ss": rho,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", default="runs/disaster_cert")
    ap.add_argument(
        "--arms",
        default="disaster,disaster_gated,disaster_elbcov,disaster_gated_elbcov",
    )
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--ckpt-name", default="checkpoint_best.eqx")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    runs = Path(args.runs_dir)
    rows = []
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        for seed in [int(s) for s in args.seeds.split(",")]:
            ckpt = runs / f"{arm}_s{seed}" / args.ckpt_name
            if not ckpt.exists():
                print(f"[skip] {ckpt} missing")
                continue
            r = probe(str(ckpt))
            r["arm"], r["seed"] = arm, seed
            rows.append(r)
            print(
                f"{arm:>24} s{seed}  rho(SS)={r['rho_ss']:.4f}  "
                f"max|dSS|={r['max_ss_rel_err']:.3%}  "
                f"drift@100={r['drift'][100]:.3%}"
            )

    if not rows:
        print("No checkpoints found.")
        return

    print(
        "\n| arm | median rho(SS) | pass rho<1 | median max SS err | median drift@100 |"
    )
    print("|---|---|---|---|---|")
    import numpy as np

    for arm in dict.fromkeys(r["arm"] for r in rows):
        ar = [r for r in rows if r["arm"] == arm]
        rho = np.array([r["rho_ss"] for r in ar])
        err = np.array([r["max_ss_rel_err"] for r in ar])
        dr = np.array([r["drift"][100] for r in ar])
        print(
            f"| {arm} | {np.median(rho):.4f} | {(rho < 1).sum()}/{len(rho)} "
            f"| {np.median(err):.3%} | {np.median(dr):.3%} |"
        )

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
