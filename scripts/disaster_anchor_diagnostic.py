"""Anchor/selection diagnostic at a trained composite-loss checkpoint.

Tests the hypothesis that the disaster training stall is the composite
loss's anchor doing its selection job too well: the net converges to the
LINEARIZED policy (the anchor target), and the stall level is the
linearization error, with base-equation gradients and auxiliary
gradients locked in a tug-of-war.

Three measurements at a fixed checkpoint (all fp64, GH quadrature at the
operating q so there is no estimator noise):

A. Equilibrium loss of the trained net vs the PURE linearized policy on
   the net's own ergodic states. If they match per equation, the net has
   converged to the anchor and the stall is the linearization floor.
B. Policy distance net-vs-linear on ergodic states, per policy variable
   (normalized by SS levels). Quantifies "the net IS the linear policy."
C. Gradient tug-of-war at the checkpoint: cosine and norm ratio between
   the base-equation gradient and each auxiliary gradient (anchor, jac,
   model aux = barriers+newton, and their weighted total). cos ~ -1 with
   norm ratio ~ 1 means the stall is a balance point, not a minimum of
   the base loss.

Usage:
    uv run python scripts/disaster_anchor_diagnostic.py \
        --checkpoint checkpoints/disaster_fresh_dgx [--out out.json]
"""

import argparse
import json
from pathlib import Path

import jax

# fp64 everywhere; must precede deqn_jax imports (see aio_bias_floor_diagnostic).
jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from deqn_jax.irf import load_policy_from_checkpoint  # noqa: E402
from deqn_jax.training.checkpointing import best_checkpoint_path  # noqa: E402
from deqn_jax.training.composite_loss import (  # noqa: E402
    make_composite_loss,
    prepare_composite_data,
)
from deqn_jax.training.linearize import linearize_model  # noqa: E402
from deqn_jax.training.loss import compute_loss, gauss_hermite_nd  # noqa: E402


def resolve_checkpoint(path_str):
    p = Path(path_str)
    return best_checkpoint_path(str(p)) if p.is_dir() else str(p)


def simulate_ergodic_states(net, model, n_states=512, sim_T=2000, burn=500, seed=0):
    n_paths = 32
    ss_state, _ = model.steady_state_fn(model.constants)
    state0 = jnp.tile(ss_state[None, :], (n_paths, 1))
    shocks = jax.random.normal(
        jax.random.PRNGKey(seed), (sim_T, n_paths, model.n_shocks)
    )
    clip = model.clip_state_fn or (lambda s: s)

    def step(state, shock):
        nxt = clip(model.step_fn(state, net(state), shock, model.constants))
        return nxt, nxt

    _, traj = jax.lax.scan(step, state0, shocks)
    pool = traj[burn:].reshape(-1, traj.shape[-1])
    idx = jax.random.choice(
        jax.random.PRNGKey(seed + 1), pool.shape[0], (n_states,), replace=False
    )
    return pool[idx]


def flat_grad(g):
    leaves = [x.ravel() for x in jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_array))]
    return jnp.concatenate(leaves)


def cos(a, b):
    return float(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = resolve_checkpoint(args.checkpoint)
    print(f"=== Anchor/selection diagnostic at {ckpt} ===")
    net, model = load_policy_from_checkpoint(ckpt)
    cfg = yaml.safe_load(open(Path(ckpt).parent / "config.yaml"))
    if cfg.get("constants"):
        model = model._replace(constants={**model.constants, **cfg["constants"]})
    comp_cfg = cfg.get("composite_loss", {})
    q = int(cfg.get("n_quadrature_points", 3))
    quad = gauss_hermite_nd(q, model.n_shocks)
    quad_nodes, quad_weights = jnp.array(quad[0]), jnp.array(quad[1])

    # Linearization + the EXACT anchor set used in training (same seed).
    P, Q = linearize_model(model, verbose=False)
    data = prepare_composite_data(
        model,
        P,
        Q,
        n_anchor_points=int(comp_cfg.get("n_anchor_points", 64)),
        anchor_sigma=float(comp_cfg.get("anchor_sigma", 1.0)),
        seed=int(cfg.get("seed", 12345)),
        verbose=False,
    )
    ss_state, ss_policy = (
        data.ss_state,
        jnp.asarray(model.steady_state_fn(model.constants)[1]),
    )

    def lin_policy_fn(states):
        return ss_policy[None, :] + (states - ss_state[None, :]) @ data.P.T

    states = simulate_ergodic_states(net, model, seed=0)
    print(f"  ergodic states: {states.shape[0]}  (GH{q}^{model.n_shocks} base loss)")

    results = {"checkpoint": ckpt}

    # --- A. net vs linear policy: equilibrium loss ---
    def base_losses(policy_fn):
        total, eq = compute_loss(
            model,
            policy_fn,
            states,
            jax.random.PRNGKey(0),
            quad_nodes=quad_nodes,
            quad_weights=quad_weights,
        )
        return {**{k: float(v) for k, v in eq.items()}, "TOTAL": float(total)}

    L_net = base_losses(net)
    L_lin = base_losses(lin_policy_fn)
    print("\n--- A. Equilibrium loss: trained net vs linearized policy ---")
    print(f"  {'eq':<26}{'net':>12}{'linear':>12}{'lin/net':>9}")
    for k in L_net:
        r = L_lin[k] / L_net[k] if L_net[k] != 0 else float("inf")
        print(f"  {k:<26}{L_net[k]:>12.3e}{L_lin[k]:>12.3e}{r:>9.2f}")
    results["loss_net"] = L_net
    results["loss_linear"] = L_lin

    # --- B. policy distance on ergodic states ---
    p_net = np.asarray(net(states))
    p_lin = np.asarray(lin_policy_fn(states))
    names = model.policy_names or tuple(f"p{i}" for i in range(p_net.shape[1]))
    print("\n--- B. Policy distance |net - linear| on ergodic states ---")
    print(f"  {'policy':<14}{'ss':>9}{'mean|d|':>11}{'max|d|':>11}{'mean rel':>10}")
    dist = {}
    for i, nm in enumerate(names):
        d = np.abs(p_net[:, i] - p_lin[:, i])
        ss = float(ss_policy[i])
        rel = float(d.mean() / (abs(ss) + 1e-12))
        print(f"  {nm:<14}{ss:>9.4f}{d.mean():>11.3e}{d.max():>11.3e}{rel:>10.2%}")
        dist[nm] = dict(
            ss=ss, mean_abs=float(d.mean()), max_abs=float(d.max()), mean_rel=rel
        )
    results["policy_distance"] = dist

    # --- C. gradient tug-of-war ---
    w_anchor = float(comp_cfg.get("anchor_weight", 1.0))
    w_jac = float(comp_cfg.get("jac_weight", 0.1))
    composite_fn = make_composite_loss(
        model,
        data,
        anchor_weight=w_anchor,
        jac_weight=w_jac,
        jac_anchor_weight=float(comp_cfg.get("jac_anchor_weight", 0.0)),
        barrier_weight=float(comp_cfg.get("barrier_weight", 0.01)),
        newton_weight=float(comp_cfg.get("newton_weight", 0.01)),
        leverage_mult=float(comp_cfg.get("leverage_mult", 1.0)),
        aux_decay_floor=float(comp_cfg.get("aux_decay_floor", 0.2)),
        history_len=1,
    )

    def g_of(fn):
        return flat_grad(eqx.filter_grad(fn)(net))

    g_base = g_of(
        lambda p: compute_loss(
            model,
            p,
            states,
            jax.random.PRNGKey(0),
            quad_nodes=quad_nodes,
            quad_weights=quad_weights,
        )[0]
    )
    g_anchor = g_of(
        lambda p: (
            w_anchor * jnp.mean((p(data.anchor_points) - data.anchor_lin_policy) ** 2)
        )
    )

    def jac_term(p):
        J = jax.jacfwd(lambda s: p(s[None, :])[0])(data.ss_state)
        return w_jac * jnp.mean((J - data.P) ** 2)

    g_jac = g_of(jac_term)
    g_comp = g_of(
        lambda p: composite_fn(
            model,
            p,
            states,
            jax.random.PRNGKey(0),
            quad_nodes=quad_nodes,
            quad_weights=quad_weights,
        )[0]
    )
    g_aux_total = g_comp - g_base  # all aux terms (anchor+jac+barriers+newton)

    nb = float(jnp.linalg.norm(g_base))
    print("\n--- C. Gradient tug-of-war at the checkpoint ---")
    rows = {
        "anchor (w=%.3g)" % w_anchor: g_anchor,
        "jac (w=%.3g)" % w_jac: g_jac,
        "aux total (comp - base)": g_aux_total,
        "composite total": g_comp,
    }
    print(f"  ||g_base|| = {nb:.3e}")
    print(f"  {'term':<26}{'cos(g_base, .)':>15}{'||.||/||g_base||':>18}")
    results["grads"] = {"g_base_norm": nb}
    for nm, g in rows.items():
        c = cos(g_base, g)
        r = float(jnp.linalg.norm(g)) / nb
        print(f"  {nm:<26}{c:>15.3f}{r:>18.3f}")
        results["grads"][nm] = dict(cos=c, norm_ratio=r)

    out = args.out or str(Path(ckpt).parent / "anchor_diagnostic.json")
    Path(out).write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
