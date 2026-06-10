"""Expectation-estimator diagnostic at a trained checkpoint (no ground truth).

Measures, at a FIXED trained policy on its own ergodic states, the
per-equation base losses under every expectation estimator the trainer
supports:

- Gauss-Hermite quadrature at q in {2,3,4,5} points/dim — deterministic;
  drift across q reveals quadrature truncation error at the operating
  q (disaster trains at q=3 -> 243 nodes).
- MC mse at N in {2,4,8,16,32} — mean over keys; the excess over the
  unbiased reference is the bias floor Var(rbar)/N.
- MC aio at N in {4,8,16,32} — unbiased for (E[r])^2 (see
  docs/dev/aio_loss_estimator.md); the largest-N aio mean is the
  unbiased reference value.

Key outputs:
  bias_ratio(N)  = (mse_mean(N) - aio_ref) / aio_ref   per equation
  quad_drift(q)  = GH_q - aio_ref                       per equation

If bias_ratio at the operating estimator is O(1), the training loss is
bias-dominated at that point and the optimizer is mostly minimizing
shock-variance, not equilibrium error.

Usage:
    uv run python scripts/aio_bias_floor_diagnostic.py \
        --checkpoint checkpoints/disaster_fresh_20260610 \
        [--n-states 512] [--n-keys 2000] [--sim-T 2000]
"""

import argparse
import json
from pathlib import Path

import jax

# Must happen before any deqn_jax import: the checkpoint loader's own fp64
# toggle fires too late once module-level jnp constants exist as float32,
# and eqx.tree_deserialise_leaves then fails on the dtype mismatch.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from deqn_jax.irf import load_policy_from_checkpoint  # noqa: E402
from deqn_jax.training.checkpointing import best_checkpoint_path  # noqa: E402
from deqn_jax.training.loss import compute_loss, gauss_hermite_nd  # noqa: E402

QUAD_POINTS = (2, 3, 4, 5)
MSE_NS = (2, 4, 8, 16, 32)
AIO_NS = (4, 8, 16, 32)
KEY_CHUNK = 100  # keys per vmap chunk (memory control)


def resolve_checkpoint(path_str):
    p = Path(path_str)
    if p.is_dir():
        return best_checkpoint_path(str(p))
    return str(p)


def load(checkpoint):
    net, model = load_policy_from_checkpoint(checkpoint)
    cfg_path = Path(checkpoint).parent / "config.yaml"
    cfg = yaml.safe_load(open(cfg_path)) if cfg_path.exists() else {}
    overrides = cfg.get("constants") or {}
    if overrides:
        model = model._replace(constants={**model.constants, **overrides})
        print(f"  applied constants overrides from config: {overrides}")
    return net, model, cfg


def simulate_ergodic_states(net, model, n_states, sim_T, burn, seed=0):
    """Batched ergodic rollout under the trained policy via lax.scan."""
    n_paths = 32
    ss_state, _ = model.steady_state_fn(model.constants)
    state0 = jnp.tile(ss_state[None, :], (n_paths, 1))
    shocks = jax.random.normal(
        jax.random.PRNGKey(seed), (sim_T, n_paths, model.n_shocks)
    )
    clip = model.clip_state_fn or (lambda s: s)

    def step(state, shock):
        nxt = model.step_fn(state, net(state), shock, model.constants)
        nxt = clip(nxt)
        return nxt, nxt

    _, traj = jax.lax.scan(step, state0, shocks)  # [T, n_paths, n_states]
    pool = traj[burn:].reshape(-1, traj.shape[-1])
    idx = jax.random.choice(
        jax.random.PRNGKey(seed + 1), pool.shape[0], (n_states,), replace=False
    )
    states = pool[idx]
    if not bool(jnp.all(jnp.isfinite(states))):
        raise RuntimeError("non-finite states in ergodic simulation")
    return states


def quad_losses(net, model, states, q):
    quad = gauss_hermite_nd(q, model.n_shocks)
    if quad is None:
        return None
    total, eq = compute_loss(
        model,
        net,
        states,
        jax.random.PRNGKey(0),
        quad_nodes=jnp.array(quad[0]),
        quad_weights=jnp.array(quad[1]),
    )
    return {k: float(v) for k, v in {**eq, "TOTAL": total}.items()}


def mc_losses(net, model, states, loss_choice, N, n_keys, seed):
    """Mean and SE per equation over n_keys MC realizations (chunked vmap)."""
    keys = jax.random.split(jax.random.PRNGKey(seed), n_keys)

    @jax.jit
    def chunk_eval(kchunk):
        def one(k):
            total, eq = compute_loss(
                model, net, states, k, mc_samples=N, loss_choice=loss_choice
            )
            return {**eq, "TOTAL": total}

        return jax.vmap(one)(kchunk)

    acc = None
    for i in range(0, n_keys, KEY_CHUNK):
        out = chunk_eval(keys[i : i + KEY_CHUNK])
        out = {k: np.asarray(v) for k, v in out.items()}
        acc = out if acc is None else {k: np.concatenate([acc[k], out[k]]) for k in acc}
    return {
        k: (float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))))
        for k, v in acc.items()
    }


def fmt(x):
    return f"{x:>11.3e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-states", type=int, default=512)
    ap.add_argument("--n-keys", type=int, default=2000)
    ap.add_argument("--sim-T", type=int, default=2000)
    ap.add_argument("--burn", type=int, default=500)
    ap.add_argument("--out", default=None, help="JSON output path")
    args = ap.parse_args()

    ckpt = resolve_checkpoint(args.checkpoint)
    print(f"=== Estimator diagnostic at {ckpt} ===")
    net, model, cfg = load(ckpt)
    print(
        f"  model={model.name} n_shocks={model.n_shocks} "
        f"p_disaster={model.constants.get('p_disaster', 0.0)}"
    )
    op_desc = (
        f"q={cfg.get('n_quadrature_points')}"
        if str(cfg.get("expectation_type", "mc")).startswith(("q", "g"))
        else f"mc N={cfg.get('mc_samples')}"
    )
    print(
        f"  training operating point: expectation={cfg.get('expectation_type')} "
        f"({op_desc})"
    )

    states = simulate_ergodic_states(net, model, args.n_states, args.sim_T, args.burn)
    print(f"  ergodic states: {states.shape[0]} (T={args.sim_T}, burn={args.burn})")

    results = {"checkpoint": ckpt, "config_operating_point": op_desc}

    # --- quadrature ladder ---
    print("\n--- Gauss-Hermite ladder (deterministic) ---")
    gh = {q: quad_losses(net, model, states, q) for q in QUAD_POINTS}
    gh = {q: v for q, v in gh.items() if v is not None}
    eqs = list(next(iter(gh.values())).keys())
    print(f"  {'eq':<22}" + "".join(f"  GH{q}^{model.n_shocks}".rjust(13) for q in gh))
    for e in eqs:
        print(f"  {e:<22}" + "".join(fmt(gh[q][e]) for q in gh))
    results["gh"] = {str(q): v for q, v in gh.items()}

    # --- MC estimators ---
    print("\n--- MC estimators (mean over keys) ---")
    mse = {
        N: mc_losses(net, model, states, "mse", N, args.n_keys, 7 + N) for N in MSE_NS
    }
    aio = {
        N: mc_losses(net, model, states, "aio", N, args.n_keys, 1007 + N)
        for N in AIO_NS
    }
    results["mse"] = {str(N): v for N, v in mse.items()}
    results["aio"] = {str(N): v for N, v in aio.items()}

    ref_N = max(AIO_NS)
    print(f"\n--- Summary per equation (unbiased ref = aio N={ref_N}) ---")
    header = f"  {'eq':<22}{'aio ref':>12}{'GH3-ref':>12}" + "".join(
        f"  bias(N={N})".rjust(12) for N in MSE_NS
    )
    print(header + "   [bias = mse_mean - ref; ratio in ()]")
    for e in eqs:
        ref, ref_se = aio[ref_N][e]
        gh3 = gh.get(3, {}).get(e, float("nan"))
        cells = ""
        for N in MSE_NS:
            b = mse[N][e][0] - ref
            ratio = b / abs(ref) if ref != 0 else float("inf")
            cells += f"{b:>10.2e}({ratio:>5.1f})"
        print(f"  {e:<22}{ref:>12.3e}{gh3 - ref:>12.2e}{cells}")

    if args.out is None:
        args.out = str(Path(ckpt).parent / "estimator_diagnostic.json")
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
