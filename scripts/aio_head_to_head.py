"""AiO vs mse loss-estimator head-to-head on Brock-Mirman.

Companion experiment to docs/dev/aio_loss_estimator.md. Three parts:

A. Training-free bias floor: evaluate both estimators AT the true policy
   (time-iteration ground truth) on ergodic + rect states. Theory:
   E[mse loss] - (E[r])^2 = Var(rbar) ~ 1/N;  E[aio loss] - (E[r])^2 = 0.

B. Training head-to-head: reference recipe (configs/brock_mirman.yaml),
   loss_choice {mse, aio} x mc_samples {2, 8} x seeds, policy error vs
   ground truth on the training rect and the ergodic set.

C. Degenerate-case sanity (delta=1): closed form sav_rate = alpha*beta;
   the residual is pointwise zero at the optimum so theory predicts NO
   aio advantage -- both must converge, neither should be hurt.

Everything runs in float64: the canonical-sigma bias floor is ~1e-10 in
r^2 units and float32 rounding noise in the residual would contaminate
the measurement.

Usage:
    uv run python scripts/aio_head_to_head.py            # full (~30-60 min)
    uv run python scripts/aio_head_to_head.py --quick    # smoke (~5 min)
    uv run python scripts/aio_head_to_head.py --skip-train   # Part A only
"""

import argparse
import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from deqn_jax.config import NetworkConfig, OptimizerConfig, TrainConfig  # noqa: E402
from deqn_jax.models import load_model  # noqa: E402
from deqn_jax.training.loss import compute_loss, gauss_hermite_nd  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "dev" / "figures"

# Ground-truth solver grid: wider than both the training rect (k in
# [0.9, 12]) and the ergodic set (k ~ 4-9 around k_ss = 6.367) so the
# eval region never touches the interpolation edges. The floor must be
# low enough that the smallest ENDOGENOUS cash-on-hand m = c(k') + k'
# undercuts the smallest exogenous m = y(k_min, z_min) + (1-d)k_min for
# every calibration we solve -- at delta=1 the optimal k' = alpha*beta*y
# reaches ~0.1, and a clamped EGM inversion poisons the whole fixed
# point (observed: c off by 70% at the low edge propagating to 50%
# interior errors).
K_MIN, K_MAX, N_K = 0.02, 25.0, 800
# z-span must cover rho*z_rect + sigma*max(GH node): 0.9*0.357 + 0.04*6.4 ~ 0.58,
# plus the same again for queries one step further out.
Z_SPAN, N_Z = 0.80, 49
N_GH_SOLVE = 15


# ---------------------------------------------------------------------------
# Ground truth: endogenous grid method (Carroll) time iteration
# ---------------------------------------------------------------------------


def solve_brock_mirman(constants, tol=1e-12, max_iter=5000):
    """Solve c(k, z) by EGM on a (log k, z) grid; returns grids + tables."""
    alpha = constants["alpha"]
    beta = constants["beta"]
    delta = constants["delta"]
    rho = constants["rho_z"]
    sigma = constants["sigma_z"]

    kgrid = np.geomspace(K_MIN, K_MAX, N_K)  # doubles as the k' grid
    zgrid = np.linspace(-Z_SPAN, Z_SPAN, N_Z)
    nodes, wts = gauss_hermite_nd(N_GH_SOLVE, 1)
    eps = nodes[:, 0]

    K, Z = np.meshgrid(kgrid, zgrid, indexing="ij")  # [N_K, N_Z]
    Y = np.exp(Z) * K**alpha
    m_exog = Y + (1.0 - delta) * K  # cash-on-hand today
    c_tab = 0.7 * Y  # initial guess

    lkgrid = np.log(kgrid)

    def interp_c(kq, zq):
        """Interp of c_tab at query arrays (clipped into the grid).

        Bilinear in (log k, z) on LOG c: log c is exactly bilinear for the
        delta=1 closed form (log c = const + z + alpha*log k) and much
        flatter than c generally, killing the e^z curvature error that
        dominates a direct bilinear on c (~dz^2/8 ~ 1e-4 relative).
        """
        lk = np.log(np.clip(kq, K_MIN, K_MAX))
        z = np.clip(zq, -Z_SPAN, Z_SPAN)
        lc = np.log(c_tab)
        i = np.clip(np.searchsorted(lkgrid, lk) - 1, 0, N_K - 2)
        j = np.clip(np.searchsorted(zgrid, z) - 1, 0, N_Z - 2)
        tk = (lk - lkgrid[i]) / (lkgrid[i + 1] - lkgrid[i])
        tz = (z - zgrid[j]) / (zgrid[j + 1] - zgrid[j])
        return np.exp(
            lc[i, j] * (1 - tk) * (1 - tz)
            + lc[i + 1, j] * tk * (1 - tz)
            + lc[i, j + 1] * (1 - tk) * tz
            + lc[i + 1, j + 1] * tk * tz
        )

    for it in range(max_iter):
        # E over z' of u'(c(k',z'))(1-delta+mpk(k',z')) on the k' grid
        Eterm = np.zeros((N_K, N_Z))
        for el in range(len(eps)):
            Zp = rho * Z + sigma * eps[el]
            mpk = alpha * np.exp(Zp) * K ** (alpha - 1.0)
            cp = interp_c(K, Zp)  # K here is the k' grid
            Eterm += wts[el] * (1.0 - delta + mpk) / cp

        c_endog = 1.0 / (beta * Eterm)  # u'(c) = beta*E  (log utility)
        m_endog = c_endog + K  # resources consistent with choosing k'=K

        # Back out c on the exogenous grid: for each z, c(m) via the
        # endogenous (m, c) pairs (monotone in k' hence in m). Log-log
        # interp for the same curvature reason as interp_c (exactly
        # linear at delta=1: log c = log(1-ab) + log m).
        c_new = np.empty_like(c_tab)
        for j in range(N_Z):
            c_new[:, j] = np.exp(
                np.interp(
                    np.log(m_exog[:, j]),
                    np.log(m_endog[:, j]),
                    np.log(c_endog[:, j]),
                )
            )

        diff = np.max(np.abs(c_new - c_tab) / np.maximum(c_tab, 1e-12))
        c_tab = c_new
        if diff < tol:
            break
    else:
        raise RuntimeError(f"EGM did not converge: last diff {diff:.2e}")

    # k' = (1-d)k + s*y  and  k' = m_exog - c  =>  s*y = m_exog - c - (1-d)k
    sav_tab = (m_exog - c_tab - (1.0 - delta) * K) / Y
    return kgrid, zgrid, c_tab, sav_tab, it + 1, diff


def make_true_policy_fn(kgrid, zgrid, c_tab, constants):
    """JAX-traceable true savings-rate policy from the solved c table.

    Interpolates LOG c bilinearly in (log k, z) -- same transform as the
    solver, exactly linear at delta=1 -- then converts via s = 1 - c/y
    (from k' = (1-d)k + s*y and k' = y + (1-d)k - c).
    """
    alpha = constants["alpha"]
    lkg = jnp.log(jnp.asarray(kgrid))
    zg = jnp.asarray(zgrid)
    lc = jnp.log(jnp.asarray(c_tab))

    def policy_fn(states):
        lk = jnp.log(jnp.clip(states[:, 0], K_MIN, K_MAX))
        z = jnp.clip(states[:, 1], -Z_SPAN, Z_SPAN)
        i = jnp.clip(jnp.searchsorted(lkg, lk) - 1, 0, len(kgrid) - 2)
        j = jnp.clip(jnp.searchsorted(zg, z) - 1, 0, len(zgrid) - 2)
        tk = (lk - lkg[i]) / (lkg[i + 1] - lkg[i])
        tz = (z - zg[j]) / (zg[j + 1] - zg[j])
        c = jnp.exp(
            lc[i, j] * (1 - tk) * (1 - tz)
            + lc[i + 1, j] * tk * (1 - tz)
            + lc[i, j + 1] * (1 - tk) * tz
            + lc[i + 1, j + 1] * tk * tz
        )
        y = jnp.exp(states[:, 1]) * states[:, 0] ** alpha
        return (1.0 - c / y)[:, None]

    return policy_fn


def simulate_ergodic(policy_fn, constants, T=6000, burn=1000, thin=4, seed=7):
    """Ergodic states under a policy (single long path, numpy loop)."""
    alpha, delta = constants["alpha"], constants["delta"]
    rho, sigma = constants["rho_z"], constants["sigma_z"]
    rng = np.random.default_rng(seed)
    k = 6.367 if delta < 1.0 else 0.2
    z = 0.0
    out = []
    for t in range(T):
        y = np.exp(z) * k**alpha
        s = float(np.asarray(policy_fn(jnp.array([[k, z]])))[0, 0])
        k = (1.0 - delta) * k + s * y
        z = rho * z + sigma * rng.standard_normal()
        if t >= burn and (t - burn) % thin == 0:
            out.append((k, z))
    return jnp.array(out)


# ---------------------------------------------------------------------------
# Part A: bias floor at the true policy
# ---------------------------------------------------------------------------


def estimator_mean_se(model, policy_fn, states, loss_choice, mc_samples, n_keys):
    # Per-cell seed: sharing one key stream across table cells correlates
    # their deviations (one unlucky draw shifts every cell coherently and
    # masquerades as a systematic offset).
    seed = 99 + mc_samples + (10_000 if loss_choice == "aio" else 0)
    keys = jax.random.split(jax.random.PRNGKey(seed), n_keys)

    def one(k):
        loss, _ = compute_loss(
            model, policy_fn, states, k, mc_samples=mc_samples, loss_choice=loss_choice
        )
        return loss

    vals = np.asarray(jax.jit(jax.vmap(one))(keys))
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(n_keys))


def exact_loss(model, policy_fn, states, n_gh=64):
    nodes, weights = gauss_hermite_nd(n_gh, model.n_shocks)
    loss, _ = compute_loss(
        model,
        policy_fn,
        states,
        jax.random.PRNGKey(0),
        quad_nodes=jnp.array(nodes),
        quad_weights=jnp.array(weights),
    )
    return float(loss)


def part_a(model, true_policy_fn, erg_states, rect_states, n_keys, results):
    print("\n=== Part A: estimator bias AT the true policy ===")
    for label, states in (("ergodic", erg_states), ("rect", rect_states)):
        ex = exact_loss(model, true_policy_fn, states)
        print(f"\n  [{label}] exact (E[r])^2 via GH64: {ex:.6e}")
        print(f"  {'N':>4} {'mse bias':>14} {'(se)':>10} {'aio bias':>14} {'(se)':>10}")
        rows = []
        for N in (2, 4, 8, 16, 32):
            m_mse, se_mse = estimator_mean_se(
                model, true_policy_fn, states, "mse", N, n_keys
            )
            m_aio, se_aio = estimator_mean_se(
                model, true_policy_fn, states, "aio", N, n_keys
            )
            print(
                f"  {N:>4} {m_mse - ex:>14.4e} {se_mse:>10.1e} "
                f"{m_aio - ex:>14.4e} {se_aio:>10.1e}"
            )
            rows.append(
                dict(
                    N=N,
                    exact=ex,
                    mse_bias=m_mse - ex,
                    mse_se=se_mse,
                    aio_bias=m_aio - ex,
                    aio_se=se_aio,
                )
            )
        results[f"part_a_{label}"] = rows


# ---------------------------------------------------------------------------
# Part B/C: training runs
# ---------------------------------------------------------------------------


def make_config(loss_choice, mc_samples, seed, episodes, constants=None):
    return TrainConfig(
        model="brock_mirman",
        episodes=episodes,
        batch_size=128,
        episode_length=1,
        mc_samples=mc_samples,
        seed=seed,
        initialize_each_episode=True,
        n_epochs_per_rollout=1,
        n_minibatches_per_epoch=1,
        network=NetworkConfig(
            type="mlp",
            hidden_sizes=(50, 50),
            activation="relu",
            init="xavier_uniform",
        ),
        optimizer=OptimizerConfig(
            name="adam",
            learning_rate=3.0e-4,
            lr_schedule="cosine",
            lr_min_factor=0.1,
        ),
        loss_choice=loss_choice,
        constants=constants or {},
        fp64=True,
        verbose=False,
        log_every=10**9,  # silence periodic logging
    )


def policy_error(net, true_policy_fn, states):
    s_net = np.asarray(net(states))[:, 0]
    s_true = np.asarray(true_policy_fn(states))[:, 0]
    abs_err = np.abs(s_net - s_true)
    # consumption-equivalent relative error: c = (1-s)y
    rel_c = abs_err / np.maximum(1.0 - s_true, 1e-12)
    return dict(
        mean_abs=float(abs_err.mean()),
        max_abs=float(abs_err.max()),
        mean_rel_c=float(rel_c.mean()),
        log10_mean_rel_c=float(np.log10(max(rel_c.mean(), 1e-300))),
    )


def run_training_grid(grid, true_policy_fn, eval_sets, episodes, results, key):
    from deqn_jax.training.trainer import train_from_config

    rows = []
    for loss_choice, N, seed, constants in grid:
        t0 = time.time()
        cfg = make_config(loss_choice, N, seed, episodes, constants)
        params, history = train_from_config(cfg)
        wall = time.time() - t0
        row = dict(loss_choice=loss_choice, mc_samples=N, seed=seed, wall_s=wall)
        for label, states in eval_sets.items():
            row[label] = policy_error(params, true_policy_fn, states)
        rows.append(row)
        errs = " ".join(f"{label}={row[label]['mean_abs']:.3e}" for label in eval_sets)
        print(
            f"  {loss_choice:>4} N={N} seed={seed}: {errs}  ({wall:.0f}s, "
            f"final loss {history['loss'][-1]:.3e})"
        )
    results[key] = rows


def summarize(results, key, eval_label):
    """Mean +/- std of mean_abs error across seeds, per (estimator, N)."""
    rows = results.get(key, [])
    combos = sorted({(r["loss_choice"], r["mc_samples"]) for r in rows})
    print(f"\n  Summary [{key}, {eval_label}] (mean |s_net - s_true| over seeds):")
    out = []
    for lc, N in combos:
        vals = [
            r[eval_label]["mean_abs"]
            for r in rows
            if r["loss_choice"] == lc and r["mc_samples"] == N
        ]
        mu, sd = float(np.mean(vals)), float(np.std(vals))
        print(f"    {lc:>4} N={N}: {mu:.4e} +/- {sd:.1e}  (n={len(vals)})")
        out.append(dict(loss_choice=lc, mc_samples=N, mean=mu, std=sd, n=len(vals)))
    results[f"{key}_summary_{eval_label}"] = out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced episodes/keys")
    ap.add_argument("--skip-train", action="store_true", help="Part A only")
    args = ap.parse_args()

    episodes = 3001 if args.quick else 20001
    n_keys = 1000 if args.quick else 4000
    seeds = (0,) if args.quick else (0, 1, 2, 3, 4)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {"episodes": episodes, "n_keys": n_keys, "seeds": list(seeds)}

    model = load_model("brock_mirman")
    constants = dict(model.constants)
    alpha, beta = constants["alpha"], constants["beta"]

    # --- Ground truth (canonical delta=0.1) ---
    print("=== Solving ground truth (EGM time iteration) ===")
    t0 = time.time()
    kgrid, zgrid, c_tab, sav_tab, n_iter, diff = solve_brock_mirman(constants)
    print(
        f"  converged in {n_iter} iterations (diff {diff:.1e}, {time.time() - t0:.1f}s)"
    )

    # Solver validation 1: delta=1 closed form sav = alpha*beta. Checked on
    # the region we evaluate on (training rect x reference z-range), not the
    # full grid -- the extreme-low-k edge is outside every eval set.
    c1 = dict(constants, delta=1.0)
    kg1, zg1, _, sav1, _, _ = solve_brock_mirman(c1)
    rect_mask = (
        (kg1[:, None] >= 0.9)
        & (kg1[:, None] <= 12.0)
        & (zg1[None, :] >= np.log(0.7))
        & (zg1[None, :] <= np.log(1.3))
    )
    err_closed = float(np.max(np.abs(sav1 - alpha * beta)[rect_mask]))
    print(f"  delta=1 validation (rect): max |sav - alpha*beta| = {err_closed:.2e}")
    results["solver_delta1_closed_form_err"] = err_closed
    if err_closed > 1e-6:
        raise RuntimeError("ground-truth solver failed delta=1 validation")

    true_policy_fn = make_true_policy_fn(kgrid, zgrid, c_tab, constants)

    # Solver validation 2: quadrature residual at the solved policy.
    erg_states = simulate_ergodic(true_policy_fn, constants)
    rect_states = model.init_state_fn(jax.random.PRNGKey(123), 2048, constants)
    ex_erg = exact_loss(model, true_policy_fn, erg_states)
    print(f"  (E[r])^2 at solved policy, ergodic states: {ex_erg:.2e}")
    results["solver_residual_sq_ergodic"] = ex_erg

    # --- Part A ---
    part_a(model, true_policy_fn, erg_states, rect_states, n_keys, results)

    if not args.skip_train:
        # --- Part B: canonical calibration ---
        print(f"\n=== Part B: training head-to-head (delta=0.1, {episodes} eps) ===")
        grid_b = [
            (lc, N, s, None) for lc in ("mse", "aio") for N in (2, 8) for s in seeds
        ]
        eval_sets = {"ergodic": erg_states, "rect": rect_states}
        run_training_grid(
            grid_b, true_policy_fn, eval_sets, episodes, results, "part_b"
        )
        summarize(results, "part_b", "ergodic")
        summarize(results, "part_b", "rect")

        # --- Part C: delta=1 degenerate case ---
        print(f"\n=== Part C: delta=1 sanity (closed form sav={alpha * beta:.4f}) ===")

        def closed_form_fn(states):
            return jnp.full((states.shape[0], 1), alpha * beta)

        rect1 = model.init_state_fn(jax.random.PRNGKey(321), 2048, c1)
        grid_c = [(lc, 2, s, {"delta": 1.0}) for lc in ("mse", "aio") for s in seeds]
        run_training_grid(
            grid_c, closed_form_fn, {"rect": rect1}, episodes, results, "part_c"
        )
        summarize(results, "part_c", "rect")

    out_path = OUT_DIR / "aio_head_to_head_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")
    make_figures(results)


def make_figures(results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Part A: bias vs N (log-log), ergodic states
    rows = results.get("part_a_ergodic")
    if rows:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        Ns = [r["N"] for r in rows]
        mse = [r["mse_bias"] for r in rows]
        aio_abs = [abs(r["aio_bias"]) for r in rows]
        aio_se = [r["aio_se"] for r in rows]
        ref = [mse[0] * Ns[0] / n for n in Ns]
        ax.loglog(Ns, mse, "o-", label="mse bias  E[L]-(E[r])²")
        ax.loglog(Ns, ref, "k--", alpha=0.5, label="∝ 1/N")
        ax.loglog(Ns, aio_abs, "s-", label="|aio bias|")
        ax.loglog(Ns, aio_se, "s:", alpha=0.5, label="aio s.e. (resolution)")
        ax.axhline(rows[0]["exact"], color="gray", lw=0.8)
        ax.text(
            Ns[-1],
            rows[0]["exact"] * 1.15,
            "(E[r])² at true policy",
            ha="right",
            fontsize=8,
            color="gray",
        )
        ax.set_xlabel("mc_samples N")
        ax.set_ylabel("estimator bias at the true policy")
        ax.set_title("Brock–Mirman δ=0.1, ergodic states")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "aio_bias_floor.png", dpi=150)
        print(f"  figure: {OUT_DIR / 'aio_bias_floor.png'}")

    # Part B: trained policy error by estimator x N
    rows = results.get("part_b")
    if rows:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        combos = sorted({(r["loss_choice"], r["mc_samples"]) for r in rows})
        for x, (lc, N) in enumerate(combos):
            vals = [
                r["ergodic"]["mean_abs"]
                for r in rows
                if r["loss_choice"] == lc and r["mc_samples"] == N
            ]
            ax.scatter([x] * len(vals), vals, alpha=0.7)
            ax.scatter([x], [np.mean(vals)], marker="_", s=600, color="k")
        ax.set_xticks(range(len(combos)))
        ax.set_xticklabels([f"{lc}\nN={N}" for lc, N in combos])
        ax.set_yscale("log")
        ax.set_ylabel("mean |sav_net − sav_true| (ergodic)")
        ax.set_title("Trained policy error vs ground truth (seeds as dots)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "aio_trained_policy_error.png", dpi=150)
        print(f"  figure: {OUT_DIR / 'aio_trained_policy_error.png'}")


if __name__ == "__main__":
    main()
