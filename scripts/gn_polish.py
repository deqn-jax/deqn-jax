"""Gauss-Newton polish toward a joint zero of the stacked residual vector.

Experiment #4 of the sharpened diagnosis (cert report, 07-10): the trainer
parks at least-squares COMPROMISE points of the scalar objective
L = ||R||^2 (∇L = 2 J_R^T R ≈ 0 with R ≠ 0). This script asks whether a
joint zero R(θ) = 0 exists within network capacity near a given
checkpoint, by switching from descent-on-L to Levenberg-Marquardt on the
stacked residual VECTOR over a FROZEN state grid (off-policy: the
evaluation measure cannot chase the policy).

    R(θ) ∈ R^{11m}:  per-equation E-residuals (tensor GH quadrature) at
                     m frozen states = SS + ergodic-shaped cloud
    Δθ = −J^T (J J^T + λ diag-scale)^{-1} R     (min-norm LM step)

Outcomes: (a) converges to a joint zero that stays stable + SS-consistent
→ first genuine solution; (b) converges to an UNSTABLE joint zero →
evidence for real multiplicity (selection conditions become necessary);
(c) stalls with R ≠ 0 → the identification wedge at network capacity.

Per-iteration certificates are printed (learned-block spectral radius at
s*, max SS policy error) and a full certification block runs at the end
(learned fixed point ŝ, ρ at ŝ, per-equation residuals at ŝ).

The polished net is saved net-only via eqx.tree_serialise_leaves (only
when at least one LM step was accepted, or --out is given). Reload
with: net, model = load_policy_from_checkpoint(<orig ckpt>);
net = eqx.tree_deserialise_leaves(<polished path>, net).
For --widen runs that recipe does NOT apply (the polished net has
different layer shapes): rebuild the wide template exactly as the
--widen branch does (checkpoint's network config with hidden_sizes
overridden to (N, N)), then deserialise into that.

Usage (DGX):
  uv run python scripts/gn_polish.py \
      --ckpt runs/disaster_cert/disaster_gated_pcgrad_s0/checkpoint_003000.eqx \
      --m 64 --iters 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from deqn_jax.irf import load_policy_from_checkpoint  # noqa: E402
from deqn_jax.training.composite_loss import prepare_composite_data  # noqa: E402
from deqn_jax.training.linearize import linearize_model  # noqa: E402
from deqn_jax.training.loss import gauss_hermite_nd  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--m", type=int, default=64, help="frozen grid size")
    ap.add_argument("--nodes", type=int, default=3, help="GH nodes per shock dim")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--lam0", type=float, default=1e-4)
    ap.add_argument("--grid-seed", type=int, default=20260710)
    ap.add_argument("--chunk", type=int, default=64, help="Jacobian VJP chunk size")
    ap.add_argument("--out", default=None, help="polished net path (.eqx)")
    ap.add_argument(
        "--w-ss",
        type=float,
        default=0.0,
        help="constrained polish: weight on SS-consistency rows "
        "(pi(s*)-pi* and T(s*)-s*); 0 = unconstrained",
    )
    ap.add_argument(
        "--w-rho",
        type=float,
        default=0.0,
        help="constrained polish: weight on stability rows "
        "softplus(log growth of J^K probes - log target); 0 = off",
    )
    ap.add_argument("--rho-target", type=float, default=0.99)
    ap.add_argument("--rho-k", type=int, default=30)
    ap.add_argument("--rho-probes", type=int, default=4)
    ap.add_argument(
        "--widen",
        type=int,
        default=0,
        help="capacity test: distill the checkpoint into a (N,N) net first",
    )
    ap.add_argument("--distill-steps", type=int, default=5000)
    args = ap.parse_args()

    net, model = load_policy_from_checkpoint(args.ckpt)

    if args.widen:
        import optax
        import yaml

        from deqn_jax.config import NetworkConfig
        from deqn_jax.networks.factory import build_policy_net

        ss_d, _ = model.steady_state_fn(model.constants)
        ss_d = jnp.asarray(ss_d)
        P_d, Q_d = linearize_model(model, verbose=False)
        data_d = prepare_composite_data(
            model, P_d, Q_d, n_anchor_points=8, verbose=False
        )
        zc = jax.random.normal(
            jax.random.PRNGKey(20260711), (4096, model.n_states), dtype=ss_d.dtype
        )
        cloud = jnp.concatenate(
            [ss_d[None, :], ss_d[None, :] + zc @ data_d.ergodic_cov_chol.T]
        )
        target = net(cloud)
        # Build the wide template from the checkpoint's FULL network config
        # (only hidden_sizes overridden): static fields (bk_pin, zlb feature,
        # reparam flags, kf_names, output_links) change the forward graph, and
        # a hand-picked kwargs subset silently produced a structurally
        # different net — the same template class of bug as the irf.py loader.
        with open(Path(args.ckpt).parent / "config.yaml") as f:
            run_cfg = yaml.safe_load(f)
        wide_cfg_dict = dict(run_cfg.get("network", {}))
        wide_cfg_dict["hidden_sizes"] = (args.widen, args.widen)
        wide_config = NetworkConfig(
            **{
                k: v
                for k, v in wide_cfg_dict.items()
                if k in NetworkConfig.model_fields
            }
        )
        wide = build_policy_net(
            model, jax.random.PRNGKey(0), (args.widen, args.widen), wide_config
        )
        opt = optax.adam(1e-3)
        w_arr = eqx.filter(wide, eqx.is_array)
        opt_state = opt.init(w_arr)

        @jax.jit
        def distill_step(w_arr, opt_state):
            def fit_loss(wa):
                w_net = eqx.combine(wa, wide)
                return jnp.mean((w_net(cloud) - target) ** 2)

            loss, g = jax.value_and_grad(fit_loss)(w_arr)
            updates, opt_state = opt.update(g, opt_state, w_arr)
            return optax.apply_updates(w_arr, updates), opt_state, loss

        fit = jnp.nan  # --distill-steps 0: report nan rather than NameError
        for step in range(args.distill_steps):
            w_arr, opt_state, fit = distill_step(w_arr, opt_state)
            if step % 1000 == 0:
                print(f"[distill {step:5d}] fit mse={float(fit):.3e}", flush=True)
        net = eqx.combine(w_arr, wide)
        max_err = float(jnp.max(jnp.abs(net(cloud) - target)))
        print(f"[distill] done: fit mse={float(fit):.3e} max|err|={max_err:.3e}")
    ss, ss_pol = model.steady_state_fn(model.constants)
    ss, ss_pol = jnp.asarray(ss), jnp.asarray(ss_pol)
    n_s, n_p, n_eq = model.n_states, model.n_policies, len(model.equation_names)

    # --- frozen grid: SS + ergodic-shaped cloud (deterministic) -------------
    P, Q = linearize_model(model, verbose=False)
    data = prepare_composite_data(model, P, Q, n_anchor_points=8, verbose=False)
    z = jax.random.normal(
        jax.random.PRNGKey(args.grid_seed), (args.m - 1, n_s), dtype=ss.dtype
    )
    grid = jnp.concatenate([ss[None, :], ss[None, :] + z @ data.ergodic_cov_chol.T])
    qn, qw = gauss_hermite_nd(args.nodes, model.n_shocks)
    qn, qw = jnp.asarray(qn), jnp.asarray(qw)
    k = qn.shape[0]
    m = grid.shape[0]
    n_r = n_eq * m
    print(f"[setup] grid m={m}, GH nodes={k}, residuals={n_r}")

    # --- flat parameter view -------------------------------------------------
    params0 = eqx.filter(net, eqx.is_array)
    theta0, unravel = jax.flatten_util.ravel_pytree(params0)
    print(
        f"[setup] params={theta0.shape[0]} ({'under' if n_r < theta0.shape[0] else 'over'}determined)"
    )

    grid_k = jnp.repeat(grid, k, axis=0)  # [m*k, n_s]
    qn_k = jnp.tile(qn, (m, 1))  # [m*k, n_shocks]

    rho_probes = (
        jax.random.normal(jax.random.PRNGKey(20260712), (args.rho_probes, n_s))
        if args.w_rho > 0
        else None
    )
    if rho_probes is not None:
        rho_probes = rho_probes / jnp.linalg.norm(rho_probes, axis=1, keepdims=True)
    zero_shock = jnp.zeros((1, model.n_shocks))

    def residuals(theta):
        net_t = eqx.combine(unravel(theta), net)
        pb = net_t(grid)  # [m, n_p]
        pb_k = jnp.repeat(pb, k, axis=0)
        s_next = model.step_fn(grid_k, pb_k, qn_k, model.constants)
        eq = model.equations_fn(grid_k, pb_k, s_next, net_t(s_next), model.constants)
        rows = [
            jnp.sum(eq[name].reshape(m, k) * qw[None, :], axis=1)
            for name in model.equation_names
        ]
        parts = [jnp.concatenate(rows)]  # [n_eq * m]

        # Constrained polish: SS-consistency rows (level + fixed point).
        if args.w_ss > 0:
            pol_rows = (net_t(ss[None, :])[0] - ss_pol) / jnp.abs(ss_pol)
            step_rows = (
                model.step_fn(
                    ss[None, :], net_t(ss[None, :]), zero_shock, model.constants
                )[0]
                - ss
            ) / (jnp.abs(ss) + 1e-12)
            parts += [args.w_ss * pol_rows, args.w_ss * step_rows]

        # Constrained polish: stability rows — smooth spectral-growth
        # penalty. softplus((1/K)·log ||J_cl^K u|| − log target) is ≈0 when
        # the closed loop contracts faster than target and grows smoothly
        # with the unstable root; differentiable end-to-end (autodiff
        # through the closed-loop Jacobian and its matrix powers).
        if rho_probes is not None:

            def T_theta(s):
                return model.step_fn(
                    s[None, :], net_t(s[None, :]), zero_shock, model.constants
                )[0]

            Jcl = jax.jacobian(T_theta)(ss)

            def grow(u):
                v = u
                for _ in range(args.rho_k):
                    v = Jcl @ v
                # sharpened softplus: ≈0 when the loop contracts faster
                # than target (plain softplus(-0.015)≈0.69 would leave a
                # large permanent residual and hijack the LM objective)
                arg = jnp.log(jnp.linalg.norm(v) + 1e-300) / args.rho_k - jnp.log(
                    args.rho_target
                )
                return jax.nn.softplus(200.0 * arg) / 200.0

            parts.append(args.w_rho * jax.vmap(grow)(rho_probes))

        return jnp.concatenate(parts)

    R_fn = jax.jit(residuals)
    n_r = (
        n_r
        + (n_p + n_s if args.w_ss > 0 else 0)
        + (args.rho_probes if args.w_rho > 0 else 0)
    )
    if args.w_ss > 0 or args.w_rho > 0:
        print(
            f"[setup] CONSTRAINED polish: w_ss={args.w_ss} w_rho={args.w_rho} "
            f"(target {args.rho_target}, K={args.rho_k}) -> residuals={n_r}"
        )

    def jacobian(theta):
        _, vjp = jax.vjp(residuals, theta)
        eye = jnp.eye(n_r, dtype=theta.dtype)
        rows = []
        for i in range(0, n_r, args.chunk):
            rows.append(jax.vmap(lambda e: vjp(e)[0])(eye[i : i + args.chunk]))
        return jnp.concatenate(rows)  # [n_r, n_params]

    # --- certificates (autonomous/learned block split, fixed rows) ----------
    zero = jnp.zeros((1, model.n_shocks))

    def closed_loop_jac(theta, sv):
        net_t = eqx.combine(unravel(theta), net)

        def T(s):
            return model.step_fn(s[None, :], net_t(s[None, :]), zero, model.constants)[
                0
            ]

        return np.asarray(jax.jacobian(T)(sv)), T

    J0, _ = closed_loop_jac(theta0, ss)
    auto = [
        i
        for i in range(n_s)
        if (lambda r: (r.__setitem__(i, 0.0) or r.max()) < 1e-10)(np.abs(J0[i]).copy())
    ]
    rest = [i for i in range(n_s) if i not in auto]

    def certs(theta):
        Jc, _ = closed_loop_jac(theta, ss)
        rho = float(np.abs(np.linalg.eigvals(Jc[np.ix_(rest, rest)])).max())
        net_t = eqx.combine(unravel(theta), net)
        pol_err = float(
            jnp.max(jnp.abs(net_t(ss[None, :])[0] - ss_pol) / jnp.abs(ss_pol))
        )
        return rho, pol_err

    # --- LM loop --------------------------------------------------------------
    theta = theta0
    R = R_fn(theta)
    lam = args.lam0
    accepted_any = False
    rho, perr = certs(theta)
    print(
        f"[iter  0] |R|inf={float(jnp.max(jnp.abs(R))):.3e} "
        f"|R|2={float(jnp.linalg.norm(R)):.3e} rho_learned={rho:.6f} ss_err={perr:.4%}"
    )
    for it in range(1, args.iters + 1):
        J = jacobian(theta)
        JJt = J @ J.T
        scale = float(jnp.mean(jnp.diag(JJt)))
        accepted = False
        for _ in range(10):
            y = jnp.linalg.solve(JJt + lam * scale * jnp.eye(n_r), R)
            cand = theta - J.T @ y
            Rc = R_fn(cand)
            if float(jnp.linalg.norm(Rc)) < float(jnp.linalg.norm(R)):
                theta, R = cand, Rc
                lam = max(lam / 3.0, 1e-12)
                accepted = True
                accepted_any = True
                break
            lam *= 10.0
            if lam > 1e8:
                break
        rho, perr = certs(theta)
        print(
            f"[iter {it:2d}] |R|inf={float(jnp.max(jnp.abs(R))):.3e} "
            f"|R|2={float(jnp.linalg.norm(R)):.3e} lam={lam:.1e} "
            f"rho_learned={rho:.6f} ss_err={perr:.4%} "
            f"{'ACCEPT' if accepted else 'STALL'}",
            flush=True,
        )
        if not accepted:
            print("[stop] LM stalled (no descent direction at max damping)")
            break
        if float(jnp.max(jnp.abs(R))) < 1e-10:
            print("[stop] joint zero reached (|R|inf < 1e-10)")
            break

    # --- final certification block --------------------------------------------
    _, T = closed_loop_jac(theta, ss)
    s_hat = ss
    for _ in range(80):
        F = T(s_hat) - s_hat
        if float(jnp.max(jnp.abs(F))) < 1e-13:
            break
        Jf = jnp.asarray(jax.jacobian(T)(s_hat)) - jnp.eye(n_s)
        s_hat = s_hat + jnp.linalg.solve(Jf, -F)
    Jh, _ = closed_loop_jac(theta, s_hat)
    rho_hat = float(np.abs(np.linalg.eigvals(Jh[np.ix_(rest, rest)])).max())
    disp = float(jnp.max(jnp.abs(s_hat - ss) / (jnp.abs(ss) + 1e-12)))

    net_f = eqx.combine(unravel(theta), net)
    sb = jnp.broadcast_to(s_hat[None, :], (k, n_s))
    pb = jnp.broadcast_to(net_f(s_hat[None, :]), (k, n_p))
    s_next = model.step_fn(sb, pb, qn, model.constants)
    eqh = model.equations_fn(sb, pb, s_next, net_f(s_next), model.constants)
    max_r_hat = max(
        abs(float(jnp.sum(qw * eqh[name]))) for name in model.equation_names
    )
    rho_ss, perr_ss = certs(theta)
    print(
        f"\n[final] |R|inf on grid = {float(jnp.max(jnp.abs(R))):.3e}\n"
        f"[final] rho_learned(s*) = {rho_ss:.6f}   ss policy err = {perr_ss:.4%}\n"
        f"[final] shat: displacement = {disp:.4%}   rho_learned(shat) = {rho_hat:.6f}"
        f"   max |E[r]| at shat = {max_r_hat:.3e}"
    )

    # Save guard: --iters 0 (cert-only probe mode) or a run that never
    # accepted an LM step leaves net_f == the unpolished checkpoint net;
    # writing it to the default path would silently clobber a previously
    # produced genuinely-polished artifact. Save only when polishing
    # happened, or when the user asked for a path explicitly.
    if accepted_any or args.out:
        out = args.out or str(Path(args.ckpt).parent / "polished_gn.eqx")
        eqx.tree_serialise_leaves(out, net_f)
        print(f"[saved] {out}")
    else:
        print("[not saved] no accepted LM step (cert-only run); pass --out to force")


if __name__ == "__main__":
    main()
