"""Risky steady state of the disaster model (CRW-style, first-order future rules).

The deterministic steady state s* solves the equations with shocks zero
FOREVER (certainty equivalence). The risky steady state (s_rss, p_rss)
is the rest point of an economy where realized shocks are zero but
agents still EXPECT shocks — Coeurdacier-Rey-Winant (2011), here with
first-order (BK) decision rules for future behavior:

    F1: step(s, p, 0) - s = 0                            [n_states]
    F2: E_eps[ eq(s, p, step(s,p,eps), pi_lin(step)) ]   [n_policies]

with E via tensor Gauss-Hermite quadrature over the model's shocks and
pi_lin(s) = p* + P (s - s*) from linearize_model. Solved by damped
Newton from (s*, p*) in fp64.

Purpose (2026-07-10): decompose the trained network's rest-point
displacement (pcgrad s0: 0.83%, concentrated in leverage) into the part
that is CORRECT ECONOMICS (the risky-SS shift the certainty-equivalent
anchor cannot express) vs genuine approximation error. Caveat: at the
shipped calibration p_disaster=0, so this measures Gaussian
business-cycle risk only, and future behavior is first-order (the
policy-curvature part of the true risky SS is not captured).

Usage (DGX):
  uv run python scripts/disaster_risky_ss.py [--nodes 3] \
      [--ckpt runs/disaster_cert/disaster_gated_pcgrad_s0/checkpoint_003000.eqx]
"""

from __future__ import annotations

import argparse

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from deqn_jax.models import load_model  # noqa: E402
from deqn_jax.training.linearize import linearize_model  # noqa: E402
from deqn_jax.training.loss import gauss_hermite_nd  # noqa: E402


def build_residual_fn(
    model, ss_state, ss_policy, P, quad_nodes, quad_weights, p_override=None
):
    """F(x) for the 24-unknown risky-SS system; quad with 1 zero node = deterministic.

    With ``constants['p_disaster'] > 0`` the F2 expectation is the Bernoulli
    mixture over the disaster indicator (independent of the Gaussian shocks):
    (1-p)·E_eps[eq | d=0] + p·E_eps[eq | d=1], where d=1 destroys capital via
    exp(-theta_disaster) inside step_fn. F1 stays the NO-realized-shock,
    NO-realized-disaster fixed point — that is the risky-SS definition.
    """
    n_s, n_p = model.n_states, model.n_policies
    nodes = jnp.asarray(quad_nodes)
    weights = jnp.asarray(quad_weights)
    k = nodes.shape[0]
    zero = jnp.zeros((1, model.n_shocks))
    p_dis = (
        float(p_override)
        if p_override is not None
        else float(model.constants.get("p_disaster", 0.0))
    )

    def pi_lin(sb):
        return ss_policy[None, :] + (sb - ss_state[None, :]) @ P.T

    def F(x):
        s, p = x[:n_s], x[n_s:]
        sb, pb = s[None, :], p[None, :]
        sb_k = jnp.broadcast_to(sb, (k, n_s))
        pb_k = jnp.broadcast_to(pb, (k, n_p))
        # F1: zero-realized-shock (and no realized disaster) fixed point
        f1 = model.step_fn(sb, pb, zero, model.constants)[0] - s

        # F2: equations in expectation over future shocks, linear future rules
        def branch(d):
            s_next = model.step_fn(sb_k, pb_k, nodes, model.constants, d_disaster=d)
            eq = model.equations_fn(sb_k, pb_k, s_next, pi_lin(s_next), model.constants)
            return jnp.stack(
                [jnp.sum(weights * eq[name]) for name in model.equation_names]
            )

        if p_dis > 0.0:
            f2 = (1.0 - p_dis) * branch(0.0) + p_dis * branch(1.0)
        else:
            f2 = branch(0.0)
        return jnp.concatenate([f1, f2])

    return jax.jit(F)


def newton(F, x0, tol=1e-12, max_iter=60):
    x = x0
    jac = jax.jit(jax.jacobian(F))
    for it in range(max_iter):
        r = F(x)
        nrm = float(jnp.max(jnp.abs(r)))
        if nrm < tol:
            return x, nrm, it, True
        step = jnp.linalg.solve(jac(x), -r)
        # backtracking damping
        t = 1.0
        for _ in range(8):
            cand = x + t * step
            if float(jnp.max(jnp.abs(F(cand)))) < nrm:
                break
            t *= 0.5
        x = x + t * step
    return x, float(jnp.max(jnp.abs(F(x)))), max_iter, False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nodes", type=int, default=3, help="GH nodes per shock dim")
    ap.add_argument(
        "--ckpt",
        default=None,
        help="Optional checkpoint: compare the network's rest point s_hat",
    )
    ap.add_argument(
        "--p-disaster",
        type=float,
        default=None,
        help="Override constants['p_disaster'] (Alex's real calibration: 0.01)",
    )
    ap.add_argument(
        "--theta",
        type=float,
        default=None,
        help="Override constants['theta_disaster'] (Alex's real calibration: 0.15)",
    )
    args = ap.parse_args()

    model = load_model("disaster")
    if args.p_disaster is not None or args.theta is not None:
        c = dict(model.constants)
        if args.p_disaster is not None:
            c["p_disaster"] = args.p_disaster
        if args.theta is not None:
            c["theta_disaster"] = args.theta
        model = model._replace(constants=c)
    print(
        f"[calib] p_disaster={model.constants.get('p_disaster', 0.0)}  "
        f"theta_disaster={model.constants.get('theta_disaster', 0.0)}"
    )
    ss_state, ss_policy = model.steady_state_fn(model.constants)
    ss_state, ss_policy = jnp.asarray(ss_state), jnp.asarray(ss_policy)
    P, _Q = linearize_model(model, verbose=False)
    P = jnp.asarray(P)
    n_s = model.n_states
    names_s = list(getattr(model, "state_names", None) or [f"s{i}" for i in range(n_s)])
    names_p = list(model.policy_names)

    x0 = jnp.concatenate([ss_state, ss_policy])
    zero_nodes, zero_w = np.zeros((1, model.n_shocks)), np.ones(1)
    p_cfg = float(model.constants.get("p_disaster", 0.0))

    # --- (i) machinery baseline: no Gaussian risk, no disaster risk --------
    # Deterministic re-solve of the same system; must sit at (s*, p*) up to
    # the soft-clip + linear-future-rule floor.
    F_mach = build_residual_fn(
        model, ss_state, ss_policy, P, zero_nodes, zero_w, p_override=0.0
    )
    x_mach, r_m, it_m, ok_m = newton(F_mach, x0)
    d_m = float(jnp.max(jnp.abs(x_mach - x0) / (jnp.abs(x0) + 1e-12)))
    print(
        f"[machinery] deterministic re-solve (p=0): converged={ok_m} "
        f"iters={it_m} residual={r_m:.2e}  max-rel drift from (s*,p*) = {d_m:.3e}"
    )

    # --- (ii) + disaster mixture only (zero Gaussian nodes) ----------------
    if p_cfg > 0.0:
        F_dis = build_residual_fn(model, ss_state, ss_policy, P, zero_nodes, zero_w)
        x_dis, r_d, it_d, ok_d = newton(F_dis, x0)
        print(
            f"[disaster-only] mixture at zero Gaussian nodes: converged={ok_d} "
            f"iters={it_d} residual={r_d:.2e}"
        )
    else:
        x_dis = x_mach

    # --- (iii) full risky: Gaussian quadrature + disaster mixture ----------
    qn, qw = gauss_hermite_nd(args.nodes, model.n_shocks)
    print(f"[quad] {args.nodes}^{model.n_shocks} = {len(qw)} GH nodes")
    F_risk = build_residual_fn(model, ss_state, ss_policy, P, qn, qw)
    wedge = F_risk(x0)
    print(
        f"[wedge] max |E-residual| at deterministic SS under full risk: "
        f"{float(jnp.max(jnp.abs(wedge[n_s:]))):.3e}"
    )
    x_rss, r_rss, it_rss, ok_rss = newton(F_risk, x0)
    print(f"[risky SS] converged={ok_rss} iters={it_rss} residual={r_rss:.2e}\n")

    # --- compare with the model's own flat-next-policy heuristic anchor -----
    if float(model.constants.get("p_disaster", 0.0)) > 0.0:
        from deqn_jax.models.disaster.steady_state import risky_steady_state

        h_s, _h_p = risky_steady_state(model.constants)
        h_s = jnp.asarray(h_s)
        d_heur = np.asarray((h_s - ss_state) / (jnp.abs(ss_state) + 1e-12))
        d_gap = np.asarray((h_s - x_rss[:n_s]) / (jnp.abs(ss_state) + 1e-12))
        print(
            f"[heuristic anchor] model's flat-next-policy risky SS: "
            f"max-rel shift vs s* = {np.abs(d_heur).max():.4%}; "
            f"max-rel gap vs CRW risky SS = {np.abs(d_gap).max():.4%} "
            f"(the flat-vs-linear future-rules wedge)"
        )

    # Component decomposition, all scaled by |s*| (resp. |p*|):
    #   machinery = (i) - (s*, p*)      soft-clip + linear-rule floor
    #   disaster  = (ii) - (i)          Bernoulli capital-destruction risk
    #   gaussian  = (iii) - (ii)        business-cycle (GH) risk
    #   total     = (iii) - (i)         the risky-SS shift proper
    scale_s = np.asarray(jnp.abs(ss_state) + 1e-12)
    scale_p = np.asarray(jnp.abs(ss_policy) + 1e-12)
    det_s = np.asarray(x_mach[:n_s] - ss_state) / scale_s
    dis_s = np.asarray(x_dis[:n_s] - x_mach[:n_s]) / scale_s
    gau_s = np.asarray(x_rss[:n_s] - x_dis[:n_s]) / scale_s
    rel_s = np.asarray(x_rss[:n_s] - x_mach[:n_s]) / scale_s
    rel_p = np.asarray(x_rss[n_s:] - x_mach[n_s:]) / scale_p
    dis_p = np.asarray(x_dis[n_s:] - x_mach[n_s:]) / scale_p

    # --- optional: network rest point for comparison ------------------------
    rel_hat = None
    if args.ckpt:
        from deqn_jax.irf import load_policy_from_checkpoint

        net, model_c = load_policy_from_checkpoint(args.ckpt)
        zero = jnp.zeros((1, model_c.n_shocks))

        def pol(sb):
            p = net(sb)
            return p if p.ndim > 1 else p[None, :]

        def T(sv):
            return model_c.step_fn(
                sv[None, :], pol(sv[None, :]), zero, model_c.constants
            )[0]

        s_hat = ss_state
        for _ in range(80):
            Fh = T(s_hat) - s_hat
            if float(jnp.max(jnp.abs(Fh))) < 1e-13:
                break
            Jf = jnp.asarray(jax.jacobian(T)(s_hat)) - jnp.eye(n_s)
            s_hat = s_hat + jnp.linalg.solve(Jf, -Fh)
        rel_hat = np.asarray((s_hat - ss_state) / (jnp.abs(ss_state) + 1e-12))

    # --- report --------------------------------------------------------------
    hdr = (
        f"{'state':>14} {'machinery':>11} {'disaster':>11} "
        f"{'gaussian':>11} {'TOTAL risk':>11}"
    )
    if rel_hat is not None:
        hdr += f" {'network':>11} {'net/total':>10}"
    print(hdr)
    order = np.argsort(-np.abs(rel_s))
    for i in order:
        line = (
            f"{names_s[i]:>14} {det_s[i]:>10.4%} {dis_s[i]:>10.4%} "
            f"{gau_s[i]:>10.4%} {rel_s[i]:>10.4%}"
        )
        if rel_hat is not None:
            ratio = rel_hat[i] / rel_s[i] if abs(rel_s[i]) > 1e-12 else float("nan")
            line += f" {rel_hat[i]:>10.4%} {ratio:>10.2f}"
        print(line)
    print(f"\n{'policy':>14} {'disaster':>11} {'TOTAL risk':>11}")
    for i in np.argsort(-np.abs(rel_p)):
        print(f"{names_p[i]:>14} {dis_p[i]:>10.4%} {rel_p[i]:>10.4%}")


if __name__ == "__main__":
    main()
