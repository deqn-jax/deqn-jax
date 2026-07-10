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


def build_residual_fn(model, ss_state, ss_policy, P, quad_nodes, quad_weights):
    """F(x) for the 24-unknown risky-SS system; quad with 1 zero node = deterministic."""
    n_s, n_p = model.n_states, model.n_policies
    nodes = jnp.asarray(quad_nodes)
    weights = jnp.asarray(quad_weights)
    k = nodes.shape[0]
    zero = jnp.zeros((1, model.n_shocks))

    def pi_lin(sb):
        return ss_policy[None, :] + (sb - ss_state[None, :]) @ P.T

    def F(x):
        s, p = x[:n_s], x[n_s:]
        sb, pb = s[None, :], p[None, :]
        # F1: zero-realized-shock fixed point
        f1 = model.step_fn(sb, pb, zero, model.constants)[0] - s
        # F2: equations in expectation over future shocks, linear future rules
        s_next = model.step_fn(
            jnp.broadcast_to(sb, (k, n_s)),
            jnp.broadcast_to(pb, (k, n_p)),
            nodes,
            model.constants,
        )
        eq = model.equations_fn(
            jnp.broadcast_to(sb, (k, n_s)),
            jnp.broadcast_to(pb, (k, n_p)),
            s_next,
            pi_lin(s_next),
            model.constants,
        )
        f2 = jnp.stack([jnp.sum(weights * eq[name]) for name in model.equation_names])
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
    args = ap.parse_args()

    model = load_model("disaster")
    ss_state, ss_policy = model.steady_state_fn(model.constants)
    ss_state, ss_policy = jnp.asarray(ss_state), jnp.asarray(ss_policy)
    P, _Q = linearize_model(model, verbose=False)
    P = jnp.asarray(P)
    n_s = model.n_states
    names_s = list(getattr(model, "state_names", None) or [f"s{i}" for i in range(n_s)])
    names_p = list(model.policy_names)

    x0 = jnp.concatenate([ss_state, ss_policy])

    # --- sanity 1: deterministic system must reproduce (s*, p*) ------------
    F_det = build_residual_fn(
        model, ss_state, ss_policy, P, np.zeros((1, model.n_shocks)), np.ones(1)
    )
    x_det, r_det, it_det, ok_det = newton(F_det, x0)
    d_det = float(jnp.max(jnp.abs(x_det - x0) / (jnp.abs(x0) + 1e-12)))
    print(
        f"[sanity] deterministic re-solve: converged={ok_det} iters={it_det} "
        f"residual={r_det:.2e}  max-rel drift from (s*,p*) = {d_det:.3e}"
    )

    # --- risky system -------------------------------------------------------
    qn, qw = gauss_hermite_nd(args.nodes, model.n_shocks)
    print(f"[quad] {args.nodes}^{model.n_shocks} = {len(qw)} GH nodes")
    F_risk = build_residual_fn(model, ss_state, ss_policy, P, qn, qw)
    wedge = F_risk(x0)
    print(
        f"[wedge] max |E-residual| at deterministic SS under risk: "
        f"{float(jnp.max(jnp.abs(wedge[n_s:]))):.3e}"
    )
    x_rss, r_rss, it_rss, ok_rss = newton(F_risk, x0)
    print(f"[risky SS] converged={ok_rss} iters={it_rss} residual={r_rss:.2e}\n")

    s_rss, p_rss = x_rss[:n_s], x_rss[n_s:]
    # Risk-ISOLATED shifts: measure the risky solution against the
    # deterministic re-solve of the SAME system (x_det), not against
    # steady_state_fn's s*. This subtracts the machinery floor
    # (soft-clip regularization, linear-future-rule wedge) which is the
    # same in both solves; scale by |s*| for readability.
    s_det, p_det = x_det[:n_s], x_det[n_s:]
    det_s = np.asarray((s_det - ss_state) / (jnp.abs(ss_state) + 1e-12))
    rel_s = np.asarray((s_rss - s_det) / (jnp.abs(ss_state) + 1e-12))
    rel_p = np.asarray((p_rss - p_det) / (jnp.abs(ss_policy) + 1e-12))

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
    hdr = f"{'state':>14} {'det machinery':>14} {'risky-SS shift':>15}"
    if rel_hat is not None:
        hdr += f" {'network shift':>15} {'ratio net/risky':>16}"
    print(hdr)
    order = np.argsort(-np.abs(rel_s))
    for i in order:
        line = f"{names_s[i]:>14} {det_s[i]:>13.4%} {rel_s[i]:>14.4%}"
        if rel_hat is not None:
            ratio = rel_hat[i] / rel_s[i] if abs(rel_s[i]) > 1e-12 else float("nan")
            line += f" {rel_hat[i]:>14.4%} {ratio:>16.2f}"
        print(line)
    print(f"\n{'policy':>14} {'risky-SS shift':>15}")
    for i in np.argsort(-np.abs(rel_p)):
        print(f"{names_p[i]:>14} {rel_p[i]:>14.4%}")


if __name__ == "__main__":
    main()
