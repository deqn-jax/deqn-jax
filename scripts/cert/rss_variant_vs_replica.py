"""Residuals of rss_trade_ez (Phase-1 variant) against rss_trade_ez_ref
(Phase-0 replica) on shared physical states, for one and the same policy.

No checkpoint is needed: the policy is a random replica-layout network with
nonzero output layers (``rss_trade_ez.parity.random_reference_net``), which
always reads the replica's scaffolding at its converged values; the variant
sees the same function through the name maps of ``rss_trade_ez.parity``.
Every residual delta must be attributable to a named variant. Per residual
block the table reports the residual's own size under the replica
("max abs r_ref", for scale) and the largest absolute difference
variant - replica in four settings:

  own        each model under its own monomial rule (replica: 36 nodes at
             +/-sqrt(18) over its 18 shocks, six of them dead; variant: 24
             nodes at +/-sqrt(12) over its 12)
  same nodes the variant under the replica's node set, mapped onto its
             shocks by name (nodes on own-country axes become the zero
             shock): isolates the equation algebra from the node set (D6)
  homotopy   own rules at mid-curriculum homotopy weights (replica: state
             columns homo = homo_1 = 0.5; variant: constants homotopy_asym =
             homotopy_trade = 0.5): scaffolding columns vs constants (D3)
  D7         variant only: its transport-map tariff step minus the
             reference's original additive-Gaussian step (relu after the
             step), same policy, same nodes. The replica already ships the
             transport map, so this is the effect of D7 against the
             reference code, not against the replica.

Grids: "normal" = the replica's own initial sampler rolled 30 steps under
the policy; "stress" = the trade-war corner (K x exp U(-0.6, 0.3),
A + U(-0.5, 0.5), off-diagonal tau + U(0.5, 2) clipped to [0, 2.5],
log-vol + U(1, 2.5)).

Usage:  uv run python scripts/cert/rss_variant_vs_replica.py [--n 256]
"""

from __future__ import annotations

import argparse

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from deqn_jax.models import load_model  # noqa: E402
from deqn_jax.models.rss_trade_ez import (  # noqa: E402
    build_constants,
    build_model,
    parity,
)
from deqn_jax.models.rss_trade_ez.dynamics import make_step  # noqa: E402
from deqn_jax.models.rss_trade_ez.variables import Layout as VarLayout  # noqa: E402
from deqn_jax.models.rss_trade_ez_ref.definitions import clip_policy  # noqa: E402
from deqn_jax.models.rss_trade_ez_ref.variables import Layout as RefLayout  # noqa: E402
from deqn_jax.training.loss import monomial_nd  # noqa: E402


def block_of(var_name: str) -> str:
    if var_name == "world_bond_clearing":
        return var_name
    return var_name.rpartition("_")[0]


def legacy_additive_step(var_layout: VarLayout):
    """The reference's original tariff step: additive Gaussian innovation,
    relu after the step (comparison-only; the variant uses the transport)."""
    base = make_step(var_layout)
    m = var_layout.n_pairs
    pairs = np.asarray(var_layout.pair_flat)

    def step(state, policy, shock, constants):
        nxt = base(state, policy, shock, constants)
        rho_t = jnp.asarray(constants["rho_tau"]).reshape(-1)[pairs]
        bar_t = jnp.asarray(constants["bar_tau"]).reshape(-1)[pairs]
        mu = (1.0 - rho_t) * bar_t + rho_t * state[:, var_layout.tau]
        sigma = jnp.exp(nxt[:, var_layout.sig])
        tau = jnp.maximum(mu + sigma * shock[:, m : 2 * m], 0.0)
        return nxt.at[:, var_layout.tau].set(tau)

    return step


def grids(ref, policy, key, n):
    lay = RefLayout(int(ref.constants["n_countries"]))
    k_init, k_roll, k_stress = jax.random.split(key, 3)
    s = ref.init_state_fn(k_init, n, ref.constants)
    for k in jax.random.split(k_roll, 30):
        shock = jax.random.normal(k, (n, ref.n_shocks), dtype=s.dtype)
        s = ref.step_fn(s, policy(s), shock, ref.constants)
    normal = parity.pin_scaffolding(s, ref)  # stage-3 scaffolding

    ks = jax.random.split(k_stress, 4)
    off = ~np.eye(lay.n, dtype=bool)
    tau_idx, sig_idx = lay.tau[off], lay.sigma_tau[off]
    st = ref.init_state_fn(k_init, n, ref.constants)
    u = jax.random.uniform(ks[0], (n, lay.n), minval=-0.6, maxval=0.3)
    st = st.at[:, lay.K].multiply(jnp.exp(u))
    st = st.at[:, lay.A].add(
        jax.random.uniform(ks[1], (n, lay.n), minval=-0.5, maxval=0.5)
    )
    tau = jax.random.uniform(ks[2], (n, len(tau_idx)), minval=0.5, maxval=2.0)
    st = st.at[:, tau_idx].set(jnp.clip(tau, 0.0, 2.5))
    st = st.at[:, sig_idx].add(
        jax.random.uniform(ks[3], (n, len(sig_idx)), minval=1.0, maxval=2.5)
    )
    return {"normal": normal, "stress": st}


def max_delta(var, r_var, r_ref):
    return {
        n: float(jnp.max(jnp.abs(r_var[n] - r_ref[parity.equation_source(n)])))
        for n in var.equation_names
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=64)
    args = ap.parse_args()

    ref = load_model("rss_trade_ez_ref")
    var = load_model("rss_trade_ez")
    lay = RefLayout(int(ref.constants["n_countries"]))
    key = jax.random.PRNGKey(args.seed)
    net = parity.random_reference_net(ref, key, (args.hidden, args.hidden))

    def raw(s):
        return jax.vmap(net)(parity.pin_scaffolding(s, ref))

    def clipped(s):
        return clip_policy(raw(s), lay)

    ref_rule = monomial_nd(ref.n_shocks)
    var_rule = monomial_nd(var.n_shocks)
    ref_on_var = (parity.replica_nodes_on_variant(ref, var, ref_rule[0]), ref_rule[1])
    var_mid = build_model(
        {**build_constants(), "homotopy_asym": 0.5, "homotopy_trade": 0.5},
        name="rss_trade_ez_mid",
    )
    var_legacy = var._replace(step_fn=legacy_additive_step(VarLayout(lay.n)))
    pol = parity.variant_policy_fn(clipped, ref, var)
    pol_mid = parity.variant_policy_fn(clipped, ref, var_mid, 0.5, 0.5)

    tables = {}
    for gname, states in grids(ref, raw, jax.random.fold_in(key, 3), args.n).items():
        vs = parity.to_variant_state(states, ref, var)
        r_ref = parity.quadrature_residuals(ref, raw, states, *ref_rule)
        r_own = parity.quadrature_residuals(var, pol, vs, *var_rule)
        r_same = parity.quadrature_residuals(var, pol, vs, *ref_on_var)
        mid = states.at[:, ref.state_names.index("homo")].set(0.5)
        mid = mid.at[:, ref.state_names.index("homo_1")].set(0.5)
        r_ref_mid = parity.quadrature_residuals(ref, raw, mid, *ref_rule)
        r_var_mid = parity.quadrature_residuals(var_mid, pol_mid, vs, *var_rule)
        r_add = parity.quadrature_residuals(var_legacy, pol, vs, *var_rule)
        tables[gname] = {
            "max abs r_ref": {
                n: float(jnp.max(jnp.abs(r_ref[parity.equation_source(n)])))
                for n in var.equation_names
            },
            "own": max_delta(var, r_own, r_ref),
            "same nodes": max_delta(var, r_same, r_ref),
            "homotopy": max_delta(var, r_var_mid, r_ref_mid),
            "D7": {
                n: float(jnp.max(jnp.abs(r_own[n] - r_add[n])))
                for n in var.equation_names
            },
        }

    blocks = list(dict.fromkeys(block_of(n) for n in var.equation_names))
    for gname, cols in tables.items():
        print(f"\n### grid: {gname} (N = {args.n}, fp64)\n")
        heads = list(cols)
        print("| block (variant) | replica | " + " | ".join(heads) + " |")
        print("|---|---|" + "---|" * len(heads))
        for b in blocks:
            names = [n for n in var.equation_names if block_of(n) == b]
            src = parity.equation_source(names[0])
            if b != "world_bond_clearing":
                src = src.rstrip("0123456789").rstrip("_")
            vals = [max(cols[h][n] for n in names) for h in heads]
            cells = " | ".join(f"{v:.1e}" for v in vals)
            print(f"| {b} ({len(names)}) | {src} | {cells} |")
    print(
        "\nNot compared (no counterpart): replica SDF_i (D5: U_store dropped), "
        "replica Wealth_i (D3: the bond mask is the constant bonds_active)."
    )


if __name__ == "__main__":
    main()
