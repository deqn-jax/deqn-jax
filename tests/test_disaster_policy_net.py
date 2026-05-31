"""Tests for DisasterPolicyNet — disaster-specific shape priors.

DisasterPolicyNet wraps the generic LinearPlusMLP residual ansatz with three
disaster-specific encodings:

  - K/F gauge mask: zero MLP delta at named auxiliary outputs (F_p, K_p, F_w, K_w).
  - ELB feature augmentation: prepend (R_lag − R_lb), raw or kink form.
  - Investment-bracket reparam (`reparam_q_as_m`): treat q output as
    M = q · 𝓑(x), recover q = M/𝓑(x) post-MLP.

Properties pinned here:
  1. Default kf_indices=()  → pure residual ansatz, no masking.
  2. With kf_names=(F_p, K_p, F_w, K_w):
     a. forward at SS returns SS policy exactly for those rows;
     b. ``jax.jacrev`` at SS for those rows equals the BK rows of P;
     c. those rows remain exactly BK-linear after arbitrary inner-MLP
        perturbations (mask is structural, not init-only);
     d. other rows are NOT BK-linear after perturbations (mask is targeted).
  3. ``init_scale=0.0`` makes the full policy exactly BK-linear at init.
  4. Invalid kf_names raises ValueError.
  5. ``zlb_feature_kind ∈ {"raw", "kink"}`` agree above and at the floor;
     diverge below the floor (synthetic states only).
  6. ``reparam_q_as_m=True`` produces q > 0 by construction; turning it on
     vs off changes the q output but leaves SS policy exact.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

jax.config.update("jax_enable_x64", True)

from deqn_jax.models import load_model  # noqa: E402
from deqn_jax.models.disaster.network import (  # noqa: E402
    DisasterPolicyNet,
    create_disaster_policy_net,
)
from deqn_jax.training.linearize import linearize_model  # noqa: E402

KF_NAMES = ("F_p", "K_p", "F_w", "K_w")


def _kf_indices_for(model) -> tuple:
    pn = list(model.policy_names)
    return tuple(pn.index(n) for n in KF_NAMES)


def _other_indices_for(model) -> tuple:
    kf = set(_kf_indices_for(model))
    return tuple(i for i in range(model.n_policies) if i not in kf)


def _build(*, kf_names=(), init_scale=0.0, seed=0) -> DisasterPolicyNet:
    model = load_model("disaster")
    return create_disaster_policy_net(
        model,
        hidden_sizes=(16,),
        activation="tanh",
        init_scale=init_scale,
        kf_names=kf_names,
        key=jr.PRNGKey(seed),
    )


# ---------------------------------------------------------------------------
# K/F gauge mask
# ---------------------------------------------------------------------------


def test_default_kf_indices_empty():
    """Without kf_names, kf_indices defaults to empty (no mask)."""
    net = _build(kf_names=())
    assert net.kf_indices == ()


def test_factory_resolves_kf_names_to_indices():
    """kf_names → kf_indices via model.policy_names lookup."""
    model = load_model("disaster")
    net = _build(kf_names=KF_NAMES)
    assert net.kf_indices == _kf_indices_for(model)


def test_init_scale_zero_gives_exact_bk_at_init():
    """init_scale=0.0: policy = π_BK exactly at every state at step 0."""
    model = load_model("disaster")
    net = _build(kf_names=(), init_scale=0.0)
    ss_state, ss_policy = model.steady_state_fn(model.constants)
    out_ss = net(ss_state)
    assert float(jnp.max(jnp.abs(out_ss - ss_policy))) < 1e-10
    P, _ = linearize_model(model, verbose=False)
    perturbed = ss_state + 1e-2 * jr.normal(jr.PRNGKey(7), ss_state.shape)
    expected = ss_policy + P @ (perturbed - ss_state)
    out = net(perturbed)
    assert float(jnp.max(jnp.abs(out - expected))) < 1e-8


def test_kf_mask_keeps_bk_under_perturbation():
    """KF positions stay exactly BK-linear regardless of inner-MLP weights."""
    model = load_model("disaster")
    net = _build(kf_names=KF_NAMES, init_scale=1.0, seed=3)

    last = net.mlp.layers[-1]
    new_w = last.weight + 0.5 * jr.normal(jr.PRNGKey(11), last.weight.shape)
    new_b = last.bias + 0.5 * jr.normal(jr.PRNGKey(12), last.bias.shape)
    new_last = eqx.tree_at(lambda l: (l.weight, l.bias), last, (new_w, new_b))
    perturbed_mlp = eqx.tree_at(lambda m: m.layers[-1], net.mlp, new_last)
    net = eqx.tree_at(lambda n: n.mlp, net, perturbed_mlp)

    ss_state, ss_policy = model.steady_state_fn(model.constants)
    P, _ = linearize_model(model, verbose=False)

    state = ss_state + 0.05 * jr.normal(jr.PRNGKey(99), ss_state.shape)
    expected_bk = ss_policy + P @ (state - ss_state)
    out = net(state)

    for kf_pos in net.kf_indices:
        diff = float(abs(out[kf_pos] - expected_bk[kf_pos]))
        assert diff < 1e-10, (
            f"KF position {kf_pos} drifted from BK by {diff:.2e} despite mask"
        )


def test_kf_mask_does_not_freeze_other_positions():
    """Non-KF positions DO receive the delta (mask is targeted, not global)."""
    model = load_model("disaster")
    net = _build(kf_names=KF_NAMES, init_scale=1.0, seed=3)
    last = net.mlp.layers[-1]
    new_w = last.weight + 0.5 * jr.normal(jr.PRNGKey(11), last.weight.shape)
    new_b = last.bias + 0.5 * jr.normal(jr.PRNGKey(12), last.bias.shape)
    new_last = eqx.tree_at(lambda l: (l.weight, l.bias), last, (new_w, new_b))
    perturbed_mlp = eqx.tree_at(lambda m: m.layers[-1], net.mlp, new_last)
    net = eqx.tree_at(lambda n: n.mlp, net, perturbed_mlp)

    ss_state, ss_policy = model.steady_state_fn(model.constants)
    P, _ = linearize_model(model, verbose=False)
    state = ss_state + 0.05 * jr.normal(jr.PRNGKey(99), ss_state.shape)
    expected_bk = ss_policy + P @ (state - ss_state)
    out = net(state)

    other = _other_indices_for(model)
    deviations = jnp.array([abs(out[i] - expected_bk[i]) for i in other])
    assert float(jnp.max(deviations)) > 1e-3, (
        "non-KF positions appear frozen at BK; mask leaked or delta is dead"
    )


def test_jacrev_kf_rows_match_linearization():
    """For KF positions, the network Jacobian at SS equals the BK rows of P."""
    model = load_model("disaster")
    net = _build(kf_names=KF_NAMES, init_scale=1.0, seed=2)
    ss_state, _ = model.steady_state_fn(model.constants)
    P, _ = linearize_model(model, verbose=False)

    J = jax.jacrev(lambda s: net(s))(ss_state)
    for kf_pos in net.kf_indices:
        diff = float(jnp.linalg.norm(J[kf_pos] - P[kf_pos]))
        assert diff < 1e-10, f"KF row {kf_pos} jacobian deviates from P by {diff:.2e}"


def test_invalid_kf_name_raises():
    """Anchoring a name not in policy_names fails loud."""
    model = load_model("disaster")
    with pytest.raises(ValueError, match="not found in model.policy_names"):
        create_disaster_policy_net(
            model,
            hidden_sizes=(8,),
            activation="tanh",
            kf_names=("F_p", "totally_not_a_policy"),
            key=jr.PRNGKey(0),
        )


def test_kf_indices_out_of_range_raises():
    """Direct construction with out-of-range kf_indices fails loud."""
    model = load_model("disaster")
    P, _ = linearize_model(model, verbose=False)
    ss_state, ss_policy = model.steady_state_fn(model.constants)
    with pytest.raises(ValueError, match="out of range for n_policies"):
        DisasterPolicyNet(
            n_states=model.n_states,
            n_policies=model.n_policies,
            hidden_sizes=(8,),
            activation="tanh",
            P=P,
            ss_state=ss_state,
            ss_policy=ss_policy,
            kf_indices=(0, 99),
            key=jr.PRNGKey(0),
        )


def test_2d_input_shape():
    """Vmapped forward returns [batch, n_policies]."""
    model = load_model("disaster")
    net = _build(kf_names=KF_NAMES)
    ss_state, _ = model.steady_state_fn(model.constants)
    batch = jnp.broadcast_to(ss_state[None, :], (5, model.n_states))
    out = net(batch)
    assert out.shape == (5, model.n_policies)
    assert bool(jnp.all(jnp.isfinite(out)))


# ---------------------------------------------------------------------------
# zlb_feature_kind: PINN-style kink input vs raw signed-distance feature
# ---------------------------------------------------------------------------


def _build_zlb(
    *, kind: str, seed: int = 0, init_scale: float = 1.0
) -> DisasterPolicyNet:
    """Build a DisasterPolicyNet with ZLB feature on, raw or kink form."""
    model = load_model("disaster")
    net = create_disaster_policy_net(
        model,
        hidden_sizes=(16,),
        activation="tanh",
        init_scale=init_scale,
        use_zlb_feature=True,
        zlb_feature_kind=kind,
        key=jr.PRNGKey(seed),
    )
    last = net.mlp.layers[-1]
    new_w = last.weight + 0.5 * jr.normal(jr.PRNGKey(seed + 1), last.weight.shape)
    new_b = last.bias + 0.5 * jr.normal(jr.PRNGKey(seed + 2), last.bias.shape)
    new_last = eqx.tree_at(lambda l: (l.weight, l.bias), last, (new_w, new_b))
    perturbed_mlp = eqx.tree_at(lambda m: m.layers[-1], net.mlp, new_last)
    return eqx.tree_at(lambda n: n.mlp, net, perturbed_mlp)


def test_zlb_feature_kind_default_is_raw():
    """Without explicit override, zlb_feature_kind defaults to 'raw'."""
    model = load_model("disaster")
    net = create_disaster_policy_net(
        model,
        hidden_sizes=(8,),
        activation="tanh",
        use_zlb_feature=True,
        key=jr.PRNGKey(0),
    )
    assert net.zlb_feature_kind == "raw"


def test_zlb_feature_kind_invalid_raises():
    """Bad kind value fails loud at construction."""
    model = load_model("disaster")
    with pytest.raises(ValueError, match="zlb_feature_kind"):
        create_disaster_policy_net(
            model,
            hidden_sizes=(8,),
            activation="tanh",
            use_zlb_feature=True,
            zlb_feature_kind="sigmoid",  # type: ignore[arg-type]
            key=jr.PRNGKey(0),
        )


def test_zlb_kind_agrees_above_floor():
    """At R_lag > R_lb, raw and kink forms produce identical output."""
    model = load_model("disaster")
    ss_state, _ = model.steady_state_fn(model.constants)
    raw_net = _build_zlb(kind="raw", seed=7)
    kink_net = _build_zlb(kind="kink", seed=7)
    out_raw = raw_net(ss_state)
    out_kink = kink_net(ss_state)
    assert float(jnp.max(jnp.abs(out_raw - out_kink))) < 1e-12


def test_zlb_kind_agrees_at_floor():
    """At R_lag = R_lb, both forms feed 0 to the MLP, outputs agree."""
    model = load_model("disaster")
    raw_net = _build_zlb(kind="raw", seed=7)
    kink_net = _build_zlb(kind="kink", seed=7)
    ss_state, _ = model.steady_state_fn(model.constants)
    state_at_floor = ss_state.at[raw_net.r_lag_idx].set(raw_net.r_lb)
    out_raw = raw_net(state_at_floor)
    out_kink = kink_net(state_at_floor)
    assert float(jnp.max(jnp.abs(out_raw - out_kink))) < 1e-12


def test_zlb_kind_diverges_below_floor():
    """Synthetic R_lag < R_lb: raw form sees negative feature, kink clips to 0."""
    model = load_model("disaster")
    raw_net = _build_zlb(kind="raw", seed=11)
    kink_net = _build_zlb(kind="kink", seed=11)
    ss_state, _ = model.steady_state_fn(model.constants)
    state_below = ss_state.at[raw_net.r_lag_idx].set(raw_net.r_lb - 0.1)
    out_raw = raw_net(state_below)
    out_kink = kink_net(state_below)
    diff = float(jnp.max(jnp.abs(out_raw - out_kink)))
    assert diff > 1e-4, (
        f"raw vs kink should diverge below the floor; got max abs diff {diff:.2e}"
    )


def test_zlb_kink_feature_is_zero_below_floor():
    """Crossing the floor: kink form output varies less than raw."""
    model = load_model("disaster")
    kink_net = _build_zlb(kind="kink", seed=13)
    raw_net = _build_zlb(kind="raw", seed=13)
    ss_state, _ = model.steady_state_fn(model.constants)
    state_at_floor = ss_state.at[kink_net.r_lag_idx].set(kink_net.r_lb)
    state_below = ss_state.at[kink_net.r_lag_idx].set(kink_net.r_lb - 0.1)

    raw_diff = float(jnp.max(jnp.abs(raw_net(state_below) - raw_net(state_at_floor))))
    kink_diff = float(
        jnp.max(jnp.abs(kink_net(state_below) - kink_net(state_at_floor)))
    )
    assert kink_diff < raw_diff


# ---------------------------------------------------------------------------
# Investment-bracket reparameterization (§3.3): output M = q · 𝓑(x); recover q
# ---------------------------------------------------------------------------


def test_qm_reparam_off_by_default():
    """Without explicit reparam_q_as_m, the flag is off."""
    net = _build(kf_names=KF_NAMES)
    assert net.reparam_q_as_m is False


def test_qm_reparam_factory_resolves_indices():
    """When on, factory resolves q/i indices and kappa/mu_z_ss from model."""
    model = load_model("disaster")
    net = create_disaster_policy_net(
        model,
        hidden_sizes=(8,),
        activation="tanh",
        reparam_q_as_m=True,
        key=jr.PRNGKey(0),
    )
    assert net.reparam_q_as_m is True
    pn = list(model.policy_names)
    sn = list(model.state_names)
    assert net.q_idx == pn.index("q")
    assert net.i_idx == pn.index("i")
    assert net.i_lag_idx == sn.index("i_lag")
    assert net.mu_z_idx == sn.index("mu_z")
    assert net.kappa == float(model.constants["kappa"])
    assert net.mu_z_ss == float(model.constants["mu_z_ss"])


def test_qm_reparam_exact_bk_at_ss():
    """At SS with init_scale=0, qm-reparam policy still equals BK exactly.

    At SS, x_ss = µ_z_ss so 𝓑(x_ss) = 1 ⇒ M_BK = q_BK. Recovery: q = M/1 = q_BK.
    """
    model = load_model("disaster")
    net = create_disaster_policy_net(
        model,
        hidden_sizes=(8,),
        activation="tanh",
        init_scale=0.0,
        reparam_q_as_m=True,
        key=jr.PRNGKey(0),
    )
    ss_state, ss_policy = model.steady_state_fn(model.constants)
    out = net(ss_state)
    assert float(jnp.max(jnp.abs(out - ss_policy))) < 1e-10


def test_qm_reparam_q_positive_under_perturbation():
    """With reparam on, q stays > 0 even when raw policy outputs are perturbed.

    Without reparam, a sufficiently large negative δ at the q slot can drive q
    negative (clipped by policy_lower). With reparam, q = M/𝓑(x) inherits its
    sign from M (always positive after softplus-equivalent training); 𝓑(x)
    is floored at 1e-3 in the recovery step, so q > 0 is guaranteed before
    clipping.
    """
    model = load_model("disaster")
    net = create_disaster_policy_net(
        model,
        hidden_sizes=(8,),
        activation="tanh",
        init_scale=1.0,
        reparam_q_as_m=True,
        key=jr.PRNGKey(11),
    )
    # Perturb final layer to give the network non-trivial output.
    last = net.mlp.layers[-1]
    new_w = last.weight + 0.5 * jr.normal(jr.PRNGKey(12), last.weight.shape)
    new_b = last.bias + 0.5 * jr.normal(jr.PRNGKey(13), last.bias.shape)
    new_last = eqx.tree_at(lambda l: (l.weight, l.bias), last, (new_w, new_b))
    perturbed = eqx.tree_at(lambda m: m.layers[-1], net.mlp, new_last)
    net = eqx.tree_at(lambda n: n.mlp, net, perturbed)

    ss_state, _ = model.steady_state_fn(model.constants)
    states = ss_state[None, :] + 0.02 * jr.normal(
        jr.PRNGKey(99), (32, ss_state.shape[0])
    )
    out = net(states)
    q_idx = list(model.policy_names).index("q")
    q_vals = out[:, q_idx]
    assert bool(jnp.all(q_vals > 0)), f"q went non-positive: min={float(q_vals.min())}"


def test_qm_reparam_requires_kappa():
    """If kappa is missing or zero, factory raises informative error."""
    model = load_model("disaster")
    bad_constants = dict(model.constants)
    bad_constants["kappa"] = 0.0
    bad_model = model._replace(constants=bad_constants)
    with pytest.raises(ValueError, match="kappa"):
        create_disaster_policy_net(
            bad_model,
            hidden_sizes=(8,),
            activation="tanh",
            reparam_q_as_m=True,
            key=jr.PRNGKey(0),
        )
