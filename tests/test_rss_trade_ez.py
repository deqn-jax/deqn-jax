"""Phase-1 variant of the RSS-2019 trade DSGE (``rss_trade_ez``).

Pinned here: the cleaned layout (18 / 70 / 76 / 12), each variant's
mechanics (constants instead of scaffolding columns, the auxiliary
transversality term, the bond projection in definitions, the off-diagonal
shock set, one tariff measure through the transport map, capital by the
accumulation identity), and residual parity with the replica on shared
physical states for one and the same policy: every residual the two models
share agrees to rounding once the node set and the capital transition are
aligned, so the remaining deltas are the variants' own. The replica-side
identities (budget, trade shares, kernel limit, transport against the closed
form) are pinned in ``test_rss_trade_ez_ref.py`` on the shared code.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deqn_jax.config import NetworkConfig
from deqn_jax.models import load_model
from deqn_jax.models.rss_trade_ez import build_constants, build_model, parity
from deqn_jax.models.rss_trade_ez.definitions import bond_projection, core
from deqn_jax.models.rss_trade_ez.variables import Layout
from deqn_jax.models.rss_trade_ez_ref.definitions import capital_identity
from deqn_jax.models.rss_trade_ez_ref.definitions import core as ref_core
from deqn_jax.models.rss_trade_ez_ref.dynamics import (
    normal_cdf,
    transport_truncated_normal,
)
from deqn_jax.models.rss_trade_ez_ref.equations import inside_keys
from deqn_jax.models.rss_trade_ez_ref.variables import Layout as RefLayout
from deqn_jax.networks.factory import build_policy_net
from deqn_jax.training.loss import monomial_nd

B = 16


def _net(model, key=1):
    cfg = NetworkConfig(hidden_sizes=(16,), activation="gelu")
    return build_policy_net(model, jax.random.PRNGKey(key), (16,), cfg)


@pytest.fixture(scope="module")
def model():
    return load_model("rss_trade_ez")


@pytest.fixture(scope="module")
def batch(model):
    """Sampler states with positive tariffs and nonzero bonds; random-MLP
    policies (its bounded head as configured by the model)."""
    lay = Layout(3)
    k_s, k_t, k_a, k_e = jax.random.split(jax.random.PRNGKey(0), 4)
    s = model.init_state_fn(k_s, B, model.constants)
    s = s.at[:, lay.tau].set(jax.random.uniform(k_t, (B, 6), maxval=0.4))
    s = s.at[:, lay.A].set(0.05 * jax.random.normal(k_a, (B, 3)))
    net = _net(model)
    p = net(s)
    s2 = model.step_fn(s, p, jax.random.normal(k_e, (B, 12)), model.constants)
    return net, s, p, s2, net(s2)


def _with(model, **overrides):
    return build_model({**build_constants(), **overrides}, name="rss_trade_ez_t")


# ---------------------------------------------------------------- layout


def test_variant_layout_dimensions(model):
    assert (model.n_states, model.n_policies, model.n_shocks) == (18, 70, 12)
    assert len(model.equation_names) == 76
    assert len(model.policy_names) == 70 and len(model.shock_names) == 12


def test_variant_names(model):
    assert model.state_names == (
        ("K_1", "A_1", "K_2", "A_2", "K_3", "A_3")
        + tuple(f"tau_{p}" for p in ("12", "13", "21", "23", "31", "32"))
        + tuple(f"sig_{p}" for p in ("12", "13", "21", "23", "31", "32"))
    )
    assert model.shock_names[:6] == tuple(
        f"eps_sigma_{p}" for p in ("12", "13", "21", "23", "31", "32")
    )
    assert model.shock_names[6:] == tuple(
        f"eps_tau_{p}" for p in ("12", "13", "21", "23", "31", "32")
    )
    for gone in ("U_store_1", "K_1"):
        assert gone not in model.policy_names
    assert {"K_prime_1", "a_1", "U_1", "mu_1", "q"} <= set(model.policy_names)
    eq = model.equation_names
    assert eq[0] == "capital_income_C_1" and eq[-1] == "value_function_3"
    assert "world_bond_clearing" in eq
    assert not any(n.startswith(("SDF", "Wealth", "Transversality")) for n in eq)
    assert [n for n in eq if n.startswith("aux_")] == [
        "aux_transversality_1",
        "aux_transversality_2",
        "aux_transversality_3",
    ]


def test_residual_keys_follow_names_and_are_finite(model, batch):
    _, s, p, s2, p2 = batch
    r = model.equations_fn(s, p, s2, p2, model.constants)
    assert tuple(r) == model.equation_names
    for name, v in r.items():
        assert v.shape == (B,) and bool(jnp.all(jnp.isfinite(v))), name
    ins = model.inside_fn(s, p, s2, p2, model.constants)
    assert tuple(ins) == inside_keys(3)  # CE / EB / EC


def test_policy_bounds_by_block(model):
    lay = Layout(3)
    lo, hi = np.asarray(model.policy_lower), np.asarray(model.policy_upper)
    s = lay.blocks["s"]
    np.testing.assert_allclose(lo[s], 1e-4)
    np.testing.assert_allclose(hi[s], 1.0 - 1e-4, rtol=1e-6)
    assert np.all(np.isinf(hi[lay.blocks["P_C"]])) and lo[lay.blocks["U"]][0] == 1e-3
    a = lay.blocks["a"]
    np.testing.assert_allclose(lo[a], -0.5)
    np.testing.assert_allclose(hi[a], 0.5)


# ---------------------------------------------- D9: projection in definitions


def test_bond_projection_clears_and_is_idempotent(model, batch):
    _, s, p, _, _ = batch
    lay = Layout(3)
    d = core(s, p, model.constants, lay)
    assert float(jnp.max(jnp.abs(d["a"]))) > 1e-3  # non-vacuous
    L = jnp.asarray(model.constants["L"])
    np.testing.assert_allclose(np.asarray(d["A"] @ L), 0.0, atol=1e-12)
    again = bond_projection(d["A"], jnp.broadcast_to(L, d["A"].shape))
    np.testing.assert_allclose(np.asarray(again), np.asarray(d["A"]), atol=1e-14)
    # the state transition carries the cleared position
    s2 = model.step_fn(s, p, jnp.zeros((B, 12)), model.constants)
    np.testing.assert_allclose(np.asarray(s2[:, lay.A]), np.asarray(d["A"]))


# ------------------------------------------------- D4: aux transversality


def test_aux_transversality_is_the_constant_weight_term(model, batch):
    _, s, p, s2, p2 = batch
    lay = Layout(3)
    m2 = _with(model, w_tv=2.5)
    r = m2.equations_fn(s, p, s2, p2, m2.constants)
    d = core(s, p, m2.constants, lay)
    for i in range(3):
        ms = float(jnp.mean(r[f"aux_transversality_{i + 1}"] ** 2))
        want = 2.5 * float(
            jnp.mean(d["muc"][:, i] / d["P_C"][:, i] * d["A"][:, i] ** 2)
        )
        assert ms == pytest.approx(want, rel=1e-10)
    # pointwise: a state's residual does not depend on the rest of the batch
    r1 = m2.equations_fn(s[:1], p[:1], s2[:1], p2[:1], m2.constants)
    np.testing.assert_allclose(
        np.asarray(r1["aux_transversality_2"]),
        np.asarray(r["aux_transversality_2"][:1]),
    )


def test_aux_residuals_reject_update_paths_that_drop_them():
    from deqn_jax.config import OptimizerConfig, TrainConfig
    from deqn_jax.training.state_init import _resolve_model_for_training

    base = dict(model="rss_trade_ez", batch_size=8, episode_length=2, verbose=False)
    _resolve_model_for_training(TrainConfig(**base))  # adam: accepted
    for bad in (
        dict(optimizer=OptimizerConfig(name="mao")),
        dict(gradient_surgery="pcgrad"),
        dict(loss_reweight="relobralo"),
    ):
        with pytest.raises(ValueError, match="auxiliary residuals"):
            _resolve_model_for_training(TrainConfig(**base, **bad))


# ------------------------------------------- D3: curriculum as constants


def test_homotopy_constants_move_the_calibration(model, batch):
    _, s, p, _, _ = batch
    lay = Layout(3)
    c = {**build_constants(), "homotopy_asym": 0.0, "homotopy_trade": 0.0}
    d = core(s, p, c, lay)
    np.testing.assert_allclose(np.asarray(d["L_eff"]), float(np.mean(c["L"])))
    # near-autarky trade costs: imports almost all domestic
    assert float(jnp.min(jnp.diagonal(d["pi"], axis1=1, axis2=2))) > 0.99
    d1 = core(s, p, model.constants, lay)
    np.testing.assert_allclose(np.asarray(d1["L_eff"][0]), np.asarray(c["L"]))


def test_bonds_shut_imposes_zero_bonds(model, batch):
    _, s, p, s2, p2 = batch
    lay = Layout(3)
    m0 = _with(model, bonds_active=0.0)
    d = core(s, p, m0.constants, lay)
    np.testing.assert_allclose(np.asarray(d["A"]), 0.0)
    r = m0.equations_fn(s, p, s2, p2, m0.constants)
    for i in range(3):  # the bond Euler becomes a_i = 0
        np.testing.assert_allclose(
            np.asarray(r[f"euler_bond_{i + 1}"]), np.asarray(d["a"][:, i])
        )
    assert len(r) == 76
    nxt = m0.step_fn(s, p, jnp.zeros((B, 12)), m0.constants)
    np.testing.assert_allclose(np.asarray(nxt[:, lay.A]), 0.0)
    with pytest.raises(ValueError, match="bonds_active"):
        m5 = _with(model, bonds_active=0.5)
        m5.equations_fn(s, p, s2, p2, m5.constants)


def test_constants_override_reaches_the_residuals():
    from deqn_jax.config import TrainConfig
    from deqn_jax.training.state_init import _resolve_model_for_training

    cfg = TrainConfig(model="rss_trade_ez", batch_size=8, episode_length=2)
    cfg = cfg.with_overrides({"constants.homotopy_asym": "0.25", "constants.w_tv": "0"})
    assert cfg.constants == {"homotopy_asym": 0.25, "w_tv": 0}
    m, _ = _resolve_model_for_training(cfg)
    assert m.constants["homotopy_asym"] == 0.25 and m.constants["w_tv"] == 0
    assert m.constants["beta"] == 0.96  # the rest of the calibration intact


# ------------------------------- D6 / D7: off-diagonal shocks, one measure


def test_tariffs_follow_the_transport_map(model, batch):
    _, s, p, _, _ = batch
    lay = Layout(3)
    c = model.constants
    z = jnp.linspace(-5.0, 5.0, 41)
    rows = []
    for zi in z:  # tau innovation on pair 13 only; log-vol innovations zero
        shock = jnp.zeros((B, 12)).at[:, 6 + 1].set(zi)
        rows.append(model.step_fn(s, p, shock, c)[:, lay.tau[1]])
    tau13 = jnp.stack(rows)  # [41, B]
    assert float(tau13.min()) >= 0.0
    assert bool(jnp.all(jnp.diff(tau13, axis=0) >= 0.0))  # node -> tau monotone
    assert float(jnp.max(tau13[-1] - tau13[0])) > 0.0
    # the other pairs did not move with it
    sig_next = 0.04 * -6.14 + 0.96 * s[:, lay.sig[1]]
    want = transport_truncated_normal(
        z[:, None], 0.99 * s[None, :, lay.tau[1]], jnp.exp(sig_next)[None, :]
    )
    np.testing.assert_allclose(np.asarray(tau13), np.asarray(want), rtol=1e-10)


def test_simulation_draws_land_on_the_truncated_law(model):
    """Standard-normal draws through step_fn: never negative, and their mean
    matches the closed-form truncated-normal mean at a heavily truncated
    point (mu = 0.99 tau ~ 0.01, sigma ~ 0.03)."""
    lay = Layout(3)
    c = model.constants
    n = 200_000
    s = jnp.zeros((n, 18)).at[:, lay.K].set(0.1)
    s = (
        s.at[:, lay.tau]
        .set(0.0101)
        .at[:, lay.sig]
        .set(jnp.log(0.03) / 0.96 + 6.14 * 0.04 / 0.96)
    )
    shock = jax.random.normal(jax.random.PRNGKey(9), (n, 12)).at[:, :6].set(0.0)
    p = jnp.tile(_net(model)(s[:1]), (n, 1))
    tau = model.step_fn(s, p, shock, c)[:, lay.tau]
    assert float(tau.min()) >= 0.0
    mu, sigma = 0.99 * 0.0101, 0.03
    a = -mu / sigma
    mean_cf = mu + sigma * np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi) / (
        1.0 - float(normal_cdf(jnp.asarray(a)))
    )
    assert float(jnp.mean(tau)) == pytest.approx(mean_cf, abs=3e-4)
    sig_next = model.step_fn(s, p, shock, c)[:, lay.sig]
    np.testing.assert_allclose(np.asarray(jnp.exp(sig_next)), sigma, rtol=1e-10)


def test_capital_follows_the_accumulation_identity(model, batch):
    _, s, p, s2, p2 = batch
    lay = Layout(3)
    d = core(s, p, model.constants, lay)
    np.testing.assert_allclose(
        np.asarray(s2[:, lay.K]),
        np.asarray(capital_identity(d["K_state"], d["X"], model.constants)),
    )
    # with K_prime set to the identity the law-of-motion residual vanishes
    pk = p.at[:, lay.blocks["K_prime"]].set(s2[:, lay.K])
    r = model.equations_fn(s, pk, s2, p2, model.constants)
    for i in range(3):
        np.testing.assert_allclose(
            np.asarray(r[f"law_of_motion_K_{i + 1}"]), 0.0, atol=1e-14
        )


# --------------------------------------------- parity with the replica


@pytest.fixture(scope="module")
def shared():
    """Shared physical states (replica sampler, positive tariffs, raised
    volatility, nonzero bonds) and a random replica policy whose capital
    column is the accumulation identity, so both transitions coincide."""
    ref = load_model("rss_trade_ez_ref")
    var = load_model("rss_trade_ez")
    rl = RefLayout(3)
    k = jax.random.split(jax.random.PRNGKey(4), 4)
    s = ref.init_state_fn(k[0], 12, ref.constants)
    off = ~np.eye(3, dtype=bool)
    s = s.at[:, rl.tau[off]].set(jax.random.uniform(k[1], (12, 6), maxval=0.5))
    s = s.at[:, rl.sigma_tau[off]].add(jax.random.uniform(k[2], (12, 6), minval=1.0, maxval=2.5))
    s = s.at[:, rl.A].set(0.05 * jax.random.normal(k[3], (12, 3)))
    net = parity.random_reference_net(ref, jax.random.PRNGKey(5))

    def ref_policy(x):
        p = jax.vmap(net)(parity.pin_scaffolding(x, ref))
        d = ref_core(x, p, ref.constants, rl)
        return p.at[:, rl.blocks["K"]].set(
            capital_identity(d["K_state"], d["X"], ref.constants)
        )

    return ref, var, s, ref_policy, rl


def test_shared_residuals_match_the_replica(shared):
    """Same policy, replica's node set mapped onto the variant's shocks: all
    73 residuals with a counterpart agree to rounding (the 3 transversality
    residuals measure different quantities, D4)."""
    from deqn_jax.models.rss_trade_ez_ref.definitions import clip_policy

    ref, var, s, ref_policy, rl = shared
    rule = monomial_nd(ref.n_shocks)
    r_ref = parity.quadrature_residuals(ref, ref_policy, s, *rule)
    pol = parity.variant_policy_fn(lambda x: clip_policy(ref_policy(x), rl), ref, var)
    nodes = parity.replica_nodes_on_variant(ref, var, rule[0])
    r_var = parity.quadrature_residuals(
        var, pol, parity.to_variant_state(s, ref, var), nodes, rule[1]
    )
    compared = 0
    for name in var.equation_names:
        if name.startswith("aux_"):
            continue
        src = parity.equation_source(name)
        np.testing.assert_allclose(
            np.asarray(r_var[name]),
            np.asarray(r_ref[src]),
            rtol=1e-9,
            atol=1e-12,
            err_msg=f"{name} vs {src}",
        )
        compared += 1
    assert compared == 73


def test_own_node_set_moves_only_the_expectation_residuals(shared):
    """Under the variant's own 24-node rule (D6) only the three
    expectation-bearing blocks change; every current-period one is equal."""
    from deqn_jax.models.rss_trade_ez_ref.definitions import clip_policy

    ref, var, s, ref_policy, rl = shared
    r_ref = parity.quadrature_residuals(ref, ref_policy, s, *monomial_nd(18))
    pol = parity.variant_policy_fn(lambda x: clip_policy(ref_policy(x), rl), ref, var)
    r_var = parity.quadrature_residuals(
        var, pol, parity.to_variant_state(s, ref, var), *monomial_nd(12)
    )
    moved = set()
    for name in var.equation_names:
        if name.startswith("aux_"):
            continue
        delta = float(
            jnp.max(jnp.abs(r_var[name] - r_ref[parity.equation_source(name)]))
        )
        if delta > 1e-9:
            moved.add(name.rpartition("_")[0])
    assert moved == {"euler_bond", "euler_capital", "certainty_equivalent"}


def test_name_maps_cover_both_layouts(shared):
    ref, var, *_ = shared
    for n in var.state_names:
        assert parity.state_source(n) in ref.state_names
    for n in var.policy_names:
        assert parity.policy_source(n) in ref.policy_names
    srcs = [parity.equation_source(n) for n in var.equation_names]
    assert set(srcs) <= set(ref.equation_names) and len(set(srcs)) == 76
    unmatched = set(ref.equation_names) - set(srcs)
    assert unmatched == {f"{b}_{i}" for b in ("SDF", "Wealth") for i in (1, 2, 3)}


# ------------------------------------------------------------ genericity


def test_two_country_instance_of_the_same_code():
    c = build_constants(
        L=(1.0, 2.0), nu_c=(0.6, 0.5), nu_m=(0.37, 0.27), nu_x=(0.45, 0.2),
        A_c=(1.0, 0.8), A_x=(1.0, 1.2), T_m=(1.0, 0.2), d=((1.0, 2.5), (3.8, 1.0)),
    )  # fmt: skip
    m = build_model(c, name="rss_trade_ez_2c")
    assert (m.n_states, m.n_policies, m.n_shocks) == (8, 47, 4)
    assert len(m.equation_names) == 51
    s = m.init_state_fn(jax.random.PRNGKey(0), 4, c)
    net = _net(m)
    p = net(s)
    s2 = m.step_fn(s, p, jax.random.normal(jax.random.PRNGKey(1), (4, 4)), c)
    r = m.equations_fn(s, p, s2, net(s2), c)
    assert tuple(r) == m.equation_names
    assert all(bool(jnp.all(jnp.isfinite(v))) for v in r.values())


# ------------------------------------------------------------- training


def test_smoke_training_with_the_standard_mlp():
    from deqn_jax.config import OptimizerConfig, TrainConfig
    from deqn_jax.training.trainer import train_from_config

    cfg = TrainConfig(
        model="rss_trade_ez",
        episodes=3,
        batch_size=16,
        episode_length=3,
        initialize_each_episode=True,
        expectation_type="monomial",
        loss_type="mse",
        warm_start=False,
        network=NetworkConfig(type="mlp", hidden_sizes=(16,), activation="gelu"),
        optimizer=OptimizerConfig(name="adam", learning_rate=1e-3, lr_warmup=1),
        verbose=False,
        seed=0,
    )
    _, history = train_from_config(cfg)
    losses = history["loss"]
    assert len(losses) == 3 and all(np.isfinite(losses))
    assert losses[-1] < losses[0]
