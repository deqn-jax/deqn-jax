"""Phase-0 replica of the RSS-2019 trade DSGE: layout, identities, dynamics.

The replica has no steady state (``steady_state_fn=None``), so the generic
model-contract tests skip its fixed-point legs. What is pinned here instead:
the reference layout (31 / 73 / 82, names and order), the algebraic
identities every residual relies on (budget, trade shares, the bond
projection), the tariff transport map against the closed-form truncated
normal, the Epstein-Zin kernel's CRRA limit, and a two-country instance of
the same code to show the country count is not hard-wired. Parity with the
reference checkpoint itself is gate A of the port and lives outside the
unit suite.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deqn_jax.models import load_model
from deqn_jax.models.rss_trade_ez_ref import build_model
from deqn_jax.models.rss_trade_ez_ref.definitions import clip_policy, core
from deqn_jax.models.rss_trade_ez_ref.dynamics import (
    normal_cdf,
    transport_truncated_normal,
)
from deqn_jax.models.rss_trade_ez_ref.equations import ez_kernel, inside_keys
from deqn_jax.models.rss_trade_ez_ref.variables import (
    POLICY_BLOCKS,
    Layout,
    build_constants,
)
from deqn_jax.networks.rss_net import create_rss_market_clearing_net

SMOKE_HIDDEN = (16, 16)


@pytest.fixture(scope="module")
def model():
    return load_model("rss_trade_ez_ref")


@pytest.fixture(scope="module")
def batch(model):
    """States from the sampler, policies from a random-init reference net."""
    k_state, k_net, k_shock = jax.random.split(jax.random.PRNGKey(0), 3)
    states = model.init_state_fn(k_state, 16, model.constants)
    net = create_rss_market_clearing_net(
        model, key=k_net, base_hidden_sizes=SMOKE_HIDDEN
    )
    policies = jax.vmap(net)(states)
    shocks = jax.random.normal(k_shock, (16, model.n_shocks))
    next_states = model.step_fn(states, policies, shocks, model.constants)
    next_policies = jax.vmap(net)(next_states)
    return net, states, policies, next_states, next_policies


# ---------------------------------------------------------------- layout


def test_reference_layout_dimensions(model):
    assert (model.n_states, model.n_policies, model.n_shocks) == (31, 73, 18)
    assert len(model.equation_names) == 82
    assert len(model.state_names) == 31
    assert len(model.policy_names) == 73
    assert len(model.shock_names) == 18


def test_reference_state_order(model):
    names = model.state_names
    # K/A interleaved per country, then the scaffolding and shock columns
    assert names[:6] == ("K_1", "A_1", "K_2", "A_2", "K_3", "A_3")
    assert names[6] == "homo"
    assert names[7:16] == tuple(f"tau_{i}{j}" for i in "123" for j in "123")
    assert names[16:25] == tuple(f"sigma_tau_{i}{j}" for i in "123" for j in "123")
    assert names[25:] == (
        "A_min",
        "U_store_1",
        "U_store_2",
        "U_store_3",
        "a_mask",
        "homo_1",
    )


def test_reference_policy_and_residual_order(model):
    layout = Layout(3)
    assert len(POLICY_BLOCKS) == 25
    expected = []
    for block in POLICY_BLOCKS:
        expected += ["q"] if block == "q" else [f"{block}_{i}" for i in (1, 2, 3)]
    assert list(model.policy_names) == expected
    assert model.policy_names[layout.blocks["q"][0]] == "q"
    eq = model.equation_names
    assert eq[:3] == ("capital_income_C1", "capital_income_C2", "capital_income_C3")
    assert eq[-2:] == ("Transversality_3", "Wealth_3")
    assert "Good_Market_Clearing_condition" in eq
    assert inside_keys(3) == (
        "ce_1", "ce_2", "ce_3", "eb_1", "eb_2", "eb_3", "ec_1", "ec_2", "ec_3",
    )  # fmt: skip


def test_equation_keys_follow_equation_names(model, batch):
    _, s, p, s2, p2 = batch
    resid = model.equations_fn(s, p, s2, p2, model.constants)
    assert tuple(resid) == tuple(model.equation_names)
    for name, val in resid.items():
        assert val.shape == (16,), name
        assert bool(jnp.all(jnp.isfinite(val))), name


# ------------------------------------------------------------- identities


def test_bond_projection_clears_the_world_market(model, batch):
    """The net's A columns are projected so that sum_i L_i A_i = 0 exactly."""
    _, s, p, _, _ = batch
    layout = Layout(3)
    supply = p[:, layout.blocks["A"]] @ jnp.asarray(model.constants["L"])
    np.testing.assert_allclose(np.asarray(supply), 0.0, atol=1e-6)


def test_budget_identity_and_trade_shares(model, batch):
    _, s, p, _, _ = batch
    d = core(s, p, model.constants, Layout(3))
    # spending exhausts wealth (the 1e-4 floor is not binding on these draws)
    np.testing.assert_allclose(
        np.asarray(d["C"] * d["P_C"] + d["X"] * d["P_X"]),
        np.asarray(d["wealth"]),
        rtol=1e-5,
    )
    assert float(jnp.min(d["wealth"])) > 1e-4
    # Eaton-Kortum shares are a probability row per importer
    np.testing.assert_allclose(np.asarray(d["pi"].sum(axis=2)), 1.0, rtol=1e-5)
    assert float(jnp.min(d["pi"])) > 0.0
    # zero tariffs at the sampler -> omega = 1 everywhere
    np.testing.assert_allclose(np.asarray(d["omega"]), 1.0)


def test_policy_clips_match_the_reference_accessor(model):
    layout = Layout(3)
    raw = jnp.full((layout.n_policies,), -5.0)
    clipped = clip_policy(raw[None, :], layout)[0]
    for block in ("s", "q"):
        assert float(clipped[layout.blocks[block]].min()) == 0.0
    for block in ("U", "mu"):
        np.testing.assert_allclose(np.asarray(clipped[layout.blocks[block]]), 1e-3)
    assert float(clipped[layout.blocks["K"]].min()) == -5.0  # K is not clipped
    raw = jnp.full((layout.n_policies,), 7.0)
    clipped = clip_policy(raw[None, :], layout)[0]
    assert float(clipped[layout.blocks["s"]].max()) == 1.0
    assert float(clipped[layout.blocks["q"]].max()) == 1.0


def test_ez_kernel_reduces_to_crra_when_gamma_is_one_over_psi():
    U_next = jnp.array([0.8, 1.2, 2.0])
    mu = jnp.array([1.0, 1.0, 1.5])
    muc_next = jnp.array([2.0, 0.5, 1.0])
    muc = jnp.array([1.0, 1.0, 2.0])
    k = ez_kernel(U_next, mu, muc_next, muc, gama=2.0, psi=0.5)
    np.testing.assert_allclose(np.asarray(k), np.asarray(muc_next / muc), rtol=1e-6)
    # and the risk adjustment bites when gama > 1/psi: a better continuation
    # (U'/mu > 1) is discounted, a worse one (U'/mu < 1) is weighted up
    k5 = ez_kernel(U_next, mu, muc_next, muc, gama=5.0, psi=0.5)
    assert float(k5[2]) < float(k[2])
    assert float(k5[0]) > float(k[0])


# --------------------------------------------------------------- dynamics


def test_transport_map_matches_the_truncated_normal():
    mu, sigma = 0.01, 0.03  # heavily truncated: P(N < 0) ~ 37%
    z = jnp.linspace(-4.0, 4.0, 2001)
    t = transport_truncated_normal(z, mu, sigma)
    assert float(t.min()) >= 0.0
    assert bool(jnp.all(jnp.diff(t) > 0.0))  # strictly monotone
    # closed-form mean of N(mu, sigma^2) | x >= 0: mu + sigma phi(a)/(1-Phi(a))
    a = -mu / sigma
    phi_a = jnp.exp(-0.5 * a * a) / jnp.sqrt(2.0 * jnp.pi)
    mean_cf = mu + sigma * phi_a / (1.0 - normal_cdf(jnp.asarray(a)))
    draws = jax.random.normal(jax.random.PRNGKey(3), (200_000,))
    mean_mc = float(jnp.mean(transport_truncated_normal(draws, mu, sigma)))
    assert abs(mean_mc - float(mean_cf)) < 3e-4  # ~4 MC standard errors
    # untruncated limit: mu >> sigma -> the map is the identity affine
    far = transport_truncated_normal(z, 1.0, 0.01)
    # float32 erfinv round-trip noise is ~1e-5 at |z| = 4
    np.testing.assert_allclose(np.asarray(far), np.asarray(1.0 + 0.01 * z), atol=5e-5)


def test_step_keeps_tariffs_nonnegative_and_diagonal_dead(model, batch):
    _, s, p, s2, _ = batch
    layout = Layout(3)
    tau2 = s2[:, layout.tau.reshape(-1)].reshape(-1, 3, 3)
    assert float(tau2.min()) >= 0.0
    np.testing.assert_allclose(np.asarray(jnp.diagonal(tau2, axis1=1, axis2=2)), 0.0)
    sig2 = s2[:, layout.sigma_tau.reshape(-1)].reshape(-1, 3, 3)
    np.testing.assert_allclose(np.asarray(jnp.diagonal(sig2, axis1=1, axis2=2)), 0.0)
    # endogenous states are taken from the policy; scaffolding is untouched
    np.testing.assert_allclose(
        np.asarray(s2[:, layout.K]), np.asarray(p[:, layout.blocks["K"]])
    )
    for col in (layout.homo, layout.homo_1, layout.A_min, layout.a_mask):
        np.testing.assert_allclose(np.asarray(s2[:, col]), np.asarray(s[:, col]))


def test_sampler_starts_at_the_reference_scaffolding(model):
    layout = Layout(3)
    s = model.init_state_fn(jax.random.PRNGKey(1), 4, model.constants)
    assert s.shape == (4, 31)
    for col in (layout.homo, layout.homo_1, layout.A_min):
        np.testing.assert_allclose(np.asarray(s[:, col]), 1.0)
    np.testing.assert_allclose(np.asarray(s[:, layout.U_store]), 1.0)
    np.testing.assert_allclose(np.asarray(s[:, layout.a_mask]), 0.0)
    np.testing.assert_allclose(np.asarray(s[:, layout.A]), 0.0)
    np.testing.assert_allclose(np.asarray(s[:, layout.tau.reshape(-1)]), 0.0)
    assert float(s[:, layout.K].min()) > 0.0


# ------------------------------------------------------------ genericity


def test_two_country_instance_of_the_same_code():
    constants = build_constants(
        L=(1.0, 2.0),
        nu_c=(0.6, 0.5),
        nu_m=(0.37, 0.27),
        nu_x=(0.45, 0.2),
        A_c=(1.0, 0.8),
        A_x=(1.0, 1.2),
        T_m=(1.0, 0.2),
        d=((1.0, 2.5), (3.8, 1.0)),
    )
    m = build_model(constants, name="rss_trade_2c")
    assert (m.n_states, m.n_policies, m.n_shocks) == (18, 49, 8)
    assert len(m.equation_names) == 55
    k_state, k_net, k_shock = jax.random.split(jax.random.PRNGKey(0), 3)
    s = m.init_state_fn(k_state, 4, constants)
    net = create_rss_market_clearing_net(m, key=k_net, base_hidden_sizes=SMOKE_HIDDEN)
    p = jax.vmap(net)(s)
    s2 = m.step_fn(s, p, jax.random.normal(k_shock, (4, m.n_shocks)), constants)
    resid = m.equations_fn(s, p, s2, jax.vmap(net)(s2), constants)
    assert tuple(resid) == tuple(m.equation_names)
    assert all(bool(jnp.all(jnp.isfinite(v))) for v in resid.values())


def test_build_constants_rejects_mismatched_calibration():
    with pytest.raises(ValueError, match="nu_c"):
        build_constants(L=(1.0, 2.0), nu_c=(0.6,))


# ------------------------------------------------------------- training


def test_smoke_training_runs_the_two_stage_path():
    from deqn_jax.config import NetworkConfig, OptimizerConfig, TrainConfig
    from deqn_jax.training.trainer import train_from_config

    cfg = TrainConfig(
        model="rss_trade_ez_ref",
        episodes=2,
        batch_size=8,
        episode_length=3,
        initialize_each_episode=True,
        expectation_type="monomial",
        loss_type="mse",
        warm_start=False,
        network=NetworkConfig(
            type="rss_market_clearing_net", hidden_sizes=SMOKE_HIDDEN
        ),
        optimizer=OptimizerConfig(name="adam", learning_rate=1e-4, lr_warmup=1),
        verbose=False,
        seed=0,
    )
    _, history = train_from_config(cfg)
    losses = history["loss"]
    assert len(losses) == 2
    assert all(np.isfinite(losses))
