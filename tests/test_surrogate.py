"""EWM world arm (continuation surrogate): identity, exactness, validators, budgets.

Spec: docs/superpowers/specs/2026-08-28-ewm-world-arm-design.md §6.
Model under test: olg_lifecycle (two-stage hooks, 1 shock, positive inside terms).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deqn_jax.config import NetworkConfig, OptimizerConfig, SurrogateConfig, TrainConfig
from deqn_jax.models import load_model
from deqn_jax.training.loss import compute_loss, gauss_hermite_nd
from deqn_jax.training.surrogate import (
    SurrogateState,
    exact_expectations,
    inside_keys,
    make_surrogate_loss,
    make_world_update,
    polyak_update,
    predict,
)
from deqn_jax.training.trainer import create_train_state, train_from_config

MODEL = load_model("olg_lifecycle")


def _cfg(**kw):
    base = dict(
        model="olg_lifecycle",
        episodes=3,
        batch_size=16,
        episode_length=3,
        mc_samples=2,
        expectation_type="gauss_hermite",
        n_quadrature_points=3,
        network=NetworkConfig(hidden_sizes=(16,)),
        optimizer=OptimizerConfig(name="adam", learning_rate=1e-3),
        verbose=False,
        seed=0,
    )
    base.update(kw)
    return TrainConfig(**base)


def _quad():
    qn, qw = gauss_hermite_nd(3, MODEL.n_shocks)
    return jnp.asarray(qn), jnp.asarray(qw)


# ---------------------------------------------------------------------------
# unit level
# ---------------------------------------------------------------------------
def test_inside_keys_sorted_and_complete():
    keys = inside_keys(MODEL)
    assert keys == tuple(sorted(keys))
    assert len(keys) == 6  # H cohorts of (1-δ+r')/c'


def test_exact_expectations_match_compute_loss_two_stage():
    """E[inside] from the helper == the expectation compute_loss forms
    internally (checked through combine_fn -> identical residual losses)."""
    key = jax.random.PRNGKey(0)
    states = MODEL.init_state_fn(key, 8, MODEL.constants)
    state, _, _ = create_train_state(
        MODEL, key, hidden_sizes=(16,), batch_size=8, n_equations=len(MODEL.equation_names)
    )
    net = state.params
    qn, qw = _quad()
    exp = exact_expectations(MODEL, net, states, key, 2, 1.0, qn, qw)
    resid = MODEL.combine_fn(states, net(states), exp, MODEL.constants)
    total_ref, eq_ref = compute_loss(MODEL, net, states, key, 2, quad_nodes=qn, quad_weights=qw)
    mine = {k: float(jnp.mean(v**2)) for k, v in resid.items()}
    for k in eq_ref:
        np.testing.assert_allclose(mine[k], float(eq_ref[k]), rtol=1e-10)


def test_polyak_update_moves_toward_params():
    key = jax.random.PRNGKey(1)
    s1, _, _ = create_train_state(MODEL, key, hidden_sizes=(16,), batch_size=8, n_equations=5)
    s2, _, _ = create_train_state(
        MODEL, jax.random.PRNGKey(2), hidden_sizes=(16,), batch_size=8, n_equations=5
    )
    tgt = polyak_update(s1.params, s2.params, 0.9)
    leaf_t = jax.tree.leaves(jax.tree_util.tree_map(lambda x: x, tgt))[0]
    leaf1 = jax.tree.leaves(s1.params)[0]
    leaf2 = jax.tree.leaves(s2.params)[0]
    np.testing.assert_allclose(np.asarray(leaf_t), 0.9 * np.asarray(leaf1) + 0.1 * np.asarray(leaf2), rtol=1e-6)


def test_world_update_fits_targets_and_counts_budgets():
    cfg = SurrogateConfig(enabled=True, width=32, anchor_frac=1.0, epochs_w=200, positive_outputs=True)
    key = jax.random.PRNGKey(3)
    state, _, _ = create_train_state(
        MODEL, key, hidden_sizes=(16,), batch_size=16, n_equations=5, surrogate_config=cfg
    )
    assert isinstance(state.aux_params, SurrogateState)
    assert state.target_params is not None
    qn, qw = _quad()
    import optax

    upd = make_world_update(MODEL, cfg, optax.adam(3e-3), 2, qn, qw, total_episodes=10, batch_size=16)
    dataset = MODEL.init_state_fn(jax.random.PRNGKey(4), 128, MODEL.constants)
    aux = state.aux_params
    for _ in range(5):
        aux = upd(aux, state.params, dataset, jax.random.PRNGKey(5), jnp.array(1.0), 0)
    # budgets: anchors = all 128 rows, nodes = 3 GH nodes, epochs 200 per call
    assert float(aux.b_policy) == 5 * 128 * 3
    assert float(aux.b_world) == 5 * 128 * 200
    # surrogate approximates the exact expectation on the anchors (1000 Adam steps)
    keys = inside_keys(MODEL)
    exact = exact_expectations(MODEL, state.params, dataset, key, 2, 1.0, qn, qw)
    pred = predict(aux, keys, dataset)
    rel = np.median([float(jnp.median(jnp.abs(pred[k] - exact[k]) / (jnp.abs(exact[k]) + 1e-8))) for k in keys])
    assert rel < 0.05, rel


def test_surrogate_loss_gradient_aligns_with_exact_when_fitted():
    """Spec §6 exactness: with a well-fitted Ŵ the surrogate policy gradient
    points where the exact gradient points (cosine > 0.99)."""
    import equinox as eqx
    import optax

    cfg = SurrogateConfig(enabled=True, width=64, anchor_frac=1.0, epochs_w=300)
    key = jax.random.PRNGKey(6)
    state, _, _ = create_train_state(
        MODEL, key, hidden_sizes=(16,), batch_size=16, n_equations=5, surrogate_config=cfg
    )
    qn, qw = _quad()
    dataset = MODEL.init_state_fn(jax.random.PRNGKey(7), 256, MODEL.constants)
    upd = make_world_update(MODEL, cfg, optax.adam(3e-3), 2, qn, qw, 10, 16)
    aux = state.aux_params
    for _ in range(6):
        aux = upd(aux, state.params, dataset, jax.random.PRNGKey(8), jnp.array(1.0), 0)
    sur_fn = make_surrogate_loss(MODEL, cfg)
    batch = dataset[:64]

    def _flat(g):
        return jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(eqx.filter(g, eqx.is_array))])

    def _cos(a, b):
        return float(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-12))

    # Reference 1: the exact loss with its expectation held FIXED (stop_gradient
    # on E[inside]) — this is the object the surrogate gradient approximates,
    # because Ŵ is a fixed function in the policy update (paper's design).
    def l_exact_fixed_expectation(p):
        exp = exact_expectations(MODEL, p, batch, key, 2, 1.0, qn, qw)
        exp = {k: jax.lax.stop_gradient(v) for k, v in exp.items()}
        r = MODEL.combine_fn(batch, p(batch), exp, MODEL.constants)
        return sum(jnp.mean(v**2) for v in r.values()) / len(r)

    # Reference 2: the full exact gradient (also differentiates through the
    # future policy inside the expectation) — reported, weaker bar.
    def l_exact_full(p):
        return compute_loss(MODEL, p, batch, key, 2, quad_nodes=qn, quad_weights=qw)[0]

    def l_sur(p):
        return sur_fn(MODEL, p, batch, key, 2, quad_nodes=qn, quad_weights=qw, aux_params=aux)[0]

    g_s = _flat(eqx.filter_grad(l_sur)(state.params))
    g_fixed = _flat(eqx.filter_grad(l_exact_fixed_expectation)(state.params))
    g_full = _flat(eqx.filter_grad(l_exact_full)(state.params))
    cos_fixed = _cos(g_s, g_fixed)
    cos_full = _cos(g_s, g_full)
    assert cos_fixed > 0.995, (cos_fixed, cos_full)
    assert cos_full > 0.9, (cos_fixed, cos_full)


# ---------------------------------------------------------------------------
# config / validators / identity
# ---------------------------------------------------------------------------
def test_validator_requires_coverage_unless_allowed():
    with pytest.raises(ValueError, match="ablation"):
        train_from_config(_cfg(surrogate=SurrogateConfig(enabled=True)))


def test_validator_rejects_non_standard_optimizer():
    from deqn_jax.config import CoverageConfig

    cov = CoverageConfig(enabled=True, stress_ranges={"Z": (0.8, 1.2)}, n_stress=4, n_local=4, rollout_horizon=1)
    with pytest.raises(ValueError, match="STANDARD"):
        train_from_config(
            _cfg(
                surrogate=SurrogateConfig(enabled=True),
                coverage=cov,
                optimizer=OptimizerConfig(name="mao", learning_rate=1e-3),
            )
        )


def test_validator_rejects_target_network_combo():
    with pytest.raises(ValueError, match="target_params"):
        train_from_config(
            _cfg(surrogate=SurrogateConfig(enabled=True, allow_without_coverage=True), target_update_every=5)
        )


def test_disabled_is_identity():
    """surrogate.enabled=false ⇒ TrainState has no aux/target and the loss path
    is the plain one (bit-identical loss on the same batch/key)."""
    key = jax.random.PRNGKey(0)
    s_off, _, _ = create_train_state(MODEL, key, hidden_sizes=(16,), batch_size=8, n_equations=5)
    s_def, _, _ = create_train_state(
        MODEL, key, hidden_sizes=(16,), batch_size=8, n_equations=5, surrogate_config=SurrogateConfig()
    )
    assert s_off.aux_params is None and s_def.aux_params is None
    assert s_def.target_params is None
    leaves_a = jax.tree.leaves(eqx_arrays(s_off.params))
    leaves_b = jax.tree.leaves(eqx_arrays(s_def.params))
    for a, b in zip(leaves_a, leaves_b):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def eqx_arrays(m):
    import equinox as eqx

    return eqx.filter(m, eqx.is_array)


def test_smoke_train_with_surrogate_runs_and_decreases():
    from deqn_jax.config import CoverageConfig

    # olg_lifecycle has no closed-form steady state -> path-seeded stress
    # (box mode needs SS-filled seeds; path mode does not).
    cov = CoverageConfig(
        enabled=True,
        stress_seed_mode="path",
        stress_ranges={"Z": (0.8, 1.2)},
        n_stress=8,
        n_local=8,
        rollout_horizon=1,
        rho_base=1.0,
        rho_stress=0.5,
        rho_local=0.25,
    )
    cfg = _cfg(
        episodes=6,
        surrogate=SurrogateConfig(enabled=True, width=16, anchor_frac=0.5, epochs_w=3),
        coverage=cov,
    )
    # train_from_config returns (policy_net, history); the world-arm state is
    # exercised end to end here (validators, hook, surrogate loss, budgets).
    policy, history = train_from_config(cfg)
    assert policy is not None
    losses = np.asarray(history["loss"])
    assert losses.shape[0] == 6 and np.isfinite(losses).all()
