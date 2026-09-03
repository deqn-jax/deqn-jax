"""Unit tests for EWM coverage rollout helpers + loss wrapper."""

import jax
import jax.numpy as jnp
import numpy as np

from deqn_jax.models import load_model
from deqn_jax.training.coverage import (
    make_local_pool,
    roll_states,
    sample_stress_seeds,
)


def _irbc():
    return load_model("irbc")


def _tiny_net(model):
    from deqn_jax.networks.factory import build_policy_net

    return build_policy_net(model, jax.random.PRNGKey(1), (8,), None)


def test_sample_stress_seeds_shape_and_box():
    model = _irbc()
    ss_state, _ = model.steady_state_fn(model.constants)
    # stress only z_0 (index 2); leave others at SS
    stress_idx = jnp.array([2])
    lows = jnp.array([-0.5])
    highs = jnp.array([-0.2])
    seeds = sample_stress_seeds(
        jax.random.PRNGKey(0), 64, model.n_states, ss_state, stress_idx, lows, highs
    )
    assert seeds.shape == (64, model.n_states)
    # z_0 column inside the box
    assert np.all(np.asarray(seeds[:, 2]) >= -0.5 - 1e-6)
    assert np.all(np.asarray(seeds[:, 2]) <= -0.2 + 1e-6)
    # non-stress dims pinned at SS
    np.testing.assert_allclose(np.asarray(seeds[:, 0]), float(ss_state[0]))


def test_roll_states_shape_excludes_raw_seed():
    model = _irbc()
    net = _tiny_net(model)
    seeds = jnp.tile(jnp.array([1.0, 1.0, -0.15, -0.15]), (16, 1))
    out = roll_states(model, net, seeds, jax.random.PRNGKey(2), horizon=5)
    assert out.shape == (16 * 5, model.n_states)
    # landings differ from the raw seed (the exact-Γ rollout moved them)
    assert not np.allclose(np.asarray(out[:16]), np.asarray(seeds))


def test_roll_states_repair_clip():
    model = _irbc()
    net = _tiny_net(model)
    seeds = jnp.tile(jnp.array([1.0, 1.0, -0.15, -0.15]), (16, 1))
    lo = jnp.array([0.99, 0.99, -0.01, -0.01])  # tight box: forces clipping
    hi = jnp.array([1.01, 1.01, 0.01, 0.01])
    out = roll_states(model, net, seeds, jax.random.PRNGKey(2), horizon=3, lo=lo, hi=hi)
    assert np.all(np.asarray(out) >= np.asarray(lo) - 1e-7)
    assert np.all(np.asarray(out) <= np.asarray(hi) + 1e-7)


def test_roll_states_stop_gradient():
    model = _irbc()
    net = _tiny_net(model)
    seeds = jnp.tile(jnp.array([1.0, 1.0, -0.15, -0.15]), (8, 1))

    def f(p):
        return jnp.sum(roll_states(model, p, seeds, jax.random.PRNGKey(2), horizon=4))

    import equinox as eqx

    grads = eqx.filter_grad(f)(net)
    leaves = [g for g in jax.tree.leaves(eqx.filter(grads, eqx.is_array))]
    # every gradient leaf is exactly zero: no grad flows through generated states
    assert all(np.all(np.asarray(g) == 0.0) for g in leaves)


def test_make_local_pool_shape_and_detach():
    model = _irbc()
    states = jnp.tile(jnp.array([1.0, 1.0, 0.0, 0.0]), (32, 1))
    local = make_local_pool(states, jax.random.PRNGKey(3), n=20, sigma=0.02)
    assert local.shape == (20, model.n_states)
    # perturbed (not identical to base rows)
    assert not np.allclose(np.asarray(local[:1]), np.asarray(states[:1]))


# ---------------------------------------------------------------------------
# make_coverage_loss wrapper
# ---------------------------------------------------------------------------

from deqn_jax.config import CoverageConfig  # noqa: E402
from deqn_jax.training.coverage import make_coverage_loss  # noqa: E402


def _cov_cfg(**kw):
    base = dict(
        enabled=True,
        rho_base=2.0,
        rho_stress=1.0,
        rho_local=1.0,
        n_stress=8,
        n_local=8,
        rollout_horizon=2,
        local_sigma=0.02,
        stress_ranges={"z_0": (-0.18, -0.05), "k_0": (1.05, 1.20)},
        repair_ranges={
            "k_0": (0.2, 5.0),
            "k_1": (0.2, 5.0),
            "z_0": (-0.2, 0.2),
            "z_1": (-0.2, 0.2),
        },
    )
    base.update(kw)
    return CoverageConfig(**base)


def test_wrapper_forwards_quad_kwargs_to_all_pools():
    model = _irbc()
    calls = []

    def spy(
        model_,
        pf,
        states,
        key,
        mc_samples=5,
        weights=None,
        shock_scale=1.0,
        quad_nodes=None,
        quad_weights=None,
        target_policy_fn=None,
        loss_choice="mse",
        huber_delta=1.0,
    ):
        calls.append({"quad_nodes": quad_nodes, "n": int(states.shape[0])})
        return jnp.array(float(len(calls)) * 10.0), {"euler": jnp.array(1.0)}

    fn = make_coverage_loss(spy, model, _cov_cfg())
    net = _tiny_net(model)
    states = jnp.tile(jnp.array([1.0, 1.0, 0.0, 0.0]), (16, 1))
    qn = jnp.zeros((4, model.n_shocks))
    qw = jnp.ones((4,)) / 4
    total, eq = fn(
        model, net, states, jax.random.PRNGKey(0), quad_nodes=qn, quad_weights=qw
    )
    # all three pools were evaluated, each got the quadrature nodes
    assert len(calls) == 3
    assert all(c["quad_nodes"] is not None for c in calls)
    # mixture weights normalized: 2/1/1 -> 0.5/0.25/0.25; spy returns 10,20,30
    assert np.isclose(float(total), 0.5 * 10 + 0.25 * 20 + 0.25 * 30)
    for k in ("aux_cov_base", "aux_cov_stress", "aux_cov_local"):
        assert k in eq


def test_wrapper_real_compute_loss_runs():
    model = _irbc()
    from deqn_jax.training.loss import compute_loss

    net = _tiny_net(model)
    fn = make_coverage_loss(compute_loss, model, _cov_cfg())
    states = jnp.tile(jnp.array([1.0, 1.0, 0.0, 0.0]), (16, 1))
    qn = jnp.zeros((4, model.n_shocks))
    qw = jnp.ones((4,)) / 4
    total, eq = fn(
        model, net, states, jax.random.PRNGKey(0), quad_nodes=qn, quad_weights=qw
    )
    assert np.isfinite(float(total))
    assert np.isfinite(float(eq["aux_cov_stress"]))


def test_kappa_zero_collapses_to_plain_loss():
    """Paper's kappa=0 identity: rho_stress=rho_local=0 => wrapper == compute_loss
    EXACTLY (under quadrature the loss key is unused, so key-splitting inside the
    wrapper cannot introduce a difference)."""
    model = _irbc()
    from deqn_jax.training.loss import compute_loss

    net = _tiny_net(model)
    cfg = CoverageConfig(enabled=True, rho_stress=0.0, rho_local=0.0)
    fn = make_coverage_loss(compute_loss, model, cfg)
    states = jnp.tile(jnp.array([1.0, 1.0, 0.0, 0.0]), (16, 1))
    qn = jnp.zeros((4, model.n_shocks))
    qw = jnp.ones((4,)) / 4
    key = jax.random.PRNGKey(0)
    t_wrap, eq_wrap = fn(model, net, states, key, quad_nodes=qn, quad_weights=qw)
    t_plain, _ = compute_loss(
        model, net, states, key, 5, quad_nodes=qn, quad_weights=qw
    )
    assert float(t_wrap) == float(t_plain)  # exact, not allclose
    assert "aux_cov_stress" not in eq_wrap  # zero-weight pools skipped at build time


def test_sample_stress_seeds_from_path_inherits_joint_coords():
    from deqn_jax.training.coverage import sample_stress_seeds_from_path

    model = _irbc()
    batch = model.init_state_fn(jax.random.PRNGKey(3), 32, model.constants)
    stress_idx = jnp.array([2])
    lows = jnp.array([-0.5])
    highs = jnp.array([-0.2])
    seeds = sample_stress_seeds_from_path(
        jax.random.PRNGKey(0), 64, batch, stress_idx, lows, highs
    )
    assert seeds.shape == (64, model.n_states)
    # stress dim inside its box
    assert np.all(np.asarray(seeds[:, 2]) >= -0.5 - 1e-6)
    assert np.all(np.asarray(seeds[:, 2]) <= -0.2 + 1e-6)
    # every NON-stress coordinate must equal some batch row's value (the seed
    # inherits realistic joint coordinates, not SS fills): check each seed's
    # non-stress slice appears verbatim in the batch.
    non_stress = [i for i in range(model.n_states) if i != 2]
    b = np.asarray(batch)[:, non_stress]
    s = np.asarray(seeds)[:, non_stress]
    for row in s:
        assert np.any(np.all(np.isclose(b, row[None, :], atol=1e-12), axis=1))


def test_roll_states_include_seed_counts():
    model = _irbc()
    net = _tiny_net(model)
    seeds = model.init_state_fn(jax.random.PRNGKey(5), 8, model.constants)
    H = 3
    excl = roll_states(model, net, seeds, jax.random.PRNGKey(6), H)
    incl = roll_states(model, net, seeds, jax.random.PRNGKey(6), H, include_seed=True)
    assert excl.shape == (8 * H, model.n_states)
    assert incl.shape == (8 * (H + 1), model.n_states)
    # with include_seed the first block is the (unclipped) seeds themselves
    np.testing.assert_allclose(np.asarray(incl[:8]), np.asarray(seeds), rtol=1e-12)


def test_wrapper_path_mode_runs_and_differs_from_box():
    model = _irbc()
    from deqn_jax.training.loss import compute_loss

    net = _tiny_net(model)
    states = model.init_state_fn(jax.random.PRNGKey(9), 16, model.constants)
    qn = jnp.zeros((4, model.n_shocks))
    qw = jnp.ones((4,)) / 4

    fn_box = make_coverage_loss(compute_loss, model, _cov_cfg())
    fn_path = make_coverage_loss(compute_loss, model, _cov_cfg(stress_seed_mode="path"))
    t_box, eq_box = fn_box(
        model, net, states, jax.random.PRNGKey(0), quad_nodes=qn, quad_weights=qw
    )
    t_path, eq_path = fn_path(
        model, net, states, jax.random.PRNGKey(0), quad_nodes=qn, quad_weights=qw
    )
    assert np.isfinite(float(t_box)) and np.isfinite(float(t_path))
    # same key, different stress measure => different stress loss
    assert float(eq_box["aux_cov_stress"]) != float(eq_path["aux_cov_stress"])
    # base pool identical in both modes
    np.testing.assert_allclose(
        float(eq_box["aux_cov_base"]), float(eq_path["aux_cov_base"]), rtol=1e-12
    )
