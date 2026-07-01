"""Unit tests for EWM coverage rollout helpers + loss wrapper."""

import jax
import jax.numpy as jnp
import numpy as np
from deqn_jax.training.coverage import (
    make_local_pool,
    roll_states,
    sample_stress_seeds,
)

from deqn_jax.models import load_model


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
