"""Degree-3 monomial quadrature tests."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from deqn_jax.models import load_model
from deqn_jax.training.loss import compute_loss, gauss_hermite_nd, monomial_nd


@pytest.mark.parametrize("dim", [1, 4, 12])
def test_monomial_integrates_standard_normal_moments(dim):
    result = monomial_nd(dim)
    assert result is not None
    nodes, weights = result

    assert nodes.shape == (2 * dim, dim)
    assert weights.shape == (2 * dim,)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-15)
    np.testing.assert_allclose(weights @ nodes, np.zeros(dim), atol=1e-15)
    np.testing.assert_allclose(
        np.einsum("n,ni,nj->ij", weights, nodes, nodes),
        np.eye(dim),
        atol=1e-15,
    )
    np.testing.assert_allclose(weights @ nodes**3, np.zeros(dim), atol=1e-14)


def test_monomial_rejects_nonpositive_dimension():
    assert monomial_nd(0) is None
    assert monomial_nd(-1) is None


def test_brock_mirman_monomial_loss_matches_two_point_gauss_hermite():
    model = load_model("brock_mirman")
    ss_state, ss_policy = model.steady_state_fn(model.constants)
    states = jnp.stack(
        [
            ss_state,
            ss_state * jnp.array([0.95, 1.0]),
            ss_state + jnp.array([0.2, 0.03]),
        ]
    )

    def policy_fn(batch):
        return jnp.broadcast_to(ss_policy, (batch.shape[0], model.n_policies))

    mono = monomial_nd(model.n_shocks)
    gh = gauss_hermite_nd(2, model.n_shocks)
    assert mono is not None and gh is not None

    key = jax.random.PRNGKey(0)
    mono_loss, mono_eq = compute_loss(
        model,
        policy_fn,
        states,
        key,
        quad_nodes=jnp.asarray(mono[0]),
        quad_weights=jnp.asarray(mono[1]),
    )
    gh_loss, gh_eq = compute_loss(
        model,
        policy_fn,
        states,
        key,
        quad_nodes=jnp.asarray(gh[0]),
        quad_weights=jnp.asarray(gh[1]),
    )

    np.testing.assert_allclose(mono_loss, gh_loss, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(mono_eq["euler"], gh_eq["euler"], rtol=0.0, atol=1e-12)
