"""Sanity tests for ``deqn_jax.interp``.

Fixture: a small, deterministic ``LinearPlusMLP`` matching brock_mirman's
shape (2 states, 1 policy) but with hand-set linearization and a default-
initialized MLP under a fixed seed.
"""

from __future__ import annotations

import equinox as eqx  # noqa: F401
import jax
import jax.numpy as jnp
import pytest  # noqa: F401

from deqn_jax.networks.linear_plus_mlp import LinearPlusMLP


def _make_fixture_net(
    hidden_sizes=(4,),
    seed: int = 0,
    output_link: str = "linear",
) -> LinearPlusMLP:
    """Build a deterministic LinearPlusMLP for brock_mirman-like shape."""
    key = jax.random.PRNGKey(seed)
    return LinearPlusMLP(
        n_states=2,
        n_policies=1,
        hidden_sizes=hidden_sizes,
        activation="tanh",
        P=jnp.array([[0.5, 0.3]]),
        ss_state=jnp.array([1.0, 0.0]),
        ss_policy=jnp.array([0.5]),
        output_links=(output_link,),
        # Wide bounds so clipping never triggers in tests:
        policy_lower=jnp.array([-1e6]),
        policy_upper=jnp.array([1e6]),
        key=key,
    )


def _sample_states(n: int = 32, seed: int = 1) -> jnp.ndarray:
    """Random (k, z) state samples near the fixture SS."""
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(key, (n, 2)) * 0.1 + jnp.array([1.0, 0.0])


def test_fixture_builds_and_evaluates():
    net = _make_fixture_net()
    states = _sample_states()
    out = net(states)
    assert out.shape == (32, 1)
    assert jnp.all(jnp.isfinite(out))
