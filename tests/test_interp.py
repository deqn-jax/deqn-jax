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

from deqn_jax.interp import branch_decompose, forward_with_activations
from deqn_jax.networks.linear_plus_mlp import LinearPlusMLP


def _make_fixture_net(
    hidden_sizes=(4,),
    seed: int = 0,
    output_link: str = "linear",
    policy_lower=None,
    policy_upper=None,
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
        # Wide bounds so clipping never triggers in tests (overridable):
        policy_lower=jnp.array([-1e6])
        if policy_lower is None
        else jnp.array(policy_lower),
        policy_upper=jnp.array([1e6])
        if policy_upper is None
        else jnp.array(policy_upper),
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


def test_branch_decompose_shapes_and_keys():
    net = _make_fixture_net()
    states = _sample_states()
    out = branch_decompose(net, states)
    assert set(out.keys()) == {"bk", "mlp_delta", "policy", "closes_numerically"}
    assert out["bk"].shape == (32, 1)
    assert out["mlp_delta"].shape == (32, 1)
    assert out["policy"].shape == (32, 1)
    assert isinstance(bool(out["closes_numerically"]), bool)


def test_branch_decompose_closes_numerically_linear_link():
    net = _make_fixture_net(output_link="linear")
    states = _sample_states()
    out = branch_decompose(net, states)
    # bk + mlp_delta should match policy exactly (no clipping)
    reconstructed = out["bk"] + out["mlp_delta"]
    assert jnp.allclose(reconstructed, out["policy"], atol=1e-6)
    assert bool(out["closes_numerically"])


def test_branch_decompose_closes_numerically_log_link():
    # ss_policy > 0 required for log link.
    # With wide bounds clipping does not fire, so closes_numerically is True;
    # bk + mlp_delta == policy follows from no-clip-fired (raw == policy).
    net = _make_fixture_net(output_link="log")
    states = _sample_states()
    out = branch_decompose(net, states)
    assert bool(out["closes_numerically"])
    reconstructed = out["bk"] + out["mlp_delta"]
    assert jnp.allclose(reconstructed, out["policy"], atol=1e-6)


def test_branch_decompose_clip_disables_closure():
    # Tight bounds force clipping; numerical closure should report False.
    net_tight = _make_fixture_net(policy_lower=[0.49], policy_upper=[0.51])
    states = _sample_states()
    out = branch_decompose(net_tight, states)
    assert not bool(out["closes_numerically"])


def test_forward_with_activations_keys_and_shapes():
    net = _make_fixture_net(hidden_sizes=(4,))
    states = _sample_states()
    acts = forward_with_activations(net.mlp, states)
    assert set(acts.keys()) == {"h0", "out"}
    assert acts["h0"].shape == (32, 4)
    assert acts["out"].shape == (32, 1)


def test_forward_with_activations_two_hidden():
    net = _make_fixture_net(hidden_sizes=(4, 3))
    states = _sample_states()
    acts = forward_with_activations(net.mlp, states)
    assert set(acts.keys()) == {"h0", "h1", "out"}
    assert acts["h0"].shape == (32, 4)
    assert acts["h1"].shape == (32, 3)
    assert acts["out"].shape == (32, 1)


def test_forward_with_activations_out_matches_call():
    net = _make_fixture_net(hidden_sizes=(4,))
    states = _sample_states()
    acts = forward_with_activations(net.mlp, states)
    direct = net.mlp(states)
    assert jnp.allclose(acts["out"], direct, atol=1e-6)
