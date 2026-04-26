"""Tests for the K/F-gauge-elimination network class.

The network outputs the model's full policy vector but pins the
``kf_names`` rows (F_p, K_p, F_w, K_w by default) to the linearization
anchor: ``K(s) = K_ss + (P_K @ (s - s_ss))``. The remaining policies
are output by an inner bounded MLP.

These tests pin three properties:
  1. Forward at SS returns the SS policy exactly for the anchored
     rows (``K_value(SS) == K_ss``).
  2. ``jax.jacrev`` of the network at SS, restricted to the anchored
     rows, equals the linearization rows exactly. By construction,
     since the anchor is `linear(state)`, this is the sharpest test
     that we've removed the gauge freedom: anchored rows have a
     known, fixed Jacobian instead of whatever the unconstrained
     network drifts to.
  3. The full Jacobian is everywhere-finite — no NaN poisoning from
     the bound layer at SS, even though the inner MLP applies sigmoid
     / softplus bounds to the non-K/F rows.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

jax.config.update("jax_enable_x64", True)

from deqn_jax.models import load_model  # noqa: E402
from deqn_jax.networks.kf_anchored_mlp import (  # noqa: E402
    KfAnchoredMLP,
    create_kf_anchored_mlp,
)


def _build(model_name: str = "disaster", seed: int = 0) -> KfAnchoredMLP:
    model = load_model(model_name)
    return create_kf_anchored_mlp(
        model,
        hidden_sizes=(16,),
        activation="tanh",
        key=jr.PRNGKey(seed),
    )


def test_construct_disaster_indices():
    """Indices for the 4 anchored rows are correctly resolved."""
    model = load_model("disaster")
    net = _build("disaster")
    pn = list(model.policy_names)
    expected_kf = (pn.index("F_p"), pn.index("K_p"), pn.index("F_w"), pn.index("K_w"))
    assert net.kf_indices == expected_kf
    # other_indices is the disjoint complement.
    assert set(net.kf_indices) | set(net.other_indices) == set(range(model.n_policies))
    assert set(net.kf_indices) & set(net.other_indices) == set()


def test_forward_at_ss_returns_ss_kf_for_anchored():
    """At SS, the four anchored outputs equal the SS policy values exactly."""
    model = load_model("disaster")
    net = _build("disaster")
    ss_state, ss_policy = model.steady_state_fn(model.constants)
    out = net(ss_state)
    # Same fp32 tolerance reasoning as test_jacrev_kf_rows_match_linearization.
    for idx in net.kf_indices:
        assert abs(float(out[idx]) - float(ss_policy[idx])) < 1e-5, (
            f"anchored output {idx} drifted from SS"
        )


def test_jacrev_kf_rows_match_linearization():
    """The Jacobian for anchored rows is exactly P_kf — not a noisy network estimate."""
    model = load_model("disaster")
    net = _build("disaster")
    ss_state, _ = model.steady_state_fn(model.constants)
    J = jax.jacrev(lambda s: net(s))(ss_state)
    # P_kf is stored in the order of net.kf_indices.
    # Tolerance set for fp32 worst-case precision; fp64 typically gets ~1e-14.
    # Either way, the anchor is exact by construction and a deviation of
    # 1e-5 would indicate a real structural bug (the anchor isn't being used,
    # or the wrong rows got pinned).
    for slot, kf_pos in enumerate(net.kf_indices):
        diff = float(jnp.linalg.norm(J[kf_pos] - net.P_kf[slot]))
        assert diff < 1e-5, (
            f"anchored row {kf_pos} jacobian deviates from linearization by {diff:.2e}"
        )


def test_jacrev_finite_everywhere_at_ss():
    """jacrev produces no non-finite entries at SS — the bound-NaN bug stays fixed."""
    net = _build("disaster")
    model = load_model("disaster")
    ss_state, _ = model.steady_state_fn(model.constants)
    J = jax.jacrev(lambda s: net(s))(ss_state)
    n_bad = int(jnp.sum(~jnp.isfinite(J)))
    assert n_bad == 0, f"jacrev produced {n_bad} non-finite entries at SS"


def test_perturbed_state_policy_finite():
    """Forward + jacrev stay finite away from SS too (small Gaussian perturbation)."""
    net = _build("disaster")
    model = load_model("disaster")
    ss_state, _ = model.steady_state_fn(model.constants)
    perturbed = ss_state + 1e-2 * jr.normal(jr.PRNGKey(7), ss_state.shape)
    out = net(perturbed)
    assert bool(jnp.all(jnp.isfinite(out))), "forward NaN off-SS"
    J = jax.jacrev(lambda s: net(s))(perturbed)
    assert int(jnp.sum(~jnp.isfinite(J))) == 0


def test_invalid_kf_name_raises():
    """Anchoring a name that isn't in policy_names fails loud, not silent."""
    model = load_model("disaster")
    with pytest.raises(ValueError, match="not found in model.policy_names"):
        create_kf_anchored_mlp(
            model,
            hidden_sizes=(8,),
            activation="tanh",
            kf_names=("F_p", "totally_not_a_policy"),
            key=jr.PRNGKey(0),
        )


def test_2d_input_shape():
    """Vmapped forward returns ``[batch, n_policies]`` in policy-name order."""
    net = _build("disaster")
    model = load_model("disaster")
    ss_state, _ = model.steady_state_fn(model.constants)
    batch = jnp.broadcast_to(ss_state[None, :], (5, model.n_states))
    out = net(batch)
    assert out.shape == (5, model.n_policies)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_only_inner_mlp_is_trainable_via_filter():
    """Equinox's array filter only sees the inner MLP's parameters as leaves.

    The anchor arrays (``P_kf``, ``ss_state``, ``ss_kf``) are stored on
    the module but in the forward they're wrapped in ``stop_gradient`` —
    they don't accumulate gradient and the optimizer therefore doesn't
    move them. We additionally pin the bounds + index tuples as
    ``static=True`` fields so they don't appear in the pytree at all.
    Inner MLP layers (4 leaves: weights + biases for 1 hidden + 1 output
    layer) plus the 3 anchor arrays make 7 array leaves total. The
    anchor arrays have stop_gradient on them in the forward so their
    update is the identity even when optax sees them.
    """
    import equinox as eqx

    net = _build("disaster")
    leaves = jax.tree_util.tree_leaves(eqx.filter(net, eqx.is_array))
    # 4 inner-MLP arrays + 3 anchor arrays = 7.
    assert len(leaves) == 7, (
        f"expected 7 array leaves (4 MLP + 3 anchor), got {len(leaves)}"
    )
    # The static-tuple bounds + index fields don't show up.
    static_attrs = ("kf_indices", "other_indices", "kf_lower", "kf_upper")
    for name in static_attrs:
        val = getattr(net, name)
        assert val is None or isinstance(val, tuple), (
            f"{name} should be None or tuple, got {type(val).__name__}"
        )
