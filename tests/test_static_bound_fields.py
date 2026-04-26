"""Tests for the static-tuple storage of bound + normalization fields.

These fields used to be plain Array-typed pytree leaves, which meant the
optimizer's eqx.apply_updates would write to them. With Adam-family
second-moment optimizers, a single NaN gradient anywhere would poison the
running variance estimate and start writing NaN-valued updates to all
params — including supposedly-frozen bounds — even though the *correct*
gradient there is zero (stop_gradient blocks the loss-side gradient,
not the update-side write).

The fix: declare these fields as `eqx.field(static=True)` and store them
as tuples-of-floats. Static fields are excluded from the pytree leaves
that eqx.filter / optimizers see; they're safe from any update path.

The defense-in-depth in `_apply_bounds`: even if some other path writes
NaN into the bounds, the "double where" pattern (replace non-finite hi
with `lo + 1.0` before the sigmoid branch) keeps both forward and
backward NaN-clean.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from deqn_jax.networks.lstm import LSTMPolicy
from deqn_jax.networks.mlp import MLP, MultiHeadMLP, ResMLP
from deqn_jax.networks.transformer import TransformerPolicy


def _make_mlp(seed: int = 0) -> MLP:
    """Tiny MLP with mixed sigmoid/softplus bounds — same shape as disaster."""
    return MLP(
        in_features=4,
        out_features=3,
        hidden_sizes=(8,),
        activations=(jax.nn.tanh,),
        output_lower=jnp.array([0.0, 0.5, 1.0]),
        output_upper=jnp.array([1.0, jnp.inf, 2.0]),  # mixed
        input_shift=jnp.zeros(4),
        input_scale=jnp.ones(4),
        key=jr.PRNGKey(seed),
    )


# ---------------------------------------------------------------------------
# Static-field declarations — the structural protection.
# ---------------------------------------------------------------------------


def test_bound_fields_are_not_in_pytree_leaves():
    """`eqx.filter(model, eqx.is_array)` must NOT include the bound fields.

    Static fields don't show up as pytree leaves; this is what stops
    optimizers from writing to them.
    """
    net = _make_mlp()
    arrays = jax.tree_util.tree_leaves(eqx.filter(net, eqx.is_array))
    # Layer weights/biases are arrays; bounds + norms are not.
    # We can't compare by identity (tuples aren't arrays anyway), so the
    # assertion is: no array in `arrays` has the same id as any of the
    # bound/norm field tuples (trivially true), AND the bound tuples
    # appear in tree_leaves with `is_leaf=lambda x: isinstance(x, tuple)`.
    # Stronger check: filter(arrays-only) leaves, count, vs filter-without
    # bounds explicitly.
    n_array_leaves = len(arrays)
    # Layer 0: weight [8,4] + bias [8]; layer 1: weight [3,8] + bias [3].
    # That's 4 array leaves total.
    assert n_array_leaves == 4, (
        f"expected 4 array leaves (2 layers × weight+bias), got {n_array_leaves}"
    )


def test_bound_fields_are_tuples_after_init():
    """`__init__` converts arrays to tuples-of-floats."""
    net = _make_mlp()
    assert isinstance(net.output_lower, tuple)
    assert isinstance(net.output_upper, tuple)
    assert isinstance(net.input_shift, tuple)
    assert isinstance(net.input_scale, tuple)
    # Values preserved (modulo float cast).
    assert net.output_lower == (0.0, 0.5, 1.0)
    assert net.input_scale == (1.0, 1.0, 1.0, 1.0)
    # _has_upper mask reflects which entries had finite upper bounds.
    assert net._has_upper == (True, False, True)


# ---------------------------------------------------------------------------
# NaN-protection.
# ---------------------------------------------------------------------------


def test_jacrev_finite_at_zero_input():
    """jacrev of a freshly-built MLP at the origin must be all finite."""
    net = _make_mlp()
    x = jnp.zeros(4)
    out = net(x)
    assert bool(jnp.all(jnp.isfinite(out)))
    J = jax.jacrev(lambda s: net(s))(x)
    assert int(jnp.sum(~jnp.isfinite(J))) == 0


def test_apply_bounds_double_where_resists_nan_in_upper():
    """Test the defense-in-depth in `_apply_bounds` directly.

    The static-field protection means `eqx.tree_at` can't even *reach*
    `output_upper` (it's not a pytree leaf — that's the structural
    protection working). So we test the double-where defense by calling
    `_apply_bounds` directly with a deliberately-NaN'd upper tuple.
    Both forward and backward must stay finite for the entries that use
    the softplus branch (where has_upper is False), and the sigmoid
    branch entry must be finite when the upper for that entry IS finite.
    """
    from deqn_jax.networks.common import _apply_bounds

    lo = (0.0, 0.5, 1.0)
    hi_with_nan = (1.0, float("nan"), float("nan"))
    has_upper = (True, False, False)

    x = jnp.array([0.1, 0.2, 0.3])
    out = _apply_bounds(x, lo, hi_with_nan, has_upper)
    assert bool(jnp.all(jnp.isfinite(out))), f"forward NaN despite double-where: {out}"

    J = jax.jacrev(lambda y: _apply_bounds(y, lo, hi_with_nan, has_upper))(x)
    assert int(jnp.sum(~jnp.isfinite(J))) == 0, (
        f"backward produced {int(jnp.sum(~jnp.isfinite(J)))} non-finite entries"
    )


def test_static_fields_are_not_pytree_reachable():
    """eqx.tree_at on a static field raises — proof the optimizer can't touch them.

    Equinox excludes `eqx.field(static=True)` from the pytree leaves, so
    `tree_at` can't address them either. This is exactly the property that
    makes them safe from any Adam-family-style write.
    """
    net = _make_mlp()
    raised = False
    try:
        eqx.tree_at(lambda m: m.output_upper, net, (1.0, float("nan"), 1.0))
    except (TypeError, ValueError):
        raised = True
    assert raised, "expected tree_at to fail on a static field"


# ---------------------------------------------------------------------------
# Same protection in every other policy class.
# ---------------------------------------------------------------------------


def test_resmlp_bounds_are_static():
    net = ResMLP(
        in_features=4,
        out_features=2,
        hidden_sizes=(8,),
        activations=(jax.nn.tanh,),
        output_lower=jnp.zeros(2),
        output_upper=jnp.array([jnp.inf, jnp.inf]),
        key=jr.PRNGKey(0),
    )
    assert isinstance(net.output_lower, tuple)
    assert isinstance(net.output_upper, tuple)


def test_multihead_mlp_bounds_are_static():
    net = MultiHeadMLP(
        in_features=4,
        out_features=2,
        hidden_sizes=(8,),
        activations=(jax.nn.tanh,),
        output_lower=jnp.zeros(2),
        output_upper=jnp.ones(2),
        key=jr.PRNGKey(0),
    )
    assert isinstance(net.output_lower, tuple)
    assert isinstance(net.output_upper, tuple)


def test_lstm_bounds_are_static():
    net = LSTMPolicy(
        in_features=4,
        out_features=2,
        hidden_sizes=(8,),
        history_len=3,
        output_lower=jnp.zeros(2),
        output_upper=jnp.ones(2),
        key=jr.PRNGKey(0),
    )
    assert isinstance(net.output_lower, tuple)
    assert isinstance(net.output_upper, tuple)


def test_transformer_bounds_are_static():
    net = TransformerPolicy(
        in_features=4,
        out_features=2,
        hidden_dim=8,
        n_layers=1,
        num_heads=2,
        history_len=3,
        output_lower=jnp.zeros(2),
        output_upper=jnp.ones(2),
        key=jr.PRNGKey(0),
    )
    assert isinstance(net.output_lower, tuple)
    assert isinstance(net.output_upper, tuple)
