"""Tests for `deqn_jax.networks.viz` graphviz visitor.

Each test instantiates a small network and asserts the emitted DOT
source contains the structural landmarks the renderer is supposed to
expose. We do NOT shell out to the `dot` binary — these tests verify
the visitor logic, not graphviz itself.
"""

import jax
import jax.numpy as jnp
import pytest

from deqn_jax.networks.linear_plus_mlp import LinearPlusMLP
from deqn_jax.networks.lstm import LSTMPolicy
from deqn_jax.networks.mlp import MLP, MultiHeadMLP, ResMLP
from deqn_jax.networks.transformer import TransformerPolicy
from deqn_jax.networks.viz import UnsupportedModelError, to_dot


def _key(seed: int = 0):
    return jax.random.PRNGKey(seed)


def test_to_dot_mlp_has_linear_and_activation_boxes():
    model = MLP(
        in_features=4,
        out_features=2,
        hidden_sizes=(8, 8),
        activations=(jax.nn.tanh, jax.nn.relu),
        key=_key(),
    )
    dot = to_dot(model)
    assert dot.startswith("digraph "), "must start with digraph header"
    assert "rankdir=TB" in dot
    # Three Linear layers (2 hidden + 1 output)
    assert dot.count('"linear_0"') >= 1
    assert dot.count('"linear_1"') >= 1
    assert dot.count('"linear_2"') >= 1
    # Per-layer activation diamonds (two hidden layers → two activations)
    assert "tanh" in dot
    assert "relu" in dot
    # Edge labels carry tensor shapes
    assert "[4]" in dot  # input feature shape on first edge
    assert "[8]" in dot  # hidden shape


def test_to_dot_mlp_with_bounds_renders_bounds_node():
    lower = jnp.array([-1.0, -1.0])
    upper = jnp.array([1.0, 1.0])
    model = MLP(
        in_features=4,
        out_features=2,
        hidden_sizes=(8,),
        activations=(jax.nn.tanh,),
        output_lower=lower,
        output_upper=upper,
        key=_key(),
    )
    dot = to_dot(model)
    assert "_apply_bounds" in dot


def test_to_dot_mlp_without_bounds_omits_bounds_node():
    model = MLP(
        in_features=4,
        out_features=2,
        hidden_sizes=(8,),
        activations=(jax.nn.tanh,),
        key=_key(),
    )
    dot = to_dot(model)
    assert "_apply_bounds" not in dot


def test_to_dot_mlp_with_input_normalization_renders_normalize_node():
    shift = jnp.zeros(4)
    scale = jnp.ones(4)
    model = MLP(
        in_features=4,
        out_features=2,
        hidden_sizes=(8,),
        activations=(jax.nn.tanh,),
        input_shift=shift,
        input_scale=scale,
        key=_key(),
    )
    dot = to_dot(model)
    assert "normalize" in dot


def test_to_dot_resmlp_has_skip_connections():
    model = ResMLP(
        in_features=4,
        out_features=2,
        hidden_sizes=(8, 8),
        activations=(jax.nn.tanh, jax.nn.tanh),
        key=_key(),
    )
    dot = to_dot(model)
    # Residual add nodes between hidden blocks
    assert '"add_0"' in dot
    # Identity-skip case: hidden_sizes (8,8) match — first skip is identity
    assert "identity" in dot
    # Skip projection case: when hidden_sizes differ → projection box
    model2 = ResMLP(
        in_features=4,
        out_features=2,
        hidden_sizes=(8, 16),
        activations=(jax.nn.tanh, jax.nn.tanh),
        key=_key(),
    )
    dot2 = to_dot(model2)
    assert "skip_proj_" in dot2
    assert "Linear (skip)" in dot2


def test_to_dot_multihead_mlp_renders_per_policy_heads():
    model = MultiHeadMLP(
        in_features=4,
        out_features=3,
        hidden_sizes=(8,),
        activations=(jax.nn.tanh,),
        key=_key(),
    )
    dot = to_dot(model)
    assert '"head_0"' in dot
    assert '"head_1"' in dot
    assert '"head_2"' in dot
    assert "concat" in dot
    # Heads cluster
    assert "per-policy heads" in dot


def test_to_dot_lstm_renders_input_proj_cells_and_extract():
    model = LSTMPolicy(
        in_features=5,
        out_features=2,
        hidden_sizes=(8, 8),
        history_len=10,
        key=_key(),
    )
    dot = to_dot(model)
    assert "input_proj" in dot
    assert "lstm_0" in dot
    assert "lstm_1" in dot
    assert "scan over H" in dot
    assert "extract_last" in dot
    assert "output_proj" in dot
    # History dimension on edge labels
    assert "H=10" in dot
    assert "stacked LSTM" in dot


def test_to_dot_transformer_renders_blocks_attn_ffn_residuals():
    model = TransformerPolicy(
        in_features=5,
        out_features=2,
        hidden_dim=16,
        n_layers=2,
        num_heads=4,
        history_len=8,
        key=_key(),
    )
    dot = to_dot(model)
    assert "input_proj" in dot
    assert "pos_embed" in dot
    # Two blocks
    assert "TransformerBlock 0" in dot
    assert "TransformerBlock 1" in dot
    # Each block's pieces
    assert "MultiHeadAttn" in dot
    assert "heads=4" in dot
    assert "FFN" in dot
    assert "LayerNorm 1" in dot
    assert "LayerNorm 2" in dot
    # Final extract + ln + output
    assert "extract_last" in dot
    assert "final_ln" in dot
    assert "output_proj" in dot
    # Residual edges
    assert "residual" in dot


def test_to_dot_linear_plus_mlp_has_two_branches():
    inner_mlp = MLP(
        in_features=4,
        out_features=2,
        hidden_sizes=(8,),
        activations=(jax.nn.tanh,),
        key=_key(),
    )
    P = jnp.zeros((2, 4))
    ss_state = jnp.ones(4)
    ss_policy = jnp.ones(2)
    model = LinearPlusMLP.__new__(LinearPlusMLP)
    object.__setattr__(model, "mlp", inner_mlp)
    object.__setattr__(model, "P", P)
    object.__setattr__(model, "ss_state", ss_state)
    object.__setattr__(model, "ss_policy", ss_policy)
    object.__setattr__(model, "policy_lower", None)
    object.__setattr__(model, "policy_upper", None)
    object.__setattr__(model, "use_zlb_feature", False)
    object.__setattr__(model, "r_lag_idx", 5)
    object.__setattr__(model, "r_lb", 1.0)

    dot = to_dot(model)
    assert "linear_branch" in dot
    assert "mlp_branch" in dot
    assert "ss_policy + P @" in dot
    assert "MLP correction" in dot


def test_to_dot_unsupported_model_raises():
    import equinox as eqx

    class Bogus(eqx.Module):
        pass

    with pytest.raises(UnsupportedModelError):
        to_dot(Bogus())


def test_to_dot_emits_balanced_braces():
    """Sanity check: emitted DOT has matching { / }."""
    model = MLP(
        in_features=2,
        out_features=2,
        hidden_sizes=(4,),
        activations=(jax.nn.tanh,),
        key=_key(),
    )
    dot = to_dot(model)
    assert dot.count("{") == dot.count("}")
