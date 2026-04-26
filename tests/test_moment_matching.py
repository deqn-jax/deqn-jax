"""Tests for the moment-matching auxiliary loss wrapper.

The wrapper composes around any base loss and adds a per-minibatch
penalty on per-variable mean and std deviation from a Dynare reference.
The aux scalar is exposed via the eq_losses dict under the key
``aux_moment_match`` so it appears in TB / metrics without polluting
the per-equation residual lineup that gradient-surgery / reweighting
operate on (the ``aux_`` prefix is the documented exclusion convention).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

jax.config.update("jax_enable_x64", True)

from deqn_jax.training.moment_loss import (  # noqa: E402
    _moment_matching_aux_loss,
    _resolve_target_indices,
    make_moment_matching_wrapper,
)

# ---------------------------------------------------------------------------
# Index resolution
# ---------------------------------------------------------------------------


def test_resolve_target_indices_with_aliases():
    """Aliased DEQN names (`i` → `i_var`) match the Dynare reference."""
    policy_names = ["c", "i", "pi"]
    target_moments = {
        "c": {"mean": 1.6, "std": 0.05},
        "i_var": {"mean": 0.8, "std": 0.06},
        # 'pi' missing from target — should be silently skipped.
    }
    out = _resolve_target_indices(
        policy_names, target_moments, name_aliases={"i": "i_var"}
    )
    assert out == {0: (1.6, 0.05), 1: (0.8, 0.06)}


def test_resolve_target_indices_without_aliases():
    """Identity mapping when no aliases provided."""
    policy_names = ["c", "h"]
    target_moments = {"c": {"mean": 1.6, "std": 0.05}}
    out = _resolve_target_indices(policy_names, target_moments)
    assert out == {0: (1.6, 0.05)}


# ---------------------------------------------------------------------------
# Aux loss numerics
# ---------------------------------------------------------------------------


def _make_constant_policy(out_value: jnp.ndarray):
    """Return a policy_fn that ignores state and outputs ``out_value``."""

    def policy(state):
        return out_value

    return policy


def test_aux_loss_zero_when_policy_matches_target():
    """Constant policy at target mean produces zero aux loss when std=0 target."""
    policy = _make_constant_policy(jnp.array([1.6, 0.8]))
    target = {0: (1.6, 0.0), 1: (0.8, 0.0)}
    states = jnp.zeros((4, 3))
    aux = _moment_matching_aux_loss(
        states, policy, target, mean_weight=1.0, std_weight=1.0, scale_eps=1e-3
    )
    # Constant policy → net_std = 0 = target std for both vars; net_mean = target.
    assert float(aux) < 1e-10


def test_aux_loss_grows_with_mean_deviation():
    """Doubling the mean deviation (relative) quadruples the aux loss."""
    states = jnp.zeros((4, 3))
    target = {0: (1.0, 0.0)}

    p1 = _make_constant_policy(jnp.array([1.1]))  # 10% off
    p2 = _make_constant_policy(jnp.array([1.2]))  # 20% off
    a1 = float(_moment_matching_aux_loss(states, p1, target, 1.0, 1.0, 1e-3))
    a2 = float(_moment_matching_aux_loss(states, p2, target, 1.0, 1.0, 1e-3))
    # (0.2/1.0)² / (0.1/1.0)² = 4
    assert abs(a2 / a1 - 4.0) < 1e-3


def test_aux_loss_handles_sequence_input():
    """3D states (sequence input) are reduced to last timestep transparently."""
    policy = _make_constant_policy(jnp.array([1.0]))
    target = {0: (1.0, 0.0)}
    seq_states = jnp.zeros((4, 3, 5))  # [batch, history, n_states]
    aux = _moment_matching_aux_loss(
        seq_states, policy, target, mean_weight=1.0, std_weight=1.0, scale_eps=1e-3
    )
    assert float(aux) < 1e-10


# ---------------------------------------------------------------------------
# Wrapper integration
# ---------------------------------------------------------------------------


def test_wrapper_layers_on_top_of_base_loss():
    """The wrapped loss returns ``base + weight * aux`` and exposes both."""

    def fake_base(model, params, states, key, *args, **kwargs):
        # base loss = 1.0 (constant); eq_losses has one residual.
        return jnp.array(1.0), {"eq1": jnp.array(1.0)}

    target = {0: (2.0, 0.0)}
    policy = _make_constant_policy(jnp.array([3.0]))  # 50% off mean
    wrapped = make_moment_matching_wrapper(
        fake_base, target, weight=0.1, mean_weight=1.0, std_weight=1.0
    )
    states = jnp.zeros((4, 3))
    total, eq_losses = wrapped(None, policy, states, jr.PRNGKey(0))
    # aux = ((3 - 2) / 2)² = 0.25; total = 1.0 + 0.1 * 0.25 = 1.025
    assert abs(float(total) - 1.025) < 1e-6
    assert "aux_moment_match" in eq_losses
    assert "eq1" in eq_losses
    assert abs(float(eq_losses["aux_moment_match"]) - 0.025) < 1e-6


def test_wrapper_passthrough_when_no_overlap():
    """Empty target_idx_to_moments → wrapper returns the base unchanged."""

    def fake_base(model, params, states, key, *args, **kwargs):
        return jnp.array(7.0), {"eq1": jnp.array(7.0)}

    wrapped = make_moment_matching_wrapper(fake_base, {}, weight=0.1)
    # Returned object is the base itself — no wrapping when nothing to match.
    assert wrapped is fake_base


# ---------------------------------------------------------------------------
# End-to-end with disaster
# ---------------------------------------------------------------------------


def test_end_to_end_train_with_moment_matching():
    """Smoke: a 4-episode disaster run with the aux enabled completes
    without NaN, and the aux key shows up in the final history."""
    from deqn_jax.config import TrainConfig
    from deqn_jax.training.trainer import train_from_config

    cfg = TrainConfig.model_validate(
        {
            "model": "disaster",
            "episodes": 3,
            "episode_length": 4,
            "batch_size": 8,
            "sim_batch": 8,
            "mc_samples": 1,
            "fp64": True,
            "verbose": False,
            "log_every": 1,
            "curriculum_episodes": 0,
            "network": {"type": "kf_anchored_mlp", "hidden_sizes": [8]},
            "optimizer": {"name": "adam", "learning_rate": 1e-3},
            "moment_matching": {
                "enabled": True,
                "weight": 0.1,
                "dynare_dir": "dynare/results",
            },
        }
    )
    _, history = train_from_config(cfg)
    losses = history["loss"]
    assert len(losses) > 0
    import math

    assert all(not math.isnan(v) for v in losses), f"NaN in losses: {losses}"
