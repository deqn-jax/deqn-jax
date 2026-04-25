"""Tests for declarative state/definition bound penalties.

Adds soft penalties to the training loss when variables leave their
declared bounds. Matches DEQN-MAO's penalty_bounds_policy mechanism.
"""

import jax.numpy as jnp
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Unit tests on _compute_bound_penalty
# ---------------------------------------------------------------------------


def test_no_violation_gives_zero_penalty():
    from deqn_jax.training.loss import _compute_bound_penalty

    values = {"c": jnp.array([0.5, 0.6, 0.7])}
    spec = {"c": {"lower": 0.1, "penalty_lower": 1.0}}
    assert float(_compute_bound_penalty(values, spec)) == 0.0


def test_lower_violation_scales_with_square():
    from deqn_jax.training.loss import _compute_bound_penalty

    values = {"c": jnp.array([-0.2, 0.05, 0.3])}  # only first two violate lower=0.1
    spec = {"c": {"lower": 0.1, "penalty_lower": 100.0}}
    expected_violations_sq = np.array([(0.1 - (-0.2)) ** 2, (0.1 - 0.05) ** 2, 0.0])
    expected = 100.0 * expected_violations_sq.mean()
    assert float(_compute_bound_penalty(values, spec)) == pytest.approx(
        expected, rel=1e-6
    )


def test_upper_violation_likewise():
    from deqn_jax.training.loss import _compute_bound_penalty

    values = {"k": jnp.array([1.2, 0.9, 1.5])}  # two violate upper=1.0
    spec = {"k": {"upper": 1.0, "penalty_upper": 10.0}}
    expected_v2 = np.array([(1.2 - 1.0) ** 2, 0.0, (1.5 - 1.0) ** 2])
    expected = 10.0 * expected_v2.mean()
    assert float(_compute_bound_penalty(values, spec)) == pytest.approx(
        expected, rel=1e-6
    )


def test_both_bounds_stack():
    from deqn_jax.training.loss import _compute_bound_penalty

    values = {"c": jnp.array([-0.1, 0.5, 1.5])}
    spec = {
        "c": {"lower": 0.0, "upper": 1.0, "penalty_lower": 1.0, "penalty_upper": 1.0}
    }
    # Lower violations: 0.1^2 only; upper: 0.5^2 only
    expected = (0.01 + 0 + 0) / 3 + (0 + 0 + 0.25) / 3
    assert float(_compute_bound_penalty(values, spec)) == pytest.approx(
        expected, rel=1e-6
    )


def test_missing_penalty_defaults_to_inverse_square():
    from deqn_jax.training.loss import _compute_bound_penalty

    values = {"c": jnp.array([0.0])}  # violates lower=1e-4 by 1e-4
    spec = {"c": {"lower": 1e-4}}
    # Default penalty = 1 / lower^2 = 1 / 1e-8 = 1e8
    # violation = 1e-4, squared = 1e-8, mean over 1 sample = 1e-8
    # penalty = 1e8 * 1e-8 = 1.0
    assert float(_compute_bound_penalty(values, spec)) == pytest.approx(1.0, rel=1e-4)


def test_unlisted_values_ignored():
    from deqn_jax.training.loss import _compute_bound_penalty

    values = {"c": jnp.array([-0.5]), "unused": jnp.array([-100.0])}
    spec = {"c": {"lower": 0.0, "penalty_lower": 1.0}}
    # Only c's violation counted; "unused" not mentioned in spec.
    assert float(_compute_bound_penalty(values, spec)) == pytest.approx(0.25, rel=1e-6)


# ---------------------------------------------------------------------------
# Integration: compute_loss pays the penalty when bounds are violated,
# doesn't pay when they aren't.
# ---------------------------------------------------------------------------


def _loss_with_bound(bound_kind: str, **kwargs):
    """Run compute_loss on brock_mirman with a state or definition bound
    injected into the ModelSpec. Returns (total_loss, eq_losses)."""
    import jax

    from deqn_jax.models.brock_mirman import MODEL as _M
    from deqn_jax.networks import mlp as mlp_mod
    from deqn_jax.training.loss import compute_loss

    if bound_kind == "state":
        model = _M._replace(state_bounds=kwargs["spec"])
    else:
        model = _M._replace(definition_bounds=kwargs["spec"])

    key = jax.random.PRNGKey(0)
    net = mlp_mod.create_mlp(
        n_states=model.n_states,
        n_policies=model.n_policies,
        hidden_sizes=(8,),
        activation="tanh",
        policy_lower=model.policy_lower,
        policy_upper=model.policy_upper,
        key=key,
    )
    # A batch of states. We can push some outside the bound using the
    # "violator" flag.
    states = kwargs["states"]
    total, eq_losses = compute_loss(
        model=model,
        policy_fn=jax.vmap(net),
        states=states,
        key=jax.random.PRNGKey(1),
        mc_samples=2,
    )
    return float(total), {k: float(v) for k, v in eq_losses.items()}


def test_state_bounds_add_aux_key_when_spec_present():
    """An aux_state_bounds entry appears in eq_losses when state_bounds
    is declared on the model, regardless of violation magnitude."""
    states = jnp.array([[0.5, 0.1], [0.8, 0.2]])  # [k, z] -- all well inside k>=1e-6
    total, eq = _loss_with_bound(
        "state",
        spec={"k": {"lower": 1e-6, "penalty_lower": 1.0}},
        states=states,
    )
    assert "aux_state_bounds" in eq
    # No violation -> the penalty term is 0, but the key is still reported.
    assert eq["aux_state_bounds"] == 0.0


def test_state_bound_penalty_fires_on_violation():
    """If state values fall below the lower bound, the penalty > 0."""
    # k=0 violates lower=1e-2 by 1e-2 each; penalty = 1 * 1e-4 = 1e-4.
    states = jnp.array([[0.0, 0.0], [0.0, 0.0]])
    _, eq = _loss_with_bound(
        "state",
        spec={"k": {"lower": 1e-2, "penalty_lower": 1.0}},
        states=states,
    )
    assert eq["aux_state_bounds"] == pytest.approx(1e-4, rel=1e-4)


def test_definition_bounds_add_aux_key_when_spec_present():
    """Definition bounds key appears when spec is declared."""
    states = jnp.array([[0.5, 0.1], [0.8, 0.2]])
    _, eq = _loss_with_bound(
        "definition",
        spec={"c": {"lower": 0.0, "penalty_lower": 1.0}},
        states=states,
    )
    assert "aux_definition_bounds" in eq


def test_no_bounds_declared_means_no_aux_key():
    """When neither state_bounds nor definition_bounds are set, no aux
    keys for them appear in eq_losses."""
    import jax

    from deqn_jax.models.brock_mirman import MODEL
    from deqn_jax.networks import mlp as mlp_mod
    from deqn_jax.training.loss import compute_loss

    key = jax.random.PRNGKey(0)
    net = mlp_mod.create_mlp(
        n_states=MODEL.n_states,
        n_policies=MODEL.n_policies,
        hidden_sizes=(8,),
        activation="tanh",
        policy_lower=MODEL.policy_lower,
        policy_upper=MODEL.policy_upper,
        key=key,
    )
    states = jnp.array([[0.5, 0.1], [0.8, 0.2]])
    _, eq = compute_loss(
        model=MODEL,
        policy_fn=jax.vmap(net),
        states=states,
        key=jax.random.PRNGKey(1),
        mc_samples=2,
    )
    assert "aux_state_bounds" not in eq
    assert "aux_definition_bounds" not in eq


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
