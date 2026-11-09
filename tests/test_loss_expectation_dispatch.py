"""The configured residual expectation and the computed one agree.

A model with a discrete shock chain (``transition_matrix`` + ``z_state_idx``)
is integrated by exact enumeration over the chain; a chain-free model never
is. ``check_expectation_matches_model`` resolves the configured
``expectation_type`` against the model before anything is traced and refuses
the combinations a model cannot honour; ``compute_loss`` refuses quadrature
nodes handed to a chain model instead of dropping them.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deqn_jax.models import load_model
from deqn_jax.training.loss import (
    check_expectation_matches_model,
    compute_loss,
    gauss_hermite_nd,
)
from deqn_jax.types import ModelSpec


def _chain_model():
    def step(state, policy, shock, constants):
        return shock.astype(state.dtype)[:, None]

    def equations(state, policy, next_state, next_policy, constants):
        return {"r": next_state[:, 0]}

    return ModelSpec(
        name="chain_toy",
        n_states=1,
        n_policies=1,
        n_shocks=1,
        constants={},
        equations_fn=equations,
        step_fn=step,
        state_names=("z",),
        policy_names=("c",),
        equation_names=("r",),
        transition_matrix=jnp.asarray(np.array([[0.7, 0.3], [0.4, 0.6]])),
        z_state_idx=0,
    )


@pytest.mark.parametrize(
    "expectation_type", ["gauss_hermite", "quadrature", "monomial"]
)
def test_deterministic_rule_on_chain_model_names_both(expectation_type):
    with pytest.raises(ValueError) as err:
        check_expectation_matches_model(expectation_type, _chain_model())
    msg = str(err.value)
    assert expectation_type in msg and "discrete" in msg and "chain_toy" in msg


def test_discrete_on_chain_free_model_names_both():
    with pytest.raises(ValueError) as err:
        check_expectation_matches_model("discrete", load_model("brock_mirman"))
    msg = str(err.value)
    assert "discrete" in msg and "brock_mirman" in msg


@pytest.mark.parametrize(
    "expectation_type,chain,resolved",
    [
        ("discrete", True, "discrete"),
        # The default 'mc' does not describe a chain; the chain's exact
        # enumeration is what runs, and the resolved name says so.
        ("mc", True, "discrete"),
        ("mc", False, "mc"),
        ("gauss_hermite", False, "gauss_hermite"),
        ("monomial", False, "monomial"),
    ],
)
def test_resolved_expectation(expectation_type, chain, resolved):
    model = _chain_model() if chain else load_model("brock_mirman")
    assert check_expectation_matches_model(expectation_type, model) == resolved


def test_compute_loss_refuses_quadrature_nodes_on_chain_model():
    nodes, weights = gauss_hermite_nd(3, 1)
    with pytest.raises(ValueError, match="discrete"):
        compute_loss(
            _chain_model(),
            lambda s: jnp.zeros_like(s),
            jnp.array([[0.0], [1.0]]),
            jax.random.PRNGKey(0),
            mc_samples=2,
            quad_nodes=jnp.asarray(nodes),
            quad_weights=jnp.asarray(weights),
        )
