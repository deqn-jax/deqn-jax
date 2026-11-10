"""Bound penalties carry the precision of the values they penalise.

The suite runs with 64-bit enabled (``tests/conftest.py``), so an fp64 run's
penalty, its ``aux_*`` entry and the total loss are float64 whether or not a
declared bound names a variable the model actually has.
"""

import jax
import jax.numpy as jnp

from deqn_jax.models.brock_mirman import MODEL as BM
from deqn_jax.training.loss import _compute_bound_penalty, compute_loss


def test_penalty_without_matching_variable_is_default_float():
    values = {"c": jnp.array([0.5, 0.6])}
    spec = {"not_a_variable": {"lower": 0.1}}
    penalty = _compute_bound_penalty(values, spec)
    assert penalty.dtype == jnp.float64
    assert float(penalty) == 0.0


def test_penalty_follows_value_dtype():
    spec = {"c": {"lower": 0.1, "penalty_lower": 1.0}}
    for dtype in (jnp.float32, jnp.float64):
        values = {"c": jnp.array([-0.2, 0.5], dtype=dtype)}
        assert _compute_bound_penalty(values, spec).dtype == dtype


def test_aux_bound_entry_and_total_stay_float64():
    model = BM._replace(state_bounds={"no_such_state": {"lower": 0.0}})
    ss, sp = model.steady_state_fn(model.constants)
    states = jnp.broadcast_to(jnp.asarray(ss), (4, model.n_states))

    def policy_fn(s):
        return jnp.broadcast_to(jnp.asarray(sp), (s.shape[0], model.n_policies))

    total, eq = compute_loss(model, policy_fn, states, jax.random.PRNGKey(0))
    assert eq["aux_state_bounds"].dtype == jnp.float64
    assert total.dtype == jnp.float64
