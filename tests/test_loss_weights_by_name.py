"""Per-equation loss weights are matched to equations by name.

``TrainConfig.loss_weights`` (and ``--loss-weights``) is a positional list in
the order of ``model.equation_names``; ``TrainState.loss_weights`` and the
reporting labels use the same order. The residuals arrive as a dict in some
other order: alphabetical on the vmapped standard and aio paths (JAX sorts
dict keys), the ``combine_fn``'s insertion order on the two-stage path. irbc
and cdice_bau declare non-alphabetical ``equation_names``. So the loss must
line weights up by name, return the per-equation losses in ``equation_names``
order, and refuse residual names it cannot match.
"""

import jax
import jax.numpy as jnp
import pytest

from deqn_jax.models import load_model
from deqn_jax.training.loss import compute_loss, eq_losses_to_array
from deqn_jax.types import ModelSpec


def _step(state, policy, shock, constants):
    return state + shock[:, :1]


def _const_residuals(names_values):
    """Residual fn returning constant residuals in the given dict order."""

    def eqs(state, policy, next_state, next_policy, constants):
        ones = jnp.ones(state.shape[0])
        return {name: value * ones for name, value in names_values}

    return eqs


def _policy_fn(states):
    return jnp.zeros((states.shape[0], 1))


def _model(eqs, equation_names=("b", "a"), **kw):
    return ModelSpec(
        name="toy",
        n_states=1,
        n_policies=1,
        n_shocks=1,
        equation_names=equation_names,
        constants={},
        equations_fn=eqs,
        step_fn=_step,
        **kw,
    )


# equation_names is (b, a); the residual dict is built (a, b), which is also
# the alphabetical order JAX gives vmapped dicts. Residual a=1, b=2, so the
# per-equation losses are a: 1, b: 4 on every path below.
_RESIDUALS = _const_residuals([("a", 1.0), ("b", 2.0)])
_STATES = jnp.array([[0.5], [1.0], [2.0]])
# Positional in equation_names order: b -> 10, a -> 0.
_WEIGHTS = jnp.array([10.0, 0.0])


def _loss_paths():
    standard = _model(_RESIDUALS)
    two_stage = _model(
        _RESIDUALS,
        inside_fn=_RESIDUALS,
        combine_fn=lambda s, p, E, c: {"a": E["a"], "b": E["b"]},
    )
    return [
        pytest.param(standard, "mse", id="standard"),
        pytest.param(standard, "huber", id="standard-huber"),
        pytest.param(standard, "aio", id="aio"),
        pytest.param(two_stage, "mse", id="two-stage"),
    ]


@pytest.mark.parametrize("model,loss_choice", _loss_paths())
def test_weights_follow_equation_names(model, loss_choice):
    total, eq = compute_loss(
        model,
        _policy_fn,
        _STATES,
        jax.random.PRNGKey(0),
        mc_samples=4,
        weights=_WEIGHTS,
        loss_choice=loss_choice,
        huber_delta=100.0,
    )
    scale = 0.5 if loss_choice == "huber" else 1.0
    assert float(eq["a"]) == pytest.approx(scale * 1.0)
    assert float(eq["b"]) == pytest.approx(scale * 4.0)
    # Weight 10 lands on "b", weight 0 on "a"; mean over the two equations.
    assert float(total) == pytest.approx(scale * (10.0 * 4.0 + 0.0 * 1.0) / 2)
    # Per-equation losses come back in equation_names order, so the vector
    # the reweighting / gradient-surgery code sees is positional too.
    assert list(eq) == ["b", "a"]
    assert eq_losses_to_array(eq).tolist() == pytest.approx([4.0 * scale, scale])


def test_unknown_residual_name_lists_model_equations():
    model = _model(_const_residuals([("a", 1.0), ("bb", 2.0)]))
    with pytest.raises(ValueError) as err:
        compute_loss(model, _policy_fn, _STATES, jax.random.PRNGKey(0), mc_samples=2)
    msg = str(err.value)
    assert "bb" in msg and "'a'" in msg and "'b'" in msg


def test_weight_count_must_match_equation_count():
    model = _model(_RESIDUALS)
    with pytest.raises(ValueError, match="2"):
        compute_loss(
            model,
            _policy_fn,
            _STATES,
            jax.random.PRNGKey(0),
            mc_samples=2,
            weights=jnp.array([1.0]),
        )


def test_irbc_losses_come_back_in_equation_names_order():
    """irbc's equations_fn emits ``arc`` first; its equation_names do not."""
    model = load_model("irbc")
    ss, sp = model.steady_state_fn(model.constants)
    states = jnp.broadcast_to(jnp.asarray(ss), (4, model.n_states))

    def policy_fn(s):
        return jnp.broadcast_to(jnp.asarray(sp), (s.shape[0], model.n_policies))

    _, eq = compute_loss(model, policy_fn, states, jax.random.PRNGKey(0), mc_samples=2)
    base = [k for k in eq if not k.startswith("aux_")]
    assert base == list(model.equation_names)


def test_reweighting_weights_follow_equation_names():
    """The per-equation losses reach the reweighting through ``has_aux``,
    where JAX hands the dict back alphabetically; the new weights must still
    be positional in equation_names (b, a), the order compute_loss reads."""
    from types import SimpleNamespace

    from deqn_jax.training.reweighting import update_reweighting
    from deqn_jax.types import make_reweight_state

    state = SimpleNamespace(
        loss_weights=jnp.ones(2), reweight_state=make_reweight_state(2)
    )
    sorted_losses = {"a": jnp.array(1.0), "b": jnp.array(4.0)}
    weights, _ = update_reweighting(
        sorted_losses, state, "lr_annealing", 0.5, 2, ("b", "a")
    )
    # EMA from 1: a -> 1.0, b -> 2.5; inverse, normalized to sum 2.
    assert weights.tolist() == pytest.approx([0.8 / 1.4, 2.0 / 1.4], rel=1e-6)
