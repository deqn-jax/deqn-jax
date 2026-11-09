"""Declarative definition-bound penalties and the five train-step variants.

``ModelSpec.definition_bounds`` adds its soft penalty to the scalar loss under
the ``aux_definition_bounds`` key. The variants that build their update from
the per-equation residual vector (MAO, the GN family, PCGrad surgery) never
see that key, so the penalty is logged but does not move the parameters. The
first test measures that per variant; the second requires setup to reject
those variants on a model that declares definition bounds.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deqn_jax.config import OptimizerConfig, TrainConfig
from deqn_jax.models import load_model
from deqn_jax.training.state_init import (
    _resolve_model_for_training,
    _validate_train_config,
    create_train_state,
    make_train_step,
)

# (optimizer, gradient_surgery) -> does the bound penalty reach the update?
VARIANTS = {
    "standard": ("adam", "none", True),
    "pcgrad": ("adam", "pcgrad", False),
    "mao": ("mao", "none", False),
    "lbfgs": ("lbfgs", "none", True),
    "gn": ("gn", "none", False),
}


def _one_cycle(model, optimizer: str, surgery: str):
    n_eq = len(model.equation_names)
    state, opt, kind = create_train_state(
        model,
        jax.random.PRNGKey(0),
        hidden_sizes=(16,),
        batch_size=16,
        n_equations=n_eq,
        optimizer_config=OptimizerConfig(name=optimizer),
    )
    step = make_train_step(model, opt, 2, 2, 16, kind=kind, gradient_surgery=surgery)
    new_state, metrics = step(state, jnp.array(1.0), jnp.array(1.0))
    arrays = jax.tree.leaves(eqx.filter(new_state.params, eqx.is_array))
    return np.concatenate([np.ravel(np.asarray(a)) for a in arrays]), metrics


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_definition_bound_penalty_reaches_update(variant):
    optimizer, surgery, reaches = VARIANTS[variant]
    base = load_model("olg_analytic_6")
    # A bound every state violates, so the penalty and its gradient are live.
    on = base._replace(definition_bounds={"c1": {"lower": 10.0, "penalty_lower": 1.0}})
    off = base._replace(definition_bounds=None)

    p_on, m_on = _one_cycle(on, optimizer, surgery)
    p_off, _ = _one_cycle(off, optimizer, surgery)

    assert float(m_on.residuals["aux_definition_bounds"]) > 0.0
    moved = not np.array_equal(p_on, p_off)
    assert moved == reaches


def _cfg(**over) -> TrainConfig:
    base = {
        "model": "olg_analytic_6",
        "verbose": False,
        "network": {"type": "mlp", "hidden_sizes": [8]},
    }
    base.update(over)
    return TrainConfig.from_dict(base)


@pytest.mark.parametrize(
    "over",
    [
        {"optimizer": {"name": "mao"}},
        {"optimizer": {"name": "gn"}},
        {"optimizer": {"name": "ign"}},
        {"optimizer": {"name": "lm"}},
        {"gradient_surgery": "pcgrad"},
        {"switch_optimizer": "gn", "switch_episode": 5},
    ],
    ids=["mao", "gn", "ign", "lm", "pcgrad", "switch_to_gn"],
)
def test_setup_rejects_variant_that_drops_definition_bounds(over):
    cfg = _cfg(**over)
    _validate_train_config(cfg)
    with pytest.raises(ValueError, match="definition_bounds"):
        _resolve_model_for_training(cfg)


@pytest.mark.parametrize("name", ["adam", "lbfgs"])
def test_setup_accepts_variant_that_keeps_definition_bounds(name):
    cfg = _cfg(optimizer={"name": name})
    _validate_train_config(cfg)
    _resolve_model_for_training(cfg)
