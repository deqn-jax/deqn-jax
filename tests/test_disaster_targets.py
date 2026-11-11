"""The disaster model's steady state sits on its own targets.

The Taylor-rule targets ``R_ss`` and ``y_ss`` are derived from the solved
steady state; hardcoded, they once put steady-state inflation at 1.0123
against ``pi_ss = 1.006``.
"""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from deqn_jax.models import load_model
from deqn_jax.models.disaster.steady_state import consistent_targets
from deqn_jax.models.disaster.variables import CONSTANTS, SPEC
from deqn_jax.training.state_init import calibrate_model


def _ss_inflation(model):
    _, policy = model.steady_state_fn(model.constants)
    return float(policy[SPEC.policy_names.index("pi")])


def test_shipped_targets_are_the_derived_ones():
    targets = consistent_targets(CONSTANTS)
    for name, value in targets.items():
        assert CONSTANTS[name] == pytest.approx(value, rel=1e-10), name


def test_steady_state_inflation_is_on_target():
    model = load_model("disaster")
    # 1e-6, not machine precision: the 1e-8 guards in the residuals' ratio
    # forms leave inflation ~2e-7 off the target.
    assert _ss_inflation(model) == pytest.approx(CONSTANTS["pi_ss"], abs=1e-6)


def test_targets_follow_a_changed_calibration():
    config = SimpleNamespace(constants={"beta": 0.9987}, verbose=False)
    model = calibrate_model(load_model("disaster"), config)
    assert model.constants["R_ss"] != CONSTANTS["R_ss"]
    assert _ss_inflation(model) == pytest.approx(CONSTANTS["pi_ss"], abs=1e-6)


def test_explicit_target_is_kept():
    config = SimpleNamespace(constants={"y_ss": 3.0308}, verbose=False)
    model = calibrate_model(load_model("disaster"), config)
    assert model.constants["y_ss"] == 3.0308
    assert model.constants["R_ss"] == CONSTANTS["R_ss"]
    assert _ss_inflation(model) > CONSTANTS["pi_ss"] + 1e-3


def test_no_wage_dispersion_at_the_steady_state():
    """Wages grow with the trend: at the steady state the wage index and
    wage inflation coincide, as the price index and inflation do."""
    model = load_model("disaster")
    state, policy = model.steady_state_fn(model.constants)
    defs = model.definitions_fn(
        jnp.asarray(state), jnp.asarray(policy), model.constants
    )
    np.testing.assert_allclose(float(defs["wage_index_ratio"]), 1.0, atol=1e-6)
    np.testing.assert_allclose(float(defs["K_w_inner"]), 1.0, atol=1e-4)
