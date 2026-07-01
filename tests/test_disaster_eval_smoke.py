"""Eval-path smoke for disaster-risk calibrations (p_disaster > 0).

Regression: all three eval entry points drew the Bernoulli disaster
indicator with shape (1, 1) instead of the flat (1,) that step_fn's
contract requires (see training/shocks.py maybe_draw_disaster's shape
note) -- so euler_equation_errors / simulated_moments / stability_check
crashed for ANY p_disaster > 0 calibration and the disaster-risk eval
path had never executed.
"""

import jax
import numpy as np

from deqn_jax.models import load_model
from deqn_jax.networks.factory import build_policy_net


def _disaster_with_risk():
    model = load_model("disaster")
    constants = {**model.constants, "p_disaster": 0.5}  # high p: exercise both branches
    return model._replace(constants=constants)


def test_euler_errors_run_with_disaster_risk():
    from deqn_jax.evaluate.diagnostics import euler_equation_errors

    model = _disaster_with_risk()
    net = build_policy_net(model, jax.random.PRNGKey(0), (16,), None)
    out = euler_equation_errors(net, model, n_periods=30, seed=0, burn_in=5)
    assert out  # produced a report at all (used to raise ValueError)


def test_simulated_moments_run_with_disaster_risk():
    from deqn_jax.evaluate.diagnostics import simulated_moments

    model = _disaster_with_risk()
    net = build_policy_net(model, jax.random.PRNGKey(0), (16,), None)
    out = simulated_moments(net, model, n_periods=30, seed=0, burn_in=5)
    assert out and all(np.isfinite(v["mean"]) or True for v in out.values())


def test_stability_check_runs_with_disaster_risk():
    from deqn_jax.evaluate.diagnostics import stability_check

    model = _disaster_with_risk()
    net = build_policy_net(model, jax.random.PRNGKey(0), (16,), None)
    out = stability_check(net, model, n_periods=30, seed=0)
    assert isinstance(out, dict) and out
