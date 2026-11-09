"""``deqn-jax evaluate`` grades Euler errors on the conditional expectation.

The grade is computed on E[residual | state] wherever the model supports it,
and the report says which measure the grade rests on; where the conditional
expectation is unsupported it says the grade is single-draw.
"""

import sys

import jax
import numpy as np

import deqn_jax.evaluate.cli as evaluate_cli
from deqn_jax.cli import main
from deqn_jax.evaluate.diagnostics import euler_equation_errors
from deqn_jax.models import load_model
from deqn_jax.networks.factory import build_policy_net


def _run_evaluate(monkeypatch, tmp_path, net, model, periods=200):
    monkeypatch.setattr(
        evaluate_cli,
        "load_policy_from_checkpoint",
        lambda checkpoint, config: (net, model),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["deqn-jax", "evaluate", str(tmp_path / "ck.eqx"), "-n", str(periods)],
    )
    main()


def test_grade_uses_conditional_expectation(monkeypatch, tmp_path, capsys):
    model = load_model("brock_mirman")
    net = build_policy_net(model, jax.random.PRNGKey(0), (16,), None)
    seen = []

    def spy(*args, **kwargs):
        result = euler_equation_errors(*args, **kwargs)
        seen.append(result)
        return result

    monkeypatch.setattr(evaluate_cli, "euler_equation_errors", spy)
    _run_evaluate(monkeypatch, tmp_path, net, model)
    out = capsys.readouterr().out

    graded = seen[0]
    expected = euler_equation_errors(
        net, model, n_periods=200, seed=123, conditional_expectation=True
    )
    np.testing.assert_array_equal(graded["residuals"], expected["residuals"])
    assert (
        expected["measure"]
        != euler_equation_errors(net, model, n_periods=200)["measure"]
    )
    assert graded["measure"] in out


def test_measure_says_single_draw_where_unsupported():
    model = load_model("disaster")
    model = model._replace(constants={**model.constants, "p_disaster": 0.5})
    net = build_policy_net(model, jax.random.PRNGKey(0), (16,), None)
    out = euler_equation_errors(
        net, model, n_periods=30, seed=0, burn_in=5, conditional_expectation=True
    )
    assert out["measure"].startswith("single draw")


def test_library_default_stays_single_draw():
    model = load_model("brock_mirman")
    net = build_policy_net(model, jax.random.PRNGKey(0), (16,), None)
    out = euler_equation_errors(net, model, n_periods=50, seed=0)
    assert out["measure"].startswith("single draw")
