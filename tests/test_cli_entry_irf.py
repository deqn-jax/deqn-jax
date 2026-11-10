"""``deqn-jax irf`` on the reader's first commands.

* A model without a closed-form steady state (olg_lifecycle,
  olg_lifecycle_56, rss_trade_ez_ref) is refused with a message naming the
  model and the reason, for IRF and GIRF alike.
* The printed table marks the pre-shock period of every equation residual as
  not computed, whatever the equation is called (brock_mirman's is ``euler``).
"""

import sys

import jax
import jax.numpy as jnp
import pytest

import deqn_jax.evaluate.irf as irf_mod
from deqn_jax.cli import main
from deqn_jax.evaluate.irf import print_irf_summary, run_girf, run_irf
from deqn_jax.models import load_model
from deqn_jax.networks import create_mlp


def _net(model):
    return create_mlp(
        n_states=model.n_states,
        n_policies=model.n_policies,
        hidden_sizes=(16,),
        policy_lower=model.policy_lower,
        policy_upper=model.policy_upper,
        key=jax.random.PRNGKey(0),
    )


def _no_policy(state):
    raise AssertionError("the policy must not be evaluated")


@pytest.mark.parametrize("runner", [run_irf, run_girf])
def test_model_without_steady_state_is_refused(runner):
    model = load_model("olg_lifecycle")
    assert model.steady_state_fn is None
    shock = (model.shock_names or ("shock_0",))[0]
    with pytest.raises(ValueError, match="olg_lifecycle.*steady state"):
        runner(_no_policy, model, shock_name=shock, horizon=3)


@pytest.mark.parametrize("girf", [False, True])
def test_cli_refuses_model_without_steady_state(girf, monkeypatch, tmp_path):
    model = load_model("olg_lifecycle")
    monkeypatch.setattr(
        irf_mod,
        "load_policy_from_checkpoint",
        lambda checkpoint, config: (jnp.zeros(1), model),
    )
    argv = ["deqn-jax", "irf", str(tmp_path / "ck.eqx"), "-o", str(tmp_path / "o")]
    monkeypatch.setattr(sys, "argv", argv + (["--girf"] if girf else []))
    with pytest.raises(ValueError, match="olg_lifecycle.*steady state"):
        main()


def test_preshock_residual_is_marked_not_computed(capsys):
    model = load_model("brock_mirman")
    assert "euler" in model.equation_names
    results = run_irf(_net(model), model, shock_name="eps_z", horizon=3)
    print_irf_summary(results, "eps_z")
    out = capsys.readouterr().out
    assert "nan" not in out.lower().split()
    euler_rows = [
        line.split() for line in out.splitlines() if line.split()[:1] == ["euler"]
    ]
    # One row, in the residual block: t=0 not computed, later periods numeric.
    assert len(euler_rows) == 1
    assert euler_rows[0][1] == "---"
    float(euler_rows[0][2])


def test_girf_summary_marks_preshock_residual(capsys):
    model = load_model("brock_mirman")
    results = run_girf(_net(model), model, shock_name="eps_z", horizon=3)
    print_irf_summary(results, "eps_z")
    assert "nan" not in capsys.readouterr().out.lower().split()


def test_cli_irf_table_has_no_nan(monkeypatch, tmp_path, capsys):
    model = load_model("brock_mirman")
    net = _net(model)
    monkeypatch.setattr(
        irf_mod, "load_policy_from_checkpoint", lambda checkpoint, config: (net, model)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "deqn-jax",
            "irf",
            str(tmp_path / "ck.eqx"),
            "--horizon",
            "3",
            "-o",
            str(tmp_path),
        ],
    )
    main()
    out = capsys.readouterr().out
    assert "nan" not in out.lower().split()
    assert (tmp_path / "irf_eps_z.csv").exists()
