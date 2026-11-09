"""``deqn-jax evaluate``: a simulation that produces NaN is a failed run."""

import sys

import jax
import pytest

import deqn_jax.evaluate.cli as evaluate_cli
from deqn_jax.cli import main
from deqn_jax.models import load_model
from deqn_jax.networks.factory import build_policy_net


def test_nan_simulation_exits_nonzero(monkeypatch, tmp_path):
    model = load_model("brock_mirman")
    net = build_policy_net(model, jax.random.PRNGKey(0), (16,), None)
    monkeypatch.setattr(
        evaluate_cli,
        "load_policy_from_checkpoint",
        lambda checkpoint, config: (net, model),
    )
    monkeypatch.setattr(
        evaluate_cli,
        "stability_check",
        lambda *args, **kwargs: {
            "nan_free": False,
            "bound_hit_pct": 0.0,
            "max_ss_deviation_pct": 0.0,
            "stable": False,
        },
    )
    monkeypatch.setattr(
        sys, "argv", ["deqn-jax", "evaluate", str(tmp_path / "ck.eqx"), "-n", "200"]
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code not in (0, None)
