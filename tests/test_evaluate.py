"""Tests for the evaluation suite."""

import jax
import numpy as np
import pytest

from deqn_jax.evaluate import (
    euler_equation_errors,
    print_moments,
    simulated_moments,
)
from deqn_jax.models import load_model
from deqn_jax.networks import create_mlp


@pytest.fixture
def tiny_model_and_net():
    """brock_mirman + 4-unit MLP: fast enough for short evaluations."""
    model = load_model("brock_mirman")
    net = create_mlp(
        n_states=model.n_states,
        n_policies=model.n_policies,
        hidden_sizes=(4,),
        policy_lower=model.policy_lower,
        policy_upper=model.policy_upper,
        key=jax.random.PRNGKey(0),
    )
    return model, net


class TestEulerEquationErrors:
    def test_short_run_auto_clamps_burn_in(self, tiny_model_and_net):
        """Previously crashed with ValueError on n_periods < burn_in default of 500."""
        model, net = tiny_model_and_net
        result = euler_equation_errors(net, model, n_periods=20, seed=0)
        assert result["residuals"].shape[0] > 0
        assert result["residuals"].shape[1] == len(model.equation_names)

    def test_burn_in_equal_to_n_periods_keeps_one_sample(self, tiny_model_and_net):
        model, net = tiny_model_and_net
        result = euler_equation_errors(
            net,
            model,
            n_periods=30,
            burn_in=30,
            seed=0,
        )
        assert result["residuals"].shape[0] >= 1

    def test_explicit_burn_in_respected(self, tiny_model_and_net):
        model, net = tiny_model_and_net
        result = euler_equation_errors(
            net,
            model,
            n_periods=100,
            burn_in=25,
            seed=0,
        )
        # n_periods - burn_in samples retained
        assert result["residuals"].shape[0] == 75

    def test_residuals_not_all_zero_for_untrained_net(self, tiny_model_and_net):
        """Sanity: a random-init network should produce nonzero Euler residuals."""
        model, net = tiny_model_and_net
        result = euler_equation_errors(net, model, n_periods=50, seed=0)
        residuals = np.asarray(result["residuals"])
        assert np.any(np.abs(residuals) > 1e-6)


class TestPrintMoments:
    def test_header_uses_actual_period_count(self, capsys):
        """Regression: was hardcoded to 'Simulated Moments (10,000 periods)' for any n."""
        moments = {
            "k": {
                "mean": 1.0,
                "std": 0.1,
                "min": 0.8,
                "max": 1.2,
                "ss": 1.0,
                "mean_dev_pct": 0.0,
            },
        }
        print_moments(moments, label="test", n_periods=2000)
        captured = capsys.readouterr()
        assert "2,000 periods" in captured.out
        assert "10,000 periods" not in captured.out

    def test_header_fallback_when_n_periods_omitted(self, capsys):
        moments = {
            "k": {
                "mean": 1.0,
                "std": 0.1,
                "min": 0.8,
                "max": 1.2,
                "ss": 1.0,
                "mean_dev_pct": 0.0,
            },
        }
        print_moments(moments, label="")
        captured = capsys.readouterr()
        # Back-compat: still prints something sane, not a stale "10,000"
        assert "10,000" not in captured.out


class TestSimulatedMoments:
    def test_short_run_returns_something(self, tiny_model_and_net):
        model, net = tiny_model_and_net
        moments = simulated_moments(net, model, n_periods=100, seed=0)
        assert isinstance(moments, dict)
        assert len(moments) > 0
        for v, stats in moments.items():
            assert "mean" in stats
            assert "std" in stats
            assert "ss" in stats
