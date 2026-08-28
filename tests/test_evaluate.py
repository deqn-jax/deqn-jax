"""Tests for the evaluation suite."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import deqn_jax.evaluate.cli as evaluate_cli
from deqn_jax.evaluate import (
    euler_equation_errors,
    print_moments,
    simulated_moments,
)
from deqn_jax.models import load_model
from deqn_jax.networks import create_mlp
from deqn_jax.types import ModelSpec


class _ZeroPolicy:
    def __call__(self, state):
        return jnp.zeros((*state.shape[:-1], 1))


def _quartic_two_stage_model(n_shocks=4):
    """Toy whose GH-3 and degree-3 monomial expectations differ."""

    def step(state, policy, shock, constants):
        del state, policy, constants
        return jnp.sum(shock**4, axis=-1, keepdims=True)

    def equations(state, policy, next_state, next_policy, constants):
        del state, policy, next_policy, constants
        return {"quartic": next_state[:, 0]}

    def inside(state, policy, next_state, next_policy, constants):
        del state, policy, next_policy, constants
        return {"quartic": next_state[:, 0]}

    def combine(state, policy, expectations, constants):
        del state, policy, constants
        return {"quartic": expectations["quartic"]}

    def steady_state(constants):
        del constants
        return jnp.zeros(1), jnp.zeros(1)

    return ModelSpec(
        name="quartic_two_stage",
        n_states=1,
        n_policies=1,
        n_shocks=n_shocks,
        equations_fn=equations,
        step_fn=step,
        constants={},
        state_names=("moment",),
        policy_names=("zero",),
        equation_names=("quartic",),
        steady_state_fn=steady_state,
        inside_fn=inside,
        combine_fn=combine,
    )


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
    def test_two_stage_honors_monomial_expectation_rule(self):
        model = _quartic_two_stage_model()

        monomial = euler_equation_errors(
            _ZeroPolicy(),
            model,
            n_periods=1,
            burn_in=0,
            expectation_type="monomial",
        )
        gauss_hermite = euler_equation_errors(
            _ZeroPolicy(),
            model,
            n_periods=1,
            burn_in=0,
            expectation_type="gauss_hermite",
            n_quadrature_points=3,
        )

        # Monomial nodes are +/-2 e_i, so sum(epsilon_i**4) is always 16.
        # GH-3 integrates the true fourth moment: 4 dimensions * 3 = 12.
        np.testing.assert_allclose(monomial["residuals"], [[16.0]])
        np.testing.assert_allclose(gauss_hermite["residuals"], [[12.0]])

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


def test_cli_threads_checkpoint_expectation_rule(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "expectation_type: monomial\nn_quadrature_points: 7\nmc_samples: 11\n"
    )
    calls = {}

    monkeypatch.setattr(
        evaluate_cli,
        "load_policy_from_checkpoint",
        lambda checkpoint, config: ((), SimpleNamespace(name="toy")),
    )
    monkeypatch.setattr(
        evaluate_cli,
        "stability_check",
        lambda *args, **kwargs: {
            "nan_free": True,
            "bound_hit_pct": 0.0,
            "max_ss_deviation_pct": 0.0,
            "stable": True,
        },
    )

    def fake_euler(*args, **kwargs):
        calls["euler"] = kwargs
        return {
            "residuals": jnp.zeros((1, 1)),
            "equation_names": ("zero",),
            "states": jnp.zeros((1, 1)),
        }

    def fake_market(*args, **kwargs):
        calls["market"] = kwargs
        return {"error": "no resource equation"}

    monkeypatch.setattr(evaluate_cli, "euler_equation_errors", fake_euler)
    monkeypatch.setattr(evaluate_cli, "market_clearing_errors", fake_market)
    monkeypatch.setattr(
        evaluate_cli, "print_euler_errors", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(evaluate_cli, "simulated_moments", lambda *args, **kwargs: {})
    monkeypatch.setattr(evaluate_cli, "print_moments", lambda *args, **kwargs: None)

    evaluate_cli.run_evaluate_cli(
        SimpleNamespace(
            checkpoint=str(tmp_path / "checkpoint.eqx"),
            config=str(config_path),
            label=None,
            periods=1,
            seed=0,
            dynare_dir=None,
            output=None,
        )
    )

    expected = {
        "expectation_type": "monomial",
        "n_quadrature_points": 7,
        "mc_samples": 11,
    }
    assert {key: calls["euler"][key] for key in expected} == expected
    assert {key: calls["market"][key] for key in expected} == expected


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


class TestSharedRollout:
    """The one rollout loop behind every eval primitive (evaluate/simulate.py)."""

    def test_eval_rollout_matches_hand_rolled_loop(self, tiny_model_and_net):
        """Same key, same clip, same shocks as the loop it replaced."""
        import jax.numpy as jnp

        from deqn_jax.evaluate.simulate import _draw_eval_shock, eval_rollout

        model, net = tiny_model_and_net
        ss_state, _ = model.steady_state_fn(model.constants)
        start = ss_state[None, :]

        def step(state, shock, _d):
            policy = net(state)
            if policy.ndim == 1:
                policy = policy[None, :]
            return model.step_fn(state, policy, shock, model.constants), state[0]

        seen = []
        eval_rollout(
            model,
            start,
            jax.random.PRNGKey(7),
            12,
            step,
            lambda t, out: seen.append(out[1]),
        )

        # Hand-rolled reference: the loop shape the diagnostics used to carry.
        key = jax.random.PRNGKey(7)
        state = start
        expected = []
        for _ in range(12):
            key, shock_key = jax.random.split(key)
            shock = _draw_eval_shock(model, shock_key, state)
            next_state, st = step(state, shock, None)
            expected.append(st)
            state = (
                model.clip_state_fn(next_state)
                if model.clip_state_fn is not None
                else next_state
            )

        np.testing.assert_array_equal(jnp.stack(seen), jnp.stack(expected))

    def test_record_can_stop_the_rollout(self, tiny_model_and_net):
        model, net = tiny_model_and_net
        ss_state, _ = model.steady_state_fn(model.constants)
        from deqn_jax.evaluate.simulate import eval_rollout

        def step(state, shock, _d):
            policy = net(state)
            if policy.ndim == 1:
                policy = policy[None, :]
            return (model.step_fn(state, policy, shock, model.constants),)

        seen = []

        def record(t, _out):
            seen.append(t)
            return t == 3  # stop here

        eval_rollout(model, ss_state[None, :], jax.random.PRNGKey(0), 50, step, record)
        assert seen == [0, 1, 2, 3]

    def test_discrete_plus_disaster_is_refused(self, tiny_model_and_net):
        """The one branch combination whose draw order was never defined."""
        import jax.numpy as jnp
        import pytest as _pytest

        from deqn_jax.evaluate.simulate import eval_rollout

        model, net = tiny_model_and_net

        def step_fn(state, policy, shock, constants, d_disaster=None):
            return state

        both = model._replace(
            step_fn=step_fn,
            transition_matrix=jnp.eye(2),
            z_state_idx=1,
            constants={**model.constants, "p_disaster": 0.1},
        )
        with _pytest.raises(NotImplementedError, match="discrete"):
            eval_rollout(
                both,
                jnp.zeros((1, model.n_states)),
                jax.random.PRNGKey(0),
                1,
                lambda s, sh, d: (s,),
                lambda t, out: None,
            )


class TestIrfCsvMode:
    """The CSV's mode column is derived from the results, not from a caller."""

    def test_mode_column_is_derived(self, tmp_path):
        from deqn_jax.irf import save_irf_csv

        irf_results = {"period": [0, 1], "k": [1.0, 1.1]}
        girf_results = {"_mode": "girf", **irf_results}

        for results, flag in ((irf_results, "0"), (girf_results, "1")):
            path = tmp_path / f"mode_{flag}.csv"
            save_irf_csv(results, str(path))
            rows = [line.strip().split(",") for line in path.read_text().splitlines()]
            # Metadata keys never become columns.
            assert rows[0] == ["period", "k", "mode"]
            assert [r[-1] for r in rows[1:]] == [flag, flag]
            # Every field stays numeric so float-parsing readers keep working.
            for r in rows[1:]:
                [float(v) for v in r]


class TestActiveSubspaceSampler:
    """The cli sampler's disaster branch (it used to have none)."""

    def test_samples_disaster_model(self):
        import jax.numpy as jnp

        from deqn_jax.cli import sample_ergodic_states
        from deqn_jax.networks.factory import build_policy_net

        model = load_model("disaster")
        model = model._replace(
            constants={**model.constants, "p_disaster": 0.5}
        )  # high p: both branches get visited
        net = build_policy_net(model, jax.random.PRNGKey(0), (16,), None)

        states = sample_ergodic_states(model, net, 20, jax.random.PRNGKey(3))
        assert states.shape[1] == model.n_states
        assert states.shape[0] > 0
        assert bool(jnp.all(jnp.isfinite(states)))
