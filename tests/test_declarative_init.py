"""Tests for declarative per-variable init state samplers.

``make_init_state_fn`` in ``models.variable_spec`` builds a
``(key, batch_size, constants) -> [batch, n_states]`` sampler from a
dict of ``{state_name: {distribution, kwargs}}``. Matches DEQN-MAO's
per-variable ``init`` spec; use is optional (models can still provide
a monolithic function).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_uniform_distribution_respects_bounds():
    from deqn_jax.models.variable_spec import make_init_state_fn

    fn = make_init_state_fn(
        state_names=("k",),
        init_specs={"k": {"distribution": "uniform",
                          "kwargs": {"minval": 0.1, "maxval": 1.0}}},
    )
    samples = fn(jax.random.PRNGKey(0), 10_000, {})
    assert samples.shape == (10_000, 1)
    assert samples.min() >= 0.1
    assert samples.max() <= 1.0


def test_normal_distribution_matches_requested_moments():
    from deqn_jax.models.variable_spec import make_init_state_fn

    fn = make_init_state_fn(
        state_names=("z",),
        init_specs={"z": {"distribution": "normal",
                          "kwargs": {"mean": 0.2, "std": 0.05}}},
    )
    samples = np.asarray(fn(jax.random.PRNGKey(0), 50_000, {}))[:, 0]
    # 50k samples -> std error of mean ~ 2e-4; std of std ~ similar.
    assert abs(samples.mean() - 0.2) < 1e-3
    assert abs(samples.std() - 0.05) < 1e-3


def test_multiple_states_each_with_own_distribution():
    from deqn_jax.models.variable_spec import make_init_state_fn

    fn = make_init_state_fn(
        state_names=("k", "z"),
        init_specs={
            "k": {"distribution": "uniform", "kwargs": {"minval": 0.5, "maxval": 1.5}},
            "z": {"distribution": "normal", "kwargs": {"mean": 0.0, "std": 0.1}},
        },
    )
    samples = np.asarray(fn(jax.random.PRNGKey(0), 5_000, {}))
    assert samples.shape == (5_000, 2)
    k_col = samples[:, 0]
    z_col = samples[:, 1]
    assert k_col.min() >= 0.5 and k_col.max() <= 1.5
    assert abs(z_col.mean()) < 0.01
    assert abs(z_col.std() - 0.1) < 0.01


def test_missing_state_defaults_to_zero():
    from deqn_jax.models.variable_spec import make_init_state_fn

    fn = make_init_state_fn(
        state_names=("k", "z"),
        init_specs={"k": {"distribution": "uniform",
                          "kwargs": {"minval": 0.0, "maxval": 1.0}}},
        # "z" missing -> zeros
    )
    samples = np.asarray(fn(jax.random.PRNGKey(0), 100, {}))
    assert np.all(samples[:, 1] == 0.0)


def test_unknown_distribution_rejected_at_build_time():
    from deqn_jax.models.variable_spec import make_init_state_fn

    with pytest.raises(ValueError, match="Unknown distribution"):
        make_init_state_fn(
            state_names=("k",),
            init_specs={"k": {"distribution": "gamma", "kwargs": {}}},
        )


def test_unknown_state_name_rejected_at_build_time():
    from deqn_jax.models.variable_spec import make_init_state_fn

    with pytest.raises(ValueError, match="unknown state"):
        make_init_state_fn(
            state_names=("k",),
            init_specs={"x": {"distribution": "uniform",
                              "kwargs": {"minval": 0.0, "maxval": 1.0}}},
        )


def test_deterministic_given_same_key():
    from deqn_jax.models.variable_spec import make_init_state_fn

    fn = make_init_state_fn(
        state_names=("k",),
        init_specs={"k": {"distribution": "uniform",
                          "kwargs": {"minval": 0.0, "maxval": 1.0}}},
    )
    key = jax.random.PRNGKey(42)
    a = fn(key, 100, {})
    b = fn(key, 100, {})
    assert jnp.array_equal(a, b)


def test_bm_deterministic_uses_declarative_init():
    """The bm_deterministic model has been migrated to declarative init;
    confirm its init_state_fn still produces the expected uniform
    distribution on [K_LB, K_UB]."""
    from deqn_jax.models.bm_deterministic import MODEL
    from deqn_jax.models.bm_deterministic.variables import K_LB, K_UB

    samples = np.asarray(MODEL.init_state_fn(jax.random.PRNGKey(0), 10_000, MODEL.constants))
    assert samples.shape == (10_000, 1)
    assert samples.min() >= K_LB
    assert samples.max() <= K_UB
    # Uniform sampler: mean should be ~ (K_LB + K_UB) / 2
    expected_mean = 0.5 * (K_LB + K_UB)
    assert abs(samples.mean() - expected_mean) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
