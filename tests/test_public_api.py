"""Tests for the stable `deqn_jax.api` surface.

What this verifies:

* every symbol listed in ``deqn_jax.api.__all__`` is actually importable,
* discovery helpers (``list_models``, ``list_optimizers``, ``list_networks``)
  return the expected built-ins,
* programmatic registration via ``register_model`` round-trips
  through ``load_model`` / ``list_models`` and respects the duplicate /
  overwrite contract,
* dropping a registered model with ``unregister_model`` actually
  removes it from the registry.

Touching anything in this file means thinking about the public-API
contract documented in ``docs/site/REFERENCE.md``.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import deqn_jax.api as api
from deqn_jax.models import unregister_model

# -- Discovery ---------------------------------------------------------


class TestDiscovery:
    def test_list_models_returns_builtin_set(self):
        names = [n for n, _ in api.list_models()]
        # Must include the three canonical reference models.
        for required in ("brock_mirman", "disaster", "irbc"):
            assert required in names, f"missing built-in model {required!r}"
        # Every entry has a (str, str) shape.
        for n, d in api.list_models():
            assert isinstance(n, str) and isinstance(d, str)

    def test_list_optimizers_returns_full_set(self):
        opts = api.list_optimizers()
        # The 13 documented optimizer names must all be present.
        for required in (
            "adam",
            "sgd",
            "adamw",
            "lion",
            "muon",
            "ngd",
            "shampoo",
            "lbfgs",
            "mao",
            "mao_kfac",
            "gn",
            "ign",
            "lm",
        ):
            assert required in opts, f"missing optimizer {required!r}"

    def test_list_networks_matches_config(self):
        nets = api.list_networks()
        # Must match the documented network types in NetworkConfig.
        assert set(nets) == set(api.NetworkConfig.VALID_TYPES)

    def test_load_model_returns_modelspec(self):
        m = api.load_model("brock_mirman")
        assert isinstance(m, api.ModelSpec)
        assert m.name == "brock_mirman"

    def test_load_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            api.load_model("nonexistent_model_12345")


# -- Registration ------------------------------------------------------


class TestRegisterModel:
    """register_model contract — agent-codegen path."""

    def teardown_method(self):
        # Clean up anything the tests below leaked.
        unregister_model("custom_test_model")
        unregister_model("dup_model")

    def test_register_then_load(self):
        """A registered model is visible to load_model / list_models."""
        existing = api.load_model("brock_mirman")
        new_spec = existing._replace(name="custom_test_model")
        api.register_model(new_spec, description="for tests")

        names = [n for n, _ in api.list_models()]
        assert "custom_test_model" in names
        assert dict(api.list_models())["custom_test_model"] == "for tests"
        assert api.load_model("custom_test_model").name == "custom_test_model"

    def test_register_duplicate_raises_by_default(self):
        existing = api.load_model("brock_mirman")
        new_spec = existing._replace(name="dup_model")
        api.register_model(new_spec)
        with pytest.raises(ValueError, match="already registered"):
            api.register_model(new_spec)

    def test_register_overwrite_true_replaces(self):
        existing = api.load_model("brock_mirman")
        new_spec = existing._replace(name="dup_model")
        api.register_model(new_spec, description="first")
        # Replace deliberately:
        api.register_model(new_spec, description="second", overwrite=True)
        assert dict(api.list_models())["dup_model"] == "second"

    def test_register_non_modelspec_raises_type_error(self):
        with pytest.raises(TypeError, match="ModelSpec"):
            api.register_model({"name": "not_a_spec"})  # type: ignore[arg-type]

    def test_register_empty_name_raises(self):
        existing = api.load_model("brock_mirman")
        bad = existing._replace(name="")
        with pytest.raises(ValueError, match="non-empty"):
            api.register_model(bad)

    def test_unregister_removes_model(self):
        existing = api.load_model("brock_mirman")
        api.register_model(existing._replace(name="custom_test_model"))
        assert "custom_test_model" in [n for n, _ in api.list_models()]
        unregister_model("custom_test_model")
        assert "custom_test_model" not in [n for n, _ in api.list_models()]


# -- Surface completeness ---------------------------------------------


class TestPublicSurface:
    """Every symbol in __all__ resolves; no broken re-exports."""

    def test_all_symbols_resolve(self):
        for name in api.__all__:
            assert hasattr(api, name), (
                f"deqn_jax.api advertises {name!r} in __all__ but it isn't importable"
            )
            obj = getattr(api, name)
            assert obj is not None

    def test_modelspec_is_namedtuple(self):
        # The user-contract type — agents construct this. Must be a NamedTuple.
        assert hasattr(api.ModelSpec, "_fields")
        # And the required fields must be there.
        required = {
            "name",
            "n_states",
            "n_policies",
            "n_shocks",
            "constants",
            "equations_fn",
            "step_fn",
        }
        assert required.issubset(set(api.ModelSpec._fields))

    def test_train_config_default_constructs(self):
        cfg = api.TrainConfig()
        assert cfg.model == "brock_mirman"  # documented default
        assert isinstance(cfg.network, api.NetworkConfig)
        assert isinstance(cfg.optimizer, api.OptimizerConfig)


# -- End-to-end: register → train (smoke) -----------------------------


class TestEndToEndRegisteredModel:
    """An agent-codegen'd model registered via register_model trains."""

    def teardown_method(self):
        unregister_model("e2e_test_model")

    def test_register_train_smoke(self):
        existing = api.load_model("brock_mirman")
        api.register_model(
            existing._replace(name="e2e_test_model"),
            description="end-to-end smoke",
        )

        cfg = api.TrainConfig(
            model="e2e_test_model",
            episodes=2,
            batch_size=8,
            episode_length=2,
            sim_batch=8,
            mc_samples=1,
            initialize_each_episode=True,
            network=api.NetworkConfig(hidden_sizes=(8,)),
            optimizer=api.OptimizerConfig(name="adam", learning_rate=1e-3),
            verbose=False,
            n_minibatches_per_epoch=1,
        )
        params, history = api.train_from_config(cfg)
        # train_from_config returns the trained network module + history dict.
        assert params is not None
        assert "loss" in history
        assert len(history["loss"]) >= 1
        # Loss must be finite (basic sanity).
        assert jnp.isfinite(jnp.array(history["loss"][-1]))
