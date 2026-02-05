"""Tests for optimizer registry and optimizer implementations."""

import jax
import jax.numpy as jnp
import pytest
import optax


class TestRegistry:
    """Test optimizer registry."""

    def test_list_optimizers(self):
        from deqn_jax.optimizers import list_optimizers
        opts = list_optimizers()
        assert "adam" in opts
        assert "sgd" in opts
        assert "adamw" in opts
        assert "ngd" in opts
        assert "mao" in opts
        assert "shampoo" in opts
        assert "lbfgs" in opts
        assert "lion" in opts
        assert "muon" in opts
        assert "kfac" not in opts

    def test_create_adam(self):
        from deqn_jax.optimizers import create_optimizer, OptimizerKind
        from deqn_jax.config import OptimizerConfig
        opt, kind = create_optimizer(OptimizerConfig(name="adam"))
        assert kind == OptimizerKind.STANDARD

    def test_create_mao(self):
        from deqn_jax.optimizers import create_optimizer, OptimizerKind
        from deqn_jax.optimizers.mao import _MAOFactory
        from deqn_jax.config import OptimizerConfig
        opt, kind = create_optimizer(OptimizerConfig(name="mao"))
        assert kind == OptimizerKind.MAO
        assert isinstance(opt, _MAOFactory)

    def test_create_lbfgs(self):
        from deqn_jax.optimizers import create_optimizer, OptimizerKind
        from deqn_jax.config import OptimizerConfig
        opt, kind = create_optimizer(OptimizerConfig(name="lbfgs"))
        assert kind == OptimizerKind.LBFGS

    def test_unknown_optimizer_raises(self):
        from deqn_jax.optimizers import create_optimizer
        from deqn_jax.config import OptimizerConfig
        with pytest.raises(ValueError, match="Unknown optimizer"):
            create_optimizer(OptimizerConfig(name="nonexistent"))

    def test_grad_clip_chained(self):
        from deqn_jax.optimizers import create_optimizer
        from deqn_jax.config import OptimizerConfig
        opt, kind = create_optimizer(OptimizerConfig(name="adam", grad_clip=1.0))
        # Should be a chained transform (clip + adam)
        params = {"w": jnp.ones(3)}
        state = opt.init(params)
        assert state is not None


class TestNGD:
    """Test Natural Gradient Descent."""

    def test_init_and_update(self):
        from deqn_jax.optimizers.ngd import ngd
        opt = ngd(learning_rate=0.1, damping=1e-4)
        params = {"w": jnp.ones((3, 2))}
        state = opt.init(params)
        grads = jax.tree.map(jnp.ones_like, params)
        updates, new_state = opt.update(grads, state, params)
        assert updates["w"].shape == (3, 2)
        assert jnp.all(jnp.isfinite(updates["w"]))

    def test_reduces_loss(self):
        """NGD reduces a simple quadratic loss."""
        from deqn_jax.optimizers.ngd import ngd
        opt = ngd(learning_rate=0.01)
        params = jnp.array([1.0, 2.0, 3.0])
        state = opt.init(params)

        for _ in range(10):
            grads = 2 * params  # grad of ||x||^2
            updates, state = opt.update(grads, state, params)
            params = optax.apply_updates(params, updates)

        assert jnp.sum(params ** 2) < 14.0  # Should decrease from 14


class TestMAO:
    """Test Multi-Adaptive Optimizer."""

    def test_init_and_update(self):
        from deqn_jax.optimizers.mao import MAOTransform
        mao = MAOTransform(learning_rate=1e-3, n_tasks=3)
        params = {"w": jnp.ones((4, 2)), "b": jnp.ones(2)}
        state = mao.init(params)

        # Simulate per-equation Jacobian (each leaf gets [n_tasks, *shape])
        eq_jac = jax.tree.map(
            lambda p: jnp.ones((3,) + p.shape) * 0.1,
            params,
        )

        updates, new_state = mao.update(eq_jac, state, params)
        assert updates["w"].shape == (4, 2)
        assert updates["b"].shape == (2,)
        assert jnp.all(jnp.isfinite(updates["w"]))

    def test_factory(self):
        from deqn_jax.optimizers.mao import _MAOFactory, MAOTransform
        from deqn_jax.config import OptimizerConfig
        factory = _MAOFactory(OptimizerConfig(name="mao", learning_rate=1e-3))
        mao = factory.with_num_tasks(5)
        assert isinstance(mao, MAOTransform)
        assert mao.n_tasks == 5


class TestShampoo:
    """Test Shampoo optimizer."""

    def test_init_and_update(self):
        from deqn_jax.optimizers.shampoo import shampoo
        opt = shampoo(learning_rate=0.01)
        params = {"w": jnp.ones((4, 3)), "b": jnp.ones(3)}
        state = opt.init(params)
        grads = jax.tree.map(jnp.ones_like, params)
        updates, new_state = opt.update(grads, state, params)
        assert updates["w"].shape == (4, 3)
        assert updates["b"].shape == (3,)
        assert jnp.all(jnp.isfinite(updates["w"]))
        assert jnp.all(jnp.isfinite(updates["b"]))


class TestLBFGS:
    """Test L-BFGS via optax."""

    def test_create(self):
        from deqn_jax.optimizers import create_optimizer, OptimizerKind
        from deqn_jax.config import OptimizerConfig
        opt, kind = create_optimizer(OptimizerConfig(name="lbfgs"))
        assert kind == OptimizerKind.LBFGS


class TestShortTraining:
    """Integration tests: short training runs with different optimizers."""

    def _train_short(self, optimizer_name, **extra_kwargs):
        from deqn_jax.training.trainer import train
        params, history = train(
            "brock_mirman",
            episodes=3,
            hidden_sizes=(16,),
            batch_size=16,
            episode_length=10,
            mc_samples=2,
            optimizer=optimizer_name,
            log_every=10,
            verbose=False,
            **extra_kwargs,
        )
        return history

    def test_adam(self):
        h = self._train_short("adam")
        assert len(h["loss"]) == 3
        assert all(jnp.isfinite(l) for l in h["loss"])

    def test_sgd(self):
        h = self._train_short("sgd")
        assert len(h["loss"]) == 3

    def test_adamw(self):
        h = self._train_short("adamw")
        assert len(h["loss"]) == 3

    def test_ngd(self):
        h = self._train_short("ngd")
        assert len(h["loss"]) == 3
        assert all(jnp.isfinite(l) for l in h["loss"])

    def test_mao(self):
        h = self._train_short("mao")
        assert len(h["loss"]) == 3
        assert all(jnp.isfinite(l) for l in h["loss"])

    def test_lion(self):
        h = self._train_short("lion")
        assert len(h["loss"]) == 3
        assert all(jnp.isfinite(l) for l in h["loss"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
