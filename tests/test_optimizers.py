"""Tests for optimizer registry and optimizer implementations."""

import jax
import jax.numpy as jnp
import optax
import pytest


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
        assert "mao_kfac" in opts
        assert "shampoo" in opts
        assert "lbfgs" in opts
        assert "lion" in opts
        assert "muon" in opts
        assert "ign" in opts
        assert "kfac" not in opts

    def test_create_adam(self):
        from deqn_jax.config import OptimizerConfig
        from deqn_jax.optimizers import OptimizerKind, create_optimizer

        opt, kind = create_optimizer(OptimizerConfig(name="adam"))
        assert kind == OptimizerKind.STANDARD

    def test_create_mao(self):
        from deqn_jax.config import OptimizerConfig
        from deqn_jax.optimizers import OptimizerKind, create_optimizer
        from deqn_jax.optimizers.mao import _MAOFactory

        opt, kind = create_optimizer(OptimizerConfig(name="mao"))
        assert kind == OptimizerKind.MAO
        assert isinstance(opt, _MAOFactory)

    def test_create_lbfgs(self):
        from deqn_jax.config import OptimizerConfig
        from deqn_jax.optimizers import OptimizerKind, create_optimizer

        opt, kind = create_optimizer(OptimizerConfig(name="lbfgs"))
        assert kind == OptimizerKind.LBFGS

    def test_unknown_optimizer_raises(self):
        from deqn_jax.config import OptimizerConfig
        from deqn_jax.optimizers import create_optimizer

        with pytest.raises(ValueError, match="Unknown optimizer"):
            create_optimizer(OptimizerConfig(name="nonexistent"))

    def test_grad_clip_chained(self):
        from deqn_jax.config import OptimizerConfig
        from deqn_jax.optimizers import create_optimizer

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

        assert jnp.sum(params**2) < 14.0  # Should decrease from 14


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
        from deqn_jax.config import OptimizerConfig
        from deqn_jax.optimizers.mao import MAOTransform, _MAOFactory

        factory = _MAOFactory(OptimizerConfig(name="mao", learning_rate=1e-3))
        mao = factory.with_num_tasks(5)
        assert isinstance(mao, MAOTransform)
        assert mao.n_tasks == 5


class TestMAOKFAC:
    """Test MAO-KFAC optimizer."""

    def test_init_and_update(self):
        from deqn_jax.optimizers.mao_kfac import MAOKFACTransform

        opt = MAOKFACTransform(learning_rate=1e-3, n_tasks=3, precond_update_freq=1)
        params = {"w": jnp.ones((4, 2)), "b": jnp.ones(2)}
        state = opt.init(params)

        # Simulate per-equation Jacobian
        eq_jac = jax.tree.map(
            lambda p: jnp.ones((3,) + p.shape) * 0.1,
            params,
        )

        updates, new_state = opt.update(eq_jac, state, params)
        assert updates["w"].shape == (4, 2)
        assert updates["b"].shape == (2,)
        assert jnp.all(jnp.isfinite(updates["w"]))
        assert jnp.all(jnp.isfinite(updates["b"]))

    def test_factory(self):
        from deqn_jax.config import OptimizerConfig
        from deqn_jax.optimizers.mao_kfac import MAOKFACTransform, _MAOKFACFactory

        factory = _MAOKFACFactory(OptimizerConfig(name="mao_kfac", learning_rate=1e-3))
        opt = factory.with_num_tasks(5)
        assert isinstance(opt, MAOKFACTransform)
        assert opt.n_tasks == 5

    def test_shared_R_per_eq_L(self):
        """Verify shared R is same for all eqs, per-eq L differs."""
        from deqn_jax.optimizers.mao_kfac import MAOKFACTransform

        opt = MAOKFACTransform(learning_rate=1e-3, n_tasks=2, precond_update_freq=1)
        params = {"w": jnp.ones((4, 3))}
        state = opt.init(params)

        # Two equations with different gradient structure
        key = jax.random.PRNGKey(0)
        j1 = jax.random.normal(key, (4, 3))
        j2 = jax.random.normal(jax.random.PRNGKey(1), (4, 3))
        eq_jac = {"w": jnp.stack([j1, j2])}  # [2, 4, 3]

        _, new_state = opt.update(eq_jac, state, params)

        # R is shared [3, 3]
        assert new_state.shared_R["w"].shape == (3, 3)
        # L is per-equation [2, 4, 4]
        assert new_state.per_eq_L["w"].shape == (2, 4, 4)
        # Per-eq L should differ between equations
        assert not jnp.allclose(new_state.per_eq_L["w"][0], new_state.per_eq_L["w"][1])

    def test_precond_update_freq(self):
        """Cached inverses only update when count % freq == 0."""
        from deqn_jax.optimizers.mao_kfac import MAOKFACTransform

        opt = MAOKFACTransform(learning_rate=1e-3, n_tasks=2, precond_update_freq=3)
        params = {"w": jnp.ones((3, 2))}
        state = opt.init(params)

        eq_jac = {"w": jnp.ones((2, 3, 2)) * 0.5}

        # Steps 1 and 2: inverses should stay at identity
        _, s1 = opt.update(eq_jac, state, params)
        _, s2 = opt.update(eq_jac, s1, params)
        assert jnp.allclose(s1.R_inv4["w"], state.R_inv4["w"])
        assert jnp.allclose(s2.R_inv4["w"], state.R_inv4["w"])

        # Step 3: inverses should update
        _, s3 = opt.update(eq_jac, s2, params)
        assert not jnp.allclose(s3.R_inv4["w"], state.R_inv4["w"])


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
        from deqn_jax.config import OptimizerConfig
        from deqn_jax.optimizers import OptimizerKind, create_optimizer

        opt, kind = create_optimizer(OptimizerConfig(name="lbfgs"))
        assert kind == OptimizerKind.LBFGS


class TestGaussNewtonIntegration:
    """Integration checks for GN/LM-specific trainer behavior."""

    @pytest.mark.parametrize("optimizer_name", ["gn", "ign", "lm"])
    def test_train_step_respects_lr_scale(self, optimizer_name):
        import equinox as eqx

        from deqn_jax.config import OptimizerConfig
        from deqn_jax.models import load_model
        from deqn_jax.training.trainer import create_train_state, make_train_step

        model = load_model("brock_mirman")
        key = jax.random.PRNGKey(0)
        n_eq = len(model.equation_names) if model.equation_names else 1

        state, opt, kind, _critic_opt = create_train_state(
            model=model,
            key=key,
            hidden_sizes=(8,),
            batch_size=4,
            n_equations=n_eq,
            optimizer_config=OptimizerConfig(name=optimizer_name, learning_rate=1.0),
        )
        train_step = make_train_step(
            model=model,
            opt=opt,
            episode_length=5,
            mc_samples=2,
            batch_size=4,
            kind=kind,
        )

        state_zero, _ = train_step(state, jnp.array(0.0), jnp.array(1.0))
        state_one, _ = train_step(state, jnp.array(1.0), jnp.array(1.0))

        params0 = eqx.filter(state.params, eqx.is_array)
        params_zero = eqx.filter(state_zero.params, eqx.is_array)
        params_one = eqx.filter(state_one.params, eqx.is_array)

        def maxdiff(lhs, rhs):
            leaves = jax.tree.leaves(
                jax.tree.map(lambda x, y: jnp.max(jnp.abs(x - y)), lhs, rhs)
            )
            return max(float(x) for x in leaves)

        assert maxdiff(params_zero, params0) == pytest.approx(0.0, abs=1e-10)
        assert maxdiff(params_one, params0) > 1e-8
        assert maxdiff(params_zero, params_one) > 1e-8

    def test_lm_rejects_loss_increase_and_raises_damping(self):
        from deqn_jax.optimizers.gauss_newton import levenberg_marquardt

        # This configuration produces an uphill tentative step; LM should
        # reject it and increase damping for the next iteration.
        params = jnp.array([-0.9749998450279236])
        opt = levenberg_marquardt(learning_rate=2.0, initial_damping=0.01)
        state = opt.init(params)

        def residual_fn(p):
            return jnp.array([p[0] ** 2 - 1.0])

        new_params, new_state = opt.update(residual_fn, params, state)
        current_loss = jnp.sum(residual_fn(params) ** 2)

        assert jnp.allclose(new_params, params)
        assert jnp.isclose(new_state.last_loss, current_loss)
        assert new_state.damping > state.damping

    def test_implicit_gn_matches_damped_gn_on_linear_residuals(self):
        from deqn_jax.optimizers.gauss_newton import (
            gauss_newton,
            implicit_gauss_newton,
        )

        A = jnp.array([[1.0, 2.0], [3.0, -1.0], [0.5, 1.0]])
        b = jnp.array([1.0, -2.0, 0.25])
        params = jnp.array([0.7, -0.3])
        damping = 1e-3

        def residual_fn(p):
            return A @ p - b

        dense = gauss_newton(learning_rate=1.0, damping=damping)
        implicit = implicit_gauss_newton(
            learning_rate=1.0,
            damping=damping,
            cg_iters=20,
            cg_tol=1e-10,
        )

        dense_params, _ = dense.update(residual_fn, params, dense.init(params))
        implicit_params, implicit_state = implicit.update(
            residual_fn, params, implicit.init(params)
        )

        assert jnp.allclose(implicit_params, dense_params, atol=1e-5, rtol=1e-5)
        assert implicit_state.last_cg_residual < 1e-6


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

    def test_mao_kfac(self):
        h = self._train_short("mao_kfac")
        assert len(h["loss"]) == 3
        assert all(jnp.isfinite(l) for l in h["loss"])

    def test_gn(self):
        h = self._train_short("gn")
        assert len(h["loss"]) == 3
        assert all(jnp.isfinite(l) for l in h["loss"])

    def test_ign(self):
        h = self._train_short("ign")
        assert len(h["loss"]) == 3
        assert all(jnp.isfinite(l) for l in h["loss"])

    def test_lm(self):
        h = self._train_short("lm")
        assert len(h["loss"]) == 3
        assert all(jnp.isfinite(l) for l in h["loss"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
