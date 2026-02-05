"""Basic tests for DEQN-JAX components."""

import jax
import jax.numpy as jnp
import pytest


class TestTypes:
    """Test type definitions."""

    def test_model_spec_creation(self):
        """ModelSpec can be created with required fields."""
        from deqn_jax.types import ModelSpec

        def dummy_equations(s, p, ns, np, c):
            return {"eq": s[:, 0]}

        def dummy_step(s, p, shock, c):
            return s

        spec = ModelSpec(
            name="test",
            n_states=2,
            n_policies=1,
            n_shocks=1,
            equations_fn=dummy_equations,
            step_fn=dummy_step,
            constants={"a": 1.0},
        )

        assert spec.name == "test"
        assert spec.n_states == 2


class TestNetworks:
    """Test neural network architectures."""

    def test_mlp_forward(self):
        """MLP forward pass produces correct shape."""
        from deqn_jax.networks import create_mlp

        key = jax.random.PRNGKey(0)
        mlp = create_mlp(
            n_states=2,
            n_policies=1,
            hidden_sizes=(32, 32),
            key=key,
        )

        x = jnp.ones((16, 2))
        y = mlp(x)

        assert y.shape == (16, 1)

    def test_mlp_bounded_output(self):
        """MLP with bounds produces outputs in range."""
        from deqn_jax.networks import create_mlp

        key = jax.random.PRNGKey(0)
        lower = jnp.array([0.1])
        upper = jnp.array([0.9])

        mlp = create_mlp(
            n_states=2,
            n_policies=1,
            hidden_sizes=(32,),
            policy_lower=lower,
            policy_upper=upper,
            key=key,
        )

        x = jax.random.normal(key, (100, 2)) * 10  # Large inputs
        y = mlp(x)

        assert jnp.all(y >= lower)
        assert jnp.all(y <= upper)


class TestBrockMirman:
    """Test Brock-Mirman model."""

    def test_model_loads(self):
        """Model can be imported."""
        from deqn_jax.models.brock_mirman import MODEL

        assert MODEL.name == "brock_mirman"
        assert MODEL.n_states == 2
        assert MODEL.n_policies == 1
        assert MODEL.n_shocks == 1

    def test_steady_state(self):
        """Steady state is computed correctly."""
        from deqn_jax.models.brock_mirman import MODEL, steady_state

        ss_state, ss_policy = steady_state(MODEL.constants)

        assert ss_state.shape == (2,)
        assert ss_policy.shape == (1,)
        assert ss_state[0] > 0  # Capital positive
        assert 0 < ss_policy[0] < 1  # Savings rate in (0,1)

    def test_equations_at_steady_state(self):
        """Euler equation approximately satisfied at steady state."""
        from deqn_jax.models.brock_mirman import MODEL, steady_state, equations

        ss_state, ss_policy = steady_state(MODEL.constants)

        # Batch dimension
        state = ss_state[None, :]
        policy = ss_policy[None, :]

        # At steady state, next state = current state (no shock)
        residuals = equations(state, policy, state, policy, MODEL.constants)

        # Should be close to zero (float32 precision ~1e-7)
        assert jnp.abs(residuals["euler"][0]) < 1e-6

    def test_step_function(self):
        """Step function produces valid next state."""
        from deqn_jax.models.brock_mirman import MODEL, step, steady_state

        ss_state, ss_policy = steady_state(MODEL.constants)
        state = ss_state[None, :]
        policy = ss_policy[None, :]
        shock = jnp.zeros((1, 1))

        next_state = step(state, policy, shock, MODEL.constants)

        assert next_state.shape == (1, 2)
        assert next_state[0, 0] > 0  # Capital positive


class TestTraining:
    """Test training components."""

    def test_loss_computation(self):
        """Loss can be computed."""
        from deqn_jax.models.brock_mirman import MODEL
        from deqn_jax.networks import create_mlp
        from deqn_jax.training.loss import compute_loss

        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)

        mlp = create_mlp(
            n_states=MODEL.n_states,
            n_policies=MODEL.n_policies,
            hidden_sizes=(32,),
            policy_lower=MODEL.policy_lower,
            policy_upper=MODEL.policy_upper,
            key=k1,
        )

        states = jax.random.uniform(k2, (16, 2), minval=0.5, maxval=1.5)

        loss, eq_losses = compute_loss(MODEL, mlp, states, key, mc_samples=3)

        assert jnp.isfinite(loss)
        assert "euler" in eq_losses

    def test_episode_simulation(self):
        """Episode can be simulated."""
        from deqn_jax.models.brock_mirman import MODEL
        from deqn_jax.networks import create_mlp
        from deqn_jax.training.episode import run_episode, sample_initial_states

        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)

        mlp = create_mlp(
            n_states=MODEL.n_states,
            n_policies=MODEL.n_policies,
            hidden_sizes=(32,),
            policy_lower=MODEL.policy_lower,
            policy_upper=MODEL.policy_upper,
            key=k1,
        )

        init_state = sample_initial_states(MODEL, k2, batch_size=8)
        trajectory, final_state = run_episode(MODEL, mlp, init_state, k3, episode_length=10)

        assert trajectory.shape == (10, 8, 2)
        assert final_state.shape == (8, 2)

    def test_short_training(self):
        """Training runs without error."""
        from deqn_jax.training.trainer import train

        params, history = train(
            "brock_mirman",
            episodes=5,
            hidden_sizes=(16,),
            batch_size=16,
            episode_length=10,
            mc_samples=2,
            log_every=10,
            verbose=False,
        )

        assert len(history["loss"]) == 5
        assert all(jnp.isfinite(l) for l in history["loss"])


class TestOptimizers:
    """Test optimizer implementations."""

    def test_gauss_newton_init(self):
        """Gauss-Newton optimizer initializes."""
        from deqn_jax.optimizers.gauss_newton import gauss_newton

        opt = gauss_newton(learning_rate=0.1)
        params = {"w": jnp.ones((3, 2))}
        state = opt.init(params)

        assert state.count == 0

    def test_gauss_newton_step(self):
        """Gauss-Newton performs optimization step."""
        from deqn_jax.optimizers.gauss_newton import gauss_newton

        opt = gauss_newton(learning_rate=1.0)
        params = jnp.array([1.0, 2.0])
        state = opt.init(params)

        # Simple quadratic residuals
        def residual_fn(p):
            return p - jnp.array([0.0, 0.0])

        new_params, new_state = opt.update(residual_fn, params, state)

        # Should move toward zero
        assert jnp.all(jnp.abs(new_params) < jnp.abs(params))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
