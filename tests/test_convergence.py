"""Convergence tests for DEQN-JAX models."""

import jax
import jax.numpy as jnp
import pytest


class TestBrockMirmanConvergence:
    """Test that Brock-Mirman model converges."""

    def test_loss_decreases(self):
        """Loss should decrease over training."""
        from deqn_jax.training.trainer import train

        params, history = train(
            "brock_mirman",
            episodes=200,
            hidden_sizes=(32, 32),
            batch_size=32,
            episode_length=50,
            mc_samples=3,
            verbose=False,
        )

        initial_loss = history["loss"][0]
        final_loss = history["loss"][-1]

        # Loss should decrease by at least 2x
        assert final_loss < initial_loss / 2, \
            f"Loss didn't decrease enough: {initial_loss:.4e} -> {final_loss:.4e}"

    def test_convergence_to_low_loss(self):
        """Model should converge to reasonably low loss."""
        from deqn_jax.training.trainer import train

        params, history = train(
            "brock_mirman",
            episodes=500,
            hidden_sizes=(64, 64),
            learning_rate=1e-3,
            batch_size=64,
            episode_length=100,
            mc_samples=5,
            verbose=False,
        )

        final_loss = history["loss"][-1]

        # Should achieve loss < 1e-3
        assert final_loss < 1e-3, f"Final loss too high: {final_loss:.4e}"

    def test_policy_near_steady_state(self):
        """Trained policy should be close to analytical steady state."""
        from deqn_jax.training.trainer import train
        from deqn_jax.models.brock_mirman import MODEL, steady_state

        params, _ = train(
            "brock_mirman",
            episodes=500,
            hidden_sizes=(64, 64),
            verbose=False,
        )

        # Get steady state
        ss_state, ss_policy = steady_state(MODEL.constants)

        # Evaluate policy at steady state
        pred_policy = params(ss_state[None, :])

        # Should be within 10% of true steady state policy
        rel_error = jnp.abs(pred_policy[0] - ss_policy) / ss_policy
        assert rel_error[0] < 0.1, f"Policy error at SS: {float(rel_error[0]):.2%}"


class TestDisasterTraining:
    """Test that Disaster model can train (convergence is harder)."""

    def test_loss_decreases(self):
        """Loss should decrease over training."""
        from deqn_jax.training.trainer import train

        params, history = train(
            "disaster",
            episodes=150,
            hidden_sizes=(64, 64),
            learning_rate=3e-4,
            batch_size=64,
            episode_length=50,
            mc_samples=3,
            verbose=False,
        )

        initial_loss = history["loss"][0]
        # Use min of last 20 episodes (loss is noisy on disaster model)
        final_loss = min(history["loss"][-20:])

        # Loss should decrease (disaster is hard, so just check it goes down)
        assert final_loss < initial_loss, \
            f"Loss didn't decrease: {initial_loss:.4e} -> {final_loss:.4e}"

    def test_no_nan_loss(self):
        """Training should not produce NaN losses."""
        from deqn_jax.training.trainer import train

        params, history = train(
            "disaster",
            episodes=50,
            hidden_sizes=(64, 64),
            learning_rate=1e-4,
            verbose=False,
        )

        # No NaN losses
        assert all(jnp.isfinite(l) for l in history["loss"]), "Got NaN/Inf losses"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
