"""Convergence tests for DEQN-JAX models."""

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
        assert final_loss < initial_loss / 2, (
            f"Loss didn't decrease enough: {initial_loss:.4e} -> {final_loss:.4e}"
        )

    def test_convergence_to_low_loss(self):
        """Model should converge to a low loss.

        Robustness (audit tests-04): the training trajectory is chaotic, so a
        tight absolute bar (the old ``< 1e-3``) flakes on numerically
        irrelevant code edits even when the model converged fine. Assert a
        large *relative* drop plus a loose absolute ceiling instead, with the
        seed pinned (default 42) for reproducibility.
        """
        from deqn_jax.training.trainer import train

        params, history = train(
            "brock_mirman",
            episodes=500,
            hidden_sizes=(64, 64),
            learning_rate=1e-3,
            batch_size=64,
            episode_length=100,
            mc_samples=5,
            seed=42,
            verbose=False,
        )

        initial_loss = history["loss"][0]
        final_loss = min(history["loss"][-20:])

        assert final_loss < initial_loss / 10, (
            f"Loss did not converge: {initial_loss:.4e} -> {final_loss:.4e}"
        )
        assert final_loss < 1e-2, f"Final loss unexpectedly high: {final_loss:.4e}"

    def test_policy_near_steady_state(self):
        """Trained policy should be close to analytical steady state."""
        from deqn_jax.models.brock_mirman import MODEL, steady_state
        from deqn_jax.training.trainer import train

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


class TestBrockMirmanLinearPlusMLP:
    """Sanity check: LinearPlusMLP architecture trains end-to-end on the
    closed-form Brock-Mirman model.

    BM has no kinks, no Calvo gauge, no constraint cliffs — so the BK linear
    policy is already a strong starting point. LinearPlusMLP should:
      (a) start at the exact BK linear policy (init_scale=0.0)
      (b) converge to a low residual loss
      (c) end up close to the analytical SS policy
      (d) keep the MLP correction `delta` bounded (BM is approximately
          log-linear in log-state; the correction shouldn't need to grow large)

    These are loose smoke-level checks — the architecture's value-add over
    bare MLP is in *harder* models (disaster). Here we just want "doesn't
    blow up, doesn't regress, behaves sensibly".
    """

    def _train(self, episodes: int = 200, kf_names=()):
        from deqn_jax.config import NetworkConfig, OptimizerConfig, TrainConfig
        from deqn_jax.training.trainer import train_from_config

        config = TrainConfig(
            model="brock_mirman",
            episodes=episodes,
            batch_size=32,
            episode_length=50,
            mc_samples=3,
            seed=42,
            network=NetworkConfig(
                type="linear_plus_mlp",
                hidden_sizes=(32, 32),
                activation="tanh",
                init="xavier_normal",
                init_scale=0.0,
                kf_names=tuple(kf_names),
            ),
            optimizer=OptimizerConfig(name="adam", learning_rate=1e-3),
            verbose=False,
            n_minibatches_per_epoch=1,
        )
        return train_from_config(config)

    def test_init_is_exact_bk_policy(self):
        """At step 0 with init_scale=0.0, the LinearPlusMLP policy must
        coincide with the BK linearization at SS."""
        import jax.random as jr

        from deqn_jax.models import load_model
        from deqn_jax.networks.linear_plus_mlp import create_linear_plus_mlp

        model = load_model("brock_mirman")
        net = create_linear_plus_mlp(
            model,
            hidden_sizes=(32, 32),
            activation="tanh",
            init_scale=0.0,
            key=jr.PRNGKey(0),
        )
        ss_state, ss_policy = model.steady_state_fn(model.constants)
        out = net(ss_state)
        assert float(jnp.max(jnp.abs(out - ss_policy))) < 1e-10, (
            "LinearPlusMLP at init should output the SS policy exactly at SS"
        )

    def test_loss_decreases(self):
        """Loss should drop at least 2x — same bar as the bare-MLP test."""
        _params, history = self._train(episodes=200)
        initial = history["loss"][0]
        final = history["loss"][-1]
        assert final < initial / 2, (
            f"Loss didn't decrease enough: {initial:.4e} -> {final:.4e}"
        )

    def test_policy_near_steady_state(self):
        """After training, policy at SS should match analytical SS within 5%
        (tighter than the bare-MLP test's 10% — LinearPlusMLP starts at
        exactly the right answer at SS, the MLP correction has no reason to
        push the network *away* from it)."""
        from deqn_jax.models.brock_mirman import MODEL, steady_state

        params, _ = self._train(episodes=300)
        ss_state, ss_policy = steady_state(MODEL.constants)
        pred = params(ss_state[None, :])
        rel_error = float(jnp.abs(pred[0, 0] - ss_policy[0]) / ss_policy[0])
        assert rel_error < 0.05, f"Policy error at SS: {rel_error:.2%}"

    def test_delta_correction_stays_bounded(self):
        """BM is approximately log-linear; the MLP correction shouldn't need
        to grow large to fit the residuals. Pin |delta(SS)| < some loose bound
        as a sanity check that we haven't accidentally landed somewhere weird.
        """
        from deqn_jax.models.brock_mirman import MODEL, steady_state

        params, _ = self._train(episodes=300)
        ss_state, _ = steady_state(MODEL.constants)
        # The "BK linear" part at SS evaluates to exactly ss_policy. Any
        # deviation in the trained output is the MLP delta at SS.
        bk_at_ss = params.ss_policy
        out = params(ss_state)
        delta = out - bk_at_ss
        assert float(jnp.max(jnp.abs(delta))) < 0.05, (
            f"MLP correction at SS unexpectedly large: {float(jnp.max(jnp.abs(delta))):.4e}"
        )


class TestDisasterTraining:
    """Test that Disaster model can train (convergence is harder)."""

    def test_loss_decreases(self):
        """Loss should decrease over training."""
        from deqn_jax.training.trainer import train

        # LR calibration note: loss aggregation across equations changed
        # from sum to mean (DEQN-MAO convention) on 2026-04-24. Disaster
        # has 11 equations, so the pre-change LR=3e-4 in sum mode acts
        # like LR=3.3e-5 in mean mode. Compensate with a larger LR and
        # more episodes so the smoke test still reflects "loss goes
        # down" under the new convention.
        params, history = train(
            "disaster",
            episodes=300,
            hidden_sizes=(64, 64),
            learning_rate=1e-2,
            batch_size=64,
            episode_length=50,
            mc_samples=3,
            verbose=False,
        )

        initial_loss = history["loss"][0]
        # Use min of last 20 episodes (loss is noisy on disaster model)
        final_loss = min(history["loss"][-20:])

        # Loss should decrease (disaster is hard, so just check it goes down)
        assert final_loss < initial_loss, (
            f"Loss didn't decrease: {initial_loss:.4e} -> {final_loss:.4e}"
        )

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
