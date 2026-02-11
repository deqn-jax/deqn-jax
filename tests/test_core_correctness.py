"""High-value correctness tests for core training mechanics.

Tests that verify mathematical invariants, not just "doesn't crash":
- Disaster model: all 11 equilibrium equations ≈ 0 at steady state
- Antithetic variates: shock pairing ε / -ε
- Gauss-Hermite quadrature: weight normalization
- eq_losses_to_array: aux_ prefix filtering
- GN loss tracking: uses post-update residuals
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


class TestDisasterSteadyState:
    """Verify disaster model equations are satisfied at steady state."""

    @pytest.fixture(autouse=True)
    def setup(self):
        jax.config.update("jax_enable_x64", True)
        from deqn_jax.models.disaster import MODEL, steady_state, equations
        self.MODEL = MODEL
        self.equations = equations
        self.ss_state, self.ss_policy = steady_state(MODEL.constants)

    def test_steady_state_shapes(self):
        assert self.ss_state.shape == (13,), f"Expected 13 states, got {self.ss_state.shape}"
        assert self.ss_policy.shape == (11,), f"Expected 11 policies, got {self.ss_policy.shape}"

    def test_steady_state_positive(self):
        """Key economic variables should be positive at SS."""
        from deqn_jax.models.disaster.variables import SPEC
        st = SPEC.unpack_state(self.ss_state)
        # Capital, consumption, investment, net worth proxy
        assert float(st.k_lag) > 0, "Capital should be positive"
        assert float(st.c_lag) > 0, "Consumption should be positive"
        assert float(st.i_lag) > 0, "Investment should be positive"
        assert float(st.q_lag) > 0, "Tobin's q should be positive"

    def test_all_equations_zero_at_ss(self):
        """All 11 equilibrium equations should be ≈ 0 at steady state."""
        state = self.ss_state[None, :]
        policy = self.ss_policy[None, :]

        residuals = self.equations(state, policy, state, policy, self.MODEL.constants)

        assert len(residuals) == 11, f"Expected 11 equations, got {len(residuals)}"

        max_resid = 0.0
        for name, r in residuals.items():
            val = float(jnp.abs(r[0]))
            max_resid = max(max_resid, val)
            assert val < 1e-4, (
                f"Equation '{name}' residual = {val:.2e} at SS (should be ≈ 0)"
            )

        # At least check that the max isn't suspiciously large
        assert max_resid < 1e-4, f"Max SS residual = {max_resid:.2e}"

    def test_step_at_ss_returns_ss(self):
        """With zero shocks, stepping from SS should return SS."""
        from deqn_jax.models.disaster import step

        state = self.ss_state[None, :]
        policy = self.ss_policy[None, :]
        zero_shock = jnp.zeros((1, self.MODEL.n_shocks))

        next_state = step(state, policy, zero_shock, self.MODEL.constants)

        # Next state should be close to current state (SS is fixed point)
        diff = jnp.max(jnp.abs(next_state[0] - self.ss_state))
        assert float(diff) < 1e-4, (
            f"SS is not a fixed point of step(): max |next - current| = {float(diff):.2e}"
        )


class TestAntitheticVariates:
    """Verify antithetic variance reduction sampling."""

    def test_pairing(self):
        """First half of shocks should be negation of second half."""
        from deqn_jax.training.loss import sample_antithetic_shocks

        key = jax.random.PRNGKey(42)
        shocks = sample_antithetic_shocks(key, n_samples=10, batch_size=4, shock_dim=3)

        assert shocks.shape == (10, 4, 3)

        first_half = shocks[:5]
        second_half = shocks[5:]
        assert jnp.allclose(first_half, -second_half), (
            "Antithetic pairs should satisfy ε_i = -ε_{i+N/2}"
        )

    def test_odd_samples(self):
        """Odd n_samples should still pair correctly for the even portion."""
        from deqn_jax.training.loss import sample_antithetic_shocks

        key = jax.random.PRNGKey(0)
        shocks = sample_antithetic_shocks(key, n_samples=7, batch_size=2, shock_dim=1)

        assert shocks.shape == (7, 2, 1)

        # First 3 should negate next 3 (the 7th is an extra unpaired sample)
        assert jnp.allclose(shocks[:3], -shocks[3:6])

    def test_shock_scale_applied(self):
        """shock_scale should multiply all shocks."""
        from deqn_jax.training.loss import sample_antithetic_shocks

        key = jax.random.PRNGKey(0)
        full = sample_antithetic_shocks(key, n_samples=4, batch_size=2, shock_dim=1, shock_scale=1.0)
        half = sample_antithetic_shocks(key, n_samples=4, batch_size=2, shock_dim=1, shock_scale=0.5)

        assert jnp.allclose(half, full * 0.5)

    def test_zero_shocks_returns_zeros(self):
        """n_samples=0 should return zeros."""
        from deqn_jax.training.loss import sample_antithetic_shocks

        key = jax.random.PRNGKey(0)
        shocks = sample_antithetic_shocks(key, n_samples=0, batch_size=4, shock_dim=2)

        assert jnp.all(shocks == 0.0)


class TestGaussHermiteQuadrature:
    """Verify Gauss-Hermite quadrature is correctly normalized."""

    def test_weights_sum_to_one(self):
        """Weights should sum to 1 after normalization to standard normal."""
        from deqn_jax.training.loss import gauss_hermite_nd

        for n_points in [3, 5, 7]:
            result = gauss_hermite_nd(n_points, dim=1)
            assert result is not None
            nodes, weights = result
            assert np.isclose(weights.sum(), 1.0, atol=1e-12), (
                f"1D weights with {n_points} points sum to {weights.sum()}, expected 1.0"
            )

    def test_multidim_weights_sum_to_one(self):
        """Tensor-product weights should also sum to 1."""
        from deqn_jax.training.loss import gauss_hermite_nd

        for dim in [2, 3]:
            result = gauss_hermite_nd(3, dim=dim)
            assert result is not None
            nodes, weights = result
            assert np.isclose(weights.sum(), 1.0, atol=1e-10), (
                f"{dim}D weights sum to {weights.sum()}, expected 1.0"
            )

    def test_nodes_shape(self):
        """Nodes should have shape [n_points^dim, dim]."""
        from deqn_jax.training.loss import gauss_hermite_nd

        result = gauss_hermite_nd(5, dim=2)
        assert result is not None
        nodes, weights = result
        assert nodes.shape == (25, 2)
        assert weights.shape == (25,)

    def test_first_moment_zero(self):
        """Weighted mean of nodes should be ≈ 0 (standard normal mean)."""
        from deqn_jax.training.loss import gauss_hermite_nd

        result = gauss_hermite_nd(7, dim=1)
        assert result is not None
        nodes, weights = result
        mean = np.sum(weights * nodes[:, 0])
        assert np.isclose(mean, 0.0, atol=1e-12), (
            f"Weighted mean = {mean}, expected 0"
        )

    def test_second_moment_one(self):
        """Weighted variance should be ≈ 1 (standard normal variance)."""
        from deqn_jax.training.loss import gauss_hermite_nd

        result = gauss_hermite_nd(7, dim=1)
        assert result is not None
        nodes, weights = result
        variance = np.sum(weights * nodes[:, 0] ** 2)
        assert np.isclose(variance, 1.0, atol=1e-10), (
            f"Weighted variance = {variance}, expected 1.0"
        )

    def test_too_many_points_returns_none(self):
        """Should return None when grid exceeds max_points."""
        from deqn_jax.training.loss import gauss_hermite_nd

        assert gauss_hermite_nd(100, dim=3, max_points=4096) is None


class TestEqLossesFiltering:
    """Verify aux_ prefix filtering in eq_losses_to_array."""

    def test_filters_aux_prefix(self):
        """Keys starting with aux_ should be excluded."""
        from deqn_jax.training.loss import eq_losses_to_array

        eq_losses = {
            "euler": jnp.array(1.0),
            "resource": jnp.array(2.0),
            "aux_barrier_n": jnp.array(0.5),
            "aux_newton_resid": jnp.array(0.1),
        }

        arr = eq_losses_to_array(eq_losses)

        assert arr.shape == (2,), f"Expected 2 base equations, got {arr.shape[0]}"
        assert float(arr[0]) == 1.0
        assert float(arr[1]) == 2.0

    def test_no_aux_keys(self):
        """All keys pass through when none have aux_ prefix."""
        from deqn_jax.training.loss import eq_losses_to_array

        eq_losses = {
            "eq1": jnp.array(1.0),
            "eq2": jnp.array(2.0),
            "eq3": jnp.array(3.0),
        }

        arr = eq_losses_to_array(eq_losses)
        assert arr.shape == (3,)

    def test_all_aux_keys_raises(self):
        """Should raise when all keys are aux_ (no base equations)."""
        from deqn_jax.training.loss import eq_losses_to_array

        eq_losses = {
            "aux_a": jnp.array(1.0),
            "aux_b": jnp.array(2.0),
        }

        # jnp.stack on empty list raises -- this is fine since
        # a model with zero base equations is invalid
        with pytest.raises(ValueError):
            eq_losses_to_array(eq_losses)


class TestGaussNewtonLossTracking:
    """Verify GN tracks post-update loss, not pre-update."""

    def test_loss_decreases_on_good_step(self):
        """After a good GN step, tracked loss should be lower than initial."""
        from deqn_jax.optimizers.gauss_newton import gauss_newton

        opt = gauss_newton(learning_rate=1.0, damping=1e-4)
        params = jnp.array([3.0, 4.0])
        state = opt.init(params)

        def residual_fn(p):
            return p  # r = p, so loss = ||p||², minimized at origin

        new_params, new_state = opt.update(residual_fn, params, state)

        old_loss = float(jnp.sum(params ** 2))
        tracked_loss = float(new_state.last_loss)
        actual_new_loss = float(jnp.sum(new_params ** 2))

        # Tracked loss should match actual post-update loss
        assert jnp.isclose(tracked_loss, actual_new_loss, rtol=1e-5), (
            f"Tracked loss {tracked_loss:.6f} != actual {actual_new_loss:.6f}"
        )

        # And it should be less than the old loss
        assert tracked_loss < old_loss, (
            f"GN step should reduce loss: {old_loss:.6f} -> {tracked_loss:.6f}"
        )


class TestAuxDecayFloorValidation:
    """Verify aux_decay_floor is validated in CompositeLossConfig."""

    def test_valid_values(self):
        from deqn_jax.config import CompositeLossConfig

        CompositeLossConfig(aux_decay_floor=0.0)  # no decay
        CompositeLossConfig(aux_decay_floor=0.5)
        CompositeLossConfig(aux_decay_floor=1.0)  # full decay

    def test_negative_raises(self):
        from deqn_jax.config import CompositeLossConfig

        with pytest.raises(ValueError, match="aux_decay_floor"):
            CompositeLossConfig(aux_decay_floor=-0.1)

    def test_above_one_raises(self):
        from deqn_jax.config import CompositeLossConfig

        with pytest.raises(ValueError, match="aux_decay_floor"):
            CompositeLossConfig(aux_decay_floor=1.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
