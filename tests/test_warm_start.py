"""Warm-start regression tests."""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest


@pytest.mark.skipif(
    not (
        Path(__file__).resolve().parents[1] / "dynare" / "results" / "dynare_ghx.csv"
    ).exists(),
    reason="Dynare reference output (dynare/results/) is local-only (gitignored); skipped on CI.",
)
def test_dynare_warm_start_runs_for_disaster_model():
    """Dynare warm-start should support current disaster policy variables."""
    from deqn_jax.models.disaster import MODEL
    from deqn_jax.networks import create_mlp
    from deqn_jax.training.warm_start import warm_start_from_dynare

    key = jax.random.PRNGKey(0)
    net = create_mlp(
        n_states=MODEL.n_states,
        n_policies=MODEL.n_policies,
        hidden_sizes=(16,),
        policy_lower=MODEL.policy_lower,
        policy_upper=MODEL.policy_upper,
        key=key,
    )

    dynare_dir = Path(__file__).resolve().parents[1] / "dynare" / "results"
    warm_net = warm_start_from_dynare(
        net,
        MODEL,
        dynare_dir=str(dynare_dir),
        n_points=64,
        max_iter=5,
        verbose=False,
        key=key,
    )

    ss_state, _ = MODEL.steady_state_fn(MODEL.constants)
    pred = warm_net(ss_state)

    assert pred.shape == (MODEL.n_policies,)
    assert jnp.all(jnp.isfinite(pred))
