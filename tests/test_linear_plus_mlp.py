"""Tests for the generic LinearPlusMLP residual ansatz.

LinearPlusMLP is a model-agnostic neural-network architecture:

    policy(state) = clip(ss_policy + P @ (state - ss_state) + δ_θ(state),
                         policy_lower, policy_upper)

Per-model shape priors (K/F gauge mask, ELB feature, q-as-M reparam) live in
the model's own ``network.py`` module, not here. See
``tests/test_disaster_policy_net.py`` for disaster-specific shape-prior tests.

Properties pinned here:
  1. With ``init_scale=0.0``, the policy at training step 0 is exactly the
     BK linearization at every state (no MLP correction).
  2. Vmapped forward returns ``[batch, n_policies]``.
  3. The class has no model-specific fields — it's a clean library module.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

jax.config.update("jax_enable_x64", True)

from deqn_jax.models import load_model  # noqa: E402
from deqn_jax.networks.linear_plus_mlp import (  # noqa: E402
    LinearPlusMLP,
    create_linear_plus_mlp,
)
from deqn_jax.training.linearize import linearize_model  # noqa: E402


def _build(*, init_scale=0.0, seed=0) -> LinearPlusMLP:
    """Construct a generic LinearPlusMLP. No model-specific knobs."""
    model = load_model("disaster")
    return create_linear_plus_mlp(
        model,
        hidden_sizes=(16,),
        activation="tanh",
        init_scale=init_scale,
        key=jr.PRNGKey(seed),
    )


def test_init_scale_zero_gives_exact_bk_at_init():
    """init_scale=0.0: policy = π_BK exactly at every state at step 0."""
    model = load_model("disaster")
    net = _build(init_scale=0.0)
    ss_state, ss_policy = model.steady_state_fn(model.constants)
    out_ss = net(ss_state)
    assert float(jnp.max(jnp.abs(out_ss - ss_policy))) < 1e-10
    P, _ = linearize_model(model, verbose=False)
    perturbed = ss_state + 1e-2 * jr.normal(jr.PRNGKey(7), ss_state.shape)
    expected = ss_policy + P @ (perturbed - ss_state)
    out = net(perturbed)
    assert float(jnp.max(jnp.abs(out - expected))) < 1e-8


def test_2d_input_shape():
    """Vmapped forward returns [batch, n_policies]."""
    model = load_model("disaster")
    net = _build()
    ss_state, _ = model.steady_state_fn(model.constants)
    batch = jnp.broadcast_to(ss_state[None, :], (5, model.n_states))
    out = net(batch)
    assert out.shape == (5, model.n_policies)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_class_has_no_model_specific_fields():
    """The generic LinearPlusMLP must NOT carry disaster-specific knobs.

    This regression test exists because the class accreted disaster-specific
    fields (kf_indices, ELB feature, q_as_m) during early development. Those
    have been moved to ``DisasterPolicyNet`` in the disaster model module.
    Adding new model-specific fields back to LinearPlusMLP should fail this
    test and force the contributor to consider whether the new field belongs
    in a model-specific subclass instead.
    """
    net = _build()
    forbidden = {
        "kf_indices",  # Calvo/disaster-specific
        "use_zlb_feature",  # ELB/disaster-specific
        "zlb_feature_kind",
        "r_lag_idx",
        "r_lb",
        "reparam_q_as_m",  # investment-Euler/disaster-specific
        "q_idx",
        "i_idx",
        "i_lag_idx",
        "mu_z_idx",
        "kappa",
        "mu_z_ss",
    }
    actual_fields = set(vars(net).keys())
    leaked = forbidden & actual_fields
    assert not leaked, (
        f"LinearPlusMLP has leaked model-specific fields: {leaked}. "
        f"These belong in a model-specific subclass (e.g. DisasterPolicyNet)."
    )
