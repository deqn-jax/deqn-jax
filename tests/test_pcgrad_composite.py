"""PCGrad × composite loss: aux-compatible gradient surgery guards.

The surgery contract (optimizers/pcgrad.py): with a custom
``compute_loss_fn``, PCGrad projects only the per-equation core
gradients (at the base loss's mean-over-equations scale) and adds the
exact auxiliary gradient grad(total) − grad(base) unprojected. Two
hand-computable limits pin it:

- NO conflict (parallel eq gradients): projection is the identity, so
  the step must equal the plain gradient step on the composite total.
- TOTAL conflict (1-param model, opposed eq gradients): PCGrad zeroes
  both core gradients, so the step must equal the aux gradient alone.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from deqn_jax.optimizers.pcgrad import make_grad_step_pcgrad
from deqn_jax.training.loss import compute_loss
from deqn_jax.types import ModelSpec, TrainState, make_reweight_state

# Single dummy quadrature node: deterministic expectations.
QN = jnp.zeros((1, 1))
QW = jnp.ones(1)
BATCH = jnp.array([[0.5], [1.0], [1.5], [2.0]])  # m2 = mean(s²) = 1.875
M2 = float(jnp.mean(BATCH**2))
A0 = 1.5
LR = 0.1
C_AUX = 0.5


class ScalePolicy(eqx.Module):
    """p(s) = a·s with a single scalar parameter — 1-D gradient space."""

    a: jax.Array

    def __call__(self, states):
        return self.a * states


def _toy(eq2_fn):
    def equations(state, policy, next_state, next_policy, constants):
        s = state[:, 0]
        p = policy[:, 0]
        return {"eq1": p - s, "eq2": eq2_fn(s, p)}

    return ModelSpec(
        name="toy_pcgrad",
        n_states=1,
        n_policies=1,
        n_shocks=1,
        equation_names=("eq1", "eq2"),
        constants={},
        equations_fn=equations,
        step_fn=lambda state, policy, shock, constants: state,
    )


# eq gradients w.r.t. a at a0: dl1/da = 2(a-1)m2, parallel dl2/da = 8(a-1)m2
NOCONFLICT = _toy(lambda s, p: 2.0 * (p - s))
# dl2/da = 2(a-2)m2: at a0=1.5 exactly opposed to dl1/da (cos = -1)
CONFLICT = _toy(lambda s, p: p - 2.0 * s)


def _custom_fn(c):
    """Composite-shaped loss: base + c·a² (aux depends on params only)."""

    def fn(
        model_,
        policy_fn,
        states,
        key,
        mc_samples=5,
        weights=None,
        shock_scale=1.0,
        quad_nodes=None,
        quad_weights=None,
        target_policy_fn=None,
    ):
        base, eq_losses = compute_loss(
            model_,
            policy_fn,
            states,
            key,
            mc_samples,
            weights=weights,
            shock_scale=shock_scale,
            quad_nodes=quad_nodes,
            quad_weights=quad_weights,
            target_policy_fn=target_policy_fn,
        )
        phi = jnp.sum(policy_fn.a**2)
        eq_losses = dict(eq_losses)
        eq_losses["aux_phi"] = phi
        return base + c * phi, eq_losses

    return fn


def _step_once(model, compute_loss_fn):
    params = ScalePolicy(a=jnp.array(A0))
    opt = optax.sgd(LR)
    opt_state = opt.init(eqx.filter(params, eqx.is_array))
    state = TrainState(
        params=params,
        opt_state=opt_state,
        episode_state=BATCH,
        key=jax.random.PRNGKey(0),
        step=0,
        episode=0,
        loss_weights=jnp.ones(2),
        reweight_state=make_reweight_state(2),
    )
    grad_step = make_grad_step_pcgrad(
        model,
        opt,
        mc_samples=2,
        quad_nodes=QN,
        quad_weights=QW,
        loss_reweight="none",
        reweight_alpha=0.9,
        use_target_network=False,
        compute_loss_fn=compute_loss_fn,
    )
    new_state, metrics = grad_step(state, BATCH, jnp.array(1.0), jnp.array(1.0))
    return float(new_state.params.a) - A0, metrics


def test_no_conflict_equals_plain_composite_gradient():
    # Parallel eq grads → projection = identity → step must be the plain
    # gradient step on base_mean + c·a²:
    #   grad = (2(a-1)m2 + 8(a-1)m2)/2 + 2c·a
    g_expected = (2 * (A0 - 1) * M2 + 8 * (A0 - 1) * M2) / 2 + 2 * C_AUX * A0
    delta, metrics = _step_once(NOCONFLICT, _custom_fn(C_AUX))
    np.testing.assert_allclose(delta, -LR * g_expected, rtol=1e-6)
    assert "aux_phi" in metrics.residuals


def test_total_conflict_leaves_exactly_the_aux_gradient():
    # At a0=1.5 the two eq grads are equal and opposite in the 1-D param
    # space; PCGrad projects both to zero, so ONLY grad(aux) = 2c·a moves.
    delta, _ = _step_once(CONFLICT, _custom_fn(C_AUX))
    np.testing.assert_allclose(delta, -LR * 2 * C_AUX * A0, rtol=1e-6)

    # Sanity: without surgery the same custom loss has zero base gradient
    # only by cancellation; with c=0 the surgery step must not move at all.
    delta0, _ = _step_once(CONFLICT, _custom_fn(0.0))
    np.testing.assert_allclose(delta0, 0.0, atol=1e-12)


def test_pure_pcgrad_path_unchanged():
    # compute_loss_fn=None keeps the legacy semantics: SUM of projected
    # grads, no mean normalization, no aux term.
    g_expected = 2 * (A0 - 1) * M2 + 8 * (A0 - 1) * M2
    delta, metrics = _step_once(NOCONFLICT, None)
    np.testing.assert_allclose(delta, -LR * g_expected, rtol=1e-6)
    assert "aux_phi" not in metrics.residuals


def test_real_composite_loss_runs_under_pcgrad():
    from deqn_jax.models import load_model
    from deqn_jax.training.composite_loss import (
        make_composite_loss,
        prepare_composite_data,
    )
    from deqn_jax.training.linearize import linearize_model
    from deqn_jax.training.loss import gauss_hermite_nd

    model = load_model("brock_mirman")
    P, Q = linearize_model(model, verbose=False)
    data = prepare_composite_data(model, P, Q, n_anchor_points=8, verbose=False)
    loss_fn = make_composite_loss(model, data)
    qn, qw = gauss_hermite_nd(3, model.n_shocks)

    from deqn_jax.networks.mlp import MLP

    net = MLP(
        in_features=model.n_states,
        out_features=model.n_policies,
        hidden_sizes=(8,),
        activations=(jax.nn.tanh,),
        key=jax.random.PRNGKey(0),
    )
    opt = optax.adam(1e-3)
    opt_state = opt.init(eqx.filter(net, eqx.is_array))
    ss_state, _ = model.steady_state_fn(model.constants)
    batch = jnp.tile(ss_state[None, :], (6, 1)) * jnp.linspace(0.9, 1.1, 6)[:, None]
    n_eq = len(model.equation_names)
    state = TrainState(
        params=net,
        opt_state=opt_state,
        episode_state=batch,
        key=jax.random.PRNGKey(1),
        step=0,
        episode=0,
        loss_weights=jnp.ones(n_eq),
        reweight_state=make_reweight_state(n_eq),
    )
    grad_step = make_grad_step_pcgrad(
        model,
        opt,
        mc_samples=2,
        quad_nodes=jnp.array(qn),
        quad_weights=jnp.array(qw),
        loss_reweight="none",
        reweight_alpha=0.9,
        use_target_network=False,
        compute_loss_fn=loss_fn,
    )
    new_state, metrics = grad_step(state, batch, jnp.array(1.0), jnp.array(1.0))
    assert "aux_anchor" in metrics.residuals
    assert jnp.isfinite(metrics.loss)
    # The step must actually move parameters.
    old_flat = jax.flatten_util.ravel_pytree(eqx.filter(net, eqx.is_array))[0]
    new_flat = jax.flatten_util.ravel_pytree(
        eqx.filter(new_state.params, eqx.is_array)
    )[0]
    assert float(jnp.max(jnp.abs(new_flat - old_flat))) > 0.0


def test_validator_gates():
    from deqn_jax.config import TrainConfig
    from deqn_jax.training.state_init import _validate_train_config

    base = {
        "model": "brock_mirman",
        "episodes": 2,
        "batch_size": 16,
        "verbose": False,
        "loss_type": "composite",
        "network": {"type": "mlp", "hidden_sizes": [8]},
    }

    # composite + pcgrad + STANDARD optimizer: now allowed.
    cfg = TrainConfig.from_dict({**base, "gradient_surgery": "pcgrad"})
    _validate_train_config(cfg)

    # composite + MAO: still rejected (aux never reaches its update).
    cfg_mao = TrainConfig.from_dict({**base, "optimizer": {"name": "mao"}})
    with pytest.raises(ValueError, match="composite"):
        _validate_train_config(cfg_mao)

    # coverage + pcgrad: still rejected (stress pools fold into the scalar).
    cfg_cov = TrainConfig.from_dict(
        {
            **base,
            "gradient_surgery": "pcgrad",
            "coverage": {
                "enabled": True,
                "stress_ranges": {"k_0": [0.5, 1.5]},
            },
        }
    )
    with pytest.raises(ValueError, match="STANDARD"):
        _validate_train_config(cfg_cov)
