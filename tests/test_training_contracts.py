"""Regression tests for high-level training-loop contracts.

These tests are intentionally small and use toy ModelSpec instances. They
pin semantics that are easy to lose in refactors without paying for full
model training runs.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from deqn_jax.types import ModelSpec, TrainState, make_reweight_state


class ZeroPolicy(eqx.Module):
    """Policy with a dummy parameter so optimizers have a real pytree leaf."""

    w: jax.Array

    def __call__(self, states):
        if states.ndim == 1:
            return jnp.array([0.0 * self.w])
        return jnp.zeros((states.shape[0], 1)) + 0.0 * self.w


def _zero_equations(state, policy, next_state, next_policy, constants):
    return {"eq": jnp.zeros(state.shape[0]) + 0.0 * jnp.sum(policy)}


def _identity_step(state, policy, shock, constants):
    return state


def _toy_model(**kwargs):
    spec = dict(
        name="toy",
        n_states=1,
        n_policies=1,
        n_shocks=1,
        equation_names=("eq",),
        constants={},
        equations_fn=_zero_equations,
        step_fn=_identity_step,
    )
    spec.update(kwargs)
    return ModelSpec(**spec)


def _train_state(params, opt_state, batch_size=4):
    return TrainState(
        params=params,
        opt_state=opt_state,
        episode_state=jnp.zeros((batch_size, 1)),
        key=jax.random.PRNGKey(0),
        step=0,
        episode=0,
        loss_weights=jnp.ones(1),
        reweight_state=make_reweight_state(1),
    )


def test_composite_loss_barrier_weight_affects_total_loss():
    from deqn_jax.training.composite_loss import CompositeData, make_composite_loss

    def definitions(state, policy, constants):
        return {
            "n": state[..., 0],
            "L": state[..., 1],
            "c": state[..., 2],
            "newton_h_prime": jnp.ones_like(state[..., 0]),
            "newton_residual": jnp.zeros_like(state[..., 0]),
        }

    def equations(state, policy, next_state, next_policy, constants):
        return {"eq": jnp.zeros(state.shape[0])}

    def step(state, policy, shock, constants):
        return state

    model = ModelSpec(
        name="barrier_toy",
        n_states=5,
        n_policies=1,
        n_shocks=1,
        equation_names=("eq",),
        constants={},
        equations_fn=equations,
        step_fn=step,
        definitions_fn=definitions,
    )
    data = CompositeData(
        P=jnp.zeros((1, 5)),
        ss_state=jnp.zeros(5),
        ss_policy=jnp.zeros(1),
        ergodic_cov_chol=jnp.eye(5),
        ss_leverage=2.0,
        anchor_points=jnp.zeros((2, 5)),
        anchor_deviations=jnp.zeros((2, 5)),
        anchor_lin_policy=jnp.zeros((2, 1)),
    )

    def policy_fn(states):
        if states.ndim == 1:
            return jnp.zeros(1)
        return jnp.zeros((states.shape[0], 1))

    states = jnp.array([
        [0.5, 12.0, 0.5, 1.0, 0.0],
        [0.25, 8.0, 0.25, 1.0, 0.0],
    ])
    common = dict(
        anchor_weight=0.0,
        jac_weight=0.0,
        jac_anchor_weight=0.0,
        newton_weight=0.0,
        leverage_mult=2.0,
    )
    loss_no_barrier = make_composite_loss(
        model, data, barrier_weight=0.0, **common,
    )
    loss_with_barrier = make_composite_loss(
        model, data, barrier_weight=3.0, **common,
    )

    total0, eq0 = loss_no_barrier(model, policy_fn, states, jax.random.PRNGKey(0), mc_samples=1)
    total3, eq3 = loss_with_barrier(model, policy_fn, states, jax.random.PRNGKey(0), mc_samples=1)

    barrier_sum = eq3["aux_barrier_n"] + eq3["aux_barrier_L"] + eq3["aux_barrier_c"]
    assert float(barrier_sum) > 0.0
    assert float(total0) == pytest.approx(0.0)
    assert float(total3 - total0) == pytest.approx(3.0 * float(barrier_sum), rel=1e-6)
    assert set(k for k in eq0 if k.startswith("aux_barrier_")) == {
        "aux_barrier_n",
        "aux_barrier_L",
        "aux_barrier_c",
    }


def test_run_episode_passes_disaster_indicator_when_probability_is_one():
    from deqn_jax.training.episode import run_episode

    def disaster_step(state, policy, shock, constants, d_disaster=0.0):
        d = jnp.asarray(d_disaster)
        d = jnp.reshape(d, (-1, 1)) if d.ndim > 0 else d
        return state + 1.0 + 9.0 * d

    model = _toy_model(constants={"p_disaster": 1.0}, step_fn=disaster_step)
    init = jnp.zeros((3, 1))
    _, final_state = run_episode(
        model,
        ZeroPolicy(jnp.array(0.0)),
        init,
        jax.random.PRNGKey(0),
        episode_length=3,
    )

    assert jnp.allclose(final_state, jnp.full((3, 1), 30.0))


def test_run_episode_disaster_indicator_broadcasts_against_per_sample_quantity():
    """Regression: maybe_draw_disaster must return [batch], not [batch, 1].

    The disaster model's step_fn computes ``k_next = defs["k"] * exp(-theta *
    d_disaster)`` where ``defs["k"]`` is shape ``[batch]``. If d_disaster
    arrives as ``[batch, 1]`` it numpy-broadcasts to ``[batch, batch]`` and
    a downstream ``jnp.stack(axis=1)`` fails. The previous toy step_fn
    defensively reshaped d_disaster, so it didn't catch this; this one
    mirrors the real model's shape-naive pattern.
    """
    from deqn_jax.training.episode import run_episode

    def disaster_step(state, policy, shock, constants, d_disaster=0.0):
        # state: [batch, 1]; treat the single state dim as a per-sample
        # capital-like quantity and apply k_next = k * exp(-theta * d).
        k = state[:, 0]                              # [batch]
        k_next = k * jnp.exp(-1.0 * d_disaster)      # must stay [batch]
        return jnp.stack([k_next], axis=1)           # [batch, 1]

    model = _toy_model(constants={"p_disaster": 1.0}, step_fn=disaster_step)
    init = jnp.full((3, 1), 10.0)
    _, final_state = run_episode(
        model,
        ZeroPolicy(jnp.array(0.0)),
        init,
        jax.random.PRNGKey(0),
        episode_length=2,
    )

    # k -> 10 * exp(-1)^2 = 10 * exp(-2). Loose tolerance since shock_scale
    # in the rollout doesn't enter this branch but we don't depend on it.
    assert final_state.shape == (3, 1)
    assert jnp.allclose(final_state[:, 0], 10.0 * jnp.exp(-2.0), atol=1e-6)


def test_train_step_shock_scale_zero_freezes_rollout_shocks():
    from deqn_jax.optimizers import OptimizerKind
    from deqn_jax.training.trainer import make_train_step

    def shock_step(state, policy, shock, constants):
        return state + shock[:, :1]

    model = _toy_model(step_fn=shock_step)
    params = ZeroPolicy(jnp.array(0.0))
    opt = optax.sgd(learning_rate=0.1)
    opt_state = opt.init(eqx.filter(params, eqx.is_array))
    state = _train_state(params, opt_state, batch_size=8)
    train_step = make_train_step(
        model=model,
        opt=opt,
        episode_length=4,
        mc_samples=1,
        batch_size=4,
        kind=OptimizerKind.STANDARD,
    )

    new_state, _ = train_step(state, jnp.array(1.0), jnp.array(0.0))

    assert jnp.allclose(new_state.episode_state, state.episode_state)


@pytest.mark.parametrize(
    ("optimizer_name", "gradient_surgery"),
    [
        ("mao", "none"),
        ("gn", "none"),
        ("lm", "none"),
        ("lbfgs", "none"),
        ("adam", "pcgrad"),
    ],
)
def test_composite_loss_rejects_update_paths_that_do_not_apply_aux_gradients(
    optimizer_name,
    gradient_surgery,
):
    from deqn_jax.config import NetworkConfig, OptimizerConfig, TrainConfig
    from deqn_jax.training.trainer import train_from_config

    cfg = TrainConfig(
        model="disaster",
        loss_type="composite",
        episodes=1,
        batch_size=2,
        episode_length=1,
        mc_samples=1,
        network=NetworkConfig(hidden_sizes=(4,)),
        optimizer=OptimizerConfig(name=optimizer_name, learning_rate=1e-3),
        gradient_surgery=gradient_surgery,
        verbose=False,
    )

    with pytest.raises(ValueError, match="loss_type='composite' is not supported"):
        train_from_config(cfg)
