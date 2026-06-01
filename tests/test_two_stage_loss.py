"""Tests for the two-stage (expectation-inside-residual) loss path.

A two-stage model provides ``inside_fn`` (shock-dependent terms, averaged to
``E[inside]``) plus ``combine_fn`` (a nonlinearity applied AFTER the
expectation). This is the only MC-correct way to wrap an expectation in a
nonlinearity -- e.g. a Fischer-Burmeister borrowing constraint on an
intertemporal Euler (Geneva Day 2 Ex 4), where ``E[fb(.)] != fb(E[.])``.

Pinned here:
  1. ``combine_fn = identity`` reproduces the standard ``(E[residual])^2`` loss
     EXACTLY (the standard path is the special case combine = identity).
  2. ``combine_fn`` is applied to ``E[inside]``, not per-shock
     (combine-AFTER-expectation), which is what makes it MC-correct.
"""

import jax
import jax.numpy as jnp

from deqn_jax.training.loss import compute_loss
from deqn_jax.types import ModelSpec


def _step(state, policy, shock, constants):
    # next_state = state + shock, so the "inside" term is shock-dependent.
    return state + shock[:, :1]


def _eqs(state, policy, next_state, next_policy, constants):
    return {"eq": next_state[:, 0]}


def _policy_fn(states):
    return jnp.zeros((states.shape[0], 1))


def _model(**kw):
    base = dict(
        name="toy",
        n_states=1,
        n_policies=1,
        n_shocks=1,
        equation_names=("eq",),
        constants={},
        equations_fn=_eqs,
        step_fn=_step,
    )
    base.update(kw)
    return ModelSpec(**base)


def test_two_stage_combine_identity_equals_standard():
    states = jnp.array([[0.5], [1.0], [2.0]])
    key = jax.random.PRNGKey(0)
    std = _model()
    two = _model(inside_fn=_eqs, combine_fn=lambda s, p, E, c: {"eq": E["eq"]})
    l_std, eq_std = compute_loss(std, _policy_fn, states, key, mc_samples=8)
    l_two, eq_two = compute_loss(two, _policy_fn, states, key, mc_samples=8)
    assert float(l_two) == float(l_std)
    assert float(eq_two["eq"]) == float(eq_std["eq"])


def test_two_stage_combine_applied_after_expectation():
    states = jnp.array([[0.5], [1.0], [2.0]])
    key = jax.random.PRNGKey(0)

    def g(x):
        return jnp.sqrt(x * x + 1.0) - 1.0

    two = _model(inside_fn=_eqs, combine_fn=lambda s, p, E, c: {"eq": g(E["eq"])})
    loss, _ = compute_loss(two, _policy_fn, states, key, mc_samples=8)

    # Antithetic shocks => E[shock] = 0 exactly => E[inside] = E[state + shock]
    # = state. So combine-after-expectation gives loss = mean(g(state)^2),
    # independent of the shock spread. (The WRONG per-shock E[g(inside)^2]
    # would depend on the shock variance via Jensen; matching the value below
    # proves combine sees the expectation.)
    expected = float(jnp.mean(g(states[:, 0]) ** 2))
    assert abs(float(loss) - expected) < 1e-6
