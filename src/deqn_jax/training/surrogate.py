"""EWM world arm: the continuation surrogate Ŵ_ψ(x) ≈ E[inside_fn](x).

Two-stage models factor their residual as ``combine_fn(x, π(x), E[inside])``.
The world arm replaces ``E[inside]`` in the POLICY update by a small network
Ŵ fitted each episode to exact expectations on sparse anchor states, computed
under a Polyak-averaged target policy. Ŵ is treated as a fixed function in the
policy gradient (stop_gradient on its output). Coverage pools keep the exact
expectation by default (``exact_in_coverage``).

Reference: Scheidegger & Schaab (2026), arXiv:2606.23463. Spec:
docs/superpowers/specs/2026-08-28-ewm-world-arm-design.md.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from deqn_jax.training.loss import compute_residuals, sample_antithetic_shocks
from deqn_jax.types import ModelSpec


class SurrogateState(NamedTuple):
    """Everything the world arm carries in ``TrainState.aux_params``.

    All leaves are arrays (or an Equinox module) so the whole tuple is a
    checkpointable pytree; the loader template must construct the same
    structure (``create_train_state(..., surrogate_config=...)``).
    """

    net: Any  # eqx.Module: standardized state -> [n_keys]
    opt_state: Any  # optax state for net
    mean: Array  # [n_states] input standardization
    std: Array  # [n_states]
    b_policy: Array  # scalar: exact quadrature evaluations paid (anchors × nodes)
    b_world: Array  # scalar: surrogate fits (anchors × epochs_w)
    fit_mse: Array  # scalar: last anchor-fit MSE (diagnostic)


class _WorldNet(eqx.Module):
    layers: tuple
    positive: bool = eqx.field(static=True)

    def __init__(self, n_in: int, width: int, n_out: int, positive: bool, key: Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = (
            eqx.nn.Linear(n_in, width, key=k1),
            eqx.nn.Linear(width, width, key=k2),
            eqx.nn.Linear(width, n_out, key=k3),
        )
        self.positive = positive

    def __call__(self, x: Array) -> Array:
        h = jnp.tanh(self.layers[0](x))
        h = jnp.tanh(self.layers[1](h))
        out = self.layers[2](h)
        return jax.nn.softplus(out) if self.positive else out


def inside_keys(model: ModelSpec) -> Tuple[str, ...]:
    """Sorted keys of ``inside_fn``'s output (the surrogate's output columns)."""
    n_s = model.n_states
    s = jnp.zeros((1, n_s)) + 1.0
    p = jnp.zeros((1, model.n_policies)) + 0.5
    out = model.inside_fn(s, p, s, p, model.constants)
    return tuple(sorted(out.keys()))


def init_surrogate(model: ModelSpec, cfg, key: Array, lr: float) -> Tuple[SurrogateState, Any]:
    """Fresh world-arm state + its optax optimizer."""
    keys = inside_keys(model)
    net = _WorldNet(model.n_states, cfg.width, len(keys), cfg.positive_outputs, key)
    opt = optax.adam(cfg.lr_w if cfg.lr_w is not None else lr)
    opt_state = opt.init(eqx.filter(net, eqx.is_array))
    zero = jnp.array(0.0)
    sstate = SurrogateState(
        net=net,
        opt_state=opt_state,
        mean=jnp.zeros(model.n_states),
        std=jnp.ones(model.n_states),
        b_policy=zero,
        b_world=zero,
        fit_mse=zero,
    )
    return sstate, opt


def predict(sstate: SurrogateState, keys: Tuple[str, ...], states: Array) -> Dict[str, Array]:
    """Ŵ(x) as the expectations dict ``combine_fn`` expects (stop-gradient'd)."""
    x = (states - sstate.mean) / jnp.maximum(sstate.std, 1e-3)
    out = jax.vmap(sstate.net)(x)  # [b, n_keys]
    out = jax.lax.stop_gradient(out)
    return {k: out[:, i] for i, k in enumerate(keys)}


def exact_expectations(
    model: ModelSpec,
    policy_fn: Callable,
    states: Array,
    key: Array,
    mc_samples: int,
    shock_scale=1.0,
    quad_nodes: Optional[Array] = None,
    quad_weights: Optional[Array] = None,
) -> Dict[str, Array]:
    """E[inside_fn](x) with the trainer's own expectation operator (quadrature
    when nodes are given, antithetic MC otherwise). Mirrors compute_loss's
    two-stage branch; kept separate so compute_loss stays verbatim."""
    b = states.shape[0]
    if quad_nodes is not None and quad_weights is not None:
        n = quad_nodes.shape[0]
        shocks = jnp.broadcast_to(quad_nodes[:, None, :], (n, b, model.n_shocks)) * shock_scale
        w = jnp.broadcast_to(quad_weights[:, None], (n, b))
    else:
        shocks = sample_antithetic_shocks(key, mc_samples, b, model.n_shocks, shock_scale)
        n = shocks.shape[0]
        w = jnp.broadcast_to((jnp.ones(n) / n)[:, None], (n, b))

    def one(shock):
        return compute_residuals(model, policy_fn, states, shock, residual_fn=model.inside_fn)

    per_node = jax.vmap(one)(shocks)  # dict of [n, b]
    return {k: jnp.einsum("sb,sb->b", w, v) for k, v in per_node.items()}


def n_expectation_nodes(mc_samples: int, quad_nodes: Optional[Array]) -> int:
    return int(quad_nodes.shape[0]) if quad_nodes is not None else int(mc_samples)


def make_world_update(
    model: ModelSpec,
    cfg,
    opt: Any,
    mc_samples: int,
    quad_nodes: Optional[Array],
    quad_weights: Optional[Array],
    total_episodes: int,
    batch_size: int,
) -> Callable:
    """Build the per-episode world update (runs OUTSIDE the policy grad JIT).

    Returns ``update(sstate, target_params, dataset, key, lr_scale, episode)
    -> sstate``: sample anchors from the episode's dataset, compute exact
    E[inside] at the Polyak target policy, fit Ŵ for ``epochs_w`` Adam steps,
    refresh input standardization from the dataset, bump the budgets.
    """
    keys = inside_keys(model)
    n_nodes = n_expectation_nodes(mc_samples, quad_nodes)

    @eqx.filter_jit
    def _targets(target_params, anchors, key):
        q = exact_expectations(
            model, target_params, anchors, key, mc_samples, 1.0, quad_nodes, quad_weights
        )
        return jnp.stack([q[k] for k in keys], axis=1)  # [n_anchor, n_keys]

    @eqx.filter_jit
    def _fit_steps(net, opt_state, mean, std, x, y, lr_scale):
        def loss(n):
            pred = jax.vmap(n)((x - mean) / jnp.maximum(std, 1e-3))
            return jnp.mean((pred - y) ** 2)

        def body(carry, _):
            n, s = carry
            l, g = eqx.filter_value_and_grad(loss)(n)
            arrays = eqx.filter(n, eqx.is_array)
            upd, s = opt.update(eqx.filter(g, eqx.is_array), s, arrays)
            upd = jax.tree.map(lambda u: lr_scale * u, upd)
            n = eqx.combine(optax.apply_updates(arrays, upd), n)
            return (n, s), l

        (net, opt_state), losses = jax.lax.scan(body, (net, opt_state), None, length=cfg.epochs_w)
        return net, opt_state, losses[-1]

    def update(sstate: SurrogateState, target_params, dataset: Array, key: Array, lr_scale, episode: int):
        progress = float(episode) / float(max(1, total_episodes))
        frac = cfg.anchor_frac_at(progress)
        n_anchor = max(int(batch_size), int(frac * dataset.shape[0]))
        n_anchor = min(n_anchor, int(dataset.shape[0]))
        k_idx, k_q = jax.random.split(key)
        idx = jax.random.randint(k_idx, (n_anchor,), 0, dataset.shape[0])
        anchors = jax.lax.stop_gradient(dataset[idx])
        y = _targets(target_params, anchors, k_q)
        mean = jnp.mean(dataset, axis=0)
        std = jnp.std(dataset, axis=0)
        net, opt_state, fit = _fit_steps(sstate.net, sstate.opt_state, mean, std, anchors, y, lr_scale)
        return sstate._replace(
            net=net,
            opt_state=opt_state,
            mean=mean,
            std=std,
            b_policy=sstate.b_policy + n_anchor * n_nodes,
            b_world=sstate.b_world + n_anchor * cfg.epochs_w,
            fit_mse=fit,
        )

    return update


def polyak_update(target_params, params, tau: float):
    """θ̄ ← τ θ̄ + (1−τ) θ over array leaves (static fields untouched)."""
    t_arr = eqx.filter(target_params, eqx.is_array)
    p_arr = eqx.filter(params, eqx.is_array)
    new = jax.tree.map(lambda t, p: tau * t + (1.0 - tau) * p, t_arr, p_arr)
    return eqx.combine(new, target_params)


def make_surrogate_loss(model: ModelSpec, cfg) -> Callable:
    """Loss with the compute_loss signature (+ ``aux_params``) that uses Ŵ in
    place of the two-stage expectation. MSE aggregation only (validated)."""
    keys = inside_keys(model)

    def surrogate_loss_fn(
        model_,
        policy_fn,
        states,
        key,
        mc_samples: int = 5,
        weights=None,
        shock_scale=1.0,
        quad_nodes=None,
        quad_weights=None,
        target_policy_fn=None,
        loss_choice: str = "mse",
        huber_delta: float = 1.0,
        aux_params: Optional[SurrogateState] = None,
    ):
        if aux_params is None:
            raise ValueError("surrogate loss called without aux_params (SurrogateState)")
        cur = states[:, -1, :] if states.ndim == 3 else states
        expectations = predict(aux_params, keys, cur)
        residuals = model_.combine_fn(cur, policy_fn(states), expectations, model_.constants)
        eq_losses = {}
        total = 0.0
        for i, (name, r) in enumerate(residuals.items()):
            l = jnp.mean(r**2)
            eq_losses[name] = l
            w = 1.0 if weights is None else weights[i]
            total = total + w * l
        n_eq = len(residuals)
        if n_eq > 1:
            total = total / n_eq
        eq_losses["aux_world_fit"] = aux_params.fit_mse
        return total, eq_losses

    return surrogate_loss_fn
