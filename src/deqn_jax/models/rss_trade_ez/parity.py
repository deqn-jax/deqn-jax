"""Replica ↔ variant correspondence for ``rss_trade_ez``, by name.

The Phase-1 variant must reproduce the replica's residuals wherever no
variant changes them, so its deltas are attributable one switch at a time.
This module holds what that comparison needs and nothing else: the name maps
between the two layouts (states, policies, shocks, residuals — matched by
name, never by position), the layout conversions (the replica's scaffolding
columns filled at given curriculum values), and a per-state two-stage
residual evaluator for an arbitrary node set.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from deqn_jax.types import ModelSpec


def state_source(var_name: str) -> str:
    """Replica state name of a variant state (``sig_ij`` was ``sigma_tau_ij``)."""
    return "sigma_tau_" + var_name[4:] if var_name.startswith("sig_") else var_name


def policy_source(var_name: str) -> str:
    """Replica policy name of a variant policy: ``K_prime_i`` was the
    replica's next-capital column ``K_i``; the raw bond ``a_i`` takes the
    replica's (already cleared) ``A_i``."""
    if var_name.startswith("K_prime_"):
        return "K_" + var_name[len("K_prime_") :]
    if var_name.startswith("a_"):
        return "A_" + var_name[2:]
    return var_name


_EQ_RENAMES = {
    "intermediate_allocation_": "intermediate_goods_allocation",
    "bop_": "BoP_",
    "euler_bond_": "EE_bond_",
    "euler_capital_": "EE_capital_",
    "aux_transversality_": "Transversality_",
}
_EQ_KEEP_UNDERSCORE = (
    "law_of_motion_K_",
    "certainty_equivalent_",
    "value_function_",
)


def equation_source(var_name: str) -> Optional[str]:
    """Replica residual corresponding to a variant residual (``None`` if the
    replica has none). ``aux_transversality_i`` maps to the replica's
    ``Transversality_i``, which measures a different quantity (see the
    variant's equations module)."""
    if var_name == "world_bond_clearing":
        return "Good_Market_Clearing_condition"
    for new, old in _EQ_RENAMES.items():
        if var_name.startswith(new):
            return old + var_name[len(new) :]
    if var_name.startswith(_EQ_KEEP_UNDERSCORE):
        return var_name
    # capital_income_C_1 -> capital_income_C1 (and the other sector blocks)
    stem, _, i = var_name.rpartition("_")
    return stem + i


def _index(names, wanted) -> np.ndarray:
    lookup = {n: k for k, n in enumerate(names)}
    missing = [w for w in wanted if w not in lookup]
    if missing:
        raise KeyError(f"names not in layout: {missing}")
    return np.array([lookup[w] for w in wanted])


def to_variant_state(ref_states: Array, ref: ModelSpec, var: ModelSpec) -> Array:
    """Variant-layout states from replica states (scaffolding dropped)."""
    idx = _index(ref.state_names, [state_source(n) for n in var.state_names])
    return ref_states[..., idx]


def _fill(ref: ModelSpec, states: Array, values: Dict[str, float]) -> Array:
    cols = _index(ref.state_names, list(values))
    return states.at[:, cols].set(
        jnp.asarray(list(values.values()), dtype=states.dtype)
    )


def pin_scaffolding(ref_states: Array, ref: ModelSpec) -> Array:
    """Replica states with the non-homotopy scaffolding at its converged
    values (``A_min = U_store = 1``, ``a_mask = 0``); the homotopy columns
    are kept."""
    values = {"A_min": 1.0, "a_mask": 0.0}
    values.update({n: 1.0 for n in ref.state_names if n.startswith("U_store_")})
    return _fill(ref, ref_states, values)


def to_replica_state(
    var_states: Array,
    var: ModelSpec,
    ref: ModelSpec,
    homo: float = 1.0,
    homo_1: float = 1.0,
) -> Array:
    """Replica-layout states from variant states: the homotopy columns at the
    given weights, the rest of the scaffolding converged
    (:func:`pin_scaffolding`), own-country tariffs and their log-vol zero."""
    b = var_states.shape[0]
    out = jnp.zeros((b, ref.n_states), dtype=var_states.dtype)
    idx = _index(ref.state_names, [state_source(n) for n in var.state_names])
    out = out.at[:, idx].set(var_states)
    out = _fill(ref, out, {"homo": homo, "homo_1": homo_1})
    return pin_scaffolding(out, ref)


def to_variant_policy(ref_policy: Array, ref: ModelSpec, var: ModelSpec) -> Array:
    """Variant-layout policies from (clipped) replica policies."""
    idx = _index(ref.policy_names, [policy_source(n) for n in var.policy_names])
    return ref_policy[..., idx]


def variant_policy_fn(
    ref_policy_fn: Callable[[Array], Array],
    ref: ModelSpec,
    var: ModelSpec,
    homo: float = 1.0,
    homo_1: float = 1.0,
) -> Callable[[Array], Array]:
    """The replica's policy function seen through the variant's layout: the
    same function of the physical state, so both models' residuals can be
    evaluated for one policy. ``ref_policy_fn`` must return clipped policies."""

    def fn(var_states: Array) -> Array:
        rs = to_replica_state(var_states, var, ref, homo, homo_1)
        return to_variant_policy(ref_policy_fn(rs), ref, var)

    return fn


def replica_nodes_on_variant(
    ref: ModelSpec, var: ModelSpec, ref_nodes: np.ndarray
) -> np.ndarray:
    """The replica's quadrature nodes expressed on the variant's shocks (by
    shock name): nodes on own-country axes, whose loadings are zero in the
    replica, become the zero shock."""
    out = np.zeros((ref_nodes.shape[0], var.n_shocks))
    idx = _index(ref.shock_names, var.shock_names)
    out[:] = ref_nodes[:, idx]
    return out


def random_reference_net(ref: ModelSpec, key: Array, hidden=(16, 16), scale=0.1):
    """A random replica-layout policy network for comparisons that need no
    trained weights. The reference architecture zero-initializes both output
    layers (a constant policy, under which every expectation is trivial), so
    their weights and biases are drawn here: the policy then depends on every
    state column, and the bond positions are nonzero."""
    import equinox as eqx

    from deqn_jax.networks.rss_net import create_rss_market_clearing_net

    k_net, k1, k2, k3 = jax.random.split(key, 4)
    net = create_rss_market_clearing_net(ref, key=k_net, base_hidden_sizes=hidden)
    w_base = net.base_layers[-1].weight
    w_ans = net.ansatz_layers[-1].weight
    b_base = net.base_layers[-1].bias
    return eqx.tree_at(
        lambda n: (
            n.base_layers[-1].weight,
            n.ansatz_layers[-1].weight,
            n.base_layers[-1].bias,
        ),
        net,
        (
            scale * jax.random.normal(k1, w_base.shape, w_base.dtype),
            scale * jax.random.normal(k2, w_ans.shape, w_ans.dtype),
            jax.random.normal(k3, b_base.shape, b_base.dtype),
        ),
    )


def quadrature_residuals(
    model: ModelSpec,
    policy_fn: Callable[[Array], Array],
    states: Array,
    nodes: Array,
    weights: Array,
) -> Dict[str, Array]:
    """Per-state residuals ``combine_fn(state, policy, sum_k w_k inside_k)``
    under the node set ``(nodes, weights)``, with ``constants`` the model's."""
    c = model.constants
    policy = policy_fn(states)

    def inside_at(node):
        shock = jnp.broadcast_to(node, (states.shape[0], node.shape[0]))
        nxt = model.step_fn(states, policy, shock, c)
        return model.inside_fn(states, policy, nxt, policy_fn(nxt), c)

    ins = jax.vmap(inside_at)(jnp.asarray(nodes, dtype=states.dtype))
    w = jnp.asarray(weights, dtype=states.dtype)
    expectations = {k: jnp.einsum("k,kb->b", w, v) for k, v in ins.items()}
    return model.combine_fn(states, policy, expectations, c)


__all__ = [
    "state_source",
    "policy_source",
    "equation_source",
    "to_variant_state",
    "to_replica_state",
    "to_variant_policy",
    "variant_policy_fn",
    "replica_nodes_on_variant",
    "quadrature_residuals",
]
