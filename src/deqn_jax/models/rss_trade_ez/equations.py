"""Equilibrium conditions of the Phase-1 variant ``rss_trade_ez`` (76 residuals).

The 70 economic conditions are the replica's
(:func:`~deqn_jax.models.rss_trade_ez_ref.equations.economic_blocks`, writeup
(1)–(29), shared, not copied), renamed to the variant's keys. What the variant
changes:

- **No scaffolding residuals.** The replica's ``SDF_i`` (EZ discount
  accumulator) and ``Wealth_i`` (bond-mask penalty) are gone; with
  ``bonds_active = 0`` the bond Euler is replaced by ``a_i = 0`` instead, so
  the residual count does not change across curriculum stages.
- **Transversality as a pointwise, constant-weight auxiliary term.**
  ``aux_transversality_i = sqrt(w_tv muc_i / P_C_i) A_i``: the framework's
  mean-square loss of it is ``w_tv mean(muc_i / P_C_i A_i^2)``. The ``aux_``
  prefix keeps it out of adaptive reweighting and gradient surgery (and the
  trainer rejects the optimizers whose update would drop it).

Two-stage hooks as in the replica: ``inside_fn`` returns the CE / EB / EC
continuation integrands, ``combine_fn`` all residuals from their expectations.
"""

from __future__ import annotations

from typing import Dict, Tuple

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.rss_trade_ez.definitions import bonds_active, core
from deqn_jax.models.rss_trade_ez.variables import Layout
from deqn_jax.models.rss_trade_ez_ref.equations import (
    economic_blocks,
    inside_keys,
    integrands,
)

# (shared block, variant residual prefix), in the variant's order.
_BLOCKS = (
    [(f"capital_income_{s}", f"capital_income_{s}_") for s in "CMX"]
    + [(f"labor_income_{s}", f"labor_income_{s}_") for s in "CMX"]
    + [(f"input_demand_{s}", f"input_demand_{s}_") for s in "CMX"]
    + [
        ("capital_allocation", "capital_allocation_"),
        ("labor_allocation", "labor_allocation_"),
        ("intermediate_allocation", "intermediate_allocation_"),
    ]
    + [(f"market_clearing_{s}", f"market_clearing_{s}_") for s in "CMX"]
    + [(f"price_index_{s}", f"price_index_{s}_") for s in "CMX"]
    + [("bop", "bop_"), ("world", None)]
    + [
        ("law_of_motion_K", "law_of_motion_K_"),
        ("euler_bond", "euler_bond_"),
        ("euler_capital", "euler_capital_"),
        ("aux_transversality", "aux_transversality_"),
        ("certainty_equivalent", "certainty_equivalent_"),
        ("value_function", "value_function_"),
    ]
)
WORLD_NAME = "world_bond_clearing"


def equation_names(n: int) -> Tuple[str, ...]:
    """The variant's residual keys: ``22 n + 1 + 3 n`` (76 for n = 3)."""
    names = []
    for _, prefix in _BLOCKS:
        if prefix is None:
            names.append(WORLD_NAME)
        else:
            names += [f"{prefix}{i}" for i in range(1, n + 1)]
    return tuple(names)


def transversality(d: Dict[str, Array], constants) -> Array:
    """``sqrt(w_tv muc / P_C) A``, so that its mean square is the
    constant-weight transversality term ``w_tv mean(muc / P_C A^2)``."""
    w_tv = float(constants["w_tv"])
    return jnp.sqrt(w_tv) * jnp.sqrt(d["muc"] / d["P_C"]) * d["A"]


def make_equations(layout: Layout):
    n = layout.n
    names = equation_names(n)
    keys = inside_keys(n)

    def inside_fn(
        state, policy, next_state, next_policy, constants
    ) -> Dict[str, Array]:
        """CE / EB / EC continuation integrands at one next-period draw."""
        cur = core(state, policy, constants, layout)
        nxt = core(next_state, next_policy, constants, layout)
        out: Dict[str, Array] = {}
        for tag, values in zip(("ce", "eb", "ec"), integrands(cur, nxt, constants)):
            for i in range(n):
                out[f"{tag}_{i + 1}"] = values[:, i]
        return out

    def combine_fn(
        state, policy, expectations: Dict[str, Array], constants
    ) -> Dict[str, Array]:
        """All 76 residuals from the current state/policy and E[inside]."""
        d = core(state, policy, constants, layout)
        E = [
            jnp.stack([expectations[k] for k in keys[m * n : (m + 1) * n]], axis=1)
            for m in range(3)
        ]
        blocks = economic_blocks(d, E[0], E[1], E[2], constants)
        if not bonds_active(constants):
            # bonds shut: the bond Euler is replaced by a_i = 0
            blocks["euler_bond"] = d["a"]
        blocks["aux_transversality"] = transversality(d, constants)

        r: Dict[str, Array] = {}
        for block, prefix in _BLOCKS:
            if prefix is None:
                r[WORLD_NAME] = blocks[block]
            else:
                for i in range(n):
                    r[f"{prefix}{i + 1}"] = blocks[block][:, i]
        return r

    def equations(
        state, policy, next_state, next_policy, constants
    ) -> Dict[str, Array]:
        """Single-draw residuals; every expectation-bearing block is affine
        in its expectation, so averaging these over draws equals
        ``combine_fn(E[inside])``."""
        inside = inside_fn(state, policy, next_state, next_policy, constants)
        return combine_fn(state, policy, inside, constants)

    return equations, inside_fn, combine_fn, names


__all__ = ["make_equations", "equation_names", "transversality", "WORLD_NAME"]
