"""RSS-2019 three-country trade DSGE with Epstein–Zin preferences: the
Phase-1 variant ``rss_trade_ez``.

Same economy as the faithful replica ``rss_trade_ez_ref`` (Ravikumar,
Santacreu & Sposi 2019) and the same economic code — definitions and
equilibrium conditions are imported from the replica package, not copied —
with the replica's reference-solution scaffolding replaced, one named change
each:

- **Curriculum scalars as constants** (``homotopy_asym``, ``homotopy_trade``,
  ``bonds_active``): the network never sees them; a curriculum is a sequence
  of resumed runs with ``--set constants.<name>=<value>``.
- **Transversality** as the constant-weight auxiliary residual
  ``aux_transversality_i`` (weight ``w_tv``), pointwise per state.
- **No EZ discount accumulators** (``U_store_i``) and no ``SDF_i`` residual.
- **Twelve shocks**: the off-diagonal tariff and log-volatility innovations;
  own-country tariffs are not in the state.
- **One tariff measure**: simulation and quadrature both map standard
  normals through the truncated-normal transport inside ``step_fn``.
- **Bond clearing in definitions**: the policy carries raw positions
  ``a_i``; the cleared ``A_i`` is computed in ``definitions_fn``, so any
  network (the standard ``mlp`` by default) can be trained.

18 states, 70 policies, 76 residuals, 12 shocks. Open accounting question,
kept as in the reference: tariff revenue is collected on imports but not
rebated to households, so the household budget and the balance of payments
differ by it (``tariff_revenue_i`` in the definitions measures the gap).
"""

import jax.numpy as jnp

from deqn_jax.models.rss_trade_ez.definitions import make_definitions
from deqn_jax.models.rss_trade_ez.dynamics import (
    make_clip_state,
    make_init_state,
    make_step,
)
from deqn_jax.models.rss_trade_ez.equations import make_equations
from deqn_jax.models.rss_trade_ez.variables import (
    CONSTANTS,
    N_COUNTRIES,
    Layout,
    build_constants,
    shock_names,
)
from deqn_jax.types import ModelSpec


def build_model(constants=None, name: str = "rss_trade_ez") -> ModelSpec:
    """Assemble the variant ModelSpec for ``len(constants['L'])`` countries."""
    constants = CONSTANTS if constants is None else constants
    layout = Layout(int(constants["n_countries"]))
    equations, inside_fn, combine_fn, names = make_equations(layout)
    clip = make_clip_state(layout)
    lower, upper = layout.policy_bounds()
    return ModelSpec(
        name=name,
        n_states=layout.n_states,
        n_policies=layout.n_policies,
        n_shocks=layout.n_shocks,
        state_names=layout.states,
        policy_names=layout.policies,
        equation_names=names,
        shock_names=shock_names(layout.n),
        constants=constants,
        equations_fn=equations,
        step_fn=make_step(layout),
        steady_state_fn=None,
        init_state_fn=make_init_state(layout),
        definitions_fn=make_definitions(layout),
        inside_fn=inside_fn,
        combine_fn=combine_fn,
        policy_lower=jnp.asarray(lower, dtype=jnp.float32),
        policy_upper=jnp.asarray(upper, dtype=jnp.float32),
        clip_state_fn=lambda s: clip(s, constants),
    )


MODEL = build_model()

__all__ = ["MODEL", "build_model", "build_constants", "Layout", "N_COUNTRIES"]
