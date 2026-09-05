"""RSS-2019 three-country trade DSGE with Epstein–Zin preferences.

``rss_trade_ez_ref`` is the Phase-0 **faithful replica** of the reference
solution's function: 31 states (economic states plus the reference's training
scaffolding held at converged values), 73 policies in the reference's order,
82 residuals with the reference's names. It exists so that a trained
reference checkpoint can be loaded into ``network.type: rss_market_clearing_net``
and its policies reproduced exactly (the port's gate A); the Phase-1 variant
model with the cleaned layout is built on top of it, one flagged change at a
time.

Two-stage loss hooks carry the three expectation-bearing blocks (certainty
equivalent, bond Euler, capital Euler); the standard ``equations_fn`` path is
equivalent because each is linear in its expectation.
"""

from deqn_jax.models.rss_trade_ez_ref.definitions import make_definitions
from deqn_jax.models.rss_trade_ez_ref.dynamics import make_clip_state, make_step
from deqn_jax.models.rss_trade_ez_ref.equations import make_equations
from deqn_jax.models.rss_trade_ez_ref.steady_state import make_init_state
from deqn_jax.models.rss_trade_ez_ref.variables import (
    CONSTANTS,
    N_COUNTRIES,
    Layout,
    build_constants,
    shock_names,
)
from deqn_jax.types import ModelSpec


def build_model(constants=None, name: str = "rss_trade_ez_ref") -> ModelSpec:
    """Assemble the replica ModelSpec for ``len(constants['L'])`` countries."""
    constants = CONSTANTS if constants is None else constants
    layout = Layout(int(constants["n_countries"]))
    equations, inside_fn, combine_fn, names = make_equations(layout)
    clip = make_clip_state(layout)
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
        # The reference network applies the activations and the bond
        # projection itself; the hard clips of the accessor layer live in
        # definitions.clip_policy. No framework-side bounds.
        policy_lower=None,
        policy_upper=None,
        clip_state_fn=lambda s: clip(s, constants),
    )


MODEL = build_model()

__all__ = ["MODEL", "build_model", "build_constants", "Layout", "N_COUNTRIES"]
