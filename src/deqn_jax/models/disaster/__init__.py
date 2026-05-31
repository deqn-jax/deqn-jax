"""CMR-style NK-DSGE with Financial Frictions ("Disaster Model").

A medium-scale New Keynesian model with:
- 13 state variables (8 endogenous + 5 exogenous)
- 11 policy variables (s, L, omega_bar computed analytically)
- 11 equilibrium equations
- Financial frictions (costly state verification banking)

Analytical eliminations (12 original -> 9):
  s (cost min), L (balance sheet), omega_bar (bank participation)
"""

from deqn_jax.models.disaster.composite_aux import (
    composite_aux,
    composite_aux_constants,
)
from deqn_jax.models.disaster.diagnostics import scalar_diagnostics
from deqn_jax.models.disaster.dynamics import clip_state, compute_state_barrier, step
from deqn_jax.models.disaster.equations import EQUATION_NAMES, definitions, equations
from deqn_jax.models.disaster.steady_state import init_state, steady_state
from deqn_jax.models.disaster.variables import (
    CONSTANTS,
    N_SHOCKS,
    POLICY_LOWER,
    POLICY_UPPER,
    SPEC,
)
from deqn_jax.types import ModelSpec


def _setup(model: ModelSpec, config) -> ModelSpec:
    """Pre-train hook for the disaster model.

    When ``p_disaster > 0`` the equilibrium contains a discrete
    capital-destruction branch and the deterministic steady state is
    no longer the relevant anchor. Composite-loss anchors and
    Blanchard-Kahn linearization should target the *risky* steady
    state instead -- the SS that solves ``E_d[F] = 0`` under the
    disaster mixture.

    This hook swaps ``steady_state_fn`` to ``risky_steady_state`` when
    both ``constants['p_disaster'] > 0`` and
    ``config.use_risky_steady_state`` is True (the default). Setting
    the flag False forces the deterministic SS even under disaster
    risk -- used for ablating the anchor/residual disagreement.

    Lives here, not in trainer.py, so the framework core stays model-
    agnostic. Triggered via ``ModelSpec.setup_fn`` from
    ``train_from_config``.
    """
    p = float(model.constants.get("p_disaster", 0.0))
    if p <= 0.0:
        return model

    use_risky = getattr(config, "use_risky_steady_state", True)
    if use_risky:
        from deqn_jax.models.disaster.steady_state import risky_steady_state

        if getattr(config, "verbose", False):
            print(f"  Anchor: risky steady state (p_disaster={p:.4f})")
        return model._replace(steady_state_fn=risky_steady_state)
    if getattr(config, "verbose", False):
        print(
            f"  Anchor: DETERMINISTIC SS forced "
            f"(use_risky_steady_state=False, p_disaster={p:.4f})"
        )
    return model


MODEL = ModelSpec(
    name="disaster",
    n_states=SPEC.n_states,
    n_policies=SPEC.n_policies,
    n_shocks=N_SHOCKS,
    state_names=SPEC.state_names,
    policy_names=SPEC.policy_names,
    equation_names=EQUATION_NAMES,
    constants=CONSTANTS,
    equations_fn=equations,
    step_fn=step,
    steady_state_fn=steady_state,
    init_state_fn=init_state,
    definitions_fn=definitions,
    policy_lower=POLICY_LOWER,
    policy_upper=POLICY_UPPER,  # None → softplus bounding (no gradient death)
    # default_output_links left None: existing disaster.yaml configs use the
    # legacy additive-linear ansatz unchanged. Opt in to the log ansatz
    # explicitly via NetworkConfig.output_links: [log, log, ..., log] (see
    # configs/disaster_log.yaml). All 11 disaster policies are positive so
    # log-link is mathematically valid; the default is conservative.
    default_output_links=None,
    clip_state_fn=clip_state,  # Hard clip for eval/irf only
    state_barrier_fn=compute_state_barrier,  # Box penalty for loss
    # Order MUST match dynamics.step()'s shock[:, i] unpacking order.
    shock_names=("eps", "mu_ups", "mu_z", "g", "m_p"),
    setup_fn=_setup,
    scalar_diagnostics_fn=scalar_diagnostics,
    composite_aux_fn=composite_aux,
    composite_aux_constants_fn=composite_aux_constants,
)
