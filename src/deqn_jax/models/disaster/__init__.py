"""CMR-style NK-DSGE with Financial Frictions ("Disaster Model").

A medium-scale New Keynesian model with:
- 13 state variables (8 endogenous + 5 exogenous)
- 11 policy variables (network outputs; see ``SPEC.policy_names``)
- 11 equilibrium equations (see ``EQUATION_NAMES``)
- Financial frictions (costly state verification banking)

Analytical eliminations (14 unknowns -> 11): three quantities and their
three equations are solved inside ``definitions()`` rather than by the
network: s (cost minimisation) and L (balance sheet) in closed form,
omega_bar (bank participation) by a fixed-iteration projected Newton.
What remains is the 11-policy / 11-residual system the solver actually
trains on.
"""

from deqn_jax.models.disaster.composite_aux import (
    composite_aux,
    composite_aux_constants,
)
from deqn_jax.models.disaster.diagnostics import scalar_diagnostics
from deqn_jax.models.disaster.dynamics import clip_state, compute_state_barrier, step
from deqn_jax.models.disaster.equations import (
    EQUATION_NAMES,
    anchor_gate,
    definitions,
    equations,
)
from deqn_jax.models.disaster.steady_state import (
    TARGET_NAMES,
    consistent_targets,
    init_state,
    steady_state,
)
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

    Two adaptations to the run's calibration.

    The Taylor-rule targets ``R_ss`` and ``y_ss`` follow the calibration:
    they are re-derived by ``consistent_targets`` from the run's constants,
    so a run that changes, say, ``beta`` still has its steady state at
    ``pi_ss``. A run that sets either target in ``config.constants`` keeps
    both as given.

    When ``p_disaster > 0`` the equilibrium contains a discrete
    capital-destruction branch and the deterministic steady state is
    no longer the relevant anchor. Composite-loss anchors and
    Blanchard-Kahn linearization should target the *risky* steady
    state instead -- the SS that solves ``E_d[F] = 0`` under the
    disaster mixture. The hook swaps ``steady_state_fn`` to
    ``risky_steady_state`` when both ``constants['p_disaster'] > 0`` and
    ``config.use_risky_steady_state`` is True (the default). Setting
    the flag False forces the deterministic SS even under disaster
    risk -- used for ablating the anchor/residual disagreement.

    Lives here, not in trainer.py, so the framework core stays model-
    agnostic. Triggered via ``ModelSpec.setup_fn`` from
    ``calibrate_model``, which training and the checkpoint loader share.
    """
    overrides = getattr(config, "constants", None) or {}
    if not any(name in overrides for name in TARGET_NAMES):
        targets = consistent_targets(model.constants)
        model = model._replace(constants={**model.constants, **targets})
        if getattr(config, "verbose", False):
            print(
                f"  Taylor targets from the steady state: "
                f"R_ss={targets['R_ss']:.7f}, y_ss={targets['y_ss']:.6f}"
            )

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
    policy_upper=POLICY_UPPER,  # inf → softplus bounding; finite (pi) → sigmoid
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
    anchor_gate_fn=anchor_gate,
)
