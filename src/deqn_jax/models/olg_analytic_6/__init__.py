"""6-agent OLG with analytic closed-form solution (Krueger & Kubler 2004).

Seven-dimensional state (k^2..k^6, eta, delta), five-dimensional policy
(savings for agents 1..5), five Euler equations, two-dimensional shock.
Log utility, i.i.d. product-of-binary shocks on (TFP, depreciation).

The analytic policy k'^h = beta_h * inc^h serves as an oracle; the
trained DEQN should recover it exactly on the full training support.
"""

from deqn_jax.models.olg_analytic_6.dynamics import step
from deqn_jax.models.olg_analytic_6.equations import (
    EQUATION_NAMES,
    definitions,
    equations,
)
from deqn_jax.models.olg_analytic_6.steady_state import (
    analytic_beta_h,
    analytic_policy,
    init_state,
    steady_state,
)
from deqn_jax.models.olg_analytic_6.variables import (
    CONSTANTS,
    N_SHOCKS,
    POLICY_LOWER,
    POLICY_UPPER,
    SPEC,
    A,
)
from deqn_jax.types import ModelSpec

__all__ = [
    "MODEL",
    "CONSTANTS",
    "EQUATION_NAMES",
    "N_SHOCKS",
    "POLICY_LOWER",
    "POLICY_UPPER",
    "SPEC",
    "A",
    "analytic_beta_h",
    "analytic_policy",
    "definitions",
    "equations",
    "init_state",
    "steady_state",
    "step",
]

# Soft feasibility penalties: c^h >= 0 and K >= 0 at every training state.
# Without these, a softplus-output network near initialization can produce
# savings > income for some agents (negative consumption), which makes u'(c)
# explode and the Euler residual unusable. Matches the reference notebook's
# opt_punish_cons / opt_punish_K pattern exactly.
_C_BOUNDS = {f"c{h + 1}": {"lower": 0.0, "penalty_lower": 1.0e4} for h in range(A)}
_K_BOUND = {"K": {"lower": 1e-4, "penalty_lower": 1.0e4}}

MODEL = ModelSpec(
    name="olg_analytic_6",
    n_states=SPEC.n_states,
    n_policies=SPEC.n_policies,
    n_shocks=N_SHOCKS,
    state_names=SPEC.state_names,
    policy_names=SPEC.policy_names,
    equation_names=EQUATION_NAMES,
    shock_names=("eps_eta", "eps_delta"),
    constants=CONSTANTS,
    equations_fn=equations,
    step_fn=step,
    steady_state_fn=steady_state,
    init_state_fn=init_state,
    definitions_fn=definitions,
    policy_lower=POLICY_LOWER,
    policy_upper=POLICY_UPPER,
    definition_bounds={**_C_BOUNDS, **_K_BOUND},
)
