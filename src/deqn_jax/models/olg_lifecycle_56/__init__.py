"""56-generation life-cycle OLG with borrowing constraints (annual frequency).

The dimensionality-scale companion of ``olg_lifecycle``: same economics, same
two-stage Fischer-Burmeister loss, H = 56 cohorts -> 57-dim state, 55-dim
policy, 55 Euler complementarities. All computational machinery is imported
from ``olg_lifecycle`` (its functions derive H from ``len(l_cycle)``); this
package only supplies the H=56 layout, the annual calibration, and the
interpolated age-efficiency profile (see ``variables.py``).

No closed-form steady state (``steady_state_fn=None``): trained over the
ergodic distribution from a random init, certified by held-out residuals,
long-horizon simulation, and cross-sectional market-clearing checks — never by
the training loss alone.
"""

from deqn_jax.models.olg_lifecycle.dynamics import step
from deqn_jax.models.olg_lifecycle.equations import (
    combine_fn,
    definitions,
    equations,
    inside_fn,
)
from deqn_jax.models.olg_lifecycle.steady_state import init_state
from deqn_jax.models.olg_lifecycle_56.variables import (
    CONSTANTS,
    N_SHOCKS,
    POLICY_LOWER,
    POLICY_UPPER,
    SPEC,
    H,
)
from deqn_jax.types import ModelSpec

EQUATION_NAMES = tuple(f"euler_{h}" for h in range(H - 1))

MODEL = ModelSpec(
    name="olg_lifecycle_56",
    n_states=SPEC.n_states,
    n_policies=SPEC.n_policies,
    n_shocks=N_SHOCKS,
    state_names=SPEC.state_names,
    policy_names=SPEC.policy_names,
    equation_names=EQUATION_NAMES,
    shock_names=("eps_z",),
    constants=CONSTANTS,
    equations_fn=equations,
    step_fn=step,
    steady_state_fn=None,
    init_state_fn=init_state,
    definitions_fn=definitions,
    inside_fn=inside_fn,
    combine_fn=combine_fn,
    policy_lower=POLICY_LOWER,
    policy_upper=POLICY_UPPER,
)
