"""CDICE business-as-usual: DICE-2016 economy with the re-calibrated CDICE climate.

Port of the business-as-usual ("gdice_baseline") model of

    Folini, D., Friedl, A., Kübler, F. and Scheidegger, S. (2024).
    "The Climate in Climate Economics". Review of Economic Studies.
    doi:10.1093/restud/rdae011 (Section 6 and Online Appendix D).

with the multi-model-mean ("mmm_mmm") calibration of the carbon cycle and the
temperature module, annual time step. Replication package:
https://github.com/ClimateChangeEcon/Climate_in_Climate_Economics
(``DEQN_for_IAMs/gdice_baseline``); the economics here is re-implemented from
the paper and that code, not copied.

Non-stationary: time is a state (``tau``, a compressed clock in [0, 1)),
and TFP, population, carbon intensity, land emissions and exogenous forcing
are deterministic functions of it (``exogenous``). No shocks, no steady
state; training runs simulated 500-year paths from the 2015 state
(``initial_state``). Seven states (capital per effective worker, three
carbon reservoirs, two temperatures, time), seven policies (next capital
and six normalized shadow prices), seven equilibrium conditions
(``equations``). ``definitions`` reports the paths, consumption, savings,
emissions and the social cost of carbon. ``scripts/cert/cdice_replication.py``
compares a trained checkpoint with the stored reference solution.
"""

from deqn_jax.models.cdice_bau.dynamics import step
from deqn_jax.models.cdice_bau.equations import (
    EQUATION_NAMES,
    definitions,
    equations,
)
from deqn_jax.models.cdice_bau.initial_state import init_state
from deqn_jax.models.cdice_bau.variables import (
    CONSTANTS,
    N_SHOCKS,
    POLICY_LOWER,
    POLICY_UPPER,
    SPEC,
)
from deqn_jax.types import ModelSpec

MODEL = ModelSpec(
    name="cdice_bau",
    n_states=SPEC.n_states,
    n_policies=SPEC.n_policies,
    n_shocks=N_SHOCKS,
    state_names=SPEC.state_names,
    policy_names=SPEC.policy_names,
    equation_names=EQUATION_NAMES,
    constants=CONSTANTS,
    equations_fn=equations,
    step_fn=step,
    steady_state_fn=None,
    init_state_fn=init_state,
    definitions_fn=definitions,
    policy_lower=POLICY_LOWER,
    policy_upper=POLICY_UPPER,
)
