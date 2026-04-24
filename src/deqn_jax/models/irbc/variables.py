"""Variables for the 2-country International Real Business Cycle model.

State: per-country capital and per-country log TFP. Two countries, so
4 states total.

Policies: per-country next-period capital + a shared aggregate-resource
shadow price + per-country irreversibility Lagrange multipliers.
5 policy outputs total. All strictly positive via softplus.

Shocks: 3 Gaussian drives -- one per country plus one aggregate that
hits both countries symmetrically.

Calibration matches Simon Scheidegger's Day 3 notebook 01
(International Real Business Cycles). Risk aversion is
heterogeneous across countries (the source of the Pareto-weighted
consumption-sharing pattern): gamma_0=0.25, gamma_1=1.0.

N=2 hardcoded. The framework's equations_fn / step_fn / definitions_fn
generalize to N>2 with a straightforward loop over the per-country
arrays, but the first port is the reference 2-country case.
"""

import jax.numpy as jnp

from deqn_jax.models.variable_spec import VariableSpec

N_COUNTRIES = 2


SPEC = VariableSpec(
    state_names=("k_0", "k_1", "z_0", "z_1"),
    policy_names=("k_0_next", "k_1_next", "lam", "mu_0", "mu_1"),
)

CONSTANTS = {
    # Preferences / technology (symmetric).
    "beta": 0.99,
    "delta": 0.01,       # intentionally low; Simon's IRBC uses 0.01 for simplicity
    "zeta": 0.36,        # capital share in Cobb-Douglas
    "kappa": 0.5,        # quadratic capital-adjustment-cost coefficient
    "rho_z": 0.95,       # TFP autocorrelation
    "sigma_eps": 0.01,   # std of both country-specific and aggregate innovations
    "A_tfp": 0.055836,   # TFP scale, calibrated so MPK_ss = 1/beta
    # Heterogeneous risk aversion. Notebook uses a linear spread [0.25, 1.0]
    # across N=2 countries; exposed here as separate keys for Pydantic
    # compatibility (constants are Dict[str, float]).
    "gamma_0": 0.25,
    "gamma_1": 1.0,
    # Pareto weights (normalized to sum to 1 across countries).
    "tau_0": 0.5,
    "tau_1": 0.5,
    # Fischer-Burmeister regularization; 0 is the pure FB function, a small
    # eps (~1e-8) keeps the sqrt smooth at the origin without distorting
    # the zero of the function away from the true complementary state.
    "fb_eps": 1.0e-8,
}

# All policy outputs strictly positive via softplus (upper=inf).
POLICY_LOWER = jnp.full(5, 1e-6)
POLICY_UPPER = jnp.full(5, jnp.inf)

N_SHOCKS = 3     # eps_0, eps_1, eps_agg

DESCRIPTION = "2-country International RBC with irreversibility (Fischer-Burmeister)"
