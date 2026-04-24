"""Steady state and init-state sampler for 2-country IRBC.

Simon calibrates the TFP scale ``A_tfp`` so that at z=0, k=1 the
marginal product equals 1/beta (the capital Euler with zero adjustment
cost, zero irreversibility slack). Under symmetric calibration that
fixes K_ss=1 exactly.

Given K_ss=1 and MPK_ss=1/beta, the SS shadow price lambda_ss can be
chosen from either country's Pareto FOC ``u'(c_j) = lambda / tau_j``.
Under symmetric calibration (both Pareto weights 0.5) we'd have
c_0 = c_1 and lambda_ss chosen so that c_j = y_j - delta * k_j at SS.

Here the calibration uses HETEROGENEOUS risk aversion
(gamma_0=0.25, gamma_1=1.0) with symmetric Pareto weights, so
consumption is actually pinned by solving a joint system. We handle
this by noting: at SS, summing the FOC c_j = (lambda/tau_j)^(-1/gamma_j)
and the aggregate resource constraint sum(c_j) = sum(y_j - delta k_j)
gives a 1-D equation in lambda that we solve by damped iteration.

mu_ss: irreversibility is slack at SS (i_ss = delta > 0), so
mu_ss = 0 exactly.
"""

from typing import Dict, Tuple

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.irbc.variables import N_COUNTRIES
from deqn_jax.models.variable_spec import make_init_state_fn

# Init rect bounds: jitter k around 1 and z around 0. Narrow range
# keeps early training in the economically sensible region (k << 1
# produces explosive MPK; z outside ~+-0.1 makes ergodic states rare).
K_LB = 0.9
K_UB = 1.1
Z_LB = -0.05
Z_UB = 0.05


def _solve_lambda_ss(constants: Dict) -> float:
    """Find lambda_ss so that aggregate consumption = aggregate output - aggregate depreciation.

    Using the Pareto-weighted FOC c_j = (lambda / tau_j)^{-1/gamma_j},
    sum_j c_j(lambda) = sum_j (A_tfp * k^zeta - delta * k) at k=1,
    which at SS (k_j=1) simplifies to:
        sum_j (lambda / tau_j)^{-1/gamma_j} = N * (A_tfp - delta).

    Damped fixed-point iteration on lambda.
    """
    A_tfp = constants["A_tfp"]
    delta = constants["delta"]
    gamma = [constants["gamma_0"], constants["gamma_1"]]
    tau = [constants["tau_0"], constants["tau_1"]]

    target = N_COUNTRIES * (A_tfp - delta)

    def agg_c(lam):
        return sum((lam / tau[j]) ** (-1.0 / gamma[j]) for j in range(N_COUNTRIES))

    lam = 1.0
    for _ in range(500):
        cur = agg_c(lam)
        # Scale lam to close the gap; damped update on log-lambda.
        # agg_c is strictly decreasing in lambda (marginal utility falls
        # as lambda rises), so the update is stable.
        ratio = cur / target
        lam_new = lam * (ratio ** 0.5)          # damped scaling
        if abs(lam_new - lam) < 1e-14:
            lam = lam_new
            break
        lam = lam_new
    return float(lam)


def steady_state(constants: Dict) -> Tuple[Array, Array]:
    """Zero-shock symmetric SS: k_j=1, z_j=0, mu_j=0.

    k_ss=1 is built into the A_tfp calibration (MPK=1/beta at k=1).
    Returns (ss_state [4], ss_policy [5]).
    """
    lam_ss = _solve_lambda_ss(constants)

    ss_state = jnp.array([1.0, 1.0, 0.0, 0.0])                 # k_0, k_1, z_0, z_1
    ss_policy = jnp.array([1.0, 1.0, lam_ss, 0.0, 0.0])        # k_0', k_1', lam, mu_0, mu_1
    return ss_state, ss_policy


INIT_SPECS = {
    "k_0": {"distribution": "uniform", "kwargs": {"minval": K_LB, "maxval": K_UB}},
    "k_1": {"distribution": "uniform", "kwargs": {"minval": K_LB, "maxval": K_UB}},
    "z_0": {"distribution": "uniform", "kwargs": {"minval": Z_LB, "maxval": Z_UB}},
    "z_1": {"distribution": "uniform", "kwargs": {"minval": Z_LB, "maxval": Z_UB}},
}

init_state = make_init_state_fn(("k_0", "k_1", "z_0", "z_1"), INIT_SPECS)
