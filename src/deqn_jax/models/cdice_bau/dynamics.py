"""State transition of the CDICE business-as-usual model.

Capital follows the policy; the climate block follows its own laws of motion
(``climate.transition``) driven by this year's industrial emissions
sigma_t A_t L_t k^alpha (zero abatement) plus land-use emissions; the clock
advances one year. Deterministic: the ``shock`` argument has width 0.
"""

from typing import Dict

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.cdice_bau import climate, exogenous
from deqn_jax.models.cdice_bau.variables import SPEC


def step(state: Array, policy: Array, shock: Array, constants: Dict) -> Array:
    c = constants
    s = SPEC.unpack_state(state)
    p = SPEC.unpack_policy(policy)

    t = exogenous.real_time(s.tau, c)
    emissions = exogenous.sigma(t, c) * exogenous.tfp(t, c) * exogenous.lab(
        t, c
    ) * s.k ** c["alpha"] + exogenous.eland(t, c)
    m_at, m_uo, m_lo, t_at, t_oc = climate.transition(
        s.m_at, s.m_uo, s.m_lo, s.t_at, s.t_oc, emissions, exogenous.fex(t, c), c
    )
    columns = [p.k_next, m_at, m_uo, m_lo, t_at, t_oc, exogenous.next_tau(s.tau, c)]
    return jnp.stack(columns, axis=-1)
