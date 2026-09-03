"""State transitions for Brock-Mirman model.

``make_step`` builds the transition from a model's ``SPEC`` and its
``definitions()``; the accumulation law itself is the same for the
Brock-Mirman family. Two models build their own step from it:
``brock_mirman`` (below) and ``bm_labor``. ``bm_labor_constrained`` and
``bm_labor_autodiff`` do NOT call the factory — they import ``bm_labor``'s
already-built ``step``, so they transition under ``bm_labor``'s SPEC and
``definitions()``; only their equations differ.
"""

from typing import Callable, Dict

import jax.numpy as jnp
from jax import Array

from deqn_jax.models.brock_mirman.equations import definitions
from deqn_jax.models.brock_mirman.variables import SPEC


def make_step(spec, definitions_fn: Callable) -> Callable:
    """Build a Brock-Mirman ``step`` bound to one model's SPEC/definitions.

    Capital: k' = (1 - delta) * k + s   (``s`` from ``definitions_fn``;
    the labor variants return savings already scaled by output)
    TFP:     z' = rho_z * z + sigma_z * eps
    """

    def step(
        state: Array,
        policy: Array,
        shock: Array,
        constants: Dict,
    ) -> Array:
        s = spec.unpack_state(state)
        defs = definitions_fn(state, policy, constants)

        delta = constants["delta"]
        rho_z = constants["rho_z"]
        sigma_z = constants["sigma_z"]

        # Capital accumulation
        k_next = (1 - delta) * s.k + defs["s"]

        # TFP shock
        eps = shock[:, 0] if shock.ndim > 1 else shock
        z_next = rho_z * s.z + sigma_z * eps

        return jnp.stack([k_next, z_next], axis=1)

    return step


step = make_step(SPEC, definitions)
