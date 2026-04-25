"""Disaster-model composite-loss auxiliary terms.

Wired into the model via ``ModelSpec.composite_aux_fn``. Reads
disaster-specific definitions (``newton_h_prime``, ``newton_residual``)
that come from the model's piecewise-smooth omega_bar Newton solver,
and turns them into ``aux_*`` loss entries for the composite loss.

Lives here, not in ``training/composite_loss.py``, so the framework
core stays model-agnostic. Other models can ship their own hook to add
their own auxiliary terms, or leave ``composite_aux_fn=None`` and get
just the generic anchor/jac/barrier terms.
"""

from typing import Any, Dict, Tuple

import jax.numpy as jnp
from jax import Array

from deqn_jax.types import ModelSpec


def composite_aux(
    model: ModelSpec,
    defs: Dict[str, Array],
    data: Any,
    weights: Dict[str, float],
) -> Tuple[Dict[str, Array], Array]:
    """Newton solver diagnostic losses for the disaster model.

    Penalises regions where the Newton solver for ``omega_bar`` is
    ill-conditioned (``h'(omega)`` near zero) or has high residual.
    """
    h_prime = defs["newton_h_prime"]
    deficit = jnp.maximum(0.1 - h_prime, 0.0)
    newton_cond = jnp.mean(deficit**2)

    newton_resid = defs["newton_residual"]
    newton_resid_loss = jnp.mean(newton_resid**2)

    entries = {
        "aux_newton_cond": newton_cond,
        "aux_newton_resid": newton_resid_loss,
    }
    newton_weight = weights.get("newton_weight", 0.0)
    total = newton_weight * (newton_cond + newton_resid_loss)
    return entries, total
