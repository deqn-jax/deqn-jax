"""Disaster-model composite-loss auxiliary terms.

Wired into the model via ``ModelSpec.composite_aux_fn`` and
``ModelSpec.composite_aux_constants_fn``. Reads disaster-specific
definitions (``n``, ``L``, ``c``, ``newton_h_prime``,
``newton_residual``) from the per-batch ``defs`` dict and produces:

- **Economic-feasibility barriers** — net worth, leverage, consumption
  log-barriers / penalties that prevent the optimizer from driving the
  model into pathological regions (negative net worth, divergent
  leverage, collapsed consumption).
- **Newton-solver diagnostics** — penalize regions where the Newton
  solver for ``omega_bar`` is ill-conditioned (``h'(omega)`` near zero)
  or has high residual.

Lives here, not in ``training/composite_loss.py``, so the framework
core stays model-agnostic. Other models can ship their own hook to add
their own auxiliary terms, or leave ``composite_aux_fn=None`` and get
just the generic anchor / Jacobian / Sobolev-anchor terms.
"""

from typing import Any, Dict, Tuple

import jax.numpy as jnp
from jax import Array

from deqn_jax.types import ModelSpec


def composite_aux_constants(model: ModelSpec) -> Dict[str, Any]:
    """Precompute disaster-specific constants for the aux hook.

    Called once at trainer setup, outside jit. The returned dict lands
    in ``CompositeData.aux_constants``; ``composite_aux`` reads it on
    every loss evaluation.

    Currently emits ``ss_leverage`` (used as the per-batch leverage
    barrier threshold via ``leverage_mult * ss_leverage``).
    """
    assert model.steady_state_fn is not None, (
        "disaster composite_aux_constants requires steady_state_fn"
    )
    assert model.definitions_fn is not None, (
        "disaster composite_aux_constants requires definitions_fn"
    )
    ss_state, ss_policy = model.steady_state_fn(model.constants)
    ss_defs = model.definitions_fn(ss_state, ss_policy, model.constants)
    return {"ss_leverage": float(ss_defs["L"])}


def composite_aux(
    model: ModelSpec,
    defs: Dict[str, Array],
    data: Any,
    weights: Dict[str, float],
) -> Tuple[Dict[str, Array], Array]:
    """Disaster aux losses: BGG barriers + Newton diagnostics.

    Args:
        model: ModelSpec (unused here but part of the hook signature).
        defs: per-batch definitions from ``model.definitions_fn``, vmapped
            over the training batch. Reads ``n``, ``L``, ``c``,
            ``newton_h_prime``, ``newton_residual``.
        data: ``CompositeData``. Reads ``aux_constants["ss_leverage"]``.
        weights: weights dict from the composite-loss config.
            ``barrier_weight`` scales all three barrier terms;
            ``leverage_mult`` is the leverage barrier threshold;
            ``newton_weight`` scales the Newton-conditioning losses.

    Returns:
        ``(entries, total_loss)`` where ``entries`` is the dict of named
        ``aux_*`` losses and ``total_loss`` is the weighted sum to add to
        the composite loss.
    """
    barrier_weight = float(weights.get("barrier_weight", 0.0))
    leverage_mult = float(weights.get("leverage_mult", 5.0))
    newton_weight = float(weights.get("newton_weight", 0.0))
    ss_leverage = float(data.aux_constants["ss_leverage"])

    # ---- Economic-feasibility barriers ----

    # Net worth barrier: max(0, -log(n))^2 — penalizes n < 1 (approaching zero)
    n = defs["n"]
    aux_barrier_n = jnp.mean(jnp.maximum(0.0, -jnp.log(jnp.maximum(n, 1e-8))) ** 2)

    # Leverage penalty: (L - L_threshold)^2 / L_ss^2 when L > leverage_mult * L_ss
    L = defs["L"]
    L_threshold = leverage_mult * ss_leverage
    excess = jnp.maximum(L - L_threshold, 0.0)
    aux_barrier_L = jnp.mean((excess / ss_leverage) ** 2)

    # Consumption barrier: max(0, -log(c))^2 — penalizes c < 1
    c = defs["c"]
    aux_barrier_c = jnp.mean(jnp.maximum(0.0, -jnp.log(jnp.maximum(c, 1e-8))) ** 2)

    # ---- Newton-solver diagnostics ----

    h_prime = defs["newton_h_prime"]
    deficit = jnp.maximum(0.1 - h_prime, 0.0)
    aux_newton_cond = jnp.mean(deficit**2)

    newton_resid = defs["newton_residual"]
    aux_newton_resid = jnp.mean(newton_resid**2)

    entries = {
        "aux_barrier_n": aux_barrier_n,
        "aux_barrier_L": aux_barrier_L,
        "aux_barrier_c": aux_barrier_c,
        "aux_newton_cond": aux_newton_cond,
        "aux_newton_resid": aux_newton_resid,
    }
    total = barrier_weight * (
        aux_barrier_n + aux_barrier_L + aux_barrier_c
    ) + newton_weight * (aux_newton_cond + aux_newton_resid)
    return entries, total
