"""Equilibrium conditions for the 6-generation life-cycle OLG (Geneva Day 2 Ex 4).

Each cohort ``h in {0,...,H-2}`` chooses a saving rate ``s^h`` out of
cash-at-hand; the last cohort consumes everything (``s^{H-1} ≡ 0``). The
optimality condition is an intertemporal Euler that holds with equality only
when the borrowing constraint ``k'^{h+1} >= 0`` is slack:

    1/c^h  >=  beta * E_t[ (1-delta+r') / c'^{h+1} ],   k'^{h+1} >= 0,  comp. slack.

Because the borrowing constraint complements an EXPECTATION, the residual wraps
``E[.]`` in a Fischer-Burmeister nonlinearity, and ``E[fb(.)] != fb(E[.])``. So
this model uses the framework's two-stage loss hooks:

  inside_fn  -> the H shock-dependent terms ``(1-delta+r')/c'_j``, averaged to E[.]
  combine_fn -> ``fb( 1/(c^h * beta * E[inside^{h+1}]) - 1,  sav^h / c^h )``, h=0..H-2

The ratio form of the stationarity arg (``a = 1/(c·β·E[·]) - 1``) and the
normalized slack (``b = sav^h / c^h``) match the notebook
(``04_DEQN_Exercises_Solutions.ipynb``, Exercise 4) exactly — note ``sav^h`` is
``k'^{h+1}``, the capital that cohort h carries into next period.
"""

from typing import Dict

import jax.numpy as jnp
from jax import Array

from deqn_jax.models._complementarity import fischer_burmeister
from deqn_jax.models.olg_lifecycle.variables import H

EQUATION_NAMES = tuple(f"euler_{h}" for h in range(H - 1))


def _cohort_block(
    Z: Array, k: Array, saving_rates: Array, constants: Dict
) -> Dict[str, Array]:
    """Within-period firm + household block.

    Args:
        Z: TFP level ``[b, 1]``.
        k: capital by cohort ``[b, H]``.
        saving_rates: network saving rates for cohorts ``0..H-2`` ``[b, H-1]``.

    Returns dict with aggregate prices ``r, w`` (``[b, 1]``) and per-cohort
    ``cah, sav, c`` (``[b, H]``). The last cohort's saving rate is pinned to 0.
    """
    alpha = constants["alpha"]
    delta = constants["delta"]
    l_cycle = jnp.asarray(constants["l_cycle"])  # [H]
    L = jnp.sum(l_cycle)

    K = jnp.sum(k, axis=1, keepdims=True)  # [b,1]
    Y = Z * K**alpha * L ** (1.0 - alpha)  # [b,1]
    r = alpha * (Y / K)  # [b,1]
    w = (1.0 - alpha) * (Y / L)  # [b,1]

    cah = l_cycle[None, :] * w + k * (1.0 - delta + r)  # [b,H]
    s_full = jnp.concatenate(
        [saving_rates, jnp.zeros((saving_rates.shape[0], 1))], axis=1
    )  # [b,H]; last cohort saves 0 (eats everything)
    sav = cah * s_full  # [b,H] = k'^{h+1}
    c = cah - sav  # [b,H]
    return {"r": r, "w": w, "cah": cah, "sav": sav, "c": c}


def inside_fn(
    state: Array,
    policy: Array,
    next_state: Array,
    next_policy: Array,
    constants: Dict,
) -> Dict[str, Array]:
    """Per-shock terms whose expectation enters the Euler: ``(1-delta+r')/c'_j``.

    Evaluated at the next period (next_state, next_policy) and averaged over
    shocks by the loss to form ``E[inside_j]``. Linear in the shock-dependent
    quantities, so the MC average is unbiased — the nonlinearity is deferred to
    ``combine_fn``.
    """
    delta = constants["delta"]
    Z_next = next_state[:, :1]
    k_next = next_state[:, 1 : 1 + H]
    blk = _cohort_block(Z_next, k_next, next_policy, constants)
    inside = (1.0 - delta + blk["r"]) / blk["c"]  # [b,H]
    return {f"inside_{j}": inside[:, j] for j in range(H)}


def combine_fn(
    state: Array,
    policy: Array,
    expectations: Dict[str, Array],
    constants: Dict,
) -> Dict[str, Array]:
    """Apply Fischer-Burmeister to ``E[inside]`` (the nonlinearity, after E[.]).

    For each cohort ``h = 0..H-2``:
        a = 1/(c^h * beta * E[inside^{h+1}]) - 1     (relative Euler error)
        b = sav^h / c^h                              (normalized borrowing slack)
        residual = fb(a, b)
    """
    beta = constants["beta"]
    Z = state[:, :1]
    k = state[:, 1 : 1 + H]
    blk = _cohort_block(Z, k, policy, constants)
    c = blk["c"]
    sav = blk["sav"]
    out: Dict[str, Array] = {}
    for h in range(H - 1):
        e_next = expectations[f"inside_{h + 1}"]  # E[(1-δ+r')/c'^{h+1}]
        a = 1.0 / (c[:, h] * beta * e_next) - 1.0
        b = sav[:, h] / c[:, h]
        out[f"euler_{h}"] = fischer_burmeister(a, b)
    return out


def equations(
    state: Array,
    policy: Array,
    next_state: Array,
    next_policy: Array,
    constants: Dict,
) -> Dict[str, Array]:
    """Single-shock residual for eval / diagnostics / the contract test.

    Training uses the MC-correct two-stage path (``inside_fn`` + ``combine_fn``).
    This composition reuses the same pieces with a single ``next_state``
    realization standing in for the expectation: it yields the correct
    ``equation_names`` and a usable (single-draw, hence biased) residual for
    ``evaluate.py`` / ``irf.py``. Keys equal ``EQUATION_NAMES``.
    """
    inside = inside_fn(state, policy, next_state, next_policy, constants)
    return combine_fn(state, policy, inside, constants)


def definitions(state: Array, policy: Array, constants: Dict) -> Dict[str, Array]:
    """Monitoring diagnostics: aggregate capital, prices, and the
    cross-sectional minimum consumption (a corner-solution canary).

    Handles both a batched state ``[b, n_states]`` (loss / definition-bounds
    path) and a single ``[n_states]`` state — the logging path ``jax.vmap``\\s
    this over the trajectory, so each call sees a 1-D state. The ``ndim`` check
    resolves at trace time (it is static under vmap)."""
    single = state.ndim == 1
    s2 = state[None, :] if single else state
    p2 = policy[None, :] if single else policy
    k = s2[:, 1 : 1 + H]
    blk = _cohort_block(s2[:, :1], k, p2, constants)
    out = {
        "K_agg": jnp.sum(k, axis=1),
        "r": blk["r"][:, 0],
        "w": blk["w"][:, 0],
        "c_min": jnp.min(blk["c"], axis=1),
    }
    return {key: v[0] for key, v in out.items()} if single else out
