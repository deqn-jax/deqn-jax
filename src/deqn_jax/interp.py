"""Mechanistic interpretability primitives for DEQN networks.

Five pure functions for inspecting a trained ``LinearPlusMLP``:

1. ``branch_decompose`` — split the policy into Blanchard-Kahn
   linearization and MLP correction.
2. ``forward_with_activations`` — capture per-layer post-activations.
3. ``neuron_contributions`` — per-neuron attribution to downstream units.
4. ``linear_probe`` — regress concept scalars on hidden activations.
5. ``ablate_neuron`` — zero out a chosen post-activation and rerun.

Companion narrated notebook: ``examples/interp_brock_mirman.ipynb``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple  # noqa: F401

import equinox as eqx  # noqa: F401
import jax.numpy as jnp
from jax import Array  # noqa: F401

from deqn_jax.networks.linear_plus_mlp import LinearPlusMLP  # noqa: F401
from deqn_jax.networks.mlp import MLP  # noqa: F401

# ---------------------------------------------------------------------------
# Primitives — populated by subsequent tasks
# ---------------------------------------------------------------------------


def branch_decompose(net: LinearPlusMLP, states: Array) -> Dict[str, Any]:
    """Split a ``LinearPlusMLP``'s policy into BK and MLP components.

    Returns a dict with arrays for ``bk`` (Blanchard-Kahn baseline in
    level space), ``mlp_delta`` (the residual the MLP contributes,
    in level space), ``policy`` (the final clipped output), and a
    boolean ``closes_numerically`` that is true iff no clipping was
    active anywhere on the input batch — equivalently, iff the pre-clip
    output equals the clipped policy to ``atol=1e-6``. When clipping
    fires somewhere, ``bk + mlp_delta`` equals the pre-clip raw output
    (not the clipped policy) at those points.

    For log-link outputs we deliberately compute ``bk`` as
    ``ss_policy * exp(P @ (s - ss_state))`` (the BK *level* prediction),
    and ``mlp_delta`` as ``policy - bk``. This keeps the additive
    decomposition meaningful in plot units even when the underlying
    forward composes multiplicatively.

    Args:
        net: A trained ``LinearPlusMLP``.
        states: Array of shape ``[batch, n_states]``.

    Returns:
        Dict with keys ``"bk"``, ``"mlp_delta"``, ``"policy"``,
        ``"closes_numerically"``.
    """
    if states.ndim != 2:
        raise ValueError(
            f"branch_decompose expects states of shape [batch, n_states], "
            f"got shape {states.shape}"
        )

    ss_state = net.ss_state
    ss_policy = net.ss_policy
    P = net.P

    # bk_corr is in the *natural* link space per row (level for linear-link
    # rows, log for log-link rows), since P was pre-converted in the factory.
    bk_corr = (states - ss_state[None, :]) @ P.T  # [batch, n_policies]

    is_log = jnp.asarray(net.output_links, dtype=jnp.int8) == 1  # [n_policies]

    # BK in level space:
    #   linear rows: ss + bk_corr
    #   log rows:    ss * exp(bk_corr)
    bk_linear = ss_policy[None, :] + bk_corr
    bk_log = ss_policy[None, :] * jnp.exp(bk_corr)
    bk = jnp.where(is_log[None, :], bk_log, bk_linear)

    # MLP delta — same raw output that _forward_single feeds into the
    # additive (linear-link) or multiplicative (log-link) composition below.
    # MLP.__call__ vmaps internally for 2-D input, so no outer vmap needed.
    delta = net.mlp(states)  # [batch, n_policies]

    # Raw (pre-clip) output in level space — mirrors _forward_single logic.
    raw_linear = ss_policy[None, :] + bk_corr + delta
    raw_log = ss_policy[None, :] * jnp.exp(bk_corr + delta)
    raw = jnp.where(is_log[None, :], raw_log, raw_linear)

    # Final policy via the model's actual forward (applies clipping).
    policy = net(states)

    # mlp_delta is the level-space residual contributed by the MLP:
    # bk + mlp_delta == raw (before clipping) by construction.
    mlp_delta = raw - bk

    # closes_numerically: true iff no clipping was active anywhere,
    # i.e. the pre-clip raw output equals the clipped policy.
    closes = jnp.allclose(raw, policy, atol=1e-6)

    return {
        "bk": bk,
        "mlp_delta": mlp_delta,
        "policy": policy,
        "closes_numerically": closes,
    }
