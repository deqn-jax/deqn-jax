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
import jax
import jax.numpy as jnp
from jax import Array  # noqa: F401

from deqn_jax.networks.linear_plus_mlp import LinearPlusMLP  # noqa: F401
from deqn_jax.networks.mlp import MLP  # noqa: F401

# ---------------------------------------------------------------------------
# Primitives — populated by subsequent tasks
# ---------------------------------------------------------------------------


def forward_with_activations(mlp: MLP, states: Array) -> Dict[str, Array]:
    """Run ``mlp`` and capture every post-activation along the way.

    Mirrors ``MLP._forward_single`` but yields each hidden layer's
    post-activation. Output keys are ``"h{i}"`` for hidden layer ``i``
    (post-activation) and ``"out"`` for the pre-bounds final output.

    Args:
        mlp: The MLP module (e.g. ``linear_plus_mlp_net.mlp``).
        states: Array of shape ``[batch, n_states]``.

    Returns:
        Dict mapping layer name to activation array. Each hidden layer
        contributes ``"h{i}"``; the final pre-bounds output is ``"out"``.
    """

    def _single(state: Array) -> Dict[str, Array]:
        from deqn_jax.networks.mlp import _normalize_input  # local: tiny helper

        x = _normalize_input(state, mlp.input_shift, mlp.input_scale)

        captures: Dict[str, Array] = {}
        for i, layer in enumerate(mlp.layers[:-1]):
            x = mlp.activations[i](layer(x))
            captures[f"h{i}"] = x

        out = mlp.layers[-1](x)
        captures["out"] = out  # pre-bounds output
        return captures

    return jax.vmap(_single)(states)


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


def neuron_contributions(mlp: MLP, states: Array) -> Dict[int, Array]:
    """Per-neuron contribution to the next layer's pre-activation.

    For hidden layer ``ℓ`` with post-activation ``h_ℓ`` of shape
    ``[batch, H_ℓ]``, and the next layer's weight ``W_{ℓ+1}`` of shape
    ``[H_{ℓ+1}, H_ℓ]``, the per-neuron contribution to downstream unit
    ``j`` from hidden unit ``i`` is ``W_{ℓ+1}[j, i] * h_ℓ[batch, i]``.

    Returns a dict keyed by hidden-layer index ``ℓ ∈ {0, …, L-1}`` (where
    ``L`` is the number of hidden layers); each value has shape
    ``[batch, H_ℓ, H_{ℓ+1}]`` (or ``[batch, H_ℓ, n_outputs]`` for the last
    hidden layer).

    The downstream layer's bias is *not* included; callers can add
    ``mlp.layers[ℓ+1].bias`` to reconstruct the full pre-activation.

    Args:
        mlp: The MLP module.
        states: Array of shape ``[batch, n_states]``.

    Returns:
        Dict mapping ``layer_idx -> Array[batch, H_layer, H_downstream]``.
    """
    acts = forward_with_activations(mlp, states)
    out: Dict[int, Array] = {}
    n_hidden = len(mlp.layers) - 1
    for layer_idx in range(n_hidden):
        h = acts[f"h{layer_idx}"]  # [batch, H_layer]
        w = mlp.layers[layer_idx + 1].weight  # [H_downstream, H_layer]
        # Per (b, i, j): w[j, i] * h[b, i]
        # Broadcast: h[:, :, None] is [batch, H_layer, 1]
        #            w.T[None, :, :] is [1, H_layer, H_downstream]
        out[layer_idx] = h[:, :, None] * w.T[None, :, :]
    return out


def linear_probe(activations: Array, concepts: Array) -> Dict[str, Array]:
    """Per-(neuron, concept) univariate linear regression.

    For each pair ``(i, j)`` of neuron ``i`` and concept ``j``, fits
    ``activations[:, i] ≈ a · concepts[:, j] + b`` and returns the slope
    ``a``, the coefficient of determination ``R²``, and the residual
    variance.

    R² uses ``1 - SS_res / SS_tot`` with sample-variance denominators.
    When a neuron's activation is constant (``SS_tot == 0``) R² is
    defined as 0 rather than NaN.

    No regularization, no joint regression across concepts. Callers
    should pre-scale concepts if they want comparable coefficients.

    Args:
        activations: Array of shape ``[batch, n_neurons]``.
        concepts: Array of shape ``[batch, n_concepts]``.

    Returns:
        Dict with:
          - ``"coef"``: ``Array[n_neurons, n_concepts]`` — slope per pair.
          - ``"r2"``: ``Array[n_neurons, n_concepts]`` — coefficient of
            determination per pair.
          - ``"residual_var"``: ``Array[n_neurons, n_concepts]`` —
            variance of the residual per pair.
    """
    if activations.ndim != 2 or concepts.ndim != 2:
        raise ValueError(
            f"activations and concepts must both be 2-D; got "
            f"activations.shape={activations.shape}, concepts.shape={concepts.shape}"
        )
    if activations.shape[0] != concepts.shape[0]:
        raise ValueError(
            f"batch size mismatch: activations has {activations.shape[0]} rows, "
            f"concepts has {concepts.shape[0]}"
        )

    n = activations.shape[0]
    a_mean = activations.mean(axis=0, keepdims=True)  # [1, n_neurons]
    c_mean = concepts.mean(axis=0, keepdims=True)  # [1, n_concepts]
    a_c = activations - a_mean
    c_c = concepts - c_mean

    cov = (a_c.T @ c_c) / n  # [n_neurons, n_concepts]
    c_var = (c_c**2).mean(axis=0)  # [n_concepts]
    a_var = (a_c**2).mean(axis=0)  # [n_neurons]

    eps = 1e-12
    coef = cov / (c_var[None, :] + eps)

    # SS_res / n = a_var - cov² / c_var (closed-form residual variance)
    residual_var = a_var[:, None] - cov**2 / (c_var[None, :] + eps)

    # R² = 1 - residual_var / a_var; defined as 0 when a_var == 0.
    r2_raw = 1.0 - residual_var / (a_var[:, None] + eps)
    r2 = jnp.where(a_var[:, None] > eps, r2_raw, jnp.zeros_like(r2_raw))

    return {
        "coef": coef,
        "r2": r2,
        "residual_var": residual_var,
    }


def ablate_neuron(
    net: LinearPlusMLP,
    layer_idx: int,
    neuron_idx: int,
    states: Array,
) -> Array:
    """Run the network with a chosen post-activation forced to zero.

    Mirrors ``LinearPlusMLP._forward_single`` and ``MLP._forward_single``
    exactly, except that after computing the post-activation of hidden
    layer ``layer_idx``, we zero entry ``neuron_idx`` before passing on.

    Args:
        net: The ``LinearPlusMLP`` to inspect (unchanged).
        layer_idx: Which hidden layer to ablate in (0-indexed).
        neuron_idx: Which neuron within that layer to zero.
        states: Array of shape ``[batch, n_states]``.

    Returns:
        The policy with the chosen post-activation forced to zero, shape
        ``[batch, n_policies]``. Full clipping + link-type semantics from
        ``LinearPlusMLP._forward_single`` are preserved.
    """
    from deqn_jax.networks.mlp import _normalize_input

    mlp = net.mlp
    n_hidden = len(mlp.layers) - 1
    if not 0 <= layer_idx < n_hidden:
        raise ValueError(
            f"layer_idx {layer_idx} out of range for {n_hidden} hidden layer(s)"
        )
    hidden_size_at_layer = mlp.layers[layer_idx].weight.shape[0]
    if not 0 <= neuron_idx < hidden_size_at_layer:
        raise ValueError(
            f"neuron_idx {neuron_idx} out of range for layer {layer_idx} "
            f"with size {hidden_size_at_layer}"
        )

    def _single(state: Array) -> Array:
        x = _normalize_input(state, mlp.input_shift, mlp.input_scale)
        for i, layer in enumerate(mlp.layers[:-1]):
            x = mlp.activations[i](layer(x))
            if i == layer_idx:
                x = x.at[neuron_idx].set(0.0)
        delta = mlp.layers[-1](x)

        ss_state = jax.lax.stop_gradient(net.ss_state)
        ss_policy = jax.lax.stop_gradient(net.ss_policy)
        P = jax.lax.stop_gradient(net.P)
        bk_corr = P @ (state - ss_state)

        if all(code == 0 for code in net.output_links):
            raw = ss_policy + bk_corr + delta
        elif all(code == 1 for code in net.output_links):
            raw = ss_policy * jnp.exp(bk_corr + delta)
        else:
            is_log = jnp.asarray(net.output_links, dtype=jnp.int8) == 1
            raw_linear = ss_policy + bk_corr + delta
            raw_log = ss_policy * jnp.exp(bk_corr + delta)
            raw = jnp.where(is_log, raw_log, raw_linear)

        if net.policy_lower is not None:
            lower = jax.lax.stop_gradient(jnp.asarray(net.policy_lower))
            raw = jnp.maximum(raw, lower)
        if net.policy_upper is not None:
            upper = jax.lax.stop_gradient(jnp.asarray(net.policy_upper))
            safe_upper = jnp.where(jnp.isinf(upper), jnp.array(1e10), upper)
            raw = jnp.minimum(raw, safe_upper)
        return raw

    return jax.vmap(_single)(states)
