"""Moment-matching auxiliary loss: penalize ergodic-moment deviation from a target.

Why this exists: residual minimization on its own can land in a *wrong-
attractor* manifold even when the structural gauge freedom is removed
(see ``networks/kf_anchored_mlp.py``). The K/F-anchor fix closes the
mean gap to ~18%, but the std gap stays high (~80%) because the network
can satisfy residuals while keeping its policies near-constant — a
state-blind solution. Direct fix: penalize ``(net_std − target_std)²``
during training. The target is Dynare's ergodic moment vector, the same
reference we already use for the eval comparison.

Interaction with other auxiliaries:
  - The per-equation ``eq_losses`` dict tracks the moment-matching term
    under the key ``aux_moment_match``. The ``aux_`` prefix means
    reweighting (``lr_annealing`` / ``relobralo``) and gradient-surgery
    paths exclude it — same convention as the composite-loss aux terms.
  - Layered cleanly with composite loss: the wrapper takes any base
    loss callable and adds the aux penalty on top.

Estimator: per-minibatch policy-output moments. The 64-state batch is
a small sample, so the per-step estimate is noisy, but Adam-family
optimizers average it out over training. Crucially, the gradient
flows through ``policy(s)`` only — the states themselves are
``stop_gradient``'d (they came from a separate rollout). This means
we're matching "policy-output distribution conditional on the visited
states" rather than the true ergodic moments. As the policy improves
and the visited-state distribution approaches the true ergodic, the
two coincide; in early training the bias toward the target moments
shapes which states the policy steers toward.

API: ``make_moment_matching_wrapper(base_loss_fn, target_moments,
weight, ...)`` returns a callable with the same signature as
``training.loss.compute_loss``.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import Array


def _resolve_target_indices(
    policy_names: Sequence[str],
    target_moments: Dict[str, Dict[str, float]],
    name_aliases: Optional[Dict[str, str]] = None,
) -> Dict[int, Tuple[float, float]]:
    """Map DEQN policy index → ``(dyn_mean, dyn_std)`` for variables that exist
    in the Dynare reference.

    ``name_aliases`` maps DEQN policy name → Dynare variable name (identity
    when absent). For the disaster model this typically holds ``{"i": "i_var"}``.
    Policies absent from ``target_moments`` are silently skipped — only the
    overlap contributes to the aux loss.
    """
    aliases = name_aliases or {}
    out: Dict[int, Tuple[float, float]] = {}
    for i, pname in enumerate(policy_names):
        dvar = aliases.get(pname, pname)
        if dvar in target_moments:
            entry = target_moments[dvar]
            out[i] = (float(entry["mean"]), float(entry["std"]))
    return out


def _moment_matching_aux_loss(
    states: Array,
    policy_fn: Callable[[Array], Array],
    target_idx_to_moments: Dict[int, Tuple[float, float]],
    mean_weight: float,
    std_weight: float,
    scale_eps: float,
) -> Array:
    """Per-minibatch moment-matching penalty as a single scalar.

    Computes the mean and std of ``policy_fn(s)`` for each ``s`` in the
    batch (using ``vmap``), per overlapping policy variable. Returns the
    sum over variables of relative squared error in mean + std.
    """
    if states.ndim == 3:
        # Sequence input — drop the history dimension and use only the
        # current-period state for moment computation.
        flat_states = states[:, -1, :]
    else:
        flat_states = states

    policies = jax.vmap(policy_fn)(flat_states)  # [B, n_policies]

    total = jnp.array(0.0, dtype=policies.dtype)
    for idx, (dm, ds) in target_idx_to_moments.items():
        col = policies[:, idx]
        net_mean = jnp.mean(col)
        net_std = jnp.std(col)
        scale_m = max(abs(dm), scale_eps)
        scale_s = max(ds, scale_eps)
        total = total + mean_weight * ((net_mean - dm) / scale_m) ** 2
        total = total + std_weight * ((net_std - ds) / scale_s) ** 2
    return total


def make_moment_matching_wrapper(
    base_loss_fn: Optional[Callable],
    target_idx_to_moments: Dict[int, Tuple[float, float]],
    weight: float = 0.1,
    mean_weight: float = 1.0,
    std_weight: float = 1.0,
    scale_eps: float = 1.0e-3,
) -> Callable:
    """Wrap a base loss callable to add a moment-matching aux term.

    The returned function has the same signature as ``compute_loss``
    (model, params, states, key, ...) -> (total_loss, eq_losses) and
    layers an additive penalty on top of ``base_loss_fn`` (or the
    default MSE residual loss when ``base_loss_fn`` is None).

    The aux loss is exposed in ``eq_losses`` under the key
    ``aux_moment_match`` (the ``aux_`` prefix excludes it from
    reweighting / gradient-surgery paths).
    """
    if base_loss_fn is None:
        from deqn_jax.training.loss import compute_loss

        base_loss_fn = compute_loss

    if not target_idx_to_moments:
        # Nothing to match — return the base unchanged. Safe default for
        # configs where the model's policies don't overlap with the
        # Dynare reference.
        return base_loss_fn

    def wrapped(model, params, states, key, *args, **kwargs):
        base_total, eq_losses = base_loss_fn(
            model, params, states, key, *args, **kwargs
        )
        aux = _moment_matching_aux_loss(
            states,
            params,
            target_idx_to_moments,
            mean_weight=mean_weight,
            std_weight=std_weight,
            scale_eps=scale_eps,
        )
        scaled = jnp.asarray(weight, dtype=aux.dtype) * aux
        new_total = base_total + scaled
        new_eq_losses = dict(eq_losses)
        new_eq_losses["aux_moment_match"] = scaled
        return new_total, new_eq_losses

    return wrapped


__all__ = [
    "make_moment_matching_wrapper",
    "_resolve_target_indices",
    "_moment_matching_aux_loss",
]
