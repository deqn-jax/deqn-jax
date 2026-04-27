"""Active-subspace analysis of a trained policy network.

Why this exists: trained DEQN policies on disaster live in a 13-dim state
space. We can't plot a 13-d function. But policies *typically* depend on
only a few directions in state space — a couple of inflation-block
combinations, a couple of capital-block combinations. The active subspace
of an output ``π_i(s)`` is the subspace spanned by the leading
eigenvectors of the gradient covariance

    C_i = E_s[ ∇π_i(s) · ∇π_i(s)^T ]

where the expectation is over the ergodic distribution. Eigenvalues
measure variation; trailing eigenvectors are directions ``π_i`` is
essentially constant in.

What this gives us:
    1. Effective dimensionality per output, via cumulative-variance
       threshold (e.g. "smallest k such that top-k eigenvalues account
       for 95% of trace") and participation ratio
       ``(Σ λ)² / Σ λ²``.
    2. The dominant directions in state space — concrete vectors we can
       project onto for 2-d "shape" plots, recovering the visual
       diagnostic that low-d HJB-PINN methods take for granted.
    3. A degeneracy detector: a state-blind policy has *all* eigenvalues
       small; a healthy policy has a clear separation between leading
       and trailing eigenvalues.

Estimator: sample-mean over states drawn from the simulated ergodic
trajectory. For each state we compute ``∇π_i(s)`` via ``jax.jacrev``,
then form the rank-1 outer product and average. With the
bound-static-fields fix the gradient is now everywhere-finite, so we
don't need extra masking in well-trained networks; we still scrub
non-finite samples defensively.

References:
    - Constantine, "Active Subspaces" (2015) — original treatment for
      uncertainty quantification on scalar functions.
    - The participation-ratio summary is the standard "effective rank"
      from random-matrix theory; gives a single number where the
      cumulative-variance threshold gives a knee point.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import Array

# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------


def estimate_gradient_covariance(
    policy_fn: Callable[[Array], Array],
    states: Array,
    output_idx: int,
) -> Tuple[Array, int]:
    """Sample-mean gradient covariance ``C_i`` for one policy output.

    Args:
        policy_fn: ``[n_states] → [n_policies]``. Pass the trained Equinox
            module directly (it's callable).
        states: ``[N, n_states]`` ergodic-trajectory samples. The longer
            the trajectory the smoother the estimate; for n_states=13 a
            few thousand samples is plenty.
        output_idx: Which policy dimension to analyse (0..n_policies-1).

    Returns:
        ``(C, n_kept)`` where ``C`` is ``[n_states, n_states]`` and
        ``n_kept`` is the number of finite gradient samples that went
        into the average. Non-finite per-state gradients are scrubbed
        before averaging — surfaces a count so callers can flag a run
        whose policy has lots of bad-gradient regions (probably stuck
        on a saturating bound somewhere).
    """

    def grad_at(s: Array) -> Array:
        # jacrev returns [n_policies, n_states]; pull the row we want.
        J = jax.jacrev(policy_fn)(s)
        if J.ndim == 3:  # sequence input -> not handled here
            J = J[0]
        return J[output_idx]

    grads = jax.vmap(grad_at)(states)  # [N, n_states]
    finite_mask = jnp.all(jnp.isfinite(grads), axis=1)  # [N]
    n_kept = int(jnp.sum(finite_mask))

    # Zero out non-finite rows so they contribute nothing to the average,
    # then renormalise by the number kept (not N).
    safe_grads = jnp.where(finite_mask[:, None], grads, jnp.zeros_like(grads))
    C = (safe_grads.T @ safe_grads) / jnp.maximum(n_kept, 1)
    return C, n_kept


def estimate_active_subspace(
    policy_fn: Callable[[Array], Array],
    states: Array,
    output_idx: int,
) -> Dict[str, Any]:
    """Eigendecomposition of the gradient covariance for one output.

    Returns a dict with:
        ``eigenvalues``: ``[n_states]``, sorted descending.
        ``eigenvectors``: ``[n_states, n_states]``, columns are the
            corresponding eigenvectors (column ``k`` matches
            ``eigenvalues[k]``).
        ``cumulative_variance_ratio``: ``[n_states]`` cumulative sum of
            ``eigenvalues / sum(eigenvalues)``. Last entry is 1.0.
        ``participation_ratio``: scalar, ``(Σ λ)² / Σ λ²``. Equal to
            the rank for a rank-deficient matrix; equal to ``n_states``
            when all eigenvalues are equal. The "soft" effective dim.
        ``trace``: ``Σ eigenvalues``. Total variation in the policy
            output explained by the gradient (in squared units).
        ``n_finite_samples``: how many states contributed (others had
            non-finite gradient).
    """
    C, n_kept = estimate_gradient_covariance(policy_fn, states, output_idx)

    # ``eigh`` is stable for symmetric PSD; returns ascending order.
    evals_asc, evecs_asc = jnp.linalg.eigh(C)
    # Reverse to descending for the conventional "leading direction" order.
    eigenvalues = evals_asc[::-1]
    eigenvectors = evecs_asc[:, ::-1]

    # Clamp tiny-negative numerical artifacts to 0 (eigh on a sample
    # covariance can produce -1e-15 entries).
    eigenvalues = jnp.maximum(eigenvalues, 0.0)

    total = jnp.sum(eigenvalues)
    cumulative = jnp.cumsum(eigenvalues) / jnp.maximum(total, 1e-30)

    sum_sq = jnp.sum(eigenvalues**2)
    participation = (total**2) / jnp.maximum(sum_sq, 1e-30)

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "cumulative_variance_ratio": cumulative,
        "participation_ratio": float(participation),
        "trace": float(total),
        "n_finite_samples": n_kept,
    }


# ---------------------------------------------------------------------------
# Effective dimensionality summaries
# ---------------------------------------------------------------------------


def effective_dimensionality(eigenvalues: Array, threshold: float = 0.95) -> int:
    """Smallest ``k`` such that the top-k eigenvalues explain ``threshold``
    of the total variation.

    For a constant-policy degenerate output the trace is ~0 and we
    report 0. For a uniformly-spread policy the result approaches
    ``n_states``.
    """
    eigenvalues = jnp.asarray(eigenvalues)
    total = float(jnp.sum(eigenvalues))
    if total <= 1e-30:
        return 0
    cumulative = jnp.cumsum(eigenvalues) / total
    # Find first index where cumulative crosses threshold.
    above = cumulative >= threshold
    # If somehow nothing crosses (numerical), return full dimensionality.
    if not bool(jnp.any(above)):
        return int(eigenvalues.shape[0])
    return int(jnp.argmax(above)) + 1


def summarize_subspace_per_policy(
    policy_fn: Callable[[Array], Array],
    states: Array,
    policy_names: Sequence[str],
    threshold: float = 0.95,
) -> Dict[str, Dict[str, Any]]:
    """Run subspace analysis for every output policy.

    Returns a dict keyed by policy name, each containing the per-policy
    eigenvalue spectrum + effective-dim numbers. Useful as a one-shot
    diagnostic on a trained checkpoint.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for i, name in enumerate(policy_names):
        sub = estimate_active_subspace(policy_fn, states, i)
        sub["effective_dim"] = effective_dimensionality(
            sub["eigenvalues"], threshold=threshold
        )
        out[name] = sub
    return out


# ---------------------------------------------------------------------------
# State-projection helpers (for downstream visualisation)
# ---------------------------------------------------------------------------


def project_states(
    states: Array,
    directions: Array,
    center: Optional[Array] = None,
) -> Array:
    """Project states onto a low-dim subspace.

    Args:
        states: ``[N, n_states]``.
        directions: ``[n_states, k]`` orthonormal columns. Typically the
            top-k eigenvectors from ``estimate_active_subspace``.
        center: ``[n_states]`` to subtract before projection (e.g.
            steady state). Default zero.

    Returns:
        ``[N, k]`` coordinates in the subspace.
    """
    if center is not None:
        states = states - center
    return states @ directions


def policy_grid_on_subspace(
    policy_fn: Callable[[Array], Array],
    center: Array,
    direction_1: Array,
    direction_2: Optional[Array],
    output_idx: int,
    grid_range: Tuple[float, float] = (-2.0, 2.0),
    n_pts: int = 50,
) -> Dict[str, Any]:
    """Evaluate ``policy_fn(center + z1·v1 + z2·v2)[output_idx]`` on a grid.

    Sets up a ``n_pts × n_pts`` grid in subspace coordinates, builds the
    full-state inputs, and evaluates the policy. Returns the grid and
    the resulting policy values, ready for matplotlib.

    If ``direction_2`` is None, falls back to a 1-d sweep along
    ``direction_1`` and returns ``z`` and ``values`` of length ``n_pts``.
    """
    z1 = jnp.linspace(grid_range[0], grid_range[1], n_pts)
    if direction_2 is None:
        states = center[None, :] + z1[:, None] * direction_1[None, :]
        values = jax.vmap(lambda s: policy_fn(s)[output_idx])(states)
        return {"z1": z1, "values": values, "shape": "1d"}

    z2 = jnp.linspace(grid_range[0], grid_range[1], n_pts)
    Z1, Z2 = jnp.meshgrid(z1, z2, indexing="xy")
    flat = (
        center[None, :]
        + Z1.reshape(-1, 1) * direction_1[None, :]
        + Z2.reshape(-1, 1) * direction_2[None, :]
    )
    flat_values = jax.vmap(lambda s: policy_fn(s)[output_idx])(flat)
    values = flat_values.reshape(n_pts, n_pts)
    return {"z1": z1, "z2": z2, "values": values, "shape": "2d"}


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------


def print_subspace_summary(
    summary: Dict[str, Dict[str, Any]],
    label: str = "",
) -> None:
    """Tabulate per-policy effective dim + participation ratio."""
    if label:
        print(f"\n{'=' * 76}\nActive-subspace summary — {label}\n{'=' * 76}")
    print(
        f"  {'policy':>10s}  {'eff_dim_95':>10s}  {'part_ratio':>10s}  "
        f"{'trace':>11s}  {'top_eig':>10s}  {'eig_ratio':>10s}  {'n_finite':>9s}"
    )
    for name, sub in summary.items():
        evals = sub["eigenvalues"]
        top = float(evals[0]) if evals.shape[0] > 0 else 0.0
        second = float(evals[1]) if evals.shape[0] > 1 else 0.0
        ratio = top / max(second, 1e-30)
        print(
            f"  {name:>10s}  {sub['effective_dim']:>10d}  "
            f"{sub['participation_ratio']:>10.3f}  {sub['trace']:>11.3e}  "
            f"{top:>10.3e}  {ratio:>10.2f}  {sub['n_finite_samples']:>9d}"
        )


__all__ = [
    "estimate_gradient_covariance",
    "estimate_active_subspace",
    "effective_dimensionality",
    "summarize_subspace_per_policy",
    "project_states",
    "policy_grid_on_subspace",
    "print_subspace_summary",
]
