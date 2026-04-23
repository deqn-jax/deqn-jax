"""Training-time diagnostic hooks for deterministic Brock-Mirman.

Provides ``make_cycle_hook(figures_dir)`` — a factory that returns a
function matching ``ModelSpec.cycle_hook``'s signature. The returned
hook plots the trained savings-rate policy against the analytic
s* = alpha * beta and writes the result to
``{figures_dir}/policy_ep{episode}.png`` every time the trainer fires
it (controlled by ``config.log_every``).

Pattern: the model owns "what diagnostics are useful for this economic
problem"; rendering primitives live in ``deqn_jax.plots`` (pure
functions, data in / Figure out); the trainer just invokes the hook at
the right moment. This matches the DEQN-MAO upstream per-model
``Hooks.py`` idiom.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from deqn_jax.models.bm_deterministic.variables import K_LB, K_UB


def make_cycle_hook(
    figures_dir: Optional[str] = "figures/bm_deterministic",
    n_grid: int = 200,
) -> Callable:
    """Build a cycle hook closed over its output directory.

    Args:
        figures_dir: where to save plots. Created on first call. If
            ``None``, the hook is a no-op (useful for silent runs).
        n_grid: number of K values on which to evaluate the policy for
            the diagnostic plot.

    Returns:
        ``hook(state, model, episode) -> None``.
    """
    if figures_dir is None:
        def _noop(state, model, episode):
            return None
        return _noop

    out = Path(figures_dir)

    def hook(state, model, episode: int) -> None:
        out.mkdir(parents=True, exist_ok=True)

        alpha = model.constants["alpha"]
        beta = model.constants["beta"]
        s_star = alpha * beta

        k_grid = jnp.linspace(K_LB, K_UB, n_grid)[:, None]
        s_pred = np.asarray(state.params(k_grid))[:, 0]
        k_np = np.asarray(k_grid)[:, 0]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].plot(k_np, s_pred, lw=2, label=r"DEQN policy $\mathcal{N}(K)$")
        axes[0].axhline(s_star, color="k", ls="--", label=fr"$\alpha\beta = {s_star:.4f}$")
        axes[0].set_xlabel("K")
        axes[0].set_ylabel("savings rate $s$")
        axes[0].set_title(f"policy (episode {episode})")
        axes[0].legend(loc="best")

        err = s_pred - s_star
        axes[1].plot(k_np, err, lw=2, color="C3")
        axes[1].axhline(0.0, color="k", ls=":", alpha=0.5)
        axes[1].set_xlabel("K")
        axes[1].set_ylabel(r"$\mathcal{N}(K) - \alpha\beta$")
        axes[1].set_title(f"error vs analytic (max |err| = {np.max(np.abs(err)):.2e})")

        fig.tight_layout()
        fig.savefig(out / f"policy_ep{episode:05d}.png", dpi=100)
        plt.close(fig)

    return hook
