"""SurrogateConfig: the EWM continuation surrogate ("world arm").

Off by default. When enabled, a small network Ŵ_ψ(x) stands in for the
two-stage expectation E[inside_fn] inside the POLICY update; Ŵ is fitted
each episode to exact expectations computed on sparse anchor states under a
Polyak-averaged target policy. Equilibrium is still defined by the exact
residual (which is what certification evaluates).

Reference: Scheidegger & Schaab (2026), "Equilibrium World Models",
arXiv:2606.23463 — the coverage + surrogate arm. Design spec:
docs/superpowers/specs/2026-08-28-ewm-world-arm-design.md.
"""

from __future__ import annotations

from typing import List, Optional, Union

from pydantic import ConfigDict, Field, field_validator, model_validator

from deqn_jax.config._base import _coerce_float, _coerce_int, _ConfigBase


class SurrogateConfig(_ConfigBase):
    """EWM world-arm (continuation surrogate) configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Master switch. When False, training is byte-identical to no surrogate.",
    )
    width: int = Field(
        default=64,
        description="Hidden width of Ŵ (two tanh hidden layers).",
    )
    anchor_frac: Union[float, List[float]] = Field(
        default=[0.1, 0.2, 0.4],
        description="Fraction of the (path ∪ coverage) batch used as exact anchors per episode. A list is staged over equal thirds of training (paper: 0.1 → 0.2 → 0.4); a scalar is constant. At least one minibatch is always used.",
    )
    polyak_tau: float = Field(
        default=0.97,
        description="Polyak coefficient of the target policy the exact anchor targets are computed at: θ̄ ← τ θ̄ + (1−τ) θ. Below ~0.95 the target chases the policy and Ŵ diverges.",
    )
    epochs_w: int = Field(
        default=12,
        description="Adam steps on Ŵ per episode (fitting the anchor targets).",
    )
    lr_w: Optional[float] = Field(
        default=None,
        description="Learning rate for Ŵ. None = the policy optimizer's learning rate.",
    )
    exact_in_coverage: bool = Field(
        default=True,
        description="Score the coverage (stress/local) pools with the EXACT expectation even in the surrogate arm; only the base (path) pool uses Ŵ. The reference reports surrogate-scored coverage diverging.",
    )
    positive_outputs: bool = Field(
        default=True,
        description="Softplus output head so Ŵ > 0 (the inside terms of the olg_lifecycle family are positive). Set False for models whose inside terms change sign.",
    )
    allow_without_coverage: bool = Field(
        default=False,
        description="Permit surrogate.enabled without coverage.enabled (the paper's ablation). Off by default so the combination is a named choice.",
    )

    @field_validator("polyak_tau", mode="before")
    @classmethod
    def _coerce_tau(cls, v, info):
        return _coerce_float(v, f"surrogate.{info.field_name}")

    @field_validator("width", "epochs_w", mode="before")
    @classmethod
    def _coerce_ints(cls, v, info):
        return _coerce_int(v, f"surrogate.{info.field_name}")

    @field_validator("anchor_frac", mode="before")
    @classmethod
    def _coerce_anchor(cls, v):
        if isinstance(v, (list, tuple)):
            return [float(x) for x in v]
        return float(v)

    def anchor_frac_at(self, progress: float) -> float:
        """Anchor fraction at training progress in [0, 1] (staged if a list)."""
        if isinstance(self.anchor_frac, list):
            n = len(self.anchor_frac)
            i = min(n - 1, int(progress * n))
            return self.anchor_frac[i]
        return float(self.anchor_frac)

    @model_validator(mode="after")
    def _validate(self):
        if self.width <= 0:
            raise ValueError("surrogate.width must be > 0")
        if self.epochs_w <= 0:
            raise ValueError("surrogate.epochs_w must be > 0")
        if not (0.0 < self.polyak_tau < 1.0):
            raise ValueError("surrogate.polyak_tau must be in (0, 1)")
        fracs = self.anchor_frac if isinstance(self.anchor_frac, list) else [self.anchor_frac]
        if not fracs or any(not (0.0 < f <= 1.0) for f in fracs):
            raise ValueError("surrogate.anchor_frac entries must be in (0, 1]")
        if self.lr_w is not None and self.lr_w <= 0:
            raise ValueError("surrogate.lr_w must be > 0")
        return self
