"""CoverageConfig: EWM coverage-sampling settings.

Off by default. When enabled, the residual loss is imposed on a mixture
measure (base + stress + local pools) instead of the base batch alone.
Stress seeds are drawn from ``stress_ranges`` and rolled ``rollout_horizon``
steps through the exact transition; generated states are repaired (clipped)
into ``repair_ranges`` before the residual is evaluated. The learned
continuation surrogate is out of scope in v1 (irbc's expectation is cheap
quadrature).

Reference: Scheidegger & Schaab (2026), "Equilibrium World Models",
arXiv:2606.23463 — the surrogate-free coverage arm.
"""

from __future__ import annotations

from typing import Dict, Literal, Tuple

from pydantic import ConfigDict, Field, field_validator, model_validator

from deqn_jax.config._base import _coerce_float, _coerce_int, _ConfigBase


class CoverageConfig(_ConfigBase):
    """EWM coverage-sampling configuration (v1: coverage-only)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Master switch. When False, training is byte-identical to no coverage.",
    )
    rho_base: float = Field(
        default=1.0,
        description="Mixture weight on the base (init-rect / on-policy) pool.",
    )
    rho_stress: float = Field(
        default=0.5,
        description="Mixture weight on the stress pool (paper's coverage-exact arm: 0.5 relative to path).",
    )
    rho_local: float = Field(
        default=0.25,
        description="Mixture weight on the local-perturbation pool (paper: 0.25). Weights are normalized over the included pools inside the wrapper.",
    )
    n_stress: int = Field(
        default=128,
        description="Number of stress seeds drawn per step (before rollout).",
    )
    n_local: int = Field(
        default=128,
        description="Number of local perturbations per step.",
    )
    rollout_horizon: int = Field(
        default=5,
        description="H: steps to roll stress seeds through the exact transition (paper: typically 3 or 5).",
    )
    local_sigma: float = Field(
        default=0.02,
        description="Std of Gaussian local perturbations, in state units.",
    )
    stress_ranges: Dict[str, Tuple[float, float]] = Field(
        default_factory=dict,
        description="Per-state-name uniform box for stress seeds. Keys are state names (validated against model.state_names at model resolution). Empty is an error when enabled with rho_stress>0.",
    )
    repair_ranges: Dict[str, Tuple[float, float]] = Field(
        default_factory=dict,
        description="Per-state-name feasible box; stress landings and local perturbations are clipped into it before the residual is evaluated (the paper's repair step). Empty = no clipping.",
    )
    stress_seed_mode: Literal["box", "path"] = Field(
        default="box",
        description=(
            "'box' (historical variant): stress seeds are SS-filled states with "
            "the stress dims uniform in stress_ranges; the raw seed is excluded "
            "from the pool. 'path' (the paper's measure): seeds are visited "
            "batch states with ONLY the stress dims overridden — every other "
            "coordinate keeps its realistic joint value — and the seed itself "
            "joins the pool alongside its rollout landings."
        ),
    )

    @field_validator(
        "rho_base", "rho_stress", "rho_local", "local_sigma", mode="before"
    )
    @classmethod
    def _coerce_float_reject_bool(cls, v, info):
        return _coerce_float(v, f"coverage.{info.field_name}")

    @field_validator("n_stress", "n_local", "rollout_horizon", mode="before")
    @classmethod
    def _coerce_int_reject_bool(cls, v, info):
        return _coerce_int(v, f"coverage.{info.field_name}")

    @model_validator(mode="after")
    def _validate(self):
        for name in ("rho_base", "rho_stress", "rho_local"):
            if getattr(self, name) < 0:
                raise ValueError(f"coverage.{name} must be >= 0")
        if self.rho_base + self.rho_stress + self.rho_local <= 0:
            raise ValueError("coverage weights must not all be zero")
        if self.local_sigma < 0:
            raise ValueError("coverage.local_sigma must be >= 0")
        for n in ("n_stress", "n_local", "rollout_horizon"):
            if getattr(self, n) < 0:
                raise ValueError(f"coverage.{n} must be >= 0")
        for field_name in ("stress_ranges", "repair_ranges"):
            for k, rng in getattr(self, field_name).items():
                if rng[0] > rng[1]:
                    raise ValueError(
                        f"coverage.{field_name}[{k!r}] must be [low <= high], got {rng}"
                    )
        if self.enabled:
            if self.rho_stress > 0:
                if self.n_stress <= 0:
                    raise ValueError("coverage.rho_stress>0 requires n_stress>0")
                if self.rollout_horizon < 1:
                    raise ValueError(
                        "coverage.rho_stress>0 requires rollout_horizon>=1"
                    )
                if not self.stress_ranges:
                    raise ValueError(
                        "coverage.rho_stress>0 requires non-empty stress_ranges"
                    )
            if self.rho_local > 0 and self.n_local <= 0:
                raise ValueError("coverage.rho_local>0 requires n_local>0")
        return self
