"""CompositeLossConfig + MomentMatchingConfig."""

from __future__ import annotations

from pydantic import (
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from deqn_jax.config._base import (
    _coerce_float,
    _coerce_int,
    _ConfigBase,
)

# ---------------------------------------------------------------------------
# CompositeLossConfig
# ---------------------------------------------------------------------------


class CompositeLossConfig(_ConfigBase):
    """Composite loss configuration (anchor + Jacobian + barrier + Newton terms).

    Anchor and Jacobian losses decay with shock_scale during curriculum
    (most useful near SS, fade as stochastic domain expands).
    Barrier and Newton losses don't decay (always useful for feasibility).
    """

    model_config = ConfigDict(extra="forbid")

    anchor_weight: float = Field(
        default=0.1,
        description="Weight on the anchor loss (||π_net(x) - π_lin(x)||² at sampled anchor points near SS).",
    )
    jac_weight: float = Field(
        default=0.01,
        description="Weight on the Jacobian-match loss (||J_net(SS) - P||² at the steady state).",
    )
    jac_anchor_weight: float = Field(
        default=0.0,
        description="Weight on the per-anchor Jacobian match (||J_net(x_i) - P||² averaged over anchors). 0 = off. ~d× more expensive than `jac_weight`.",
    )
    barrier_weight: float = Field(
        default=0.01,
        description="Weight on economic feasibility barriers (net worth, leverage, consumption positivity).",
    )
    newton_weight: float = Field(
        default=0.01,
        description="Weight on Newton-step auxiliary losses (condition number, residual) for kink-approximation stabilization.",
    )
    n_anchor_points: int = Field(
        default=64,
        description="Number of anchor points sampled near SS at setup time (deterministic).",
    )
    anchor_gate: bool = Field(
        default=False,
        description="Kink-aware anchor: down-weight anchor points where the model declares its linearization invalid (requires the model to define `anchor_gate_fn`, e.g. disaster's interest-rate floor). Off = legacy unweighted anchor, bit-identical.",
    )
    drift_weight: float = Field(
        default=0.0,
        description="Certificate-in-the-loop stability loss: penalize average per-period log growth of the deterministic closed loop from small SS perturbations (≈ log rho(SS)). 0 = off, bit-identical. Does not decay with curriculum.",
    )
    drift_horizon: int = Field(
        default=20,
        description="Closed-loop rollout length T for the drift loss (gradient flows through all T steps).",
    )
    drift_eps: float = Field(
        default=1.0e-3,
        description="Scale of the ergodic-shaped SS perturbations used as drift probes.",
    )
    drift_n_probes: int = Field(
        default=4,
        description="Number of fixed probe directions (drawn once at build time, a priori).",
    )
    drift_target: float = Field(
        default=0.99,
        description="Growth-factor target: the hinge fires when the per-period growth exceeds log(drift_target). 0.99 asks for mild contraction.",
    )
    res_sobolev_weight: float = Field(
        default=0.0,
        description="Residual-Sobolev loss: penalize directional derivatives of the per-state EXPECTED residual toward zero (the true policy zeroes E[r] on a neighborhood; impostors keep values small with finite gradients). Adds selection information the value loss cannot see. 0 = off, bit-identical. Quadrature expectations + single-stage models only.",
    )
    res_sobolev_n_states: int = Field(
        default=16,
        description="Batch subsample size for the residual-Sobolev term (cost control).",
    )
    res_sobolev_n_dirs: int = Field(
        default=2,
        description="Number of fixed ergodic-shaped unit directions for the JVPs (drawn once at build time).",
    )
    anchor_sigma: float = Field(
        default=1.0,
        description="Scale of the Gaussian spread around SS for anchor-point sampling.",
    )
    leverage_mult: float = Field(
        default=5.0,
        description="Leverage barrier fires when `L > leverage_mult * L_ss`. Higher = more permissive.",
    )
    aux_decay_floor: float = Field(
        default=0.2,
        description="Minimum retained weight of anchor+jac auxiliaries as curriculum progresses. Set to 1.0 to keep aux terms fully active throughout.",
    )

    @field_validator(
        "anchor_weight",
        "jac_weight",
        "jac_anchor_weight",
        "barrier_weight",
        "newton_weight",
        "anchor_sigma",
        "leverage_mult",
        "aux_decay_floor",
        "drift_weight",
        "drift_eps",
        "drift_target",
        "res_sobolev_weight",
        mode="before",
    )
    @classmethod
    def _coerce_float_reject_bool(cls, v, info):
        return _coerce_float(v, f"composite_loss.{info.field_name}")

    @field_validator(
        "n_anchor_points", "drift_horizon", "drift_n_probes", mode="before"
    )
    @classmethod
    def _coerce_int_reject_bool(cls, v, info):
        return _coerce_int(v, f"composite_loss.{info.field_name}")

    @model_validator(mode="after")
    def _validate_ranges(self):
        for name in (
            "anchor_weight",
            "jac_weight",
            "jac_anchor_weight",
            "barrier_weight",
            "newton_weight",
        ):
            val = getattr(self, name)
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val}")
        if self.n_anchor_points <= 0:
            raise ValueError(f"n_anchor_points must be > 0, got {self.n_anchor_points}")
        if self.anchor_sigma <= 0:
            raise ValueError(f"anchor_sigma must be > 0, got {self.anchor_sigma}")
        if self.leverage_mult <= 0:
            raise ValueError(f"leverage_mult must be > 0, got {self.leverage_mult}")
        if not (0 <= self.aux_decay_floor <= 1):
            raise ValueError(
                f"aux_decay_floor must be in [0, 1], got {self.aux_decay_floor}"
            )
        if self.res_sobolev_weight < 0:
            raise ValueError(
                f"res_sobolev_weight must be >= 0, got {self.res_sobolev_weight}"
            )
        if self.res_sobolev_weight > 0:
            if self.res_sobolev_n_states <= 0:
                raise ValueError(
                    f"res_sobolev_n_states must be > 0, got {self.res_sobolev_n_states}"
                )
            if self.res_sobolev_n_dirs <= 0:
                raise ValueError(
                    f"res_sobolev_n_dirs must be > 0, got {self.res_sobolev_n_dirs}"
                )
        if self.drift_weight < 0:
            raise ValueError(f"drift_weight must be >= 0, got {self.drift_weight}")
        if self.drift_weight > 0:
            if self.drift_horizon <= 0:
                raise ValueError(f"drift_horizon must be > 0, got {self.drift_horizon}")
            if self.drift_eps <= 0:
                raise ValueError(f"drift_eps must be > 0, got {self.drift_eps}")
            if self.drift_n_probes <= 0:
                raise ValueError(
                    f"drift_n_probes must be > 0, got {self.drift_n_probes}"
                )
            if not (0 < self.drift_target <= 1.5):
                raise ValueError(
                    f"drift_target must be in (0, 1.5], got {self.drift_target}"
                )
        return self


# ---------------------------------------------------------------------------
# MomentMatchingConfig
# ---------------------------------------------------------------------------


class MomentMatchingConfig(_ConfigBase):
    """Aux loss that penalizes ergodic-moment deviation from a Dynare reference.

    Composes with any base loss (residual MSE, composite, etc). Uses
    per-minibatch policy-output moments as the estimator; the gradient
    flows through ``policy(s)`` only, with states ``stop_gradient``-ed
    (they came from a separate rollout). See
    ``training/moment_loss.py`` for the design rationale.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Master switch. When False, training behaviour is identical to the base loss.",
    )
    weight: float = Field(
        default=0.1,
        description="Multiplier on the aux loss term added to the total loss.",
    )
    mean_weight: float = Field(
        default=1.0,
        description="Within the aux, weight on the squared mean-deviation term.",
    )
    std_weight: float = Field(
        default=1.0,
        description="Within the aux, weight on the squared std-deviation term.",
    )
    dynare_dir: str = Field(
        default="dynare/results",
        description="Directory containing dynare_moments.csv (the target moments).",
    )
    scale_eps: float = Field(
        default=1.0e-3,
        description="Floor on the per-variable scale used for relative comparison; prevents division blowup for variables with near-zero target.",
    )

    @field_validator("weight", "mean_weight", "std_weight", "scale_eps", mode="before")
    @classmethod
    def _coerce_float_reject_bool(cls, v, info):
        return _coerce_float(v, f"moment_matching.{info.field_name}")

    @model_validator(mode="after")
    def _validate_ranges(self):
        if self.weight < 0:
            raise ValueError(f"weight must be >= 0, got {self.weight}")
        if self.mean_weight < 0:
            raise ValueError(f"mean_weight must be >= 0, got {self.mean_weight}")
        if self.std_weight < 0:
            raise ValueError(f"std_weight must be >= 0, got {self.std_weight}")
        if self.scale_eps <= 0:
            raise ValueError(f"scale_eps must be > 0, got {self.scale_eps}")
        return self
