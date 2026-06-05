"""OptimizerConfig: optimizer name + hyperparameters + LR schedule."""

from __future__ import annotations

from typing import ClassVar, Optional

from pydantic import (
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from deqn_jax.config._base import (
    _coerce_float,
    _coerce_int,
    _coerce_optional_float,
    _ConfigBase,
)

# ---------------------------------------------------------------------------
# OptimizerConfig
# ---------------------------------------------------------------------------


class OptimizerConfig(_ConfigBase):
    """Optimizer configuration."""

    model_config = ConfigDict(extra="forbid", coerce_numbers_to_str=False)

    name: str = Field(
        default="adam",
        description="Optimizer name. Options: `adam`, `sgd`, `adamw`, `lion`, `muon`, `ngd`, `shampoo`, `lbfgs`, `mao`, `mao_kfac`, `gn`, `ign`, `lm`.",
    )
    learning_rate: float = Field(
        default=1e-3,
        description="Peak learning rate (or constant LR when `lr_schedule='constant'`).",
    )
    grad_clip: Optional[float] = Field(
        default=None, description="Global gradient-norm clipping. None disables."
    )
    weight_decay: float = Field(
        default=0.0, description="L2 weight decay (used by adamw / adam / sgd)."
    )
    beta1: float = Field(default=0.9, description="Adam / MAO first-moment decay.")
    beta2: float = Field(default=0.999, description="Adam / MAO second-moment decay.")
    epsilon: float = Field(default=1e-8, description="Adam / MAO numerical floor.")
    damping: float = Field(
        default=1e-4,
        description="Preconditioner damping for NGD / GN / IGN / LM.",
    )
    decay: float = Field(
        default=0.999, description="NGD / Shampoo preconditioner EMA decay."
    )
    block_size: int = Field(default=64, description="Shampoo Kronecker block size.")
    precond_update_freq: int = Field(
        default=10, description="Shampoo preconditioner update frequency."
    )
    memory_size: int = Field(default=10, description="L-BFGS history size.")
    ns_steps: int = Field(default=5, description="Muon Newton-Schulz iteration count.")
    cg_iters: int = Field(
        default=20,
        description="Implicit Gauss-Newton conjugate-gradient iteration cap.",
    )
    cg_tol: float = Field(
        default=1e-6,
        description="Implicit Gauss-Newton relative conjugate-gradient residual tolerance.",
    )
    lr_schedule: str = Field(
        default="constant",
        description="LR schedule: `constant`, `cosine`, or `reduce_on_plateau`.",
    )
    lr_warmup: int = Field(
        default=0, description="Linear warmup episodes before `lr_schedule` kicks in."
    )
    lr_min_factor: float = Field(
        default=0.0,
        description="Minimum LR as a fraction of peak (cosine / reduce_on_plateau floor).",
    )

    lr_reduce_factor: float = Field(
        default=0.5,
        description="ReduceLROnPlateau: multiply LR by this factor on plateau.",
    )
    lr_reduce_patience: int = Field(
        default=500,
        description="ReduceLROnPlateau: episodes without improvement before decay.",
    )
    lr_reduce_cooldown: int = Field(
        default=100,
        description="ReduceLROnPlateau: episodes to wait after a decay before resuming monitoring.",
    )
    lr_reduce_min_delta: float = Field(
        default=1e-6,
        description="ReduceLROnPlateau: minimum loss drop that counts as improvement.",
    )
    # Lower bound on LR as a fraction of initial. Reusing lr_min_factor
    # (already present for cosine) keeps config surface small.

    VALID_NAMES: ClassVar[frozenset] = frozenset(
        {
            "adam",
            "sgd",
            "adamw",
            "lion",
            "muon",
            "ngd",
            "shampoo",
            "lbfgs",
            "mao",
            "mao_kfac",
            "gn",
            "ign",
            "lm",
        }
    )
    VALID_LR_SCHEDULES: ClassVar[frozenset] = frozenset(
        {"constant", "cosine", "reduce_on_plateau"}
    )

    @field_validator(
        "learning_rate",
        "weight_decay",
        "beta1",
        "beta2",
        "epsilon",
        "damping",
        "decay",
        "cg_tol",
        "lr_min_factor",
        "lr_reduce_factor",
        "lr_reduce_min_delta",
        mode="before",
    )
    @classmethod
    def _coerce_float_reject_bool(cls, v, info):
        return _coerce_float(v, f"optimizer.{info.field_name}")

    @field_validator("grad_clip", mode="before")
    @classmethod
    def _coerce_grad_clip(cls, v, info):
        return _coerce_optional_float(v, f"optimizer.{info.field_name}")

    @field_validator(
        "block_size",
        "precond_update_freq",
        "memory_size",
        "ns_steps",
        "cg_iters",
        "lr_warmup",
        "lr_reduce_patience",
        "lr_reduce_cooldown",
        mode="before",
    )
    @classmethod
    def _coerce_int_reject_bool(cls, v, info):
        return _coerce_int(v, f"optimizer.{info.field_name}")

    @field_validator("name", mode="before")
    @classmethod
    def _check_name_type(cls, v):
        if not isinstance(v, str):
            raise TypeError(
                f"OptimizerConfig.name: expected str, got {type(v).__name__} ({v!r})"
            )
        return v

    @model_validator(mode="after")
    def _validate_ranges(self):
        if self.name not in self.VALID_NAMES:
            raise ValueError(
                f"Unknown optimizer '{self.name}'. Valid: {sorted(self.VALID_NAMES)}"
            )
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError(f"grad_clip must be > 0, got {self.grad_clip}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if not (0 < self.beta1 < 1):
            raise ValueError(f"beta1 must be in (0, 1), got {self.beta1}")
        if not (0 < self.beta2 < 1):
            raise ValueError(f"beta2 must be in (0, 1), got {self.beta2}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if self.name in {"gn", "ign", "lm"}:
            if self.damping < 0:
                raise ValueError(
                    f"damping must be >= 0 for {self.name}, got {self.damping}"
                )
        elif self.damping <= 0:
            raise ValueError(f"damping must be > 0, got {self.damping}")
        if not (0 < self.decay < 1):
            raise ValueError(f"decay must be in (0, 1), got {self.decay}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")
        if self.precond_update_freq <= 0:
            raise ValueError(
                f"precond_update_freq must be > 0, got {self.precond_update_freq}"
            )
        if self.memory_size <= 0:
            raise ValueError(f"memory_size must be > 0, got {self.memory_size}")
        if self.ns_steps <= 0:
            raise ValueError(f"ns_steps must be > 0, got {self.ns_steps}")
        if self.cg_iters <= 0:
            raise ValueError(f"cg_iters must be > 0, got {self.cg_iters}")
        if self.cg_tol <= 0:
            raise ValueError(f"cg_tol must be > 0, got {self.cg_tol}")
        if self.lr_schedule not in self.VALID_LR_SCHEDULES:
            raise ValueError(
                f"Unknown lr_schedule '{self.lr_schedule}'. "
                f"Valid: {sorted(self.VALID_LR_SCHEDULES)}"
            )
        if self.lr_warmup < 0:
            raise ValueError(f"lr_warmup must be >= 0, got {self.lr_warmup}")
        if not (0 <= self.lr_min_factor <= 1):
            raise ValueError(
                f"lr_min_factor must be in [0, 1], got {self.lr_min_factor}"
            )
        return self
