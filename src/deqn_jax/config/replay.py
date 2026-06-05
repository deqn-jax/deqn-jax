"""ReplayBufferConfig: prioritized state-replay buffer settings."""

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
# ReplayBufferConfig
# ---------------------------------------------------------------------------


class ReplayBufferConfig(_ConfigBase):
    """Prioritized state-replay buffer configuration.

    Off by default. When enabled, each cycle's just-rolled-out trajectory
    states are written to a fixed-shape ring buffer with per-state priorities
    (= sum-of-squared equilibrium residuals at write time). Each gradient
    minibatch then mixes ``mix_ratio`` fraction of priority-weighted
    buffered samples in with the current trajectory.

    Anti-forgetting (states from older policies stay in the gradient signal)
    + spectral-bias mitigation (high-residual states get oversampled).

    Sequence networks (``network.history_len > 1``) are not supported in v1
    and raise ``NotImplementedError`` if enabled together.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Master switch. When False, the cycle path is byte-identical to no-replay training.",
    )
    capacity: int = Field(
        default=65536,
        description="Number of past states retained in the ring buffer. Memory: capacity × n_states × 4B.",
    )
    mix_ratio: float = Field(
        default=0.5,
        description="Fraction of each minibatch dataset drawn from the buffer (0=none, 1=all-buffer). 0.5 is the natural default.",
    )
    min_fill_frac: float = Field(
        default=0.25,
        description="Buffer must reach this fraction of capacity before sampling activates. Until then, training uses current trajectory only.",
    )
    priority_alpha: float = Field(
        default=0.6,
        description="PER's α: sampling probability ∝ (priority + eps)^α. α=0 is uniform, α=1 is fully proportional. 0.6 is the original PER default.",
    )
    priority_eps: float = Field(
        default=1.0e-6,
        description="Floor added to priorities before exponentiation. Prevents zero-priority states from being completely starved.",
    )
    eviction: str = Field(
        default="fifo",
        description="Eviction policy. v1 only supports `fifo` (ring overwrite). Reservoir sampling is a v2 follow-up.",
    )

    @field_validator(
        "mix_ratio",
        "min_fill_frac",
        "priority_alpha",
        "priority_eps",
        mode="before",
    )
    @classmethod
    def _coerce_float_reject_bool(cls, v, info):
        return _coerce_float(v, f"replay_buffer.{info.field_name}")

    @field_validator("capacity", mode="before")
    @classmethod
    def _coerce_int_reject_bool(cls, v, info):
        return _coerce_int(v, f"replay_buffer.{info.field_name}")

    @model_validator(mode="after")
    def _validate_ranges(self):
        if self.capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {self.capacity}")
        if not (0 <= self.mix_ratio <= 1):
            raise ValueError(f"mix_ratio must be in [0, 1], got {self.mix_ratio}")
        if not (0 <= self.min_fill_frac <= 1):
            raise ValueError(
                f"min_fill_frac must be in [0, 1], got {self.min_fill_frac}"
            )
        if self.priority_alpha < 0:
            raise ValueError(f"priority_alpha must be >= 0, got {self.priority_alpha}")
        if self.priority_eps <= 0:
            raise ValueError(f"priority_eps must be > 0, got {self.priority_eps}")
        if self.eviction not in {"fifo"}:
            raise ValueError(
                f"eviction must be 'fifo' (v1 only), got {self.eviction!r}"
            )
        return self
