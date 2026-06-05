"""NetworkConfig: policy-network architecture + bounds + reparams."""

from __future__ import annotations

from typing import ClassVar, Literal, Optional, Tuple

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
# NetworkConfig
# ---------------------------------------------------------------------------


class NetworkConfig(_ConfigBase):
    """Neural network configuration."""

    model_config = ConfigDict(extra="forbid")

    VALID_TYPES: ClassVar[frozenset] = frozenset(
        {
            "mlp",
            "lstm",
            "transformer",
            "linear_plus_mlp",
            "kf_anchored_mlp",
            "disaster_policy_net",
        }
    )
    VALID_ACTIVATIONS: ClassVar[frozenset] = frozenset(
        {"tanh", "relu", "gelu", "silu", "softplus"}
    )
    VALID_INITS: ClassVar[frozenset] = frozenset(
        {
            "default",
            "xavier_normal",
            "xavier_uniform",
            "he_normal",
            "he_uniform",
            "lecun_normal",
        }
    )

    type: str = Field(
        default="mlp",
        description="Network architecture: `mlp` (feedforward), `lstm`, `transformer`, `linear_plus_mlp` (generic residual ansatz), `disaster_policy_net` (residual ansatz + disaster-specific shape priors), or `kf_anchored_mlp` (legacy K/F gauge elimination).",
    )
    hidden_sizes: Tuple[int, ...] = Field(
        default=(64, 64),
        description="Hidden layer widths. E.g. `(64, 64)` = two 64-unit hidden layers.",
    )
    activation: str = Field(
        default="tanh",
        description="Per-layer activation: `tanh`, `relu`, `gelu`, `silu`, `softplus`.",
    )
    activations: Optional[Tuple[str, ...]] = Field(
        default=None,
        description="Per-layer activations if different per layer. None = use `activation` uniformly. Length = `len(hidden_sizes)`.",
    )
    init: str = Field(
        default="default",
        description="Weight init scheme: `default` (Equinox default), `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`.",
    )
    multi_head: bool = Field(
        default=False,
        description="If True, use separate output heads per policy dimension (experimental).",
    )
    skip_connections: bool = Field(
        default=False,
        description="If True, add residual connections between matching-width hidden layers.",
    )
    history_len: int = Field(
        default=1,
        description="History window length for sequence policies. 1 = MLP (no history). >1 = LSTM / Transformer.",
    )
    num_heads: int = Field(
        default=4, description="Transformer: attention heads per layer."
    )
    n_layers: int = Field(
        default=2, description="Transformer: number of transformer blocks."
    )
    init_scale: float = Field(
        default=0.0,
        description="`linear_plus_mlp` and `disaster_policy_net`: init scale of the MLP delta's final layer. 0.0 = policy starts exactly at the linear solution.",
    )

    use_zlb_feature: bool = Field(
        default=False,
        description="`disaster_policy_net` only: prepend `(R_lag - R_lb)` as an extra MLP input feature.",
    )

    zlb_feature_kind: Literal["raw", "kink"] = Field(
        default="raw",
        description="`disaster_policy_net` only, when use_zlb_feature=true: 'raw' = signed distance R_lag - R_lb; 'kink' = max(R_lag - R_lb, 0), PINN-style explicit kink at the floor.",
    )

    kf_names: Tuple[str, ...] = Field(
        default=("F_p", "K_p", "F_w", "K_w"),
        description="`kf_anchored_mlp` and `disaster_policy_net`: policy names whose MLP delta is masked to zero (gauge fix). Default targets the four CMR Calvo Phillips-curve auxiliaries.",
    )

    reparam_q_as_m: bool = Field(
        default=False,
        description="`disaster_policy_net` only: treat the network's `q` output as `M = q · 𝓑(x)` where 𝓑(x) = 1 - S(x) - x·S'(x) is the investment-Euler bracket; recover q = M/𝓑(x) post-MLP. Eliminates the eq 7 sign-flip pathology by parameterization. (§3.3 of disaster_equation_shape_priors.md)",
    )

    reparam_pi_as_kp_inner: bool = Field(
        default=False,
        description="`disaster_policy_net` only: treat the network's `pi` output as K_p_inner ∈ (0, 1/(1−ξ_p)); derive π via the inverse Calvo formula post-clip. Encodes the Calvo asymptote in the parameterization so the MLP only learns smooth K_p_inner. (§3.1 of disaster_equation_shape_priors.md)",
    )

    reparam_wtilda_as_kw_inner: bool = Field(
        default=False,
        description="`disaster_policy_net` only: treat the network's `w_tilda` output as K_w_inner ∈ (0, 1/(1−ξ_w)); derive w_tilda via the inverse eq 4a formula post-clip. Wage-side mirror of reparam_pi_as_kp_inner; combine with that flag for symmetric Calvo reparam. (§3.1' of disaster_equation_shape_priors.md)",
    )

    output_links: Optional[Tuple[str, ...]] = Field(
        default=None,
        description="Per-policy output parameterization for residual networks. Each entry must be 'linear' (additive: π_i = ss_i + BK + MLP) or 'log' (multiplicative: π_i = ss_i·exp(BK_log + MLP), bakes in positivity). Length must equal n_policies. None = use the model's default_output_links (or all-linear if model doesn't specify).",
    )

    @field_validator("output_links", mode="before")
    @classmethod
    def _coerce_output_links(cls, v):
        if v is None:
            return None
        if isinstance(v, list):
            v = tuple(v)
        if not isinstance(v, tuple):
            raise TypeError(
                f"NetworkConfig.output_links: expected list/tuple of str, "
                f"got {type(v).__name__} ({v!r})"
            )
        valid = {"linear", "log"}
        for entry in v:
            if entry not in valid:
                raise ValueError(
                    f"NetworkConfig.output_links: each entry must be 'linear' "
                    f"or 'log', got {entry!r}"
                )
        return v

    @field_validator("hidden_sizes", mode="before")
    @classmethod
    def _coerce_hidden_sizes(cls, v):
        if isinstance(v, list):
            return tuple(v)
        if isinstance(v, str) or isinstance(v, int):
            raise TypeError(
                f"NetworkConfig.hidden_sizes: expected Tuple[int, ...], "
                f"got {type(v).__name__} ({v!r})"
            )
        return v

    @field_validator("activations", mode="before")
    @classmethod
    def _coerce_activations(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_validator("kf_names", mode="before")
    @classmethod
    def _coerce_kf_names(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_validator("history_len", "num_heads", "n_layers", mode="before")
    @classmethod
    def _coerce_int_reject_bool(cls, v, info):
        return _coerce_int(v, f"network.{info.field_name}")

    @field_validator("init_scale", mode="before")
    @classmethod
    def _coerce_init_scale(cls, v, info):
        return _coerce_float(v, f"network.{info.field_name}")

    @field_validator("type", "activation", "init", mode="before")
    @classmethod
    def _check_str_type(cls, v, info):
        if not isinstance(v, str):
            raise TypeError(
                f"NetworkConfig.{info.field_name}: expected str, got {type(v).__name__} ({v!r})"
            )
        return v

    @field_validator("multi_head", "skip_connections", "use_zlb_feature", mode="before")
    @classmethod
    def _check_bool_type(cls, v, info):
        if not isinstance(v, bool):
            raise TypeError(
                f"NetworkConfig.{info.field_name}: expected bool, got {type(v).__name__} ({v!r})"
            )
        return v

    @model_validator(mode="after")
    def _validate_ranges(self):
        if self.type not in self.VALID_TYPES:
            raise ValueError(
                f"Unknown network type '{self.type}'. Valid: {sorted(self.VALID_TYPES)}"
            )
        if self.activation not in self.VALID_ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{self.activation}'. "
                f"Valid: {sorted(self.VALID_ACTIVATIONS)}"
            )
        if self.init not in self.VALID_INITS:
            raise ValueError(
                f"Unknown init '{self.init}'. Valid: {sorted(self.VALID_INITS)}"
            )
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes must be non-empty")
        if any(s <= 0 for s in self.hidden_sizes):
            raise ValueError(f"All hidden_sizes must be > 0, got {self.hidden_sizes}")
        if self.activations is not None and len(self.activations) != len(
            self.hidden_sizes
        ):
            raise ValueError(
                f"activations length ({len(self.activations)}) must match "
                f"hidden_sizes length ({len(self.hidden_sizes)})"
            )
        if self.history_len < 1:
            raise ValueError(f"history_len must be >= 1, got {self.history_len}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {self.num_heads}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {self.n_layers}")
        if self.type == "transformer":
            hidden_dim = self.hidden_sizes[0]
            if hidden_dim % self.num_heads != 0:
                raise ValueError(
                    f"For transformer, hidden_dim ({hidden_dim}) must be divisible "
                    f"by num_heads ({self.num_heads})"
                )
        return self
