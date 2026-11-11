"""Options of the disaster policy network (``network.type: disaster_policy_net``).

Set under ``network.options`` in a config; ``NetworkConfig`` validates the
block against this class and ``networks.factory`` passes it to
``create_disaster_policy_net``. What each option does to the forward pass is
described in ``network.py``.
"""

from __future__ import annotations

from typing import Literal, Tuple

from pydantic import Field, field_validator

from deqn_jax.config._base import _ConfigBase


class DisasterNetOptions(_ConfigBase):
    """Shape priors and selection devices of ``DisasterPolicyNet``."""

    kf_names: Tuple[str, ...] = Field(
        default=("F_p", "K_p", "F_w", "K_w"),
        description="Policy names whose MLP delta is masked to zero (a restriction that holds them linear; not a gauge fix — graph @aleph/deqn #34). Default targets the four CMR Calvo Phillips-curve auxiliaries.",
    )
    use_zlb_feature: bool = Field(
        default=False,
        description="Append `(R_lag - R_lb)` as an extra MLP input feature.",
    )
    zlb_feature_kind: Literal["raw", "kink"] = Field(
        default="raw",
        description="When use_zlb_feature=true: 'raw' = signed distance R_lag - R_lb; 'kink' = max(R_lag - R_lb, 0), PINN-style explicit kink at the floor.",
    )
    bk_pin: bool = Field(
        default=False,
        description="Blanchard-Kahn selection by construction — subtract the MLP delta's value and tangent at the steady state, so pi(s*)=pi* and dpi/ds(s*)=P hold exactly for every parameter value. The residual loss then shapes only second-order-and-beyond deviations.",
    )
    reparam_q_as_m: bool = Field(
        default=False,
        description="Treat the network's `q` output as `M = q · 𝓑(x)` where 𝓑(x) = 1 - S(x) - x·S'(x) is the investment-Euler bracket; recover q = M/𝓑(x) post-MLP (𝓑 floored at 1e-3 in the division). Aimed at the eq 7 sign-flip pathology; 𝓑(x) itself is not constrained.",
    )
    reparam_pi_as_kp_inner: bool = Field(
        default=False,
        description="Treat the network's `pi` output as K_p_inner ∈ (0, 1/(1−ξ_p)); derive π via the inverse Calvo formula post-clip. Encodes the Calvo asymptote in the parameterization so the MLP only learns smooth K_p_inner.",
    )
    reparam_wtilda_as_kw_inner: bool = Field(
        default=False,
        description="Treat the network's `w_tilda` output as K_w_inner ∈ (0, 1/(1−ξ_w)); derive w_tilda via the inverse eq 4a formula post-clip. Wage-side mirror of reparam_pi_as_kp_inner; combine with that flag for symmetric Calvo reparam.",
    )

    @field_validator("kf_names", mode="before")
    @classmethod
    def _coerce_kf_names(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_validator(
        "use_zlb_feature",
        "bk_pin",
        "reparam_q_as_m",
        "reparam_pi_as_kp_inner",
        "reparam_wtilda_as_kw_inner",
        mode="before",
    )
    @classmethod
    def _check_bool_type(cls, v, info):
        if not isinstance(v, bool):
            raise TypeError(
                f"network.options.{info.field_name}: expected bool, got {type(v).__name__} ({v!r})"
            )
        return v
