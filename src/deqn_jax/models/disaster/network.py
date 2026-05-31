"""Disaster-specific policy network: shape priors layered on the residual ansatz.

The generic ``LinearPlusMLP`` in ``networks/linear_plus_mlp.py`` provides
``π = clip(π_BK + δ_θ, lower, upper)`` and nothing else. The disaster model
needs additional shape encodings to handle pathological residual functions
that smooth approximators struggle with (vertical asymptotes in the Calvo
bracket, sign flips in the investment Euler, sharp kinks at the ELB). Those
encodings are model-specific and live here, not in the generic library
module.

Three shape priors are implemented, each independently toggleable:

1. **K/F gauge mask** (``kf_names``). Zeros the MLP delta at the named
   policy positions so those outputs stay exactly equal to the BK linear
   policy. Targets the gauge near-degeneracy in the Calvo recursive
   aggregates ``F_p, K_p, F_w, K_w``.

2. **ELB feature augmentation** (``use_zlb_feature`` + ``zlb_feature_kind``).
   Prepends ``R_lag - R_lb`` (raw) or ``max(R_lag - R_lb, 0)`` (kink) to
   the MLP's input so the network has explicit access to ELB-regime
   information without needing tanh layers to learn the kink shape.

3. **Investment-bracket reparam** (``reparam_q_as_m``, §3.3 of the
   shape-priors doc). Treats the network's ``q`` output as
   ``M = q · 𝓑(x)`` where ``𝓑(x) = 1 − S(x) − x·S'(x)`` is the
   investment-Euler bracket and ``x = µ_z·i/i_lag``. Recovers
   ``q = M / 𝓑(x)`` post-MLP. Eliminates the sign-flip pathology in
   eq 7 — the LHS ``µ_Υ q 𝓑(x)`` is forced positive by parameterization.

All three are off by default. Configs that need them set the relevant
``NetworkConfig`` fields explicitly.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.networks.common import _resolve_activation, _to_array, _to_tuple
from deqn_jax.networks.mlp import MLP


class DisasterPolicyNet(eqx.Module):
    """LinearPlusMLP residual ansatz augmented with disaster-specific shape priors.

    Forward pass (each step optional, controlled by configuration):

      1. ``mlp_input = augment_with_zlb_feature(state)``     (if use_zlb_feature)
      2. ``linear = π_BK(state)``                              (always)
      3. ``M_BK = transform_q_to_m(linear)``                  (if reparam_q_as_m)
      4. ``δ = mlp(mlp_input)``                                (always)
      5. ``δ = mask_kf(δ)``                                   (if kf_indices)
      6. ``raw = linear + δ``                                  (always)
      7. ``raw = recover_q_from_m(raw, state)``               (if reparam_q_as_m)
      8. ``policy = clip(raw, lower, upper)``                  (always)

    Each step is a thin transformation; the core residual ansatz is
    the same as in the generic LinearPlusMLP.
    """

    # Generic ansatz components (mirror LinearPlusMLP)
    mlp: MLP
    P: Array
    ss_state: Array
    ss_policy: Array
    # Per-policy output parameterization (mirror of LinearPlusMLP). Tuple of
    # ints, 0=linear, 1=log. Reparam-affected slots (q if reparam_q_as_m, pi
    # if reparam_pi_as_kp_inner, w_tilda if reparam_wtilda_as_kw_inner) are
    # forced to linear and validated in __init__ — log + reparam at the same
    # slot is incoherent (the reparam logic operates in level space).
    output_links: tuple = eqx.field(static=True)
    policy_lower: Optional[tuple] = eqx.field(static=True)
    policy_upper: Optional[tuple] = eqx.field(static=True)

    # K/F gauge mask
    kf_indices: tuple = eqx.field(static=True)

    # ELB feature augmentation
    use_zlb_feature: bool = eqx.field(static=True)
    zlb_feature_kind: str = eqx.field(static=True)
    r_lag_idx: int = eqx.field(static=True)
    r_lb: float = eqx.field(static=True)

    # Investment-bracket reparameterization
    reparam_q_as_m: bool = eqx.field(static=True)
    q_idx: int = eqx.field(static=True)
    i_idx: int = eqx.field(static=True)
    i_lag_idx: int = eqx.field(static=True)
    mu_z_idx: int = eqx.field(static=True)
    kappa: float = eqx.field(static=True)
    mu_z_ss: float = eqx.field(static=True)

    # Calvo K_p_inner reparameterization (§3.1, price side):
    # treat the network's `π` output slot as K_p_inner_t ∈ (0, 1/(1−ξ_p))
    # via repurposed bounds; derive π_t from the inverse map post-clip,
    # AND override raw[K_p_idx] with K_p = F_p · K_p_inner^{1−λ_f} so eq 2a
    # becomes an identity (no residual). Encodes the Calvo asymptote in
    # the parameterization so the MLP only has to learn smooth K_p_inner.
    reparam_pi_as_kp_inner: bool = eqx.field(static=True)
    pi_idx: int = eqx.field(static=True)
    pi_lag_idx: int = eqx.field(static=True)
    F_p_idx: int = eqx.field(static=True)
    K_p_idx: int = eqx.field(static=True)
    xi_p: float = eqx.field(static=True)
    lambda_f: float = eqx.field(static=True)
    iota: float = eqx.field(static=True)
    pi_ss: float = eqx.field(static=True)

    # Calvo K_w_inner reparameterization (§3.1', wage side, mirror of §3.1):
    # treat the network's `w_tilda` output slot as K_w_inner_t. Derive
    # w_tilda from the inverse eq 4a formula post-clip:
    #     w_tilda = ψ_L · K_w / (F_w · K_w_inner^{1−λ_w(1+σ_L)})
    # which makes eq 4a an identity by construction. Encodes the wage-side
    # Calvo asymptote symmetrically to the price-side reparam.
    reparam_wtilda_as_kw_inner: bool = eqx.field(static=True)
    w_tilda_idx: int = eqx.field(static=True)
    w_tilda_lag_idx: int = eqx.field(static=True)
    F_w_idx: int = eqx.field(static=True)
    K_w_idx: int = eqx.field(static=True)
    xi_w: float = eqx.field(static=True)
    lambda_w: float = eqx.field(static=True)
    sigma_L: float = eqx.field(static=True)
    iota_w: float = eqx.field(static=True)
    psi_L: float = eqx.field(static=True)

    def __init__(
        self,
        n_states: int,
        n_policies: int,
        hidden_sizes: Sequence[int],
        activation: str,
        P: Array,
        ss_state: Array,
        ss_policy: Array,
        policy_lower: Optional[Array] = None,
        policy_upper: Optional[Array] = None,
        init: str = "default",
        init_scale: float = 0.01,
        input_shift: Optional[Array] = None,
        input_scale: Optional[Array] = None,
        kf_indices: Sequence[int] = (),
        use_zlb_feature: bool = False,
        zlb_feature_kind: Literal["raw", "kink"] = "raw",
        r_lag_idx: int = 5,
        r_lb: float = 1.0,
        reparam_q_as_m: bool = False,
        q_idx: int = -1,
        i_idx: int = -1,
        i_lag_idx: int = -1,
        mu_z_idx: int = -1,
        kappa: float = 0.0,
        mu_z_ss: float = 1.0,
        reparam_pi_as_kp_inner: bool = False,
        pi_idx: int = -1,
        pi_lag_idx: int = -1,
        F_p_idx: int = -1,
        K_p_idx: int = -1,
        xi_p: float = 0.0,
        lambda_f: float = 0.0,
        iota: float = 0.0,
        pi_ss: float = 1.0,
        reparam_wtilda_as_kw_inner: bool = False,
        w_tilda_idx: int = -1,
        w_tilda_lag_idx: int = -1,
        F_w_idx: int = -1,
        K_w_idx: int = -1,
        xi_w: float = 0.0,
        lambda_w: float = 0.0,
        sigma_L: float = 1.0,
        iota_w: float = 0.0,
        psi_L: float = 1.0,
        output_links: Optional[Sequence[str]] = None,
        *,
        key: Array,
    ):
        # ELB feature augmentation: extra MLP input dim if enabled.
        self.use_zlb_feature = bool(use_zlb_feature)
        if zlb_feature_kind not in ("raw", "kink"):
            raise ValueError(
                f"zlb_feature_kind must be 'raw' or 'kink', got {zlb_feature_kind!r}"
            )
        self.zlb_feature_kind = str(zlb_feature_kind)
        self.r_lag_idx = int(r_lag_idx)
        self.r_lb = float(r_lb)
        n_extra = 1 if self.use_zlb_feature else 0

        # When the ZLB regime feature is on, the MLP's input dimension grows
        # by 1. Pad caller-supplied input_shift/input_scale so they match
        # (shift 0, scale 1 on the extra dim — no rescaling on regime coord).
        if n_extra > 0 and input_shift is not None:
            input_shift = jnp.concatenate(
                [jnp.asarray(input_shift), jnp.zeros(n_extra)]
            )
        if n_extra > 0 and input_scale is not None:
            input_scale = jnp.concatenate([jnp.asarray(input_scale), jnp.ones(n_extra)])

        act_fn = _resolve_activation(activation)
        self.mlp = MLP(  # pyright: ignore[reportAssignmentType]  # ty: ignore[invalid-assignment]
            in_features=n_states + n_extra,
            out_features=n_policies,
            hidden_sizes=hidden_sizes,
            activations=[act_fn] * len(hidden_sizes),
            output_lower=None,
            output_upper=None,
            input_shift=input_shift,
            input_scale=input_scale,
            init=init,
            key=key,
        )

        # Final-layer scaling for zero-init delta: at training step 0 the
        # MLP output is exactly zero, so policy = π_BK exactly.
        last = self.mlp.layers[-1]
        scaled_w = last.weight * init_scale
        zero_b = jnp.zeros_like(last.bias)
        new_last = eqx.tree_at(
            lambda l: (l.weight, l.bias),
            last,
            (scaled_w, zero_b),
        )
        self.mlp = eqx.tree_at(lambda m: m.layers[-1], self.mlp, new_last)

        self.P = jnp.asarray(P)
        self.ss_state = jnp.asarray(ss_state)
        self.ss_policy = jnp.asarray(ss_policy)
        self.policy_lower = _to_tuple(policy_lower)
        self.policy_upper = _to_tuple(policy_upper)

        # K/F gauge mask: empty tuple disables masking entirely.
        kf_idx_tuple = tuple(int(i) for i in kf_indices)
        for i in kf_idx_tuple:
            if not (0 <= i < n_policies):
                raise ValueError(
                    f"kf_indices entry {i} out of range for n_policies={n_policies}"
                )
        self.kf_indices = kf_idx_tuple

        # Investment-bracket reparam (§3.3): when True, the q output of
        # the network is interpreted as M = q · 𝓑(x), and q is recovered
        # post-MLP as M / 𝓑(x). Eliminates the sign-flip pathology in
        # eq 7 (the LHS µ_Υ q 𝓑(x) is forced positive by construction).
        self.reparam_q_as_m = bool(reparam_q_as_m)
        self.q_idx = int(q_idx)
        self.i_idx = int(i_idx)
        self.i_lag_idx = int(i_lag_idx)
        self.mu_z_idx = int(mu_z_idx)
        self.kappa = float(kappa)
        self.mu_z_ss = float(mu_z_ss)
        if self.reparam_q_as_m:
            for name, idx, bound in [
                ("q_idx", self.q_idx, n_policies),
                ("i_idx", self.i_idx, n_policies),
                ("i_lag_idx", self.i_lag_idx, n_states),
                ("mu_z_idx", self.mu_z_idx, n_states),
            ]:
                if not (0 <= idx < bound):
                    raise ValueError(
                        f"reparam_q_as_m=True requires valid {name}, got {idx} "
                        f"(bound {bound})"
                    )
            if self.kappa <= 0.0:
                raise ValueError(
                    f"reparam_q_as_m=True requires kappa > 0, got {self.kappa}"
                )

        # Calvo K_p_inner reparam (§3.1, price side): the network's π output
        # slot is repurposed as K_p_inner_t. We OVERRIDE the bounds at that
        # slot from the model's pi-bounds [0.95, 1.10] to K_p_inner-bounds
        # (0.01, 1/(1−ξ_p) − 0.01) so the existing clip mechanism enforces
        # the K_p_inner domain. Post-clip, the inverse map recovers π_t.
        self.reparam_pi_as_kp_inner = bool(reparam_pi_as_kp_inner)
        self.pi_idx = int(pi_idx)
        self.pi_lag_idx = int(pi_lag_idx)
        self.F_p_idx = int(F_p_idx)
        self.K_p_idx = int(K_p_idx)
        self.xi_p = float(xi_p)
        self.lambda_f = float(lambda_f)
        self.iota = float(iota)
        self.pi_ss = float(pi_ss)
        if self.reparam_pi_as_kp_inner:
            for name, idx, bound in [
                ("pi_idx", self.pi_idx, n_policies),
                ("F_p_idx", self.F_p_idx, n_policies),
                ("K_p_idx", self.K_p_idx, n_policies),
                ("pi_lag_idx", self.pi_lag_idx, n_states),
            ]:
                if not (0 <= idx < bound):
                    raise ValueError(
                        f"reparam_pi_as_kp_inner=True requires valid {name}, got "
                        f"{idx} (bound {bound})"
                    )
            if not (0.0 < self.xi_p < 1.0):
                raise ValueError(
                    f"reparam_pi_as_kp_inner=True requires 0 < xi_p < 1, got {self.xi_p}"
                )
            if self.lambda_f <= 1.0:
                raise ValueError(
                    f"reparam_pi_as_kp_inner=True requires lambda_f > 1, got "
                    f"{self.lambda_f}"
                )
            # Compute reference values from the solved SS for bound derivation.
            pi_lag_ss = float(self.ss_state[self.pi_lag_idx])
            pi_tilda_ss = (self.pi_ss**self.iota) * (pi_lag_ss ** (1.0 - self.iota))
            a_exp = 1.0 / (1.0 - self.lambda_f)

            def _kp_inner_from_pi(pi_val: float) -> float:
                """Forward Calvo: π → K_p_inner at SS pi_tilda (for bounds)."""
                return (1.0 - self.xi_p * (pi_tilda_ss / pi_val) ** a_exp) / (
                    1.0 - self.xi_p
                )

            # Derive K_p_inner bounds FROM THE ORIGINAL pi bounds (not the
            # full domain). High π → low K_p_inner; low π → high K_p_inner.
            # This preserves the model's physical pi range while encoding
            # the Calvo-edge asymptote at the lower bound (where π →
            # 1.108·π̃). The Calvo full domain (0, 1/(1−ξ_p)) is much
            # wider than physically plausible π — using it lets the
            # optimizer push K_p_inner toward extreme values that yield
            # π outside the model's calibrated range.
            orig_pi_lower = (
                float(self.policy_lower[self.pi_idx])
                if self.policy_lower is not None
                else 0.5
            )
            orig_pi_upper = (
                float(self.policy_upper[self.pi_idx])
                if self.policy_upper is not None
                else float("inf")
            )
            kp_inner_lo = (
                _kp_inner_from_pi(orig_pi_upper)
                if orig_pi_upper < float("inf")
                else 0.01
            )
            kp_inner_hi = _kp_inner_from_pi(orig_pi_lower)
            kp_inner_lo = max(kp_inner_lo, 0.01)
            kp_inner_hi = min(kp_inner_hi, 1.0 / (1.0 - self.xi_p) - 0.01)

            new_lower = (
                list(self.policy_lower)
                if self.policy_lower is not None
                else [-float("inf")] * n_policies
            )
            new_upper = (
                list(self.policy_upper)
                if self.policy_upper is not None
                else [float("inf")] * n_policies
            )
            new_lower[self.pi_idx] = kp_inner_lo
            new_upper[self.pi_idx] = kp_inner_hi
            self.policy_lower = tuple(new_lower)
            self.policy_upper = tuple(new_upper)

            # NOTE: do not override ss_policy[pi_idx]. We instead compute
            # K_p_inner_BK on-the-fly in the forward pass via the EXACT
            # Calvo formula applied to pi_BK = linear[pi_idx]. This avoids
            # the off-SS bias that arises when ss_policy[pi_idx] is set to
            # K_p_inner_ss but P[pi_idx] is still the pi-space derivative.

        # Calvo K_w_inner reparam (§3.1', wage side): symmetric to §3.1.
        # Treat the network's `w_tilda` output slot as K_w_inner_t. Override
        # bounds at w_tilda_idx to a wider Calvo-domain range. After MLP +
        # clip, recover w_tilda via the inverse eq 4a formula.
        self.reparam_wtilda_as_kw_inner = bool(reparam_wtilda_as_kw_inner)
        self.w_tilda_idx = int(w_tilda_idx)
        self.w_tilda_lag_idx = int(w_tilda_lag_idx)
        self.F_w_idx = int(F_w_idx)
        self.K_w_idx = int(K_w_idx)
        self.xi_w = float(xi_w)
        self.lambda_w = float(lambda_w)
        self.sigma_L = float(sigma_L)
        self.iota_w = float(iota_w)
        self.psi_L = float(psi_L)
        if self.reparam_wtilda_as_kw_inner:
            for name, idx, bound in [
                ("w_tilda_idx", self.w_tilda_idx, n_policies),
                ("F_w_idx", self.F_w_idx, n_policies),
                ("K_w_idx", self.K_w_idx, n_policies),
                ("w_tilda_lag_idx", self.w_tilda_lag_idx, n_states),
            ]:
                if not (0 <= idx < bound):
                    raise ValueError(
                        f"reparam_wtilda_as_kw_inner=True requires valid {name}, "
                        f"got {idx} (bound {bound})"
                    )
            if not (0.0 < self.xi_w < 1.0):
                raise ValueError(
                    f"reparam_wtilda_as_kw_inner requires 0 < xi_w < 1, got {self.xi_w}"
                )
            if self.lambda_w <= 1.0:
                raise ValueError(
                    f"reparam_wtilda_as_kw_inner requires lambda_w > 1, got "
                    f"{self.lambda_w}"
                )
            # Override bounds at w_tilda_idx to the K_w_inner domain
            # (0.01, 1/(1−ξ_w) − 0.01). The original w_tilda bounds (lower
            # ≈ 1.0, upper = ∞) don't translate to a meaningful K_w_inner
            # range — they'd constrain wages to the deflationary side only.
            kw_inner_lo = 0.01
            kw_inner_hi = 1.0 / (1.0 - self.xi_w) - 0.01
            new_lower = (
                list(self.policy_lower)
                if self.policy_lower is not None
                else [-float("inf")] * n_policies
            )
            new_upper = (
                list(self.policy_upper)
                if self.policy_upper is not None
                else [float("inf")] * n_policies
            )
            new_lower[self.w_tilda_idx] = kw_inner_lo
            new_upper[self.w_tilda_idx] = kw_inner_hi
            self.policy_lower = tuple(new_lower)
            self.policy_upper = tuple(new_upper)

        # output_links: per-policy "linear"/"log". Reparam-affected slots are
        # forced to "linear" (incoherent to apply log on top of a slot whose
        # raw output is the K_inner / M reparam target). We raise on conflict
        # rather than silently overriding so the user sees the issue.
        if output_links is None:
            link_codes_tuple: tuple = (0,) * n_policies
        else:
            if len(output_links) != n_policies:
                raise ValueError(
                    f"output_links length {len(output_links)} != n_policies {n_policies}"
                )
            mapping = {"linear": 0, "log": 1}
            try:
                link_codes_tuple = tuple(mapping[link] for link in output_links)
            except KeyError as e:
                raise ValueError(
                    f"unknown output_link {e.args[0]!r}; "
                    f"valid choices: {list(mapping.keys())}"
                ) from None
            ss_arr = jnp.asarray(ss_policy)
            bad_pos = [
                i
                for i, code in enumerate(link_codes_tuple)
                if code == 1 and float(ss_arr[i]) <= 0.0
            ]
            if bad_pos:
                raise ValueError(
                    f"output_link='log' requires ss_policy > 0; non-positive at "
                    f"indices {bad_pos} (ss values {[float(ss_arr[i]) for i in bad_pos]})"
                )
            # Reparam ↔ log incompatibility checks
            reparam_slots = []
            if self.reparam_q_as_m and self.q_idx >= 0:
                reparam_slots.append(("reparam_q_as_m", self.q_idx))
            if self.reparam_pi_as_kp_inner and self.pi_idx >= 0:
                reparam_slots.append(("reparam_pi_as_kp_inner", self.pi_idx))
            if self.reparam_wtilda_as_kw_inner and self.w_tilda_idx >= 0:
                reparam_slots.append(("reparam_wtilda_as_kw_inner", self.w_tilda_idx))
            for flag_name, slot_idx in reparam_slots:
                if link_codes_tuple[slot_idx] == 1:
                    raise ValueError(
                        f"{flag_name}=True forces output_links[{slot_idx}]='linear' "
                        f"(the reparam logic operates in level space); got 'log'"
                    )
        self.output_links = link_codes_tuple

    def _forward_single(self, state: Array) -> Array:
        # BK linear part. ``self.P`` is stored *per-row in natural space*
        # (level for linear/reparam slots, log for log slots — conversion
        # happens in the factory). ``bk_corr[i]`` is therefore in the
        # natural unit of policy i. For linear slots: ss_i + bk_corr[i]
        # gives the BK level. For log slots: exp(bk_corr[i]) is the
        # multiplicative deviation, so ss_i * exp(bk_corr[i]) gives BK.
        ss_state = jax.lax.stop_gradient(self.ss_state)
        ss_policy = jax.lax.stop_gradient(self.ss_policy)
        P = jax.lax.stop_gradient(self.P)
        bk_corr = P @ (state - ss_state)

        # Capture pi_BK / w_tilda_BK in *level* space at the pre-substitution
        # point. Reparam slots are forced linear-link, so pi_BK = ss + bk_corr.
        pi_BK_captured = (
            ss_policy[self.pi_idx] + bk_corr[self.pi_idx]
            if (self.reparam_pi_as_kp_inner or self.reparam_wtilda_as_kw_inner)
            else None
        )
        w_tilda_BK_captured = (
            ss_policy[self.w_tilda_idx] + bk_corr[self.w_tilda_idx]
            if self.reparam_wtilda_as_kw_inner
            else None
        )

        # Calvo K_p_inner reparam (§3.1, BK conversion): substitute the
        # pi-slot's BK contribution so that ss + bk_corr = K_p_inner_BK at
        # the pi slot. (Always level-space; pi_idx is forced linear-link.)
        if self.reparam_pi_as_kp_inner:
            pi_lag = state[self.pi_lag_idx]
            pi_tilda_t = (self.pi_ss**self.iota) * (pi_lag ** (1.0 - self.iota))
            a_exp_fwd = 1.0 / (1.0 - self.lambda_f)
            kp_inner_BK = (
                1.0 - self.xi_p * (pi_tilda_t / pi_BK_captured) ** a_exp_fwd
            ) / (1.0 - self.xi_p)
            bk_corr = bk_corr.at[self.pi_idx].set(kp_inner_BK - ss_policy[self.pi_idx])

        # Calvo K_w_inner reparam (§3.1', wage side BK conversion).
        if self.reparam_wtilda_as_kw_inner:
            pi_lag_for_w = state[self.pi_lag_idx]
            w_tilda_lag = state[self.w_tilda_lag_idx]
            pi_w_BK = pi_BK_captured * w_tilda_BK_captured / (w_tilda_lag + 1e-12)
            pi_w_tilda_t = (self.pi_ss**self.iota_w) * (
                pi_lag_for_w ** (1.0 - self.iota_w)
            )
            aw_exp = 1.0 / (1.0 - self.lambda_w)
            kw_inner_BK = (
                1.0 - self.xi_w * (pi_w_tilda_t * self.mu_z_ss / pi_w_BK) ** aw_exp
            ) / (1.0 - self.xi_w)
            bk_corr = bk_corr.at[self.w_tilda_idx].set(
                kw_inner_BK - ss_policy[self.w_tilda_idx]
            )

        # Investment-bracket reparam: q-slot's BK contribution is
        # M_BK - ss[q_idx] (q_idx forced linear-link).
        if self.reparam_q_as_m:
            mu_z_t = state[self.mu_z_idx]
            i_lag = state[self.i_lag_idx]
            i_BK = ss_policy[self.i_idx] + bk_corr[self.i_idx]
            x_BK = mu_z_t * i_BK / (i_lag + 1e-12)
            S_val_BK = 0.5 * self.kappa * (x_BK - self.mu_z_ss) ** 2
            S_prime_BK = self.kappa * (x_BK - self.mu_z_ss)
            B_BK = 1.0 - S_val_BK - x_BK * S_prime_BK
            q_BK = ss_policy[self.q_idx] + bk_corr[self.q_idx]
            M_BK = q_BK * B_BK
            bk_corr = bk_corr.at[self.q_idx].set(M_BK - ss_policy[self.q_idx])

        # ELB feature augmentation on the MLP input.
        if self.use_zlb_feature:
            raw_prox = state[self.r_lag_idx] - self.r_lb
            if self.zlb_feature_kind == "kink":
                zlb_prox = jnp.maximum(raw_prox, jnp.asarray(0.0, dtype=raw_prox.dtype))
            else:
                zlb_prox = raw_prox
            mlp_input = jnp.concatenate([state, jnp.array([zlb_prox])])
        else:
            mlp_input = state

        delta = self.mlp(mlp_input)

        # K/F gauge mask: zero delta at named output positions.
        if self.kf_indices:
            mask = (
                jnp.ones(delta.shape[0], dtype=delta.dtype)
                .at[jnp.asarray(self.kf_indices)]
                .set(jnp.asarray(0.0, dtype=delta.dtype))
            )
            delta = delta * mask

        # Per-policy raw assembly. Reparam slots are forced linear-link, so
        # the additive form recovers the level-space K_inner / M values
        # injected via bk_corr above. Log-link slots use multiplicative form
        # for natural positivity + log-deviation parameterization.
        if all(code == 0 for code in self.output_links):
            raw = ss_policy + bk_corr + delta
        elif all(code == 1 for code in self.output_links):
            raw = ss_policy * jnp.exp(bk_corr + delta)
        else:
            is_log = jnp.asarray(self.output_links, dtype=jnp.int8) == 1
            raw_linear = ss_policy + bk_corr + delta
            raw_log = ss_policy * jnp.exp(bk_corr + delta)
            raw = jnp.where(is_log, raw_log, raw_linear)

        # Investment-bracket reparam: recover q_t = M_t / 𝓑(x_t) using the
        # POST-delta i_t to compute x_t. 𝓑 floored at 1e-3 to keep the
        # division well-conditioned if x approaches the sign-flip point.
        if self.reparam_q_as_m:
            mu_z_t = state[self.mu_z_idx]
            i_lag = state[self.i_lag_idx]
            i_t = raw[self.i_idx]
            x_t = mu_z_t * i_t / (i_lag + 1e-12)
            S_val_t = 0.5 * self.kappa * (x_t - self.mu_z_ss) ** 2
            S_prime_t = self.kappa * (x_t - self.mu_z_ss)
            B_t = 1.0 - S_val_t - x_t * S_prime_t
            B_safe = jnp.where(B_t > 1e-3, B_t, jnp.asarray(1e-3, dtype=B_t.dtype))
            M_t = raw[self.q_idx]
            q_t = M_t / B_safe
            raw = raw.at[self.q_idx].set(q_t)

        # Hard clip per-output bounds. When reparam_pi_as_kp_inner is True,
        # the bounds at pi_idx have been overridden in __init__ to enforce
        # the K_p_inner domain (0.01, 1/(1−ξ_p)−0.01).
        if self.policy_lower is not None:
            lower = jax.lax.stop_gradient(_to_array(self.policy_lower))
            raw = jnp.maximum(raw, lower)
        if self.policy_upper is not None:
            upper = jax.lax.stop_gradient(_to_array(self.policy_upper))
            safe_upper = jnp.where(jnp.isinf(upper), jnp.array(1e10), upper)
            raw = jnp.minimum(raw, safe_upper)

        # Calvo K_p_inner reparam (§3.1, price side): post-clip, raw[pi_idx]
        # is K_p_inner_t bounded to (0.01, 1/(1−ξ_p)−0.01). Recover π_t via
        # the inverse map AND override raw[K_p_idx] with K_p = F_p ·
        # K_p_inner^{1−λ_f} so eq 2a is satisfied as an algebraic identity
        # (no residual). The (1−(1−ξ_p)·K_p_inner)/ξ_p factor is positive
        # everywhere on the bounded K_p_inner domain.
        if self.reparam_pi_as_kp_inner:
            pi_lag = state[self.pi_lag_idx]
            pi_tilda_t = (self.pi_ss**self.iota) * (pi_lag ** (1.0 - self.iota))
            kp_inner_t = raw[self.pi_idx]
            inner_term = (1.0 - (1.0 - self.xi_p) * kp_inner_t) / self.xi_p
            pi_t = pi_tilda_t * inner_term ** (self.lambda_f - 1.0)
            F_p_t = raw[self.F_p_idx]
            K_p_t = F_p_t * kp_inner_t ** (1.0 - self.lambda_f)
            raw = raw.at[self.pi_idx].set(pi_t)
            raw = raw.at[self.K_p_idx].set(K_p_t)

        # Calvo K_w_inner reparam (§3.1', wage side): post-clip, raw[w_tilda_idx]
        # is K_w_inner_t. Recover w_tilda via the *Calvo formula inversion*
        # (mirror of price-side), then override K_w to make eq 4a identically
        # zero. Critical: the eq-4a-direct inversion (w_tilda = ψ_L·K_w·
        # K_w_inner^{1−λ_w(1+σ_L)}/F_w) is NOT correct off-SS, because then
        # equations.definitions() recomputes K_w_inner_eq from
        # (pi_w_tilda·µ_z_ss/pi_w) and gets a value ≠ K_w_inner_t, breaking
        # eq 4a. The fix: invert the Calvo formula to get pi_w_t such that
        # K_w_inner_eq = K_w_inner_t identically, then derive w_tilda_t from
        # pi_w_t = pi_t·w_tilda_t/w_tilda_lag, then override K_w_t to satisfy
        # eq 4a.
        if self.reparam_wtilda_as_kw_inner:
            pi_lag_for_w = state[self.pi_lag_idx]
            w_tilda_lag = state[self.w_tilda_lag_idx]
            pi_w_tilda_t = (self.pi_ss**self.iota_w) * (
                pi_lag_for_w ** (1.0 - self.iota_w)
            )
            kw_inner_t = raw[self.w_tilda_idx]
            inner_term_w = (1.0 - (1.0 - self.xi_w) * kw_inner_t) / self.xi_w
            # Calvo inversion: pi_w = pi_w_tilda · µ_z_ss · inner_term^{λ_w−1}
            pi_w_t = pi_w_tilda_t * self.mu_z_ss * inner_term_w ** (self.lambda_w - 1.0)
            # raw[pi_idx] is already the recovered pi_t (price-side ran above
            # if enabled, otherwise it's the network's clipped output for pi).
            pi_t = raw[self.pi_idx]
            w_tilda_t = pi_w_t * w_tilda_lag / (pi_t + 1e-12)
            # K_w override: makes eq 4a identity (K_w = (1/ψ_L)·K_w_inner^{1−λ_w(1+σ_L)}·w_tilda·F_w)
            kw_exponent = 1.0 - self.lambda_w * (1.0 + self.sigma_L)
            F_w_t = raw[self.F_w_idx]
            K_w_t = (1.0 / self.psi_L) * (kw_inner_t**kw_exponent) * w_tilda_t * F_w_t
            raw = raw.at[self.w_tilda_idx].set(w_tilda_t)
            raw = raw.at[self.K_w_idx].set(K_w_t)

        return raw

    def __call__(self, x: Array) -> Array:
        if x.ndim == 1:
            return self._forward_single(x)
        return jax.vmap(self._forward_single)(x)


def create_disaster_policy_net(
    model,
    hidden_sizes: Sequence[int] = (128, 128),
    activation: str = "tanh",
    init: str = "default",
    init_scale: float = 0.01,
    input_shift: Optional[Array] = None,
    input_scale: Optional[Array] = None,
    kf_names: Sequence[str] = (),
    use_zlb_feature: bool = False,
    zlb_feature_kind: Literal["raw", "kink"] = "raw",
    reparam_q_as_m: bool = False,
    reparam_pi_as_kp_inner: bool = False,
    reparam_wtilda_as_kw_inner: bool = False,
    output_links: Optional[Sequence[str]] = None,
    *,
    key: Array,
) -> DisasterPolicyNet:
    """Factory: build DisasterPolicyNet for the disaster model.

    Resolves disaster-specific indices and constants from ``model``:

    - ``kf_names`` → ``kf_indices`` via ``model.policy_names`` lookup
    - ``r_lag_idx`` → position of ``R_lag`` in ``model.state_names``
    - ``r_lb`` → ``model.constants["R_lb"]`` (default 1.0)
    - ``q_idx, i_idx`` → positions of ``q, i`` in ``model.policy_names``
    - ``i_lag_idx, mu_z_idx`` → positions in ``model.state_names``
    - ``kappa`` → ``model.constants["kappa"]``
    - ``mu_z_ss`` → ``model.constants["mu_z_ss"]``

    All shape priors default off; configs enable them by setting the
    relevant fields on ``NetworkConfig``.
    """
    from deqn_jax.training.linearize import linearize_model

    if model.steady_state_fn is None:
        raise ValueError(
            "DisasterPolicyNet requires model.steady_state_fn to compute "
            "the linearization anchor."
        )
    P_level, _Q = linearize_model(model, verbose=False)
    ss_state, ss_policy = model.steady_state_fn(model.constants)

    # Pre-convert P per-row to its output_link's natural space (mirror of
    # create_linear_plus_mlp). For log-link rows: P_log_i = P_level_i / ss_i.
    if output_links is None:
        P = P_level
    else:
        from deqn_jax.networks.linear_plus_mlp import _convert_p_per_link

        P = _convert_p_per_link(P_level, ss_policy, output_links)

    policy_names = list(model.policy_names)
    state_names = list(model.state_names) if model.state_names is not None else []

    # Resolve K/F gauge-mask indices from policy_names.
    kf_indices: tuple = ()
    if kf_names:
        missing = [n for n in kf_names if n not in policy_names]
        if missing:
            raise ValueError(
                f"kf_names {missing!r} not found in model.policy_names "
                f"{policy_names!r}; cannot mask those outputs."
            )
        kf_indices = tuple(policy_names.index(n) for n in kf_names)

    # Resolve ELB feature parameters.
    r_lag_idx = 5
    r_lb = 1.0
    if use_zlb_feature:
        if "R_lag" in state_names:
            r_lag_idx = state_names.index("R_lag")
        r_lb = float(model.constants.get("R_lb", 1.0))

    # Resolve investment-bracket reparam parameters.
    q_idx = -1
    i_idx = -1
    i_lag_idx = -1
    mu_z_idx = -1
    kappa = 0.0
    mu_z_ss = 1.0
    if reparam_q_as_m:
        if "q" not in policy_names:
            raise ValueError(
                f"reparam_q_as_m=True requires 'q' in policy_names; got {policy_names}"
            )
        if "i" not in policy_names:
            raise ValueError(
                f"reparam_q_as_m=True requires 'i' in policy_names; got {policy_names}"
            )
        if "i_lag" not in state_names:
            raise ValueError(
                f"reparam_q_as_m=True requires 'i_lag' in state_names; got "
                f"{state_names}"
            )
        if "mu_z" not in state_names:
            raise ValueError(
                f"reparam_q_as_m=True requires 'mu_z' in state_names; got {state_names}"
            )
        q_idx = policy_names.index("q")
        i_idx = policy_names.index("i")
        i_lag_idx = state_names.index("i_lag")
        mu_z_idx = state_names.index("mu_z")
        kappa = float(model.constants.get("kappa", 0.0))
        mu_z_ss = float(model.constants.get("mu_z_ss", 1.0))
        if kappa <= 0.0:
            raise ValueError(
                f"reparam_q_as_m=True requires model.constants['kappa'] > 0, "
                f"got {kappa}"
            )

    # Resolve Calvo K_p_inner reparam parameters (§3.1, price side).
    pi_idx = -1
    pi_lag_idx = -1
    F_p_idx = -1
    K_p_idx = -1
    xi_p = 0.0
    lambda_f = 0.0
    iota = 0.0
    pi_ss = 1.0
    if reparam_pi_as_kp_inner:
        for required, where in [
            ("pi", "policy_names"),
            ("F_p", "policy_names"),
            ("K_p", "policy_names"),
            ("pi_lag", "state_names"),
        ]:
            names = policy_names if where == "policy_names" else state_names
            if required not in names:
                raise ValueError(
                    f"reparam_pi_as_kp_inner=True requires {required!r} in "
                    f"{where}; got {names}"
                )
        pi_idx = policy_names.index("pi")
        F_p_idx = policy_names.index("F_p")
        K_p_idx = policy_names.index("K_p")
        pi_lag_idx = state_names.index("pi_lag")
        xi_p = float(model.constants.get("xi_p", 0.0))
        lambda_f = float(model.constants.get("lambda_f", 0.0))
        iota = float(model.constants.get("iota", 0.0))
        pi_ss = float(model.constants.get("pi_ss", 1.0))
        if not (0.0 < xi_p < 1.0):
            raise ValueError(
                f"reparam_pi_as_kp_inner requires 0 < xi_p < 1, got {xi_p}"
            )
        if lambda_f <= 1.0:
            raise ValueError(
                f"reparam_pi_as_kp_inner requires lambda_f > 1, got {lambda_f}"
            )

    # Resolve Calvo K_w_inner reparam parameters (§3.1', wage side).
    w_tilda_idx = -1
    w_tilda_lag_idx = -1
    F_w_idx = -1
    K_w_idx = -1
    xi_w = 0.0
    lambda_w = 0.0
    sigma_L = 1.0
    iota_w = 0.0
    psi_L = 1.0
    if reparam_wtilda_as_kw_inner:
        for required, where in [
            ("w_tilda", "policy_names"),
            ("F_w", "policy_names"),
            ("K_w", "policy_names"),
            ("w_tilda_lag", "state_names"),
        ]:
            names = policy_names if where == "policy_names" else state_names
            if required not in names:
                raise ValueError(
                    f"reparam_wtilda_as_kw_inner=True requires {required!r} in "
                    f"{where}; got {names}"
                )
        # Also need pi_lag for pi_w_tilda (wage indexation) and pi for pi_w.
        # The latter is checked when reparam_pi_as_kp_inner is on; if off, we
        # still need pi_idx to access pi_BK on-the-fly.
        if "pi_lag" not in state_names:
            raise ValueError(
                f"reparam_wtilda_as_kw_inner requires 'pi_lag' in state_names; "
                f"got {state_names}"
            )
        if "pi" not in policy_names:
            raise ValueError(
                f"reparam_wtilda_as_kw_inner requires 'pi' in policy_names; "
                f"got {policy_names}"
            )
        w_tilda_idx = policy_names.index("w_tilda")
        F_w_idx = policy_names.index("F_w")
        K_w_idx = policy_names.index("K_w")
        w_tilda_lag_idx = state_names.index("w_tilda_lag")
        xi_w = float(model.constants.get("xi_w", 0.0))
        lambda_w = float(model.constants.get("lambda_w", 0.0))
        sigma_L = float(model.constants.get("sigma_L", 1.0))
        iota_w = float(model.constants.get("iota_w", 0.0))
        psi_L = float(model.constants.get("psi_L", 1.0))
        if not (0.0 < xi_w < 1.0):
            raise ValueError(
                f"reparam_wtilda_as_kw_inner requires 0 < xi_w < 1, got {xi_w}"
            )
        if lambda_w <= 1.0:
            raise ValueError(
                f"reparam_wtilda_as_kw_inner requires lambda_w > 1, got {lambda_w}"
            )
        # Wage-side reparam needs pi_idx and pi_lag_idx populated even if
        # the price-side reparam isn't on (pi_BK is captured in the forward
        # pass for the wage K_w_inner BK conversion, and pi_lag is needed
        # for pi_w_tilda).
        if pi_idx == -1:
            pi_idx = policy_names.index("pi")
        if pi_lag_idx == -1:
            pi_lag_idx = state_names.index("pi_lag")
        # pi_ss is also needed for pi_w_tilda; populate if not already set.
        if pi_ss == 1.0 and "pi_ss" in model.constants:
            pi_ss = float(model.constants["pi_ss"])
        # mu_z_ss enters the K_w_inner BK formula (pi_w_tilda * mu_z_ss / pi_w).
        # Only resolved by the q_as_m branch above, so populate here too if
        # that branch didn't run.
        if mu_z_ss == 1.0 and "mu_z_ss" in model.constants:
            mu_z_ss = float(model.constants["mu_z_ss"])

    return DisasterPolicyNet(
        n_states=model.n_states,
        n_policies=model.n_policies,
        hidden_sizes=hidden_sizes,
        activation=activation,
        P=P,
        ss_state=ss_state,
        ss_policy=ss_policy,
        policy_lower=model.policy_lower,
        policy_upper=model.policy_upper,
        init=init,
        init_scale=init_scale,
        input_shift=input_shift,
        input_scale=input_scale,
        kf_indices=kf_indices,
        use_zlb_feature=use_zlb_feature,
        zlb_feature_kind=zlb_feature_kind,
        r_lag_idx=r_lag_idx,
        r_lb=r_lb,
        reparam_q_as_m=reparam_q_as_m,
        q_idx=q_idx,
        i_idx=i_idx,
        i_lag_idx=i_lag_idx,
        mu_z_idx=mu_z_idx,
        kappa=kappa,
        mu_z_ss=mu_z_ss,
        reparam_pi_as_kp_inner=reparam_pi_as_kp_inner,
        pi_idx=pi_idx,
        pi_lag_idx=pi_lag_idx,
        F_p_idx=F_p_idx,
        K_p_idx=K_p_idx,
        xi_p=xi_p,
        lambda_f=lambda_f,
        iota=iota,
        pi_ss=pi_ss,
        reparam_wtilda_as_kw_inner=reparam_wtilda_as_kw_inner,
        w_tilda_idx=w_tilda_idx,
        w_tilda_lag_idx=w_tilda_lag_idx,
        F_w_idx=F_w_idx,
        K_w_idx=K_w_idx,
        xi_w=xi_w,
        lambda_w=lambda_w,
        sigma_L=sigma_L,
        iota_w=iota_w,
        psi_L=psi_L,
        output_links=output_links,
        key=key,
    )


__all__ = ["DisasterPolicyNet", "create_disaster_policy_net"]


# Register this model's diagram renderer with the generic viz package so that
# networks/viz.py does not import this model (audit networks-03). A
# DisasterPolicyNet renders identically to a LinearPlusMLP (same residual
# ansatz), so it reuses that renderer. Wrapped defensively: a viz import
# problem must never break loading the disaster model.
def _register_viz_renderer() -> None:
    try:
        from deqn_jax.networks.viz import (
            _render_linear_plus_mlp,
            register_network_renderer,
        )
    except Exception:
        return
    register_network_renderer(
        "DisasterPolicyNet",
        lambda m: isinstance(m, DisasterPolicyNet),
        _render_linear_plus_mlp,
    )


_register_viz_renderer()
