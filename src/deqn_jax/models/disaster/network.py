"""Disaster-specific policy network: shape priors layered on the residual ansatz.

The generic ``LinearPlusMLP`` in ``networks/linear_plus_mlp.py`` provides
``π = clip(π_BK + δ_θ, lower, upper)`` and nothing else; ``DisasterPolicyNet``
subclasses it and overrides the three steps of its forward pass. The disaster model
needs additional shape encodings to handle pathological residual functions
that smooth approximators struggle with (vertical asymptotes in the Calvo
bracket, sign flips in the investment Euler, sharp kinks at the ELB). Those
encodings are model-specific and live here, not in the generic library
module.

Shape priors and selection devices, each independently toggleable through
``network.options`` (``net_options.DisasterNetOptions``; the
order in which ``DisasterPolicyNet._forward_single`` applies them is listed in
the class docstring):

1. **K/F restriction** (``kf_names``, ON by default). Zeros the MLP delta at
   the named policy positions so those outputs stay exactly equal to the BK
   linear policy: the four Calvo recursive aggregates ``F_p, K_p, F_w, K_w``
   are linear forever (unless a Calvo reparam below overwrites ``K_p`` /
   ``K_w``). This is a restriction that selects a basin, not a gauge
   fix: the Calvo recursions are affine with a current-period source, so a
   common rescaling of K/F is not a symmetry of the equilibrium system
   (measured 2026-09-10; graph @aleph/deqn, node #34). Its cost is unmeasured.

2. **ELB feature augmentation** (``use_zlb_feature`` + ``zlb_feature_kind``).
   Appends ``R_lag - R_lb`` (raw) or ``max(R_lag - R_lb, 0)`` (kink) to
   the MLP's input so the network has explicit access to ELB-regime
   information without needing tanh layers to learn the kink shape.

3. **BK pin** (``bk_pin``, the certified arm). Subtracts the delta's value
   and tangent at the steady state (JVP at s*), so ``pi(s*) = pi*`` and
   ``dpi/ds(s*) = P`` hold for every parameter value; training shapes
   second-order-and-beyond deviations only.

4. **Output-slot reparametrizations** (``reparam_q_as_m``,
   ``reparam_pi_as_kp_inner``, ``reparam_wtilda_as_kw_inner``): the ``q`` /
   ``pi`` / ``w_tilda`` slots carry a better-conditioned quantity (the
   investment-bracket ``M``, the Calvo inner terms ``K_p_inner`` /
   ``K_w_inner``) and the policy is recovered from it. Their logic, indices
   and constants live in ``reparams.py``. Experimental: no shipped config
   enables them.

5. **Per-policy output links** (``output_links``): ``linear`` (additive) or
   ``log`` (multiplicative around SS); reparam slots must be linear.

Defaults: only the K/F restriction is on; everything else is off.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.models.disaster.reparams import (
    CalvoPrice,
    CalvoWage,
    InvestmentBracket,
    constants_of,
    index_of,
)
from deqn_jax.networks.common import _apply_hard_clip
from deqn_jax.networks.linear_plus_mlp import LinearPlusMLP, anchor_kwargs


class DisasterPolicyNet(LinearPlusMLP):
    """LinearPlusMLP residual ansatz augmented with disaster-specific shape priors.

    Forward pass (each step optional, controlled by configuration), by the
    ``LinearPlusMLP`` step that carries it:

      ``_bk_term``
        1. ``bk = P @ (state - ss_state)``                        (always)
        2. replace the BK term at the ``pi`` / ``w_tilda`` / ``q`` slots by
           its K_p_inner / K_w_inner / M value     (if the matching reparam)
      ``_delta``
        3. ``δ = mlp(augment_with_zlb_feature(state))``  (feature if use_zlb_feature)
        4. ``δ = mask_kf(δ)``                                    (if kf_indices)
        5. ``δ = δ - δ(ss_state) - Jδ(ss_state) (state - ss_state)``  (if bk_pin)
      (base) ``raw = combine(ss_policy, bk, δ)`` per output link  (always)
      ``_finish``
        6. ``raw = recover_q_from_m(raw, state)``               (if reparam_q_as_m)
        7. ``policy = clip(raw, lower, upper)``                  (always)
        8. recover ``pi`` / ``K_p`` and ``w_tilda`` / ``K_w`` from the clipped
           K_p_inner / K_w_inner                  (if the matching reparam)

    With no prior enabled, the result is the generic LinearPlusMLP ansatz.
    """

    # K/F restriction (mask)
    kf_indices: tuple = eqx.field(static=True)

    # ELB feature augmentation
    use_zlb_feature: bool = eqx.field(static=True)
    zlb_feature_kind: str = eqx.field(static=True)
    bk_pin: bool = eqx.field(static=True)
    r_lag_idx: int = eqx.field(static=True)
    r_lb: Optional[float] = eqx.field(static=True)

    # Output-slot reparametrizations (``reparams.py``); None = off.
    q_as_m: Optional[InvestmentBracket] = eqx.field(static=True)
    calvo_price: Optional[CalvoPrice] = eqx.field(static=True)
    calvo_wage: Optional[CalvoWage] = eqx.field(static=True)

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
        bk_pin: bool = False,
        zlb_feature_kind: Literal["raw", "kink"] = "raw",
        r_lag_idx: int = -1,
        r_lb: Optional[float] = None,
        q_as_m: Optional[InvestmentBracket] = None,
        calvo_price: Optional[CalvoPrice] = None,
        calvo_wage: Optional[CalvoWage] = None,
        output_links: Optional[Sequence[str]] = None,
        *,
        key: Array,
    ):
        self.use_zlb_feature = bool(use_zlb_feature)
        self.bk_pin = bool(bk_pin)
        if zlb_feature_kind not in ("raw", "kink"):
            raise ValueError(
                f"zlb_feature_kind must be 'raw' or 'kink', got {zlb_feature_kind!r}"
            )
        self.zlb_feature_kind = str(zlb_feature_kind)
        self.r_lag_idx = int(r_lag_idx)
        self.r_lb = None if r_lb is None else float(r_lb)
        if self.use_zlb_feature:
            if not (0 <= self.r_lag_idx < n_states):
                raise ValueError(
                    f"use_zlb_feature=True requires valid r_lag_idx, got "
                    f"{self.r_lag_idx} (bound {n_states})"
                )
            if self.r_lb is None:
                raise ValueError("use_zlb_feature=True requires r_lb")

        # The ELB feature is one extra MLP input.
        super().__init__(
            n_states=n_states,
            n_policies=n_policies,
            hidden_sizes=hidden_sizes,
            activation=activation,
            P=P,
            ss_state=ss_state,
            ss_policy=ss_policy,
            output_links=output_links,
            policy_lower=policy_lower,
            policy_upper=policy_upper,
            init=init,
            init_scale=init_scale,
            input_shift=input_shift,
            input_scale=input_scale,
            extra_inputs=1 if self.use_zlb_feature else 0,
            key=key,
        )

        # K/F restriction: empty tuple disables the mask entirely.
        self.kf_indices = tuple(int(i) for i in kf_indices)
        for i in self.kf_indices:
            if not (0 <= i < n_policies):
                raise ValueError(
                    f"kf_indices entry {i} out of range for n_policies={n_policies}"
                )

        # Output-slot reparametrizations: each repurposed slot's clip bounds
        # become the domain of the quantity it carries (Calvo slots), and its
        # link must be linear (the reparam logic operates in level space).
        self.q_as_m = q_as_m
        self.calvo_price = calvo_price
        self.calvo_wage = calvo_wage
        for reparam in self._reparams():
            reparam.check_indices(n_states, n_policies)
            self.policy_lower, self.policy_upper = reparam.override_bounds(
                self.policy_lower, self.policy_upper, self.ss_state, n_policies
            )
            if self.output_links[reparam.slot] == 1:
                raise ValueError(
                    f"{reparam.flag}=True forces output_links[{reparam.slot}]="
                    f"'linear' (the reparam logic operates in level space); "
                    f"got 'log'"
                )

    def _reparams(self) -> tuple:
        """The enabled reparametrizations, in the order the forward applies
        them (the wage side reads the pi the price side recovers)."""
        return tuple(
            r for r in (self.calvo_price, self.calvo_wage, self.q_as_m) if r is not None
        )

    def _bk_term(self, state: Array) -> Array:
        bk_corr = super()._bk_term(state)

        # Reparametrized slots: replace the BK term by the BK policy in the
        # slot's reparametrized space, so ss + bk_corr there is K_p_inner_BK /
        # K_w_inner_BK / M_BK. Each reads the BK *level* captured before any
        # substitution (reparam slots are forced linear-link).
        reparams = self._reparams()
        if reparams:
            ss_policy = self.ss_policy
            bk_level = ss_policy + bk_corr
            for reparam in reparams:
                bk_corr = bk_corr.at[reparam.slot].set(
                    reparam.bk_target(bk_level, state) - ss_policy[reparam.slot]
                )
        return bk_corr

    def _delta_of_state(self, state: Array) -> Array:
        """MLP delta as a function of the raw state (feature building +
        MLP + K/F restriction). Split out so the BK pin can take its value
        and JVP at the steady state."""
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

        # K/F restriction: zero delta at named output positions.
        if self.kf_indices:
            mask = (
                jnp.ones(delta.shape[0], dtype=delta.dtype)
                .at[jnp.asarray(self.kf_indices)]
                .set(jnp.asarray(0.0, dtype=delta.dtype))
            )
            delta = delta * mask
        return delta

    def _delta(self, state: Array) -> Array:
        delta = self._delta_of_state(state)

        # BK pin (selection by construction): remove the delta's value AND
        # tangent at the steady state, so pi(s*) = pi* and dpi/ds(s*) = P
        # hold exactly for every parameter value — no training step can
        # unlearn the Blanchard-Kahn selection. The residual objective then
        # only shapes second-order-and-beyond deviations.
        if self.bk_pin:
            ss_state = self.ss_state
            d_ss, d_tan = jax.jvp(
                self._delta_of_state, (ss_state,), (state - ss_state,)
            )
            delta = delta - d_ss - d_tan
        return delta

    def _finish(self, raw: Array, state: Array) -> Array:
        # q = M / 𝓑(x) is recovered before the clip, so the q bounds apply to q.
        if self.q_as_m is not None:
            raw = self.q_as_m.recover(raw, state)

        # Hard clip per-output bounds; the Calvo slots carry their inner
        # terms' domains (set in __init__).
        raw = _apply_hard_clip(raw, self.policy_lower, self.policy_upper)

        # Calvo recoveries from the clipped inner terms; the wage side reads
        # the pi the price side recovers.
        if self.calvo_price is not None:
            raw = self.calvo_price.recover(raw, state)
        if self.calvo_wage is not None:
            raw = self.calvo_wage.recover(raw, state)
        return raw


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
    bk_pin: bool = False,
    zlb_feature_kind: Literal["raw", "kink"] = "raw",
    reparam_q_as_m: bool = False,
    reparam_pi_as_kp_inner: bool = False,
    reparam_wtilda_as_kw_inner: bool = False,
    output_links: Optional[Sequence[str]] = None,
    *,
    key: Array,
) -> DisasterPolicyNet:
    """Factory: build DisasterPolicyNet for the disaster model.

    Resolves the indices and constants each enabled prior reads from
    ``model`` — names from ``model.policy_names`` / ``model.state_names``,
    calibration from ``model.constants`` — before linearizing, and raises
    naming whatever is missing: ``kf_names`` → ``kf_indices``;
    ``use_zlb_feature`` reads state ``R_lag`` and constant ``R_lb``; each
    ``reparam_*`` flag reads what its class in ``reparams.py`` declares.

    All shape priors default off here; a config enables them under
    ``network.options`` (``DisasterNetOptions``, where the K/F restriction
    defaults on).
    """
    policy_names = list(model.policy_names)
    state_names = list(model.state_names) if model.state_names is not None else []

    # Resolve K/F restriction indices from policy_names.
    kf_indices: tuple = ()
    if kf_names:
        missing = [n for n in kf_names if n not in policy_names]
        if missing:
            raise ValueError(
                f"kf_names {missing!r} not found in model.policy_names "
                f"{policy_names!r}; cannot mask those outputs."
            )
        kf_indices = tuple(policy_names.index(n) for n in kf_names)

    zlb_kw: dict = {}
    if use_zlb_feature:
        flag = "use_zlb_feature"
        zlb_kw["r_lag_idx"] = index_of(state_names, "R_lag", "state_names", flag)
        (zlb_kw["r_lb"],) = constants_of(model, flag, "R_lb")

    q_as_m = InvestmentBracket.from_model(model) if reparam_q_as_m else None
    calvo_price = CalvoPrice.from_model(model) if reparam_pi_as_kp_inner else None
    calvo_wage = CalvoWage.from_model(model) if reparam_wtilda_as_kw_inner else None

    return DisasterPolicyNet(
        **anchor_kwargs(model, output_links),
        hidden_sizes=hidden_sizes,
        activation=activation,
        init=init,
        init_scale=init_scale,
        input_shift=input_shift,
        input_scale=input_scale,
        kf_indices=kf_indices,
        use_zlb_feature=use_zlb_feature,
        bk_pin=bk_pin,
        zlb_feature_kind=zlb_feature_kind,
        q_as_m=q_as_m,
        calvo_price=calvo_price,
        calvo_wage=calvo_wage,
        key=key,
        **zlb_kw,
    )


__all__ = ["DisasterPolicyNet", "create_disaster_policy_net"]
