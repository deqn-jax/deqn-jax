"""Disaster-specific policy network: shape priors layered on the residual ansatz.

The generic ``LinearPlusMLP`` in ``networks/linear_plus_mlp.py`` provides
``π = clip(π_BK + δ_θ, lower, upper)`` and nothing else. The disaster model
needs additional shape encodings to handle pathological residual functions
that smooth approximators struggle with (vertical asymptotes in the Calvo
bracket, sign flips in the investment Euler, sharp kinks at the ELB). Those
encodings are model-specific and live here, not in the generic library
module.

Shape priors and selection devices, each independently toggleable (the
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
from deqn_jax.networks.common import (
    _apply_hard_clip,
    _apply_output_links,
    _resolve_activation,
    _to_array,
    _to_tuple,
)
from deqn_jax.networks.mlp import MLP


class DisasterPolicyNet(eqx.Module):
    """LinearPlusMLP residual ansatz augmented with disaster-specific shape priors.

    Forward pass (each step optional, controlled by configuration):

      1. ``bk = P @ (state - ss_state)``                        (always)
      2. replace the BK term at the ``pi`` / ``w_tilda`` / ``q`` slots by
         its K_p_inner / K_w_inner / M value     (if the matching reparam)
      3. ``δ = mlp(augment_with_zlb_feature(state))``  (feature if use_zlb_feature)
      4. ``δ = mask_kf(δ)``                                    (if kf_indices)
      5. ``δ = δ - δ(ss_state) - Jδ(ss_state) (state - ss_state)``  (if bk_pin)
      6. ``raw = combine(ss_policy, bk, δ)`` per output link  (always)
      7. ``raw = recover_q_from_m(raw, state)``               (if reparam_q_as_m)
      8. ``policy = clip(raw, lower, upper)``                  (always)
      9. recover ``pi`` / ``K_p`` and ``w_tilda`` / ``K_w`` from the clipped
         K_p_inner / K_w_inner                  (if the matching reparam)

    With no prior enabled, the result is the generic LinearPlusMLP ansatz.
    """

    # Generic ansatz components (mirror LinearPlusMLP). The BK anchors are
    # architecture constants stored as static tuples (the ``networks.common``
    # convention), outside the parameter tree: as array leaves they reached
    # the optimizer, and decoupled weight decay shrinks a leaf even at zero
    # gradient, which voids the BK pin. Read them through the ``P`` /
    # ``ss_state`` / ``ss_policy`` properties.
    mlp: MLP
    _P: tuple = eqx.field(static=True)
    _ss_state: tuple = eqx.field(static=True)
    _ss_policy: tuple = eqx.field(static=True)
    # Per-policy output parameterization (mirror of LinearPlusMLP). Tuple of
    # ints, 0=linear, 1=log. Reparam slots are forced to linear and
    # validated in __init__ — log + reparam at the same slot is incoherent
    # (the reparam logic operates in level space).
    output_links: tuple = eqx.field(static=True)
    policy_lower: Optional[tuple] = eqx.field(static=True)
    policy_upper: Optional[tuple] = eqx.field(static=True)

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
        # ELB feature augmentation: extra MLP input dim if enabled.
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

        # Final-layer scaling: the bias is zeroed and the weights scaled by
        # init_scale, so with init_scale=0 the MLP output is exactly zero at
        # training step 0 (and, with no reparam enabled, policy = π_BK).
        last = self.mlp.layers[-1]
        scaled_w = last.weight * init_scale
        zero_b = jnp.zeros_like(last.bias)
        new_last = eqx.tree_at(
            lambda l: (l.weight, l.bias),
            last,
            (scaled_w, zero_b),
        )
        self.mlp = eqx.tree_at(lambda m: m.layers[-1], self.mlp, new_last)

        self._P = tuple(_to_tuple(row) for row in P)
        self._ss_state = _to_tuple(ss_state)
        self._ss_policy = _to_tuple(ss_policy)
        self.policy_lower = _to_tuple(policy_lower)
        self.policy_upper = _to_tuple(policy_upper)

        # K/F restriction: empty tuple disables the mask entirely.
        kf_idx_tuple = tuple(int(i) for i in kf_indices)
        for i in kf_idx_tuple:
            if not (0 <= i < n_policies):
                raise ValueError(
                    f"kf_indices entry {i} out of range for n_policies={n_policies}"
                )
        self.kf_indices = kf_idx_tuple

        # Output-slot reparametrizations: each repurposed slot's clip bounds
        # become the domain of the quantity it carries (Calvo slots).
        self.q_as_m = q_as_m
        self.calvo_price = calvo_price
        self.calvo_wage = calvo_wage
        for reparam in self._reparams():
            reparam.check_indices(n_states, n_policies)
            self.policy_lower, self.policy_upper = reparam.override_bounds(
                self.policy_lower, self.policy_upper, self.ss_state, n_policies
            )

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
            for reparam in self._reparams():
                if link_codes_tuple[reparam.slot] == 1:
                    raise ValueError(
                        f"{reparam.flag}=True forces output_links[{reparam.slot}]="
                        f"'linear' (the reparam logic operates in level space); "
                        f"got 'log'"
                    )
        self.output_links = link_codes_tuple

    def _reparams(self) -> tuple:
        """The enabled reparametrizations, in the order the forward applies
        them (the wage side reads the pi the price side recovers)."""
        return tuple(
            r for r in (self.calvo_price, self.calvo_wage, self.q_as_m) if r is not None
        )

    @property
    def P(self) -> Array:
        """BK policy slope, per row in its output link's natural space."""
        return _to_array(self._P)

    @property
    def ss_state(self) -> Array:
        return _to_array(self._ss_state)

    @property
    def ss_policy(self) -> Array:
        return _to_array(self._ss_policy)

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

    def _forward_single(self, state: Array) -> Array:
        # BK linear part. ``self.P`` is stored *per-row in natural space*
        # (level for linear/reparam slots, log for log slots — conversion
        # happens in the factory). ``bk_corr[i]`` is therefore in the
        # natural unit of policy i. For linear slots: ss_i + bk_corr[i]
        # gives the BK level. For log slots: exp(bk_corr[i]) is the
        # multiplicative deviation, so ss_i * exp(bk_corr[i]) gives BK.
        ss_state = self.ss_state
        ss_policy = self.ss_policy
        P = self.P
        bk_corr = P @ (state - ss_state)

        # Reparametrized slots: replace the BK term by the BK policy in the
        # slot's reparametrized space, so ss + bk_corr there is K_p_inner_BK /
        # K_w_inner_BK / M_BK. Each reads the BK *level* captured before any
        # substitution (reparam slots are forced linear-link).
        reparams = self._reparams()
        if reparams:
            bk_level = ss_policy + bk_corr
            for reparam in reparams:
                bk_corr = bk_corr.at[reparam.slot].set(
                    reparam.bk_target(bk_level, state) - ss_policy[reparam.slot]
                )

        delta = self._delta_of_state(state)

        # BK pin (selection by construction): remove the delta's value AND
        # tangent at the steady state, so pi(s*) = pi* and dpi/ds(s*) = P
        # hold exactly for every parameter value — no training step can
        # unlearn the Blanchard-Kahn selection. The residual objective then
        # only shapes second-order-and-beyond deviations.
        if self.bk_pin:
            d_ss, d_tan = jax.jvp(
                self._delta_of_state, (ss_state,), (state - ss_state,)
            )
            delta = delta - d_ss - d_tan

        # Per-policy raw assembly. Reparam slots are forced linear-link, so
        # the additive form recovers the level-space K_inner / M values
        # injected via bk_corr above. Log-link slots use multiplicative form
        # for natural positivity + log-deviation parameterization.
        raw = _apply_output_links(ss_policy, bk_corr, delta, self.output_links)

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

    All shape priors default off; configs enable them by setting the
    relevant fields on ``NetworkConfig``.
    """
    from deqn_jax.training.linearize import linearize_model

    if model.steady_state_fn is None:
        raise ValueError(
            "DisasterPolicyNet requires model.steady_state_fn to compute "
            "the linearization anchor."
        )

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

    P_level, _Q = linearize_model(model, verbose=False)
    ss_state, ss_policy = model.steady_state_fn(model.constants)

    # Pre-convert P per-row to its output_link's natural space (mirror of
    # create_linear_plus_mlp). For log-link rows: P_log_i = P_level_i / ss_i.
    if output_links is None:
        P = P_level
    else:
        from deqn_jax.networks.linear_plus_mlp import _convert_p_per_link

        P = _convert_p_per_link(P_level, ss_policy, output_links)

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
        bk_pin=bk_pin,
        zlb_feature_kind=zlb_feature_kind,
        q_as_m=q_as_m,
        calvo_price=calvo_price,
        calvo_wage=calvo_wage,
        output_links=output_links,
        key=key,
        **zlb_kw,
    )


__all__ = ["DisasterPolicyNet", "create_disaster_policy_net"]
