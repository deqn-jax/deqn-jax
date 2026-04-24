"""Residual policy: analytically-added linear policy + MLP correction.

    policy(state) = ss_policy + P @ (state - ss_state) + delta_mlp(state)

The linear part is the Dynare-linearized (Blanchard-Kahn) solution,
which solves the model to first order and is correct by construction
near SS. The MLP starts at zero (via a scaled final layer) so at
initialization the full policy IS the linear policy. Training only
learns corrections on top.

This architecture solves a specific PINN pathology observed in the
disaster model: a bare MLP trained against equation residuals
converges to a degenerate fixed point where policies collapse to
low values, even when that produces dynamics far from the true SS.
A residual parameterization inherits the linear policy's correctness
as a floor — the network can only help, never hurt.

The MLP is UNBOUNDED (no sigmoid/softplus on output). Instead, the
linear baseline + small MLP corrections keep policies in the valid
region. Hard clipping to policy bounds happens at the very end to
prevent catastrophic policy outputs during early training.
"""

from typing import Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.networks.common import _resolve_activation
from deqn_jax.networks.mlp import MLP


class LinearPlusMLP(eqx.Module):
    """Policy = linear(state) + mlp_correction(state[, regime_feature]).

    Attributes:
        mlp: Unbounded MLP that outputs the correction (delta) in output space.
        P: Policy rule matrix from linearization [n_policies, n_states], static.
        ss_state: Steady-state state vector [n_states], static.
        ss_policy: Steady-state policy vector [n_policies], static.
        policy_lower / policy_upper: Hard clipping bounds (safety fence; the
            linear + delta should normally keep policies well inside these).
        use_zlb_feature: If True, prepend (state[r_lag_idx] - r_lb) as an
            extra MLP input so the delta can learn ELB-regime-dependent
            corrections. Linear part is unchanged (it operates on raw state).
        r_lag_idx / r_lb: Index into the state vector for the lagged interest
            rate, and the ELB floor. Disaster-model defaults (5 and 1.0).
    """

    mlp: MLP
    # P, ss_state, ss_policy are fixed throughout training but participate
    # in the forward pass as JAX arrays (not static): marking them static
    # would cache them as pytree metadata and require stop_gradient.
    P: Array
    ss_state: Array
    ss_policy: Array
    policy_lower: Optional[Array]
    policy_upper: Optional[Array]
    use_zlb_feature: bool = eqx.field(static=True)
    r_lag_idx: int = eqx.field(static=True)
    r_lb: float = eqx.field(static=True)

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
        use_zlb_feature: bool = False,
        r_lag_idx: int = 5,
        r_lb: float = 1.0,
        *,
        key: Array,
    ):
        self.use_zlb_feature = bool(use_zlb_feature)
        self.r_lag_idx = int(r_lag_idx)
        self.r_lb = float(r_lb)
        n_extra = 1 if self.use_zlb_feature else 0

        # When the ZLB regime feature is on, the MLP's input dimension grows
        # by 1. Pad any caller-supplied input_shift/input_scale so they
        # match (shift 0, scale 1 on the extra dim — no rescaling applied
        # to the regime coordinate, which is already O(1e-2)).
        if n_extra > 0 and input_shift is not None:
            input_shift = jnp.concatenate([jnp.asarray(input_shift), jnp.zeros(n_extra)])
        if n_extra > 0 and input_scale is not None:
            input_scale = jnp.concatenate([jnp.asarray(input_scale), jnp.ones(n_extra)])

        # UNBOUNDED MLP for the correction. The linear component provides
        # the "raw" output, so we don't want sigmoid/softplus on top.
        act_fn = _resolve_activation(activation)
        self.mlp = MLP(
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

        # Shrink final layer so delta ≈ 0 at initialization: the policy IS
        # the linear policy at init, so training only moves away from it
        # to the extent that doing so reduces residuals.
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
        self.policy_lower = (
            jnp.asarray(policy_lower) if policy_lower is not None else None
        )
        self.policy_upper = (
            jnp.asarray(policy_upper) if policy_upper is not None else None
        )

    def _forward_single(self, state: Array) -> Array:
        # stop_gradient on the linearization constants: they are frozen
        # parameters of the architecture, not trainable. Matches the
        # pattern used for input_shift/scale and output bounds in MLP.
        ss_state = jax.lax.stop_gradient(self.ss_state)
        ss_policy = jax.lax.stop_gradient(self.ss_policy)
        P = jax.lax.stop_gradient(self.P)
        linear = ss_policy + P @ (state - ss_state)

        if self.use_zlb_feature:
            # Regime feature: distance of lagged rate from the ELB. Tells
            # the delta MLP how close we are to the kink so it can learn
            # regime-specific corrections. Linear part still operates on
            # raw state only.
            zlb_prox = state[self.r_lag_idx] - self.r_lb
            mlp_input = jnp.concatenate([state, jnp.array([zlb_prox])])
        else:
            mlp_input = state

        delta = self.mlp(mlp_input)
        raw = linear + delta
        if self.policy_lower is not None:
            lower = jax.lax.stop_gradient(self.policy_lower)
            raw = jnp.maximum(raw, lower)
        if self.policy_upper is not None:
            upper = jax.lax.stop_gradient(self.policy_upper)
            safe_upper = jnp.where(jnp.isinf(upper), jnp.array(1e10), upper)
            raw = jnp.minimum(raw, safe_upper)
        return raw

    def __call__(self, x: Array) -> Array:
        if x.ndim == 1:
            return self._forward_single(x)
        return jax.vmap(self._forward_single)(x)


def create_linear_plus_mlp(
    model,
    hidden_sizes: Sequence[int] = (128, 128),
    activation: str = "tanh",
    init: str = "default",
    init_scale: float = 0.01,
    input_shift: Optional[Array] = None,
    input_scale: Optional[Array] = None,
    use_zlb_feature: bool = False,
    *,
    key: Array,
) -> LinearPlusMLP:
    """Factory: build a LinearPlusMLP using the model's linearization.

    When ``use_zlb_feature`` is set and the model exposes ``R_lb`` and an
    ``R_lag`` state, the MLP receives an extra input (R_lag - R_lb) so it
    can learn regime-dependent corrections near the ELB.
    """
    from deqn_jax.training.linearize import linearize_model

    P, _Q = linearize_model(model, verbose=False)
    ss_state, ss_policy = model.steady_state_fn(model.constants)

    # Disaster-specific regime-feature setup. Other models can set
    # use_zlb_feature=False (the default) and this is inert.
    r_lag_idx = 5
    r_lb = 1.0
    if use_zlb_feature:
        if model.state_names is not None and "R_lag" in model.state_names:
            r_lag_idx = list(model.state_names).index("R_lag")
        r_lb = float(model.constants.get("R_lb", 1.0))

    return LinearPlusMLP(
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
        use_zlb_feature=use_zlb_feature,
        r_lag_idx=r_lag_idx,
        r_lb=r_lb,
        key=key,
    )
