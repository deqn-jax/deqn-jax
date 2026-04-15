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

from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.networks.common import ACTIVATION_FNS, INIT_FNS, _apply_init, _resolve_activation
from deqn_jax.networks.mlp import MLP


class LinearPlusMLP(eqx.Module):
    """Policy = linear(state) + mlp_correction(state).

    Attributes:
        mlp: Unbounded MLP that outputs the correction (delta) in output space.
        P: Policy rule matrix from linearization [n_policies, n_states], static.
        ss_state: Steady-state state vector [n_states], static.
        ss_policy: Steady-state policy vector [n_policies], static.
        policy_lower / policy_upper: Hard clipping bounds (safety fence; the
            linear + delta should normally keep policies well inside these).
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
        *,
        key: Array,
    ):
        # UNBOUNDED MLP for the correction. The linear component provides
        # the "raw" output, so we don't want sigmoid/softplus on top.
        act_fn = _resolve_activation(activation)
        self.mlp = MLP(
            in_features=n_states,
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

        delta = self.mlp(state)
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
    *,
    key: Array,
) -> LinearPlusMLP:
    """Factory: build a LinearPlusMLP using the model's linearization."""
    from deqn_jax.training.linearize import linearize_model

    P, _Q = linearize_model(model, verbose=False)
    ss_state, ss_policy = model.steady_state_fn(model.constants)

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
        key=key,
    )
