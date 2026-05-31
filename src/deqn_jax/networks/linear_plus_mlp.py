"""Residual policy: analytically-added linear policy + MLP correction.

Two output parameterizations are supported, selectable per-policy via
``output_links``:

  - ``"linear"`` (default, original behavior):
        policy_i(s) = ss_i + P_i^level @ (s - ss_state) + delta_mlp_i(s)

  - ``"log"`` (positive-multiplicative around SS):
        policy_i(s) = ss_i * exp(P_i^log @ (s - ss_state) + delta_mlp_i(s))

    where P_i^log = P_i^level / ss_i (delta-method conversion of the BK
    matrix). At init (init_scale=0) and at s=ss_state, both forms reduce
    to ss_i exactly. The log form bakes in positivity and is the natural
    parameterization for log-deviations-from-SS, which is the standard
    DSGE convention (cf. Dynare's log-linearized solutions).

The linear part is the Dynare-linearized (Blanchard-Kahn) solution,
which solves the model to first order and is correct by construction
near SS. The MLP starts at zero (via a scaled final layer) so at
initialization the full policy IS the linear policy (in level or log
space depending on the per-policy link). Training only learns
corrections on top.

This architecture solves a specific PINN pathology observed when
networks are trained directly against equilibrium residuals: a bare
MLP can converge to a degenerate fixed point where policies collapse
to low values, even when that produces dynamics far from the true SS.
A residual parameterization inherits the linear policy's correctness
as a floor — the network can only help, never hurt.

The MLP is UNBOUNDED (no sigmoid/softplus on output). Instead, the
linear baseline + small MLP corrections keep policies in the valid
region. Hard clipping to policy bounds happens at the very end to
prevent catastrophic policy outputs during early training.

This module is **model-agnostic**. Per-model shape priors (e.g. K/F
gauge masking, ELB feature augmentation, output-space reparameterizations
to encode equation-specific curvature) live in the model's own
``network.py`` module, not here. See ``models/disaster/network.py`` for
the disaster-specific shape priors layered on top of this core.
"""

from typing import Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.networks.common import _resolve_activation, _to_array, _to_tuple
from deqn_jax.networks.mlp import MLP


class LinearPlusMLP(eqx.Module):
    """Generic residual ansatz: ``policy = clip(π_BK + δ_θ, lower, upper)``.

    Attributes:
        mlp: Unbounded MLP that outputs the correction (delta) in output space.
        P: Policy rule matrix from linearization [n_policies, n_states].
        ss_state: Steady-state state vector [n_states].
        ss_policy: Steady-state policy vector [n_policies].
        policy_lower / policy_upper: Hard clipping bounds (safety fence; the
            linear + delta should normally keep policies well inside these).

    The forward pass is intentionally minimal: linearization, MLP delta,
    sum, hard clip. Per-model shape priors should be implemented in a
    model-specific subclass or wrapper that lives next to the model
    definition, not bolted onto this generic core.
    """

    mlp: MLP
    # P, ss_state, ss_policy are fixed throughout training but participate
    # in the forward pass as JAX arrays (not static): marking them static
    # would cache them as pytree metadata and require stop_gradient.
    # P is stored *per-policy in the appropriate space*: row i is in level
    # units for linear-output policies, in log units for log-output policies
    # (i.e. log-row i = level-row i / ss_i). Conversion happens in the
    # factory so _forward_single doesn't branch on link type for the BK row.
    P: Array
    ss_state: Array
    ss_policy: Array
    # output_links: tuple of ints [n_policies], 0=linear, 1=log. Stored as
    # Python tuple for true staticness (mirrors policy_lower/upper pattern);
    # converted to a JAX array in _forward_single.
    output_links: tuple = eqx.field(static=True)
    policy_lower: Optional[tuple] = eqx.field(static=True)
    policy_upper: Optional[tuple] = eqx.field(static=True)

    def __init__(
        self,
        n_states: int,
        n_policies: int,
        hidden_sizes: Sequence[int],
        activation: str,
        P: Array,
        ss_state: Array,
        ss_policy: Array,
        output_links: Optional[Sequence[str]] = None,
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
        # eqx-typing: MLP() is typed as returning Module (the base class);
        # the runtime is MLP and the field annotation is correct.
        act_fn = _resolve_activation(activation)
        self.mlp = MLP(  # pyright: ignore[reportAssignmentType]  # ty: ignore[invalid-assignment]
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
            # Log-link requires positive ss_policy at that slot
            ss_arr = jnp.asarray(ss_policy)
            bad_idxs = [
                i
                for i, code in enumerate(link_codes_tuple)
                if code == 1 and float(ss_arr[i]) <= 0.0
            ]
            if bad_idxs:
                raise ValueError(
                    f"output_link='log' requires ss_policy > 0; non-positive at "
                    f"indices {bad_idxs} (ss values {[float(ss_arr[i]) for i in bad_idxs]})"
                )
        self.output_links = link_codes_tuple
        self.policy_lower = _to_tuple(policy_lower)
        self.policy_upper = _to_tuple(policy_upper)

    def _forward_single(self, state: Array) -> Array:
        # stop_gradient on the linearization constants: they are frozen
        # parameters of the architecture, not trainable. Matches the
        # pattern used for input_shift/scale and output bounds in MLP.
        ss_state = jax.lax.stop_gradient(self.ss_state)
        ss_policy = jax.lax.stop_gradient(self.ss_policy)
        P = jax.lax.stop_gradient(self.P)
        bk_corr = P @ (state - ss_state)  # in per-row natural space (level or log)

        delta = self.mlp(state)
        # Per-policy: linear policies use additive, log policies use
        # multiplicative. Resolve at Python time from the static link tuple:
        # if all-linear (common case) skip exp(); if all-log skip where().
        # Mixed case computes both and selects via jnp.where.
        if all(code == 0 for code in self.output_links):
            raw = ss_policy + bk_corr + delta
        elif all(code == 1 for code in self.output_links):
            raw = ss_policy * jnp.exp(bk_corr + delta)
        else:
            is_log = jnp.asarray(self.output_links, dtype=jnp.int8) == 1
            raw_linear = ss_policy + bk_corr + delta
            raw_log = ss_policy * jnp.exp(bk_corr + delta)
            raw = jnp.where(is_log, raw_log, raw_linear)

        if self.policy_lower is not None:
            lower = jax.lax.stop_gradient(_to_array(self.policy_lower))
            raw = jnp.maximum(raw, lower)
        if self.policy_upper is not None:
            upper = jax.lax.stop_gradient(_to_array(self.policy_upper))
            safe_upper = jnp.where(jnp.isinf(upper), jnp.array(1e10), upper)
            raw = jnp.minimum(raw, safe_upper)
        return raw

    def __call__(self, x: Array) -> Array:
        if x.ndim == 1:
            return self._forward_single(x)
        return jax.vmap(self._forward_single)(x)


def _convert_p_per_link(
    P_level: Array, ss_policy: Array, output_links: Sequence[str]
) -> Array:
    """Pre-convert each row of P to its output-link's natural space.

    log-link row i: P_log_i = P_level_i / ss_policy_i  (delta-method)
    linear-link row i: keep P_level_i as-is

    With this conversion, _forward_single can compute bk_corr = P @ ds as
    a single matvec without per-row branching.
    """
    n_pol = P_level.shape[0]
    if len(output_links) != n_pol:
        raise ValueError(
            f"output_links length {len(output_links)} != n_policies {n_pol}"
        )
    is_log = jnp.array([link == "log" for link in output_links], dtype=bool)
    ss = jnp.asarray(ss_policy)
    # Per-row scale factor: 1/ss for log rows, 1.0 for linear rows
    scale = jnp.where(is_log, 1.0 / (ss + 1e-12), 1.0)
    return jnp.asarray(P_level) * scale[:, None]


def create_linear_plus_mlp(
    model,
    hidden_sizes: Sequence[int] = (128, 128),
    activation: str = "tanh",
    init: str = "default",
    init_scale: float = 0.01,
    input_shift: Optional[Array] = None,
    input_scale: Optional[Array] = None,
    output_links: Optional[Sequence[str]] = None,
    *,
    key: Array,
) -> LinearPlusMLP:
    """Factory: build a generic LinearPlusMLP using the model's linearization.

    Args:
        output_links: per-policy parameterization. None defaults to all-linear
            (legacy behavior). Each entry must be ``"linear"`` or ``"log"``.
            Log-output policies require ``ss_policy[i] > 0``.

    This factory has NO model-specific knobs. For disaster-specific shape
    priors (K/F gauge mask, ELB feature, q-as-M reparameterization), use
    ``create_disaster_policy_net`` from ``deqn_jax.models.disaster.network``.
    """
    from deqn_jax.training.linearize import linearize_model

    P_level, _Q = linearize_model(model, verbose=False)
    ss_state, ss_policy = model.steady_state_fn(model.constants)

    if output_links is None:
        # Backward-compat: pass P unchanged, link defaults to all-linear inside __init__.
        P = P_level
    else:
        P = _convert_p_per_link(P_level, ss_policy, output_links)

    return LinearPlusMLP(
        n_states=model.n_states,
        n_policies=model.n_policies,
        hidden_sizes=hidden_sizes,
        activation=activation,
        P=P,
        ss_state=ss_state,
        ss_policy=ss_policy,
        output_links=output_links,
        policy_lower=model.policy_lower,
        policy_upper=model.policy_upper,
        init=init,
        init_scale=init_scale,
        input_shift=input_shift,
        input_scale=input_scale,
        key=key,
    )
