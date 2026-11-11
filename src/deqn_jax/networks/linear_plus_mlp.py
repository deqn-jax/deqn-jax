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

The linear part is the first-order Blanchard-Kahn solution computed by
``deqn_jax.training.linearize.linearize_model``, correct to first order
near SS. The MLP's final layer has zero bias and weights scaled by
``init_scale``, so with ``init_scale=0`` the full policy at initialization
IS the linear policy (in level or log space depending on the per-policy
link). Training learns corrections on top.

This architecture solves a specific PINN pathology observed when
networks are trained directly against equilibrium residuals: a bare
MLP can converge to a degenerate fixed point where policies collapse
to low values, even when that produces dynamics far from the true SS.
A residual parameterization starts training from the linear policy
instead of from an arbitrary network.

The MLP is UNBOUNDED (no sigmoid/softplus on output). Instead, the
linear baseline + small MLP corrections keep policies in the valid
region. Hard clipping to policy bounds happens at the very end to
prevent catastrophic policy outputs during early training.

This module is **model-agnostic**. A model that needs shape priors on top
(a masked delta, an extra input feature, a reparametrized output slot)
subclasses ``LinearPlusMLP`` next to the model and overrides the three
steps of the forward pass (``_bk_term``, ``_delta``, ``_finish``); see
``models/disaster/network.py``.
"""

from typing import Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.networks.common import (
    _apply_hard_clip,
    _apply_output_links,
    _resolve_activation,
    _to_array,
    _to_tuple,
)
from deqn_jax.networks.mlp import MLP

_LINK_CODES = {"linear": 0, "log": 1}


class LinearPlusMLP(eqx.Module):
    """Generic residual ansatz: ``policy = clip(π_BK + δ_θ, lower, upper)``.

    Attributes:
        mlp: Unbounded MLP that outputs the correction (delta) in output
            space. The only trainable part.
        P: Policy rule matrix from linearization [n_policies, n_states], each
            row in its output link's natural space (level for linear rows,
            log for log rows — the factory converts).
        ss_state: Steady-state state vector [n_states].
        ss_policy: Steady-state policy vector [n_policies].
        output_links: per-policy link code, 0 = linear, 1 = log.
        policy_lower / policy_upper: Hard clipping bounds (safety fence; the
            linear + delta should normally keep policies well inside these).

    ``P`` / ``ss_state`` / ``ss_policy`` are architecture constants stored as
    static tuples (the ``networks.common`` convention), outside the parameter
    tree: as array leaves they reach the optimizer, and decoupled weight
    decay shrinks a leaf even at zero gradient.

    The forward pass is three steps a model-specific subclass may override:
    ``_bk_term`` (the linear part), ``_delta`` (the MLP correction) and
    ``_finish`` (what happens to the combined raw policy; here the hard
    clip).
    """

    mlp: MLP
    _P: tuple = eqx.field(static=True)
    _ss_state: tuple = eqx.field(static=True)
    _ss_policy: tuple = eqx.field(static=True)
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
        extra_inputs: int = 0,
        *,
        key: Array,
    ):
        # ``extra_inputs``: features a subclass appends to the state before
        # the MLP. They are not rescaled (shift 0, scale 1).
        if extra_inputs and input_shift is not None:
            input_shift = jnp.concatenate(
                [jnp.asarray(input_shift), jnp.zeros(extra_inputs)]
            )
        if extra_inputs and input_scale is not None:
            input_scale = jnp.concatenate(
                [jnp.asarray(input_scale), jnp.ones(extra_inputs)]
            )

        # UNBOUNDED MLP for the correction: the linear component provides
        # the level, so no sigmoid/softplus on top.
        act_fn = _resolve_activation(activation)
        self.mlp = MLP(  # pyright: ignore[reportAssignmentType]  # ty: ignore[invalid-assignment]
            in_features=n_states + extra_inputs,
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

        # Final layer: bias zeroed, weights scaled by init_scale, so with
        # init_scale=0 the delta is exactly zero at initialization.
        last = self.mlp.layers[-1]
        new_last = eqx.tree_at(
            lambda l: (l.weight, l.bias),
            last,
            (last.weight * init_scale, jnp.zeros_like(last.bias)),
        )
        self.mlp = eqx.tree_at(lambda m: m.layers[-1], self.mlp, new_last)

        self._P = tuple(_to_tuple(row) for row in P)
        self._ss_state = _to_tuple(ss_state)
        self._ss_policy = _to_tuple(ss_policy)
        self.policy_lower = _to_tuple(policy_lower)
        self.policy_upper = _to_tuple(policy_upper)

        if output_links is None:
            self.output_links = (0,) * n_policies
            return
        if len(output_links) != n_policies:
            raise ValueError(
                f"output_links length {len(output_links)} != n_policies {n_policies}"
            )
        try:
            self.output_links = tuple(_LINK_CODES[link] for link in output_links)
        except KeyError as e:
            raise ValueError(
                f"unknown output_link {e.args[0]!r}; valid choices: {list(_LINK_CODES)}"
            ) from None
        bad = [
            i
            for i, code in enumerate(self.output_links)
            if code == 1 and self._ss_policy[i] <= 0.0
        ]
        if bad:
            raise ValueError(
                f"output_link='log' requires ss_policy > 0; non-positive at "
                f"indices {bad} (ss values {[self._ss_policy[i] for i in bad]})"
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

    def _bk_term(self, state: Array) -> Array:
        """Linear part, per row in its link's natural space (level or log)."""
        return self.P @ (state - self.ss_state)

    def _delta(self, state: Array) -> Array:
        """The trainable correction."""
        return self.mlp(state)

    def _finish(self, raw: Array, state: Array) -> Array:
        """Map the combined raw policy to the returned one: the hard clip."""
        return _apply_hard_clip(raw, self.policy_lower, self.policy_upper)

    def _forward_single(self, state: Array) -> Array:
        bk_corr = self._bk_term(state)
        delta = self._delta(state)
        # Linear policies combine additively, log policies multiplicatively;
        # resolved at trace time from the static link tuple.
        raw = _apply_output_links(self.ss_policy, bk_corr, delta, self.output_links)
        return self._finish(raw, state)

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

    With this conversion, the forward computes bk_corr = P @ ds as a single
    matvec without per-row branching.
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


def anchor_kwargs(model, output_links: Optional[Sequence[str]]) -> dict:
    """The constructor arguments a BK-anchored network takes from ``model``:
    sizes, the linearization ``P`` (rows converted per output link), the
    steady state, the links and the policy bounds."""
    from deqn_jax.training.linearize import linearize_model

    if model.steady_state_fn is None:
        raise ValueError(
            "a network anchored at the linear policy requires model.steady_state_fn"
        )
    P, _Q = linearize_model(model, verbose=False)
    ss_state, ss_policy = model.steady_state_fn(model.constants)
    if output_links is not None:
        P = _convert_p_per_link(P, ss_policy, output_links)
    return dict(
        n_states=model.n_states,
        n_policies=model.n_policies,
        P=P,
        ss_state=ss_state,
        ss_policy=ss_policy,
        output_links=output_links,
        policy_lower=model.policy_lower,
        policy_upper=model.policy_upper,
    )


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
        output_links: per-policy parameterization. None defaults to
            all-linear. Each entry must be ``"linear"`` or ``"log"``.
            Log-output policies require ``ss_policy[i] > 0``.
    """
    return LinearPlusMLP(
        **anchor_kwargs(model, output_links),
        hidden_sizes=hidden_sizes,
        activation=activation,
        init=init,
        init_scale=init_scale,
        input_shift=input_shift,
        input_scale=input_scale,
        key=key,
    )
