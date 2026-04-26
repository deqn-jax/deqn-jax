"""Policy network that pins K/F-style auxiliary outputs to the linearization.

Why this exists: residual-only training on Calvo-Phillips-curve NK-DSGE
models (CMR, disaster) drifts into a wrong-attractor manifold. The
mechanism is gauge freedom in the auxiliary policy outputs that show
up as discounted forward sums (``F_p``, ``F_w``) and marginal-cost
aggregates (``K_p``, ``K_w``). Each appears in only its own definition
and recursion equations; large families of self-consistent ``(F_p,
K_p, F_w, K_w)`` tuples satisfy the residuals at scales far from the
true equilibrium. Diagnostic results on a 22-cell disaster sweep:
across all completed runs, ``K_p_net ≈ 9 × K_p_ss`` (vs. Dynare),
``c_mean ≈ 0.4 × c_dyn``. Residual minimization on its own can't
distinguish these basins.

Fix: don't have the network output the auxiliaries at all. Take the
Blanchard-Kahn linearization (``training/linearize.linearize_model``)
which Dynare also produces analytically, extract the rows for the
named auxiliaries, and pin those four outputs to a linear function of
state:

    K(s) = K_ss + (P_K @ (s - s_ss))     for K in {F_p, K_p, F_w, K_w}

The linearization is correct to first order near SS. Empirically the
auxiliaries are tightly clustered around their SS values even in
nonlinear sims (they're contractive discounted sums), so first-order
is rarely the binding error here. The network outputs the *other*
seven policies through a bounded MLP, free of the gauge.

Forward output is the full 11-element policy vector in
``model.policy_names`` order so the rest of the training and eval
pipeline is untouched.
"""

from __future__ import annotations

from typing import Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.networks.common import _resolve_activation, _to_tuple
from deqn_jax.networks.mlp import MLP


class KfAnchoredMLP(eqx.Module):
    """Inner MLP for ``other`` policies + linearization anchor for K/F.

    Attributes:
        inner_mlp: Bounded MLP outputting the ``other`` (non-K/F) policies.
            Inherits the model's POLICY_LOWER/POLICY_UPPER for those
            indices and the standard sigmoid/softplus bounding from MLP.
        P_kf: ``[n_kf, n_states]`` linearization rows for the anchored
            outputs. Frozen via ``stop_gradient`` at forward time.
        ss_state, ss_kf: SS state and SS values for the anchored outputs.
        n_policies, n_kf: Static dimensions.
        kf_indices, other_indices: Static index tuples — positions of
            anchored / inner-MLP outputs in the assembled policy vector.
        kf_lower, kf_upper: Static lower / upper bounds for the anchored
            outputs (hard clip at forward time, defense against the
            linearization predicting infeasible values far from SS).
    """

    inner_mlp: MLP
    P_kf: Array
    ss_state: Array
    ss_kf: Array
    n_policies: int = eqx.field(static=True)
    n_kf: int = eqx.field(static=True)
    kf_indices: tuple = eqx.field(static=True)
    other_indices: tuple = eqx.field(static=True)
    kf_lower: Optional[tuple] = eqx.field(static=True)
    kf_upper: Optional[tuple] = eqx.field(static=True)

    def __init__(
        self,
        model,
        P: Array,
        ss_state: Array,
        ss_policy: Array,
        hidden_sizes: Sequence[int],
        activation: str,
        kf_names: Sequence[str] = ("F_p", "K_p", "F_w", "K_w"),
        init: str = "default",
        input_shift: Optional[Array] = None,
        input_scale: Optional[Array] = None,
        *,
        key: Array,
    ):
        policy_names = list(model.policy_names)
        # Validate kf_names appear in policy_names; fail loud otherwise.
        missing = [n for n in kf_names if n not in policy_names]
        if missing:
            raise ValueError(
                f"kf_names {missing!r} not found in model.policy_names "
                f"{policy_names!r}; cannot anchor those outputs."
            )

        kf_idx = tuple(policy_names.index(n) for n in kf_names)
        kf_set = set(kf_idx)
        other_idx = tuple(i for i in range(len(policy_names)) if i not in kf_set)

        self.kf_indices = kf_idx
        self.other_indices = other_idx
        self.n_policies = int(model.n_policies)
        self.n_kf = len(kf_idx)

        # Anchor data: SS values + linearization rows for K/F. Stored as
        # JAX arrays (not static tuples) because they participate in the
        # forward as P_kf @ (state - ss_state) — a static tuple would be
        # constant-folded fine in JIT but feeding it through stop_gradient
        # is cleaner via Array storage. The fields aren't trainable: in
        # forward we explicitly stop_gradient them.
        self.ss_state = jnp.asarray(ss_state)
        self.ss_kf = jnp.asarray(jnp.asarray(ss_policy)[jnp.asarray(kf_idx)])
        self.P_kf = jnp.asarray(P[jnp.asarray(kf_idx), :])

        # Bounds for hard clipping the anchored outputs. Linearization can
        # produce values that violate the model's policy bounds for states
        # far from SS; we clip back into the feasible region as a safety
        # fence (these are auxiliary discount sums; they should never go
        # near zero under sane dynamics anyway).
        if model.policy_lower is not None:
            self.kf_lower = tuple(float(model.policy_lower[i]) for i in kf_idx)
        else:
            self.kf_lower = None
        if model.policy_upper is not None:
            self.kf_upper = tuple(float(model.policy_upper[i]) for i in kf_idx)
        else:
            self.kf_upper = None

        # Inner MLP outputs only the non-K/F policies, with their bounds.
        if model.policy_lower is not None:
            other_lower = jnp.array([float(model.policy_lower[i]) for i in other_idx])
        else:
            other_lower = None
        if model.policy_upper is not None:
            other_upper = jnp.array([float(model.policy_upper[i]) for i in other_idx])
        else:
            other_upper = None

        act_fn = _resolve_activation(activation)
        self.inner_mlp = MLP(  # pyright: ignore[reportAssignmentType]  # ty: ignore[invalid-assignment]
            in_features=int(model.n_states),
            out_features=len(other_idx),
            hidden_sizes=hidden_sizes,
            activations=[act_fn] * len(hidden_sizes),
            output_lower=other_lower,
            output_upper=other_upper,
            input_shift=input_shift,
            input_scale=input_scale,
            init=init,
            key=key,
        )

    def _forward_single(self, state: Array) -> Array:
        seven = self.inner_mlp(state)

        # K/F from linear anchor (frozen).
        ss_state = jax.lax.stop_gradient(self.ss_state)
        ss_kf = jax.lax.stop_gradient(self.ss_kf)
        P_kf = jax.lax.stop_gradient(self.P_kf)
        kf = ss_kf + P_kf @ (state - ss_state)

        # Hard clip to bounds. Bounds are static tuples of floats; convert
        # to arrays for math (XLA constant-folds inside JIT).
        if self.kf_lower is not None:
            lower = jnp.asarray(self.kf_lower)
            kf = jnp.maximum(kf, lower + 1e-4)
        if self.kf_upper is not None:
            upper = jnp.asarray(self.kf_upper)
            safe_upper = jnp.where(jnp.isinf(upper), jnp.array(1e10), upper)
            kf = jnp.minimum(kf, safe_upper - 1e-4)

        # Assemble in policy_names order.
        kf_dtype_aligned = kf.astype(seven.dtype)
        out = jnp.zeros(self.n_policies, dtype=seven.dtype)
        out = out.at[jnp.asarray(self.other_indices)].set(seven)
        out = out.at[jnp.asarray(self.kf_indices)].set(kf_dtype_aligned)
        return out

    def __call__(self, x: Array) -> Array:
        if x.ndim == 1:
            return self._forward_single(x)
        return jax.vmap(self._forward_single)(x)


def create_kf_anchored_mlp(
    model,
    hidden_sizes: Sequence[int] = (64, 64),
    activation: str = "tanh",
    init: str = "default",
    kf_names: Sequence[str] = ("F_p", "K_p", "F_w", "K_w"),
    input_shift: Optional[Array] = None,
    input_scale: Optional[Array] = None,
    *,
    key: Array,
) -> KfAnchoredMLP:
    """Factory: build a KfAnchoredMLP using the model's linearization.

    Computes the Blanchard-Kahn linearization in-house via
    ``training/linearize.linearize_model`` (no Dynare files required),
    then pins the rows corresponding to ``kf_names`` to the resulting
    P matrix. The remaining policies come from a fresh MLP.
    """
    from deqn_jax.training.linearize import linearize_model

    if model.steady_state_fn is None:
        raise ValueError(
            "network.type='kf_anchored_mlp' requires model.steady_state_fn "
            "to compute the linearization anchor."
        )
    P, _Q = linearize_model(model, verbose=False)
    ss_state, ss_policy = model.steady_state_fn(model.constants)

    return KfAnchoredMLP(
        model=model,
        P=P,
        ss_state=ss_state,
        ss_policy=ss_policy,
        hidden_sizes=hidden_sizes,
        activation=activation,
        kf_names=kf_names,
        init=init,
        input_shift=input_shift,
        input_scale=input_scale,
        key=key,
    )


__all__ = ["KfAnchoredMLP", "create_kf_anchored_mlp"]


# Mark unused import so ruff is happy if the auxiliary _to_tuple helper
# turns out unneeded after final refactoring.
_ = _to_tuple
