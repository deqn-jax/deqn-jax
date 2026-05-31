"""Composite loss: anchor + Jacobian + Sobolev-anchor + model-supplied aux.

Drop-in replacement for compute_loss() — returns the same (total_loss, eq_losses_dict)
signature, with auxiliary losses keyed with "aux_" prefix so adaptive reweighting
and per-equation gradient surgery only see the base equilibrium residuals.

The generic terms here are MODEL-AGNOSTIC:

- ``aux_anchor``      = ||π_θ(s) − π_BK(s)||² at sampled-near-SS points
- ``aux_jac``         = ||J_π_θ(s_ss) − P||²_F
- ``aux_jac_anchor``  = same as aux_jac but at every anchor point (Sobolev)

Per-model auxiliary terms (e.g. economic-feasibility barriers, Newton-solver
diagnostics) flow through ``ModelSpec.composite_aux_fn``. The hook receives
the per-batch ``defs`` dict, the precomputed ``CompositeData``, and a
``weights`` dict containing every weight knob the trainer was given (so the
hook can pick the ones it cares about, e.g. ``barrier_weight``,
``leverage_mult``, ``newton_weight``). See ``models/disaster/composite_aux.py``
for the canonical pattern (BGG net-worth barrier, leverage barrier,
consumption barrier, Newton-conditioning diagnostics).

Usage:
    data = prepare_composite_data(model, P, Q)
    loss_fn = make_composite_loss(model, data, config.composite_loss)
    # loss_fn has the same signature as compute_loss
"""

from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.training.loss import compute_loss
from deqn_jax.types import ModelSpec


class CompositeData(NamedTuple):
    """Pre-computed linearization data for composite loss terms.

    Attributes:
        P: Policy rule matrix [n_policies, n_states] from Blanchard-Kahn
        ss_state: Steady state [n_states]
        ss_policy: Steady state policy [n_policies]
        ergodic_cov_chol: Cholesky of ergodic covariance [n_states, n_states]
        anchor_points: Pre-sampled states near SS [n_anchor, n_states]
        anchor_deviations: anchor_points - ss_state [n_anchor, n_states]
        anchor_lin_policy: Linear policy at anchor points [n_anchor, n_policies]
        aux_constants: Generic dict for model-specific precomputed constants
            (e.g. disaster's ss_leverage). Populated by the model's
            ``composite_aux_fn`` (or left empty when the model declares no
            aux terms). Read by the same hook at loss-evaluation time.
    """

    P: Array
    ss_state: Array
    ss_policy: Array
    ergodic_cov_chol: Array
    anchor_points: Array
    anchor_deviations: Array
    anchor_lin_policy: Array
    aux_constants: Dict[str, Any]


def prepare_composite_data(
    model: ModelSpec,
    P: Array,
    Q: Array,
    n_anchor_points: int = 64,
    anchor_sigma: float = 1.0,
    seed: int = 12345,
    verbose: bool = True,
) -> CompositeData:
    """Build CompositeData from linearization results.

    Pre-computes anchor sample points from the ergodic distribution so the
    anchor loss is deterministic (no per-step randomness = no gradient noise).

    Args:
        model: Model specification
        P: Policy rule matrix from linearize_model
        Q: Transition matrix from linearize_model
        n_anchor_points: Number of fixed sample points near SS
        anchor_sigma: Scale factor for sampling spread
        seed: RNG seed for anchor point sampling
        verbose: Print diagnostic info
    """
    from deqn_jax.training.linearize import compute_ergodic_covariance

    assert model.steady_state_fn is not None, (
        "composite loss requires a model with steady_state_fn defined "
        "(needed for linearization + ergodic covariance)"
    )
    ss_state, ss_policy = model.steady_state_fn(model.constants)
    ergodic_cov = compute_ergodic_covariance(Q, model, verbose=verbose)

    # Cholesky with regularization for numerical stability
    n = ergodic_cov.shape[0]
    ergodic_cov_chol = jnp.linalg.cholesky(ergodic_cov + 1e-8 * jnp.eye(n))

    # Pre-sample anchor points: x = ss + sigma * L @ z, z ~ N(0, I)
    key = jax.random.PRNGKey(seed)
    z = jax.random.normal(key, (n_anchor_points, ss_state.shape[0]))
    deviations = anchor_sigma * z @ ergodic_cov_chol.T
    anchor_points = ss_state + deviations
    anchor_lin_policy = ss_policy + deviations @ P.T

    # Per-model precomputed constants for the aux hook (barrier thresholds,
    # SS reference values, etc). Models opt in by setting
    # ``ModelSpec.composite_aux_constants_fn``; default empty.
    aux_constants: Dict[str, Any] = {}
    aux_const_fn = getattr(model, "composite_aux_constants_fn", None)
    if aux_const_fn is not None:
        aux_constants = dict(aux_const_fn(model))

    if verbose:
        print(f"  Anchor: {n_anchor_points} fixed points, sigma={anchor_sigma}")
        if aux_constants:
            print(f"  Aux constants: {list(aux_constants.keys())}")

    return CompositeData(
        P=P,
        ss_state=ss_state,
        ss_policy=ss_policy,
        ergodic_cov_chol=ergodic_cov_chol,
        anchor_points=anchor_points,
        anchor_deviations=deviations,
        anchor_lin_policy=anchor_lin_policy,
        aux_constants=aux_constants,
    )


def _make_markov_wrapper(
    policy_fn: Callable[[Array], Array],
    history_len: int,
) -> Callable[[Array], Array]:
    """Wrap a sequence policy to accept plain state vectors.

    For MLP (history_len=1), returns policy_fn unchanged.
    For LSTM/Transformer (history_len>1), tiles the state into a constant-history
    window [H, n_states] so the policy can be called on a single state vector.
    """
    if history_len <= 1:
        return policy_fn

    def wrapper(state: Array) -> Array:
        # state: [n_states] -> [H, n_states] constant window
        window = jnp.broadcast_to(state, (history_len, state.shape[-1]))
        return policy_fn(window)

    return wrapper


def _anchor_loss(
    policy_fn: Callable[[Array], Array],
    data: CompositeData,
    history_len: int = 1,
) -> Array:
    """Anchor loss: ||f_net(x) - f_lin(x)||^2 at pre-sampled points near SS.

    Uses fixed sample points (precomputed in prepare_composite_data) so the
    anchor loss is deterministic — no per-step random sampling noise in gradients.
    """
    markov_fn = _make_markov_wrapper(policy_fn, history_len)
    net_policy = jax.vmap(markov_fn)(data.anchor_points)  # [n_anchor, n_policies]
    return jnp.mean((net_policy - data.anchor_lin_policy) ** 2)


def _jac_loss(
    policy_fn: Callable[[Array], Array],
    data: CompositeData,
    history_len: int = 1,
) -> Array:
    """Jacobian loss: ||J_net(ss) - P||^2_F.

    Penalizes deviation of the neural network Jacobian at the steady state
    from the linearized policy rule matrix P. This ensures the net has
    the correct first-order response to state perturbations.
    """
    markov_fn = _make_markov_wrapper(policy_fn, history_len)
    # Jacobian of net at SS: [n_policies, n_states]
    J_net = jax.jacfwd(markov_fn)(data.ss_state)
    return jnp.mean((J_net - data.P) ** 2)


def _sobolev_anchor_loss(
    policy_fn: Callable[[Array], Array],
    data: CompositeData,
    history_len: int = 1,
) -> Array:
    """Sobolev-style anchor loss: ||J_net(x_i) - P||² averaged over anchors.

    Generalises ``_jac_loss`` from the single steady-state point to every
    anchor point. Matches the first-order behaviour of the network to the
    Blanchard-Kahn P matrix across a whole neighbourhood of SS, not only
    at SS itself. Roughly d× more information per anchor than value-only
    matching (where d = n_states), and it disciplines the network's
    directional response in every local frame.

    Reference: Czarnecki et al. "Sobolev Training for Neural Networks"
    (NeurIPS 2017). The P matrix is treated as a constant target; only
    the per-anchor Jacobians vary.
    """
    markov_fn = _make_markov_wrapper(policy_fn, history_len)
    jac_single = jax.jacfwd(markov_fn)
    # Jacobians at every anchor: [n_anchor, n_policies, n_states]
    J_all = jax.vmap(jac_single)(data.anchor_points)
    return jnp.mean((J_all - data.P[None, :, :]) ** 2)


def make_composite_loss(
    model: ModelSpec,
    data: CompositeData,
    anchor_weight: float = 0.1,
    jac_weight: float = 0.01,
    jac_anchor_weight: float = 0.0,
    barrier_weight: float = 0.01,
    newton_weight: float = 0.01,
    leverage_mult: float = 5.0,
    aux_decay_floor: float = 0.2,
    history_len: int = 1,
    loss_choice: str = "mse",
    huber_delta: float = 1.0,
) -> Callable:
    """Create composite loss function as drop-in replacement for compute_loss.

    Returns a function with the same signature as compute_loss():
        (model, policy_fn, states, key, mc_samples, weights, shock_scale,
         quad_nodes, quad_weights) -> (total_loss, eq_losses_dict)

    Anchor and Jacobian losses decay with shock_scale but maintain a floor:
        decay = max(floor, 1 - shock_scale)
    During curriculum (shock_scale ramps 0.1 → 1.0), they fade from 90% → floor.
    With floor=0.2, anchor/jac stay active throughout training to prevent
    the network from drifting into degenerate far-from-SS basins.

    Auxiliary loss entries are keyed with "aux_" prefix.
    """

    def composite_loss_fn(
        model_: ModelSpec,
        policy_fn: Callable[[Array], Array],
        states: Array,
        key: Array,
        mc_samples: int = 5,
        weights: Optional[Array] = None,
        shock_scale: float = 1.0,
        quad_nodes: Optional[Array] = None,
        quad_weights: Optional[Array] = None,
        target_policy_fn: Optional[Callable[[Array], Array]] = None,
    ) -> Tuple[Array, Dict[str, Array]]:
        # NOTE: barrier_weight is NOT a parameter here. It's captured from
        # the enclosing make_composite_loss closure (line above in the
        # signature). An earlier version shadowed the closure var with a
        # barrier_weight=0.0 default, which silently dropped the configured
        # barrier weight from composite training. Do not reintroduce it
        # as a parameter here -- the trainer does not thread it through.
        # 1. Base residual loss — MSE or Huber on per-state mean residual.
        base_loss, eq_losses = compute_loss(
            model_,
            policy_fn,
            states,
            key,
            mc_samples,
            weights=weights,
            shock_scale=shock_scale,
            quad_nodes=quad_nodes,
            quad_weights=quad_weights,
            target_policy_fn=target_policy_fn,
            loss_choice=loss_choice,
            huber_delta=huber_delta,
        )

        # Anchor + jac decay: fade as curriculum progresses, but keep a floor
        # shock_scale may be a vector [n_shocks] when shock_mask is active; use mean
        _ss = jnp.mean(shock_scale) if jnp.ndim(shock_scale) > 0 else shock_scale
        aux_decay = jnp.maximum(aux_decay_floor, 1.0 - _ss)

        # 2. Anchor loss: net should match linearized policy near SS
        anchor = _anchor_loss(policy_fn, data, history_len=history_len)
        eq_losses["aux_anchor"] = anchor

        # 3. Jacobian loss: net Jacobian at SS should match P
        jac = _jac_loss(policy_fn, data, history_len=history_len)
        eq_losses["aux_jac"] = jac

        # 3b. Sobolev-anchor loss: match J_net(x_i) ≈ P at EVERY anchor
        # point (not just SS). Disabled by default (weight=0); enable by
        # setting composite_loss.jac_anchor_weight > 0. More expensive
        # than aux_jac (one jacfwd per anchor, vmap'd).
        if jac_anchor_weight > 0.0:
            jac_anchor = _sobolev_anchor_loss(policy_fn, data, history_len=history_len)
            eq_losses["aux_jac_anchor"] = jac_anchor
        else:
            jac_anchor = jnp.array(0.0)

        # 4. Per-batch defs for the model-specific aux hook (barriers,
        # Newton diagnostics, etc.). Only computed when the model declares
        # an aux hook — generic-only models skip the vmap entirely.
        # TODO: redundant vmap — base loss already evaluates definitions() internally.
        # Fixing this requires changing compute_loss to return intermediate defs.
        defs = None
        if model_.composite_aux_fn is not None:
            current_states = states[:, -1, :] if states.ndim == 3 else states
            assert model_.definitions_fn is not None, (
                "composite loss aux hook requires a model with definitions_fn"
            )
            defs_fn_ = model_.definitions_fn
            defs = jax.vmap(
                lambda s: defs_fn_(
                    s, _make_markov_wrapper(policy_fn, history_len)(s), model_.constants
                )
            )(current_states)

        # 5. Weighted total (anchor/jac decay with curriculum)
        total = base_loss
        total = total + aux_decay * anchor_weight * anchor
        total = total + aux_decay * jac_weight * jac
        if jac_anchor_weight > 0.0:
            total = total + aux_decay * jac_anchor_weight * jac_anchor

        # Model-specific auxiliary terms (barriers, Newton diagnostics, etc).
        # Hook applies its own weighting via ``weights``; generic side just
        # threads every weight through so models can opt in to whichever it
        # cares about.
        if model_.composite_aux_fn is not None:
            aux_entries, aux_total = model_.composite_aux_fn(
                model_,
                defs,
                data,
                {
                    "newton_weight": newton_weight,
                    "barrier_weight": barrier_weight,
                    "leverage_mult": leverage_mult,
                },
            )
            eq_losses.update(aux_entries)
            total = total + aux_total

        return total, eq_losses

    return composite_loss_fn
