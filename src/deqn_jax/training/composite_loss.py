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
    loss_fn = make_composite_loss(model, data, anchor_weight=1.0, jac_weight=0.1, ...)
    # loss_fn takes compute_loss's positional/keyword arguments (see
    # composite_loss_fn below for the exact accepted kwargs); the builder
    # in _build_custom_loss_fn passes every CompositeLossConfig knob by name.
"""

from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.training.history import current_states, make_constant_history
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
    # Per-anchor-point weights in [0, 1] (kink-aware anchor gate), or None
    # for the legacy unweighted anchor. Computed ONCE at build time from
    # ModelSpec.anchor_gate_fn when composite_loss.anchor_gate is enabled.
    anchor_weights: Optional[Array] = None


def prepare_composite_data(
    model: ModelSpec,
    P: Array,
    Q: Array,
    n_anchor_points: int = 64,
    anchor_sigma: float = 1.0,
    seed: int = 12345,
    verbose: bool = True,
    anchor_gate_fn: Optional[Callable[..., Array]] = None,
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

    # Kink-aware anchor gate: per-point weights computed ONCE on the fixed
    # cloud (build time, pre-JIT — zero runtime cost). None = legacy path.
    anchor_weights = None
    if anchor_gate_fn is not None:
        anchor_weights = jnp.clip(
            jnp.asarray(
                anchor_gate_fn(anchor_points, anchor_lin_policy, model.constants)
            ).reshape(-1),
            0.0,
            1.0,
        )
        assert anchor_weights.shape[0] == n_anchor_points, (
            f"anchor_gate_fn returned {anchor_weights.shape[0]} weights for "
            f"{n_anchor_points} anchor points"
        )

    if verbose:
        print(f"  Anchor: {n_anchor_points} fixed points, sigma={anchor_sigma}")
        if anchor_weights is not None:
            n_down = int(jnp.sum(anchor_weights < 0.5))
            print(
                f"  Anchor gate: mean weight {float(jnp.mean(anchor_weights)):.3f}, "
                f"{n_down}/{n_anchor_points} points down-weighted (<0.5)"
            )
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
        anchor_weights=anchor_weights,
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
        return policy_fn(make_constant_history(state, history_len))

    return wrapper


def _anchor_loss(
    policy_fn: Callable[[Array], Array],
    data: CompositeData,
    history_len: int = 1,
) -> Array:
    """Anchor loss: ||f_net(x) - f_lin(x)||^2 at pre-sampled points near SS.

    Uses fixed sample points (precomputed in prepare_composite_data) so the
    anchor loss is deterministic — no per-step random sampling noise in gradients.

    When ``data.anchor_weights`` is set (kink-aware anchor gate), each
    point's squared error is scaled by its weight and the mean is taken
    over the weight mass — points where the linearization is the wrong
    local model (e.g. beyond the disaster model's rate floor) stop
    teaching. ``None`` reproduces the legacy unweighted mean exactly.
    """
    markov_fn = _make_markov_wrapper(policy_fn, history_len)
    net_policy = jax.vmap(markov_fn)(data.anchor_points)  # [n_anchor, n_policies]
    sq = (net_policy - data.anchor_lin_policy) ** 2
    if data.anchor_weights is None:
        return jnp.mean(sq)
    w = data.anchor_weights[:, None]
    return jnp.sum(w * sq) / (jnp.sum(w) * sq.shape[1] + 1e-12)


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
    sq = (J_all - data.P[None, :, :]) ** 2
    if data.anchor_weights is None:
        return jnp.mean(sq)
    w = data.anchor_weights[:, None, None]
    return jnp.sum(w * sq) / (
        jnp.sum(data.anchor_weights) * sq.shape[1] * sq.shape[2] + 1e-12
    )


def _drift_loss(
    model: ModelSpec,
    policy_fn: Callable[[Array], Array],
    data: CompositeData,
    probes: Array,
    horizon: int,
    log_target: float,
    history_len: int = 1,
) -> Array:
    """Certificate-in-the-loop stability loss ("the new loss", 2026-07-07).

    Rolls the DETERMINISTIC closed loop (zero shocks) `horizon` steps from
    small ergodic-shaped perturbations of the SS and penalizes the average
    per-period log growth of the relative deviation above ``log_target``.
    For moderate horizons this approximates log rho(SS) — the exact
    certificate the 2026-07-07 disaster experiment showed no
    anchor/sampling treatment could move (shared basin rho = 1.057 ± 0.008
    across five treated runs, while the Frobenius aux_jac term is blind to
    the spectrum of the non-normal closed-loop map). Smooth, eig-free,
    non-normality-robust; hinge is linear above threshold.

    Gradients flow through the policy at every rollout step
    (backprop-through-scan). Probes are FIXED at build time (a priori,
    never tuned). Deviations are measured relative per state dim
    (heterogeneous scales: k ~ 27 vs m_p ~ 0).
    """
    markov_fn = _make_markov_wrapper(policy_fn, history_len)
    ss = data.ss_state
    scale = jnp.abs(ss) + 1e-8
    zero_shock = jnp.zeros((probes.shape[0], model.n_shocks))
    s0 = ss[None, :] + probes  # [n_probes, n_states]

    def _step(s, _):
        pol = jax.vmap(markov_fn)(s)
        return model.step_fn(s, pol, zero_shock, model.constants), None

    sT, _ = jax.lax.scan(_step, s0, None, length=horizon)
    rel0 = jnp.linalg.norm((s0 - ss[None, :]) / scale[None, :], axis=1)
    relT = jnp.linalg.norm((sT - ss[None, :]) / scale[None, :], axis=1)
    g = (jnp.log(relT + 1e-30) - jnp.log(rel0 + 1e-30)) / horizon
    # linear hinge above log_target (softplus with sharpness 50 ≈ max(,0))
    return jnp.mean(jax.nn.softplus(50.0 * (g - log_target)) / 50.0)


def _residual_sobolev_loss(
    model: ModelSpec,
    policy_fn: Callable[[Array], Array],
    states: Array,
    quad_nodes: Array,
    quad_weights: Array,
    shock_scale,
    dirs: Array,
    history_len: int = 1,
) -> Array:
    """Residual-Sobolev loss: ||∇_s E[r]||² along fixed directions ("Simon's
    Sobolev", implemented 2026-07-07).

    The true policy zeroes the expected residual on a NEIGHBORHOOD, so its
    residual-gradient is zero too; an impostor keeps residual VALUES small
    at sampled states while its residual-gradient stays finite. The
    directional derivative d/dt E[r](s + t v) therefore carries selection
    information the value-based (E[r])² loss cannot see — it chains through
    the policy AND dynamics Jacobians, constraining the closed-loop
    eigenstructure through the equations themselves, with no linearization
    oracle and no validity region. Cheap form: forward-mode JVPs along a few
    fixed ergodic-shaped unit directions, on a subsample of the batch.

    v1: Gaussian-quadrature expectations only; single-stage models only
    (equations_fn is called directly). Gated in _validate_train_config.
    """
    markov_fn = _make_markov_wrapper(policy_fn, history_len)
    eq_names = list(model.equation_names)

    def e_resid(s):
        p = markov_fn(s)

        def per_node(eps):
            sb = s[None, :]
            pb = p[None, :]
            s_next = model.step_fn(
                sb, pb, (eps * shock_scale)[None, :], model.constants
            )
            p_next = markov_fn(s_next[0])[None, :]
            r = model.equations_fn(sb, pb, s_next, p_next, model.constants)
            return jnp.stack(
                [
                    jnp.reshape(r[k], ())
                    if jnp.ndim(r[k]) == 0
                    else r[k].reshape(-1)[0]
                    for k in eq_names
                ]
            )

        r_nodes = jax.vmap(per_node)(quad_nodes)  # [n_nodes, n_eq]
        return quad_weights @ r_nodes  # [n_eq]

    def dir_deriv(s, v):
        _, jv = jax.jvp(e_resid, (s,), (v,))
        return jv

    jvs = jax.vmap(lambda s: jax.vmap(lambda v: dir_deriv(s, v))(dirs))(states)
    return jnp.mean(jvs**2)


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
    base_loss_fn: Optional[Callable] = None,
    drift_weight: float = 0.0,
    drift_horizon: int = 20,
    drift_eps: float = 1e-3,
    drift_n_probes: int = 4,
    drift_target: float = 0.99,
    res_sobolev_weight: float = 0.0,
    res_sobolev_n_states: int = 16,
    res_sobolev_n_dirs: int = 2,
) -> Callable:
    """Create composite loss function as drop-in replacement for compute_loss.

    Returns a function with the same signature as compute_loss():
        (model, policy_fn, states, key, mc_samples, weights, shock_scale,
         quad_nodes, quad_weights) -> (total_loss, eq_losses_dict)

    ``base_loss_fn`` (default: plain ``compute_loss``) computes the base
    residual term; it must accept the full compute_loss signature including
    ``loss_choice``/``huber_delta``. Passing the EWM coverage wrapper here
    imposes the residual on the base+stress+local mixture while the
    anchor/jac terms are added ONCE on top (composite ∘ coverage) — the
    reverse order would multiply-count the state-independent anchor terms
    across pools.

    Anchor and Jacobian losses decay with shock_scale but maintain a floor:
        decay = max(floor, 1 - shock_scale)
    During curriculum (shock_scale ramps 0.1 → 1.0), they fade from 90% → floor.
    With floor=0.2, anchor/jac stay active throughout training to prevent
    the network from drifting into degenerate far-from-SS basins.

    Auxiliary loss entries are keyed with "aux_" prefix.
    """
    _base_loss_fn = base_loss_fn if base_loss_fn is not None else compute_loss

    # Drift-loss probes: fixed a priori at build time (deterministic key),
    # ergodic-shaped so heterogeneous state scales are respected. Built only
    # when the term is on — drift_weight == 0 is bit-identical to before.
    _drift_probes = None
    _drift_log_target = 0.0
    if drift_weight > 0.0:
        _dz = jax.random.normal(
            jax.random.PRNGKey(20260707), (drift_n_probes, data.ss_state.shape[0])
        )
        _drift_probes = drift_eps * _dz @ data.ergodic_cov_chol.T
        _drift_log_target = float(jnp.log(drift_target))

    # Residual-Sobolev directions: fixed a priori, ergodic-shaped, unit norm.
    _rsob_dirs = None
    if res_sobolev_weight > 0.0:
        _rq = jax.random.normal(
            jax.random.PRNGKey(20260708),
            (res_sobolev_n_dirs, data.ss_state.shape[0]),
        )
        _rd = _rq @ data.ergodic_cov_chol.T
        _rsob_dirs = _rd / (jnp.linalg.norm(_rd, axis=1, keepdims=True) + 1e-12)

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
        base_loss, eq_losses = _base_loss_fn(
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
            cur_states = current_states(states)
            assert model_.definitions_fn is not None, (
                "composite loss aux hook requires a model with definitions_fn"
            )
            defs_fn_ = model_.definitions_fn
            defs = jax.vmap(
                lambda s: defs_fn_(
                    s, _make_markov_wrapper(policy_fn, history_len)(s), model_.constants
                )
            )(cur_states)

        # 4b. Certificate-in-the-loop drift loss: penalize closed-loop
        # growth from SS probes. No curriculum decay (stability is not a
        # warm-up concern); build-time skip keeps drift_weight=0 exact.
        if _drift_probes is not None:
            drift = _drift_loss(
                model_,
                policy_fn,
                data,
                _drift_probes,
                drift_horizon,
                _drift_log_target,
                history_len=history_len,
            )
            eq_losses["aux_drift"] = drift

        # 4c. Residual-Sobolev: ||directional d/ds E[r]||² on a batch
        # subsample. No curriculum decay; build-time skip when weight 0.
        # Quadrature-only v1 (validated in _validate_train_config).
        if _rsob_dirs is not None:
            _cur = current_states(states)
            rsob = _residual_sobolev_loss(
                model_,
                policy_fn,
                _cur[:res_sobolev_n_states],
                quad_nodes,
                quad_weights,
                shock_scale,
                _rsob_dirs,
                history_len=history_len,
            )
            eq_losses["aux_res_sobolev"] = rsob

        # 5. Weighted total (anchor/jac decay with curriculum)
        total = base_loss
        total = total + aux_decay * anchor_weight * anchor
        total = total + aux_decay * jac_weight * jac
        if jac_anchor_weight > 0.0:
            total = total + aux_decay * jac_anchor_weight * jac_anchor
        if _drift_probes is not None:
            total = total + drift_weight * drift
        if _rsob_dirs is not None:
            total = total + res_sobolev_weight * rsob

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


# ---------------------------------------------------------------------------
# Loss-object builder for train_from_config (moved from trainer.py)
# ---------------------------------------------------------------------------


def _build_custom_loss_fn(config, model: ModelSpec, history_len: int):
    """Build the wrapped loss function for non-default loss configurations.

    Returns the custom loss callable (or None if the default MSE
    `compute_loss` should be used as-is). Handles three layered cases:
    composite loss, state-barrier penalty, and Huber loss for the bare
    path — plus EWM coverage sampling, which wraps plain compute_loss.
    Coverage composes with the composite loss (composite ∘ coverage: the
    base residual term becomes the base+stress+local mixture, anchor/jac
    added once on top); the remaining exclusions are validated in
    _validate_train_config.
    """
    from functools import partial

    cov_enabled = (
        getattr(config, "coverage", None) is not None and config.coverage.enabled
    )
    if cov_enabled and config.loss_type != "composite":
        from deqn_jax.training.coverage import make_coverage_loss

        if config.verbose:
            print("  Coverage sampling: base + stress + local pools")
        return make_coverage_loss(compute_loss, model, config.coverage)

    custom_loss_fn = None
    if config.loss_type == "composite":
        from deqn_jax.training.composite_loss import (
            make_composite_loss,
            prepare_composite_data,
        )
        from deqn_jax.training.linearize import linearize_model

        gate_fn = None
        if config.composite_loss.anchor_gate:
            gate_fn = getattr(model, "anchor_gate_fn", None)
            if gate_fn is None:
                raise ValueError(
                    f"composite_loss.anchor_gate=true but model "
                    f"'{model.name}' declares no anchor_gate_fn. Only models "
                    "with a known linearization-invalid region define one "
                    "(e.g. disaster's interest-rate floor)."
                )

        if config.verbose:
            print("  Building composite loss (linearize + ergodic cov)...")
        P, Q = linearize_model(model, verbose=config.verbose)

        cov_base_loss_fn = None
        if cov_enabled:
            from deqn_jax.training.coverage import make_coverage_loss

            if config.verbose:
                print(
                    "  Coverage sampling (composite base): base + stress + local pools"
                )
            cov_base_loss_fn = make_coverage_loss(compute_loss, model, config.coverage)

        comp_cfg = config.composite_loss
        comp_data = prepare_composite_data(
            model,
            P,
            Q,
            n_anchor_points=comp_cfg.n_anchor_points,
            anchor_sigma=comp_cfg.anchor_sigma,
            seed=config.seed,
            verbose=config.verbose,
            anchor_gate_fn=gate_fn,
        )
        custom_loss_fn = make_composite_loss(
            model,
            comp_data,
            anchor_weight=comp_cfg.anchor_weight,
            jac_weight=comp_cfg.jac_weight,
            jac_anchor_weight=comp_cfg.jac_anchor_weight,
            barrier_weight=comp_cfg.barrier_weight,
            newton_weight=comp_cfg.newton_weight,
            leverage_mult=comp_cfg.leverage_mult,
            aux_decay_floor=comp_cfg.aux_decay_floor,
            history_len=history_len,
            loss_choice=config.loss_choice,
            huber_delta=config.huber_delta,
            base_loss_fn=cov_base_loss_fn,
            drift_weight=comp_cfg.drift_weight,
            drift_horizon=comp_cfg.drift_horizon,
            drift_eps=comp_cfg.drift_eps,
            drift_n_probes=comp_cfg.drift_n_probes,
            drift_target=comp_cfg.drift_target,
            res_sobolev_weight=comp_cfg.res_sobolev_weight,
            res_sobolev_n_states=comp_cfg.res_sobolev_n_states,
            res_sobolev_n_dirs=comp_cfg.res_sobolev_n_dirs,
        )
        if config.verbose:
            extras = []
            if config.loss_choice != "mse":
                extras.append(
                    f"loss_choice={config.loss_choice} (δ={config.huber_delta})"
                )
            if comp_cfg.jac_anchor_weight > 0:
                extras.append(f"sobolev-anchor w={comp_cfg.jac_anchor_weight}")
            extras_str = " · ".join(extras)
            print(
                f"  Composite loss ready.{(' · ' + extras_str) if extras_str else ''}"
            )

    barrier_weight = config.barrier_weight
    if (
        barrier_weight > 0
        and custom_loss_fn is None
        and model.state_barrier_fn is not None
    ):
        custom_loss_fn = partial(
            compute_loss,
            barrier_weight=barrier_weight,
            loss_choice=config.loss_choice,
            huber_delta=config.huber_delta,
        )
        if config.verbose:
            print(f"  State barrier: weight={barrier_weight}")

    if custom_loss_fn is None and config.loss_choice != "mse":
        custom_loss_fn = partial(
            compute_loss,
            loss_choice=config.loss_choice,
            huber_delta=config.huber_delta,
        )
        if config.verbose:
            print(f"  Loss choice: {config.loss_choice} (δ={config.huber_delta})")

    # Moment-matching aux loss layered on top of whatever was chosen above.
    # Uses Dynare's reference moments as the target. See
    # training/moment_loss.py for the design rationale.
    if (
        getattr(config, "moment_matching", None) is not None
        and config.moment_matching.enabled
    ):
        from deqn_jax.dynare_io import deqn_policy_to_dynare, load_dynare_moments
        from deqn_jax.training.moment_loss import (
            _resolve_target_indices,
            make_moment_matching_wrapper,
        )

        mom_cfg = config.moment_matching
        target_moments = load_dynare_moments(mom_cfg.dynare_dir)
        # DEQN ↔ Dynare name aliases (currently just `i` -> `i_var`); reuse
        # the canonical mapping from dynare_io.
        aliases = {p: deqn_policy_to_dynare(p) for p in model.policy_names}
        target_idx = _resolve_target_indices(
            policy_names=list(model.policy_names),
            target_moments=target_moments,
            name_aliases=aliases,
        )
        if config.verbose:
            print(
                f"  Moment-matching aux loss: weight={mom_cfg.weight}, "
                f"matching {len(target_idx)} policies against {mom_cfg.dynare_dir}"
            )
        custom_loss_fn = make_moment_matching_wrapper(
            custom_loss_fn,
            target_idx_to_moments=target_idx,
            weight=mom_cfg.weight,
            mean_weight=mom_cfg.mean_weight,
            std_weight=mom_cfg.std_weight,
            scale_eps=mom_cfg.scale_eps,
        )

    return custom_loss_fn
