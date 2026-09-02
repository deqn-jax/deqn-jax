"""Pre-loop construction for DEQN-JAX training.

Everything built BEFORE the per-episode loop runs, extracted from trainer.py so
the orchestrator stays readable:

- ``create_train_state`` : policy net + optimizer + initial TrainState
- ``make_train_step``    : the single-JIT train-step dispatcher (5 variants)
- ``_validate_train_config`` / ``_resolve_model_for_training`` : config + model
  validation that doesn't / does depend on the loaded model
- ``_build_initial_state`` : resume-or-build-fresh + optional warm start

Pure move (no logic changes); trainer.py re-imports these under the same names
so ``from deqn_jax.training.trainer import create_train_state`` etc. keep working.
"""

import os
from typing import Any, Callable, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.networks.factory import build_policy_net
from deqn_jax.optimizers.gauss_newton import make_grad_step_gn as _make_grad_step_gn
from deqn_jax.optimizers.lbfgs import make_grad_step_lbfgs as _make_grad_step_lbfgs
from deqn_jax.optimizers.mao import make_grad_step_mao as _make_grad_step_mao
from deqn_jax.optimizers.pcgrad import make_grad_step_pcgrad as _make_grad_step_pcgrad
from deqn_jax.optimizers.registry import (
    OptimizerKind,
    create_optimizer,
    get_optimizer_kind,
)
from deqn_jax.optimizers.standard import (
    make_grad_step_standard as _make_grad_step_standard,
)
from deqn_jax.training.checkpointing import resume_from as _resume_from_checkpoint
from deqn_jax.training.cycle import make_cycle_step as _make_cycle_step
from deqn_jax.training.cycle import make_rollout_fn as _make_rollout_fn
from deqn_jax.training.episode import sample_initial_states
from deqn_jax.training.history import get_history_len, make_constant_history
from deqn_jax.types import ModelSpec, TrainState, make_reweight_state


def create_train_state(
    model: ModelSpec,
    key: Array,
    hidden_sizes: Tuple[int, ...] = (64, 64),
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    optimizer: str = "adam",
    grad_clip: Optional[float] = None,
    loss_weights: Optional[List[float]] = None,
    n_equations: int = 1,
    optimizer_config=None,
    network_config=None,
    sim_batch: Optional[int] = None,
    replay_config=None,
    surrogate_config=None,
) -> Tuple[TrainState, Any, OptimizerKind]:
    """Initialize training state and optimizer.

    Args:
        model: Model specification
        key: PRNG key
        hidden_sizes: MLP hidden layer sizes
        learning_rate: Optimizer learning rate
        batch_size: Batch size for states
        optimizer: Optimizer name (used if optimizer_config is None)
        grad_clip: Global gradient clipping norm
        loss_weights: Manual per-equation weights
        n_equations: Number of equations
        optimizer_config: OptimizerConfig (if provided, overrides optimizer/learning_rate/grad_clip)
        network_config: NetworkConfig (if provided, overrides hidden_sizes and adds activations/init)

    Returns:
        Tuple of (TrainState, optimizer, OptimizerKind)
    """
    key, net_key, state_key = jax.random.split(key, 3)

    policy_net = build_policy_net(model, net_key, hidden_sizes, network_config)

    # Create optimizer via registry or legacy path
    if optimizer_config is not None:
        opt, kind = create_optimizer(optimizer_config)
    else:
        # Legacy path: build OptimizerConfig from individual args
        from deqn_jax.config import OptimizerConfig

        opt_cfg = OptimizerConfig(
            name=optimizer,
            learning_rate=learning_rate,
            grad_clip=grad_clip,
        )
        opt, kind = create_optimizer(opt_cfg)

    # Resolve MAO factory and init optimizer state
    if kind == OptimizerKind.MAO:
        if hasattr(opt, "with_num_tasks"):
            opt = opt.with_num_tasks(n_equations)
        opt_state = opt.init(eqx.filter(policy_net, eqx.is_array))
    elif kind == OptimizerKind.GN:
        opt_state = opt.init(eqx.filter(policy_net, eqx.is_array))
    else:
        opt_state = opt.init(eqx.filter(policy_net, eqx.is_array))

    # Sample initial states. When sim_batch is set, the rollout carries
    # sim_batch parallel trajectories; otherwise fall back to batch_size
    # (so trajectory count == minibatch size).
    n_sim = sim_batch if sim_batch is not None else batch_size
    init_states = sample_initial_states(model, state_key, n_sim)

    # Loss weights
    if loss_weights is not None:
        weights = jnp.array(loss_weights)
    else:
        weights = jnp.ones(n_equations)

    # Seed the history window for sequence policies (LSTM/Transformer).
    # For MLP (history_len=1) keep history_state=None -- the rollout
    # path never touches it in that case. make_constant_history tiles
    # init_states across the time axis so the first rollout sees a
    # well-defined but uninformative prefix; subsequent rollouts persist
    # the actual final window via TrainState.history_state.
    from deqn_jax.training.history import get_history_len, make_constant_history

    hist_len = get_history_len(policy_net)
    if hist_len > 1:
        init_history = make_constant_history(init_states, hist_len)
    else:
        init_history = None

    if replay_config is not None and getattr(replay_config, "enabled", False):
        from deqn_jax.types import make_replay_state

        replay_state = make_replay_state(replay_config.capacity, model.n_states)
    else:
        replay_state = None

    # EWM world arm: Ŵ + its optimizer state live in aux_params; the Polyak
    # target policy the anchor targets are computed at lives in target_params.
    aux_params = None
    target_params = None
    if surrogate_config is not None and getattr(surrogate_config, "enabled", False):
        from deqn_jax.training.surrogate import init_surrogate

        key, sur_key = jax.random.split(key)
        lr = (
            optimizer_config.learning_rate
            if optimizer_config is not None
            else learning_rate
        )
        aux_params, _ = init_surrogate(model, surrogate_config, sur_key, lr)
        target_params = policy_net

    state = TrainState(
        params=policy_net,
        opt_state=opt_state,
        episode_state=init_states,
        key=key,
        step=0,
        episode=0,
        loss_weights=weights,
        reweight_state=make_reweight_state(n_equations),
        target_params=target_params,
        aux_params=aux_params,
        history_state=init_history,
        replay_state=replay_state,
    )

    return state, opt, kind


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def make_train_step(
    model: ModelSpec,
    opt: Any,
    episode_length: int,
    mc_samples: int,
    batch_size: int,
    loss_reweight: str = "none",
    reweight_alpha: float = 0.9,
    kind: OptimizerKind = OptimizerKind.STANDARD,
    gradient_surgery: str = "none",
    grad_clip: Optional[float] = None,
    quad_nodes: Optional[Array] = None,
    quad_weights: Optional[Array] = None,
    history_len: int = 1,
    compute_loss_fn: Optional[Callable] = None,
    ss_reset_frac: float = 0.0,
    use_target_network: bool = False,
    n_epochs_per_rollout: int = 1,
    n_minibatches_per_epoch: Optional[int] = None,
    initialize_each_episode: bool = False,
    sorted_within_batch: bool = False,
    replay_cfg: Any = None,
    surrogate_cfg: Any = None,
    total_episodes: int = 1,
    world_lr: Optional[float] = None,
):
    """Create a JIT-compiled training step function.

    Dispatches to the correct step variant based on OptimizerKind.

    Args:
        model: Model specification
        opt: Optimizer (optax or MAOTransform)
        episode_length: Steps per episode
        mc_samples: MC samples for loss
        batch_size: Batch size
        loss_reweight: Adaptive strategy
        reweight_alpha: EMA decay
        kind: OptimizerKind determining step variant
        gradient_surgery: "none" or "pcgrad"
        grad_clip: Global norm clipping for MAO (STANDARD handles via optax.chain)
        quad_nodes: Quadrature nodes [n_nodes, shock_dim] (None -> use MC)
        quad_weights: Quadrature weights [n_nodes] (None -> use MC)
        history_len: History window size (1=MLP, >1=LSTM/Transformer)
        compute_loss_fn: Optional custom loss function (e.g. composite loss)

    Returns:
        JIT-compiled train_step function
    """
    # All kinds now use the DEQN-style rollout + minibatch-sweep cycle.
    # Per outer iteration: 1 rollout (fills state_episode) + n_epochs ×
    # n_minibatches gradient updates over the full trajectory. The kind
    # determines only the per-batch grad step (standard / pcgrad / mao /
    # lbfgs / gn); the rollout + sweep wrapper is shared.
    rollout_fn = _make_rollout_fn(
        model,
        episode_length,
        history_len,
        ss_reset_frac,
        initialize_each_episode=initialize_each_episode,
    )

    if gradient_surgery == "pcgrad" and kind == OptimizerKind.STANDARD:
        grad_step = _make_grad_step_pcgrad(
            model,
            opt,
            mc_samples,
            quad_nodes,
            quad_weights,
            loss_reweight,
            reweight_alpha,
            use_target_network,
            compute_loss_fn,
        )
    elif kind == OptimizerKind.MAO:
        grad_step = _make_grad_step_mao(
            model,
            opt,
            mc_samples,
            quad_nodes,
            quad_weights,
            loss_reweight,
            reweight_alpha,
            use_target_network,
            compute_loss_fn,
            grad_clip,
        )
    elif kind == OptimizerKind.LBFGS:
        grad_step = _make_grad_step_lbfgs(
            model,
            opt,
            mc_samples,
            quad_nodes,
            quad_weights,
            loss_reweight,
            reweight_alpha,
            use_target_network,
            compute_loss_fn,
        )
    elif kind == OptimizerKind.GN:
        grad_step = _make_grad_step_gn(
            model,
            opt,
            mc_samples,
            batch_size,
            quad_nodes,
            quad_weights,
            loss_reweight,
            reweight_alpha,
            use_target_network,
            compute_loss_fn,
        )
    else:
        grad_step = _make_grad_step_standard(
            model,
            opt,
            mc_samples,
            quad_nodes,
            quad_weights,
            loss_reweight,
            reweight_alpha,
            use_target_network,
            compute_loss_fn,
        )

    pre_sweep_hook = None
    if surrogate_cfg is not None and getattr(surrogate_cfg, "enabled", False):
        # EWM world arm: once per cycle, Polyak-update the target policy, fit
        # Ŵ on anchors from this cycle's dataset (exact E[inside] at the
        # target), and store it back — all outside the per-minibatch JIT.
        from deqn_jax.training.surrogate import (
            make_world_update,
            polyak_update,
            world_optimizer,
        )

        w_opt = world_optimizer(surrogate_cfg, world_lr or 1e-3)
        world_update = make_world_update(
            model,
            surrogate_cfg,
            w_opt,
            mc_samples,
            quad_nodes,
            quad_weights,
            total_episodes,
            batch_size,
        )
        tau = float(surrogate_cfg.polyak_tau)

        def pre_sweep_hook(state, dataset, lr_scale):
            target = polyak_update(state.target_params, state.params, tau)
            hook_key, new_key = jax.random.split(state.key)
            aux = world_update(
                state.aux_params,
                target,
                dataset,
                hook_key,
                lr_scale,
                int(state.episode),
            )
            return state._replace(target_params=target, aux_params=aux, key=new_key)

    return _make_cycle_step(
        rollout_fn=rollout_fn,
        grad_step=grad_step,
        model=model,
        batch_size=batch_size,
        n_epochs_per_rollout=n_epochs_per_rollout,
        n_minibatches_per_epoch=n_minibatches_per_epoch,
        history_len=history_len,
        sorted_within_batch=sorted_within_batch,
        replay_cfg=replay_cfg,
        pre_sweep_hook=pre_sweep_hook,
    )


# ---------------------------------------------------------------------------
# train_from_config setup helpers
# ---------------------------------------------------------------------------


def _validate_train_config(config) -> None:
    """Validate config invariants that don't depend on the loaded model.

    Currently: fp64 toggle + composite-loss/optimizer-combo gate +
    episode_length=1 / initialize_each_episode requirement.
    """
    if config.fp64 and not jax.config.read("jax_enable_x64"):
        jax.config.update("jax_enable_x64", True)

    # Reject composite loss combined with optimizers whose update paths
    # only see base-equation gradients. Ordered first because this is
    # the more specific / silent-correctness class of mistake. LBFGS is
    # deliberately NOT in this set: its grad step differentiates
    # `compute_loss_fn or compute_loss` (optimizers/lbfgs.py), so a
    # composite/custom loss genuinely reaches its gradient and line search.
    if config.loss_type == "composite":
        # PCGrad × composite is supported since 2026-07-07: the PCGrad step
        # projects only the per-equation core gradients and adds the exact
        # auxiliary gradient grad(total) − grad(base) unprojected
        # (optimizers/pcgrad.py), so the aux stack genuinely reaches the
        # update. MAO/GN/LM/IGN still differentiate only the base residual
        # vector — the aux terms would appear in logs but not in updates.
        _bad_opts = {"mao", "lm", "gn", "ign"}
        _opt_name = config.optimizer.name.lower()
        if _opt_name in _bad_opts:
            raise ValueError(
                f"loss_type='composite' is not supported with optimizer "
                f"'{config.optimizer.name}'. Composite auxiliary losses "
                "(anchor, Jacobian, barriers, Newton) would appear in logs "
                "but not affect parameter updates on this path. Use optimizer "
                "'adam'/'sgd'/'adamw'/'lion'/'muon'/'ngd'/'shampoo' (the "
                "STANDARD variant, with or without gradient_surgery='pcgrad') "
                "or 'lbfgs', or switch to loss_type='mse'."
            )
        # PCGrad×composite reconstructs the projected core from UNWEIGHTED
        # per-equation gradients at mean scale; a common non-unit weight
        # would scale the base inside grad(total)-grad(base) but not the
        # projected core, silently breaking the no-conflict equivalence
        # with the standard composite step (2026-07-10 referee finding).
        if (
            config.gradient_surgery == "pcgrad"
            and config.loss_weights is not None
            and any(w != 1.0 for w in config.loss_weights)
        ):
            raise ValueError(
                "gradient_surgery='pcgrad' with loss_type='composite' "
                "requires unit loss_weights (or none): the surgery path "
                "reconstructs the core gradient unweighted, so non-unit "
                "weights would be silently inconsistent between the core "
                "and auxiliary terms."
            )

    # PCGrad is only wired for the STANDARD grad-step variant (the dispatch
    # in make_train_step requires kind == STANDARD); with any other optimizer
    # the setting silently does nothing. Reject unconditionally rather than
    # only when some other ignored feature happens to be configured too.
    if config.gradient_surgery == "pcgrad" and config.optimizer.name.lower() in {
        "mao",
        "lm",
        "gn",
        "ign",
        "lbfgs",
    }:
        raise ValueError(
            f"gradient_surgery='pcgrad' has no effect with optimizer "
            f"'{config.optimizer.name}': PCGrad is only wired for the STANDARD "
            "grad-step variant, so the setting would be silently ignored. Use "
            "a STANDARD optimizer (adam/sgd/adamw/lion/muon/ngd/shampoo) or "
            "set gradient_surgery='none'."
        )

    # Top-level barrier_weight is consumed by the bare-MSE loss builder only;
    # under loss_type='composite' the composite loss is built first and the
    # top-level knob is never forwarded (composite has its own, DIFFERENT
    # composite_loss.barrier_weight knob for the model aux hook). Reject
    # instead of silently dropping it.
    if config.loss_type == "composite" and config.barrier_weight > 0:
        raise ValueError(
            "barrier_weight>0 is silently ignored under loss_type='composite' "
            "(the composite loss never forwards the top-level knob). Use "
            "composite_loss.barrier_weight for the composite aux hook, or "
            "loss_type='mse' for the state-barrier penalty."
        )

    # grad_clip is chained into the optax pipeline for STANDARD kinds and
    # forwarded to MAO, but the LBFGS and GN/LM/IGN update paths never apply
    # it. Reject an explicitly-set value rather than silently dropping it.
    if config.optimizer.grad_clip is not None and config.optimizer.name.lower() in {
        "lbfgs",
        "gn",
        "lm",
        "ign",
    }:
        raise ValueError(
            f"optimizer.grad_clip is not applied on the '{config.optimizer.name}' "
            "update path (clipping is chained only for STANDARD optimizers and "
            "MAO). Remove grad_clip or use a STANDARD optimizer."
        )

    # Reject weighting / custom-loss features combined with optimizers whose
    # update paths only see base, UNWEIGHTED MSE residuals (PCGrad differentiates
    # the raw per-equation vector; MAO passes weights=None; GN builds a raw
    # residual vector). These options would appear in logs/config but never
    # affect parameter updates -- the same silent-correctness class as the
    # composite gate above (audit JAX-SILENT-02/03). LBFGS is deliberately NOT
    # in this set: it consumes weights=state.loss_weights, runs
    # update_reweighting, and differentiates any custom loss fn.
    _bad_opts = {"mao", "lm", "gn", "ign"}
    _opt_name = config.optimizer.name.lower()
    _is_pcgrad = config.gradient_surgery == "pcgrad"
    if _opt_name in _bad_opts or _is_pcgrad:
        _ignored = []
        if config.loss_weights is not None and len(set(config.loss_weights)) > 1:
            _ignored.append("loss_weights (non-uniform)")
        if config.loss_reweight != "none":
            _ignored.append(f"loss_reweight='{config.loss_reweight}'")
        if config.loss_choice != "mse":
            _ignored.append(f"loss_choice='{config.loss_choice}'")
        if config.barrier_weight > 0:
            _ignored.append("barrier_weight>0")
        if config.moment_matching.enabled:
            _ignored.append("moment_matching.enabled")
        if _ignored:
            _surgery = " + gradient_surgery='pcgrad'" if _is_pcgrad else ""
            raise ValueError(
                f"optimizer '{config.optimizer.name}'{_surgery} ignores these "
                f"configured options on its update path: {', '.join(_ignored)}. "
                "They appear in logs/config but do NOT affect parameter updates "
                "(PCGrad/MAO/GN/IGN/LM update from base, unweighted MSE "
                "residuals). Use a STANDARD optimizer (adam/sgd/adamw/lion/muon/"
                "ngd/shampoo with gradient_surgery='none') or 'lbfgs' to use "
                "these options, or remove them."
            )

    if config.coverage.enabled:
        # coverage × composite COMPOSES since 2026-07-06: the composite
        # loss's base residual term becomes the coverage mixture
        # (composite ∘ coverage in _build_custom_loss_fn); anchor/jac
        # terms are added once on top. The v1 mutual exclusion is gone.
        _cov_bad = {"mao", "lm", "gn", "ign", "lbfgs"}
        _cov_pcgrad = config.gradient_surgery == "pcgrad"
        if config.optimizer.name.lower() in _cov_bad or _cov_pcgrad:
            raise ValueError(
                f"coverage.enabled requires a STANDARD optimizer; "
                f"'{config.optimizer.name}'"
                + (" + gradient_surgery='pcgrad'" if _cov_pcgrad else "")
                + " differentiates the per-equation/residual vector, so the "
                "stress/local pools (folded into the scalar total) would be "
                "silently dropped from the gradient. Use adam/sgd/adamw/lion/"
                "muon/ngd/shampoo."
            )
        if config.loss_type != "composite" and (
            config.barrier_weight > 0 or config.loss_choice != "mse"
        ):
            # Under loss_type='mse' the coverage wrapper replaces the
            # partial() that would thread these through; under composite
            # they thread through make_composite_loss (barrier_weight>0 is
            # separately rejected for composite above).
            raise ValueError(
                "coverage.enabled wraps plain MSE compute_loss when "
                "loss_type='mse'; disable barrier_weight / loss_choice!='mse' "
                "(they would be silently dropped on the coverage path)."
            )
        if config.moment_matching.enabled:
            raise ValueError(
                "coverage.enabled with moment_matching.enabled is untested; "
                "disable one."
            )
        if config.network.history_len > 1:
            raise NotImplementedError(
                "coverage.enabled is v1-only-MLP. Sequence networks "
                "(network.history_len > 1) train on [batch, H, n_states] "
                "history windows, but the stress/local pools are flat "
                "[n, n_states] states -- pool construction for windows is a "
                "follow-up. Disable coverage or use an MLP."
            )
        if config.replay_buffer.enabled:
            raise ValueError(
                "coverage.enabled is incompatible with replay_buffer.enabled "
                "in v1: the buffer concatenates old-policy states into the "
                "batch, muddying the base-pool semantics of the coverage "
                "mixture. Disable one."
            )

    if getattr(config, "surrogate", None) is not None and config.surrogate.enabled:
        # EWM world arm: Ŵ replaces E[inside_fn] in the policy update. It
        # needs the two-stage hooks, a STANDARD optimizer (the surrogate
        # loss is a scalar-total wrapper), plain MSE, and — unless the
        # paper's ablation is explicitly requested — coverage on.
        sur = config.surrogate
        if (
            get_optimizer_kind(config.optimizer.name) != OptimizerKind.STANDARD
            or config.gradient_surgery == "pcgrad"
        ):
            raise ValueError(
                "surrogate.enabled requires a STANDARD optimizer without "
                "gradient surgery (the world-arm loss is a scalar-total wrapper; "
                "per-equation/residual-vector optimizers would silently drop Ŵ)."
            )
        if not sur.exact_in_coverage:
            raise NotImplementedError(
                "surrogate.exact_in_coverage=false: Ŵ is fitted on anchors drawn "
                "from the path batch only (v1), so it has no support on the "
                "stress/local pools; scoring them with Ŵ would extrapolate. Keep "
                "the coverage pools exact."
            )
        if config.loss_type != "mse" or config.loss_choice != "mse":
            raise ValueError(
                "surrogate.enabled supports loss_type='mse' with loss_choice='mse' "
                "only (v1); composite/huber/aio would be silently bypassed."
            )
        if not config.coverage.enabled and not sur.allow_without_coverage:
            raise ValueError(
                "surrogate.enabled without coverage.enabled is the paper's ablation; "
                "set surrogate.allow_without_coverage=true to run it deliberately."
            )
        if config.target_update_every > 0:
            raise ValueError(
                "surrogate.enabled owns TrainState.target_params (Polyak target "
                "policy); disable target_update_every."
            )
        if config.network.history_len > 1:
            raise NotImplementedError("surrogate.enabled is v1-only-MLP.")
        if sur.polyak_tau < 0.95:
            import warnings

            warnings.warn(
                f"surrogate.polyak_tau={sur.polyak_tau} < 0.95: the target policy "
                "chases the live policy and Ŵ can diverge (reference: 0.90 diverges).",
                stacklevel=2,
            )

    if (
        config.loss_type == "composite"
        and config.composite_loss.res_sobolev_weight > 0
        and config.expectation_type not in ("quadrature", "gh", "gauss_hermite")
    ):
        raise ValueError(
            "composite_loss.res_sobolev_weight > 0 requires quadrature "
            "expectations in v1 (the residual-Sobolev term rebuilds the "
            "per-state expected residual from quadrature nodes). Set "
            "expectation_type: gauss_hermite."
        )

    if config.episode_length == 1 and not config.initialize_each_episode:
        raise ValueError(
            "episode_length=1 requires initialize_each_episode=True. "
            "With T=1 and no re-initialization the cycle re-seeds from "
            "trajectory[-1] = s_0 and the state never advances between "
            "cycles; training collapses to a single-state regression. "
            "If you want fresh uniform-from-init draws each cycle, set "
            "initialize_each_episode: true. If you want rollout-based "
            "training, use episode_length > 1."
        )

    if config.replay_buffer.enabled:
        # Sequence networks deferred to v2 (see replay.py module docstring).
        if config.network.history_len > 1:
            raise NotImplementedError(
                "replay_buffer.enabled=true is v1-only-MLP. Sequence networks "
                "(network.history_len > 1) need a [capacity, H, n_states] "
                "buffer shape — follow-up. Disable replay or use an MLP."
            )
        if config.sorted_within_batch:
            raise ValueError(
                "replay_buffer.enabled=true is incompatible with "
                "sorted_within_batch=true: buffer rows break the trajectory-"
                "contiguous-chunk semantics that sorted_within_batch relies "
                "on. Disable one."
            )


def _resolve_model_for_training(config) -> Tuple[ModelSpec, int]:
    """Load the model, validate sizes, apply constants override and setup_fn.

    Returns ``(model, n_equations)``. Done as one helper because the
    validation steps depend on the loaded model and we want to apply
    all model-side adaptations (constants override, setup_fn) before
    computing ``n_equations``.
    """
    from deqn_jax.models import load_model

    model = load_model(config.model)

    sim_batch_eff = (
        config.sim_batch if config.sim_batch is not None else config.batch_size
    )
    trajectory_pool = config.episode_length * sim_batch_eff
    if trajectory_pool < config.batch_size:
        raise ValueError(
            f"Trajectory pool (episode_length * sim_batch = "
            f"{config.episode_length} * {sim_batch_eff} = {trajectory_pool}) "
            f"is smaller than batch_size ({config.batch_size}). The minibatch "
            f"sweep would either draw partial batches or reuse samples. "
            f"Increase episode_length or sim_batch, or decrease batch_size."
        )

    if config.shock_mask is not None and len(config.shock_mask) != model.n_shocks:
        raise ValueError(
            f"shock_mask length ({len(config.shock_mask)}) must equal the "
            f"model's n_shocks ({model.n_shocks}). model={model.name} has "
            f"shock_names={model.shock_names!r}."
        )

    if config.coverage.enabled:
        unknown = (
            set(config.coverage.stress_ranges) | set(config.coverage.repair_ranges)
        ) - set(model.state_names)
        if unknown:
            raise ValueError(
                f"coverage.stress_ranges/repair_ranges names {sorted(unknown)} are "
                f"not in model.state_names {model.state_names!r} (model={model.name})."
            )

    # GN-family optimizers build their residual vector from equations_fn
    # averaged over shocks, ignoring a model's two-stage inside_fn/combine_fn
    # hooks — on a two-stage model they would minimize the biased E[f(r)]
    # objective (the exact bias the two-stage path exists to remove) while
    # LOGGING the correct f(E[r]) loss. Reject rather than silently training
    # the wrong objective. MAO/LBFGS differentiate compute_loss (which takes
    # the two-stage path) and are fine.
    if config.loss_type == "composite" and config.composite_loss.res_sobolev_weight > 0:
        if model.combine_fn is not None:
            raise ValueError(
                "composite_loss.res_sobolev_weight > 0 is v1-single-stage: "
                f"model '{model.name}' declares inside_fn/combine_fn (two-stage "
                "E[fb] vs fb(E) path), which the residual-Sobolev term bypasses."
            )
        if getattr(model, "transition_matrix", None) is not None:
            raise ValueError(
                "composite_loss.res_sobolev_weight > 0 does not support "
                "discrete-chain shock models in v1 (Gaussian quadrature only)."
            )

    if (
        config.optimizer.name.lower() in {"gn", "lm", "ign"}
        and model.combine_fn is not None
    ):
        raise ValueError(
            f"optimizer '{config.optimizer.name}' is not supported on two-stage "
            f"model '{model.name}' (combine_fn set): the GN residual vector uses "
            "equations_fn averaged over shocks, which optimizes the biased "
            "E[f(r)] objective while the logged loss is the correct f(E[r]). "
            "Use a STANDARD optimizer, MAO, or LBFGS."
        )

    if config.constants:
        # Surface exactly which calibration constants change (old -> new). A
        # silent override here shifts the analytical SS / warm-start anchor and
        # was a source of the historical Brock-Mirman "SS mismatch" confusion
        # (audit bm-ss-02).
        prev = {k: model.constants.get(k) for k in config.constants}
        model = model._replace(constants={**model.constants, **config.constants})
        if config.verbose:
            changes = ", ".join(
                f"{k}: {prev[k]} -> {v}" for k, v in dict(config.constants).items()
            )
            print(f"  Constants override ({model.name}): {changes}")

    if model.setup_fn is not None:
        model = model.setup_fn(model, config)

    n_equations = len(model.equation_names) if model.equation_names else 1

    if config.loss_weights is not None and len(config.loss_weights) != n_equations:
        raise ValueError(
            f"loss_weights has {len(config.loss_weights)} entries but model "
            f"has {n_equations} equations"
        )

    return model, n_equations


def _build_initial_state(
    config,
    model: ModelSpec,
    key,
    n_equations: int,
    effective_opt_cfg,
):
    """Resume from checkpoint or build fresh state, then optionally warm-start.

    Returns ``(state, opt, kind, start_episode, total_for_schedule)``.
    ``total_for_schedule`` is the episode count to feed an LR schedule
    (config.episodes for both fresh and resume; kept here so the caller
    doesn't need to recompute it).
    """
    from deqn_jax.config import TrainConfig

    hidden_sizes = config.network.hidden_sizes
    start_episode = 0
    total_for_schedule = config.episodes

    if config.resume:
        ckpt_dir = os.path.dirname(config.resume)
        orig_cfg_path = os.path.join(ckpt_dir, "config.yaml")
        if os.path.exists(orig_cfg_path):
            orig_config = TrainConfig.from_yaml(orig_cfg_path)
        else:
            orig_config = config

        template_state, _orig_opt, _orig_kind = create_train_state(
            model,
            key,
            hidden_sizes=orig_config.network.hidden_sizes,
            batch_size=orig_config.batch_size,
            loss_weights=config.loss_weights,
            n_equations=n_equations,
            optimizer_config=orig_config.optimizer,
            network_config=orig_config.network,
            sim_batch=orig_config.sim_batch,
            replay_config=orig_config.replay_buffer,
            surrogate_config=orig_config.surrogate,
        )

        state = _resume_from_checkpoint(template_state, config.resume)
        start_episode = int(state.episode)
        total_for_schedule = config.episodes

        optimizer_changed = config.optimizer.name != orig_config.optimizer.name
        if optimizer_changed:
            new_opt, new_kind = create_optimizer(effective_opt_cfg)
            if new_kind == OptimizerKind.MAO and hasattr(new_opt, "with_num_tasks"):
                new_opt = new_opt.with_num_tasks(n_equations)
            new_opt_state = new_opt.init(eqx.filter(state.params, eqx.is_array))
            state = state._replace(opt_state=new_opt_state)
            opt, kind = new_opt, new_kind
            if config.verbose:
                print(f"  Resumed from {config.resume} (episode {start_episode})")
                print(
                    f"  Switched optimizer: {orig_config.optimizer.name} -> {config.optimizer.name}"
                )
        else:
            opt, kind = create_optimizer(effective_opt_cfg)
            if kind == OptimizerKind.MAO and hasattr(opt, "with_num_tasks"):
                opt = opt.with_num_tasks(n_equations)
            if config.verbose:
                print(f"  Resumed from {config.resume} (episode {start_episode})")
        return state, opt, kind, start_episode, total_for_schedule

    state, opt, kind = create_train_state(
        model,
        key,
        hidden_sizes=hidden_sizes,
        batch_size=config.batch_size,
        loss_weights=config.loss_weights,
        n_equations=n_equations,
        optimizer_config=effective_opt_cfg,
        network_config=config.network,
        sim_batch=config.sim_batch,
        replay_config=config.replay_buffer,
        surrogate_config=config.surrogate,
    )

    is_linear_plus_mlp = config.network.type == "linear_plus_mlp"
    if config.warm_start and is_linear_plus_mlp:
        if config.verbose:
            print(
                "  Warm start skipped: linear_plus_mlp architecture starts at linear policy by construction."
            )
    elif config.warm_start:
        _hl = get_history_len(state.params)
        if _hl > 1:
            if model.steady_state_fn is not None:
                ss_state, ss_policy = model.steady_state_fn(model.constants)
                ws_key = jax.random.PRNGKey(0)
                noise = jax.random.uniform(
                    ws_key, (256, model.n_states), minval=-0.2, maxval=0.2
                )
                sample_states = ss_state * (1 + noise)
                sample_history = make_constant_history(sample_states, _hl)
                targets = jnp.tile(ss_policy, (256, 1))

                def _ws_loss(params):
                    pred = jax.vmap(params)(sample_history)
                    return jnp.mean((pred - targets) ** 2)

                from deqn_jax.training.warm_start import _lbfgs_minimize

                final_params, n_iters, final_loss = _lbfgs_minimize(
                    _ws_loss,
                    state.params,
                    max_iter=100,
                    tol=1e-6,
                )
                if config.verbose:
                    print(
                        f"  Warm start (sequence net, constant-SS): loss={final_loss:.2e}, iters={n_iters}"
                    )
                state = state._replace(params=final_params)
        elif config.warm_start_dynare:
            from deqn_jax.training.warm_start import warm_start_from_dynare

            state = state._replace(
                params=warm_start_from_dynare(
                    state.params,
                    model,
                    dynare_dir=config.warm_start_dynare,
                    verbose=config.verbose,
                )
            )
        else:
            from deqn_jax.training.warm_start import warm_start_network

            state = state._replace(
                params=warm_start_network(
                    state.params,
                    model,
                    verbose=config.verbose,
                    linearize=config.warm_start_linearize,
                )
            )

    return state, opt, kind, start_episode, total_for_schedule
