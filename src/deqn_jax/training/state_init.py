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

from deqn_jax.networks import (
    create_linear_plus_mlp,
    create_lstm,
    create_mlp,
    create_transformer,
)
from deqn_jax.optimizers.gauss_newton import make_grad_step_gn as _make_grad_step_gn
from deqn_jax.optimizers.lbfgs import make_grad_step_lbfgs as _make_grad_step_lbfgs
from deqn_jax.optimizers.mao import make_grad_step_mao as _make_grad_step_mao
from deqn_jax.optimizers.pcgrad import make_grad_step_pcgrad as _make_grad_step_pcgrad
from deqn_jax.optimizers.registry import OptimizerKind, create_optimizer
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

    # Extract network params from config or use defaults
    activation = "tanh"
    activations = None
    init = "xavier_normal"
    multi_head = False
    skip_connections = False
    net_type = "mlp"
    history_len = 1
    num_heads = 4
    n_layers = 2
    if network_config is not None:
        hidden_sizes = network_config.hidden_sizes
        activation = network_config.activation
        activations = network_config.activations
        init = network_config.init
        multi_head = getattr(network_config, "multi_head", False)
        skip_connections = getattr(network_config, "skip_connections", False)
        net_type = getattr(network_config, "type", "mlp")
        history_len = getattr(network_config, "history_len", 1)
        num_heads = getattr(network_config, "num_heads", 4)
        n_layers = getattr(network_config, "n_layers", 2)

    # Compute input normalization from steady state
    input_shift = None
    input_scale = None
    if model.steady_state_fn is not None:
        ss_state, _ = model.steady_state_fn(model.constants)
        input_shift = ss_state
        input_scale = jnp.maximum(jnp.abs(ss_state), 0.01)

    # Create policy network based on type
    if net_type == "lstm":
        policy_net = create_lstm(
            n_states=model.n_states,
            n_policies=model.n_policies,
            hidden_sizes=hidden_sizes,
            history_len=history_len,
            policy_lower=model.policy_lower,
            policy_upper=model.policy_upper,
            input_shift=input_shift,
            input_scale=input_scale,
            key=net_key,
        )
    elif net_type == "transformer":
        # For Transformer, hidden_sizes is a single value (hidden_dim)
        # Handle case where hidden_sizes is a single int (from --set override)
        if isinstance(hidden_sizes, int):
            hidden_dim = hidden_sizes
        else:
            hidden_dim = hidden_sizes[0] if hidden_sizes else 64
        policy_net = create_transformer(
            n_states=model.n_states,
            n_policies=model.n_policies,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            num_heads=num_heads,
            history_len=history_len,
            policy_lower=model.policy_lower,
            policy_upper=model.policy_upper,
            input_shift=input_shift,
            input_scale=input_scale,
            key=net_key,
        )
    elif net_type == "linear_plus_mlp":
        # Generic residual parameterization: policy = linear(state) + mlp(state).
        # Model-agnostic; for disaster-specific shape priors (K/F mask, ELB
        # feature, q-as-M reparam) use network.type='disaster_policy_net'.
        if model.steady_state_fn is None:
            raise ValueError(
                "network.type='linear_plus_mlp' requires model.steady_state_fn"
            )
        init_scale = getattr(network_config, "init_scale", 0.0)
        # output_links: explicit YAML setting wins, then model.default_output_links,
        # then None (factory defaults to all-linear, legacy behavior).
        output_links = getattr(network_config, "output_links", None)
        if output_links is None:
            output_links = getattr(model, "default_output_links", None)
        policy_net = create_linear_plus_mlp(
            model=model,
            hidden_sizes=hidden_sizes,
            activation=activation,
            init=init,
            init_scale=init_scale,
            input_shift=input_shift,
            input_scale=input_scale,
            output_links=output_links,
            key=net_key,
        )
    elif net_type == "disaster_policy_net":
        # Disaster-specific residual ansatz: linear_plus_mlp + the three
        # disaster shape priors (K/F gauge mask, ELB feature, q-as-M reparam).
        # Each prior is independently toggleable via NetworkConfig fields.
        from deqn_jax.models.disaster.network import create_disaster_policy_net

        if model.steady_state_fn is None:
            raise ValueError(
                "network.type='disaster_policy_net' requires model.steady_state_fn"
            )
        init_scale = getattr(network_config, "init_scale", 0.0)
        use_zlb_feature = getattr(network_config, "use_zlb_feature", False)
        zlb_feature_kind = getattr(network_config, "zlb_feature_kind", "raw")
        kf_names = getattr(network_config, "kf_names", ())
        reparam_q_as_m = getattr(network_config, "reparam_q_as_m", False)
        reparam_pi_as_kp_inner = getattr(
            network_config, "reparam_pi_as_kp_inner", False
        )
        reparam_wtilda_as_kw_inner = getattr(
            network_config, "reparam_wtilda_as_kw_inner", False
        )
        output_links = getattr(network_config, "output_links", None)
        if output_links is None:
            output_links = getattr(model, "default_output_links", None)
        policy_net = create_disaster_policy_net(
            model=model,
            hidden_sizes=hidden_sizes,
            activation=activation,
            init=init,
            init_scale=init_scale,
            input_shift=input_shift,
            input_scale=input_scale,
            kf_names=kf_names,
            use_zlb_feature=use_zlb_feature,
            zlb_feature_kind=zlb_feature_kind,
            reparam_q_as_m=reparam_q_as_m,
            reparam_pi_as_kp_inner=reparam_pi_as_kp_inner,
            reparam_wtilda_as_kw_inner=reparam_wtilda_as_kw_inner,
            output_links=output_links,
            key=net_key,
        )
    elif net_type == "kf_anchored_mlp":
        # K/F gauge elimination: network outputs only non-K/F policies; K/F
        # values come from the model's Blanchard-Kahn linearization at each
        # state. See networks/kf_anchored_mlp.py for the rationale.
        from deqn_jax.networks.kf_anchored_mlp import create_kf_anchored_mlp

        kf_names = getattr(network_config, "kf_names", ("F_p", "K_p", "F_w", "K_w"))
        policy_net = create_kf_anchored_mlp(
            model=model,
            hidden_sizes=hidden_sizes,
            activation=activation,
            init=init,
            kf_names=kf_names,
            input_shift=input_shift,
            input_scale=input_scale,
            key=net_key,
        )
    else:
        policy_net = create_mlp(
            n_states=model.n_states,
            n_policies=model.n_policies,
            hidden_sizes=hidden_sizes,
            activation=activation,
            activations=activations,
            init=init,
            policy_lower=model.policy_lower,
            policy_upper=model.policy_upper,
            multi_head=multi_head,
            skip_connections=skip_connections,
            input_shift=input_shift,
            input_scale=input_scale,
            key=net_key,
        )

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

    state = TrainState(
        params=policy_net,
        opt_state=opt_state,
        episode_state=init_states,
        key=key,
        step=0,
        episode=0,
        loss_weights=weights,
        reweight_state=make_reweight_state(n_equations),
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
    # the more specific / silent-correctness class of mistake.
    if config.loss_type == "composite":
        _bad_opts = {"mao", "lm", "gn", "ign", "lbfgs"}
        _opt_name = config.optimizer.name.lower()
        _is_pcgrad = config.gradient_surgery == "pcgrad"
        if _opt_name in _bad_opts or _is_pcgrad:
            raise ValueError(
                f"loss_type='composite' is not supported with optimizer "
                f"'{config.optimizer.name}'"
                + (" + gradient_surgery='pcgrad'" if _is_pcgrad else "")
                + ". Composite auxiliary losses (anchor, Jacobian, barriers, "
                "Newton) would appear in logs but not affect parameter updates "
                "on this path. Use optimizer 'adam'/'sgd'/'adamw'/'lion'/'muon'/"
                "'ngd'/'shampoo' with gradient_surgery='none' (the STANDARD "
                "variant), or switch to loss_type='mse'."
            )

    # Reject weighting / custom-loss features combined with optimizers whose
    # update paths only see base, UNWEIGHTED MSE residuals (PCGrad differentiates
    # the raw per-equation vector; MAO passes weights=None; GN builds a raw
    # residual vector). These options would appear in logs/config but never
    # affect parameter updates -- the same silent-correctness class as the
    # composite gate above (audit JAX-SILENT-02/03).
    _bad_opts = {"mao", "lm", "gn", "ign", "lbfgs"}
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
                "ngd/shampoo with gradient_surgery='none') to use these options, "
                "or remove them."
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
