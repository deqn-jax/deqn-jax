"""Main training loop for DEQN-JAX.

Key design: single JIT boundary around entire train_step for maximum performance.
Three step variants dispatched at construction time (before JIT):

- STANDARD: normal jax.grad + opt.update(grads, state, params)
- MAO: jax.jacrev(per_eq_loss_vector) -> per-equation Jacobian -> mao.update(eq_jac, state, params)
- LBFGS: optax.lbfgs (GradientTransformationExtraArgs) -- needs value + value_fn for line search
"""

import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from deqn_jax.metrics import create_logger
from deqn_jax.networks import create_linear_plus_mlp, create_lstm, create_mlp, create_transformer
from deqn_jax.optimizers.gauss_newton import make_grad_step_gn as _make_grad_step_gn
from deqn_jax.optimizers.lbfgs import make_grad_step_lbfgs as _make_grad_step_lbfgs
from deqn_jax.optimizers.mao import make_grad_step_mao as _make_grad_step_mao
from deqn_jax.optimizers.pcgrad import make_grad_step_pcgrad as _make_grad_step_pcgrad
from deqn_jax.optimizers.registry import OptimizerKind, create_optimizer
from deqn_jax.optimizers.standard import make_grad_step_standard as _make_grad_step_standard
from deqn_jax.training.checkpointing import (
    best_checkpoint_path as _best_checkpoint_path,
)
from deqn_jax.training.checkpointing import (
    prune_checkpoints as _prune_checkpoints,
)
from deqn_jax.training.checkpointing import (
    resume_from as _resume_from_checkpoint,
)
from deqn_jax.training.checkpointing import (
    save_best_checkpoint as _save_best_checkpoint,
)
from deqn_jax.training.checkpointing import (
    save_checkpoint as _save_checkpoint,
)
from deqn_jax.training.episode import run_episode, run_episode_with_history, sample_initial_states
from deqn_jax.training.history import build_history_windows, get_history_len, make_constant_history
from deqn_jax.training.loss import compute_loss, gauss_hermite_nd
from deqn_jax.training.reporting import (
    count_params as _count_params,
)
from deqn_jax.training.reporting import (
    print_final as _print_final,
)
from deqn_jax.training.reporting import (
    print_header as _print_header,
)
from deqn_jax.training.reporting import (
    print_residual_table as _print_residual_table,
)
from deqn_jax.training.reporting import (
    strip_eq_prefix as _strip_eq_prefix,
)
from deqn_jax.types import Metrics, ModelSpec, TrainState, make_reweight_state

# Console banners and residual formatting now live in
# training/reporting.py; periodic / best-snapshot checkpointing lives
# in training/checkpointing.py. Both are imported above under their
# previous private names so call sites read unchanged.


# ---------------------------------------------------------------------------
# State + optimizer construction
# ---------------------------------------------------------------------------

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
        # Residual parameterization: policy = linear(state) + mlp(state).
        # Requires model.steady_state_fn to compute linearization.
        if model.steady_state_fn is None:
            raise ValueError(
                "network.type='linear_plus_mlp' requires model.steady_state_fn"
            )
        init_scale = getattr(network_config, "init_scale", 0.0)
        use_zlb_feature = getattr(network_config, "use_zlb_feature", False)
        policy_net = create_linear_plus_mlp(
            model=model,
            hidden_sizes=hidden_sizes,
            activation=activation,
            init=init,
            init_scale=init_scale,
            input_shift=input_shift,
            input_scale=input_scale,
            use_zlb_feature=use_zlb_feature,
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
        if hasattr(opt, 'with_num_tasks'):
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
    )

    return state, opt, kind


# ---------------------------------------------------------------------------
# Common episode + batch logic (shared by all step variants)
# ---------------------------------------------------------------------------

def _run_episode_and_sample(model, state, episode_length, batch_size, history_len=1,
                            ss_reset_frac: float = 0.0,
                            initialize_each_episode: bool = False,
                            shock_scale: Array = jnp.array(1.0),
                            shock_mask: Optional[Array] = None):
    """Run episode and sample training batch.

    For history_len=1 (MLP): returns [batch, n_states]
    For history_len>1 (LSTM/Transformer): returns [batch, H, n_states]

    history_len is captured in closure and resolves at trace time.

    ss_reset_frac: fraction of batch to reset to SS-neighborhood each episode.
    Prevents on-policy drift into degenerate basins.
    initialize_each_episode: if True, replace the entire batch with a fresh
    init_state_fn draw every rollout (Simon nb01 style; appropriate for
    deterministic / strongly-attracting models).
    """
    key = state.key
    key, episode_key, loss_key, shuffle_key, reset_key = jax.random.split(key, 5)

    # Either resample the whole batch (deterministic / no-rollout mode) or
    # mix in fresh SS-neighborhood states to prevent trajectory drift.
    ep_states = state.episode_state
    if initialize_each_episode and model.init_state_fn is not None:
        ep_states = model.init_state_fn(reset_key, ep_states.shape[0], model.constants)
    elif ss_reset_frac > 0.0:
        batch_n = ep_states.shape[0]
        n_reset = int(ss_reset_frac * batch_n)
        if n_reset > 0:
            ss_state, _ = model.steady_state_fn(model.constants)
            noise = jax.random.uniform(reset_key, (n_reset, model.n_states), minval=-0.05, maxval=0.05)
            fresh = ss_state * (1 + noise)
            ep_states = ep_states.at[:n_reset].set(fresh)

    if history_len > 1:
        # Sequence path: run episode with history, build windows.
        # state.history_state persists the sliding window across rollouts;
        # when None (first-ever rollout) run_episode_with_history rebuilds
        # from ep_states via make_constant_history.
        trajectory, final_state, final_history = run_episode_with_history(
            model,
            state.params,
            ep_states,
            episode_key,
            episode_length,
            history_len,
            shock_scale=shock_scale,
            shock_mask=shock_mask,
            init_history=state.history_state,
        )
        # Build sliding windows from trajectory: [(T-H+1)*B, H, D]
        all_windows = build_history_windows(trajectory, history_len)
        n_windows = all_windows.shape[0]
        indices = jax.random.permutation(shuffle_key, n_windows)[:batch_size]
        train_batch = all_windows[indices]  # [batch, H, D]
    else:
        # MLP path
        trajectory, final_state = run_episode(
            model,
            state.params,
            ep_states,
            episode_key,
            episode_length,
            shock_scale=shock_scale,
            shock_mask=shock_mask,
        )
        all_states = trajectory.reshape(-1, model.n_states)
        n_states = all_states.shape[0]
        indices = jax.random.permutation(shuffle_key, n_states)[:batch_size]
        train_batch = all_states[indices]  # [batch, D]
        final_history = state.history_state  # None for MLP; pass through

    return train_batch, final_state, final_history, loss_key, key


# ---------------------------------------------------------------------------
# Rollout-and-sweep path (matches reference DEQN: one rollout → many
# minibatch gradient steps over state_episode).
# ---------------------------------------------------------------------------

def _make_rollout_fn(
    model: ModelSpec,
    episode_length: int,
    history_len: int,
    ss_reset_frac: float,
    initialize_each_episode: bool = False,
):
    """JIT'd: run one rollout → return (trajectory, final_state, new_key).

    The trajectory has shape [episode_length, batch, n_states] (MLP) or
    needs to be wrapped into windows for sequence nets. The outer loop
    then draws minibatches from it and calls the grad step many times.

    initialize_each_episode: if True, replace the entire batch with a fresh
    init_state_fn draw every rollout (deterministic-model "fresh sample
    per episode" semantics; see TrainConfig docstring).
    """
    @jax.jit
    def rollout_fn(
        state: TrainState,
        shock_scale: Array = jnp.array(1.0),
    ) -> Tuple[Array, Array, Array]:
        key, episode_key, reset_key = jax.random.split(state.key, 3)
        ep_states = state.episode_state
        if initialize_each_episode and model.init_state_fn is not None:
            ep_states = model.init_state_fn(reset_key, ep_states.shape[0], model.constants)
        elif ss_reset_frac > 0.0:
            batch_n = ep_states.shape[0]
            n_reset = int(ss_reset_frac * batch_n)
            if n_reset > 0:
                ss_state, _ = model.steady_state_fn(model.constants)
                noise = jax.random.uniform(
                    reset_key, (n_reset, model.n_states), minval=-0.05, maxval=0.05,
                )
                fresh = ss_state * (1 + noise)
                ep_states = ep_states.at[:n_reset].set(fresh)
        if history_len > 1:
            trajectory, final_state, final_history = run_episode_with_history(
                model, state.params, ep_states, episode_key, episode_length, history_len,
                shock_scale=shock_scale,
                init_history=state.history_state,
            )
        else:
            trajectory, final_state = run_episode(
                model, state.params, ep_states, episode_key, episode_length,
                shock_scale=shock_scale,
            )
            final_history = state.history_state  # None for MLP; pass through unchanged
        return trajectory, final_state, final_history, key
    return rollout_fn



def _make_cycle_step(
    rollout_fn: Callable,
    grad_step: Callable,
    model: ModelSpec,
    batch_size: int,
    n_epochs_per_rollout: int,
    n_minibatches_per_epoch: Optional[int],
    history_len: int = 1,
    sorted_within_batch: bool = False,
):
    """Generic DEQN-style cycle: 1 rollout + n_epochs × minibatch sweep.

    Works for any ``grad_step`` with signature
    ``(state, batch, lr_scale, shock_scale) → (new_state, metrics)``.
    The rollout is kind-agnostic; only the per-batch grad step differs.

    ``sorted_within_batch`` (default False): when True and history_len==1,
    minibatches are contiguous slices of single trajectories (RL-style)
    rather than IID-shuffled samples. Batch order is shuffled; intra-batch
    order is preserved. See TrainConfig docstring.
    """
    def cycle_step(
        state: TrainState,
        lr_scale: Array,
        shock_scale: Array = jnp.array(1.0),
    ) -> Tuple[TrainState, Metrics]:
        # 1. Rollout — one JIT'd call.
        # Note: `rollout_fn` returns trajectory = [s_0, ..., s_{T-1}] and
        # final_carry_state = s_T (T transitions total). The *reference*
        # DEQN_MAO stores [s_0, ..., s_{T-1}] in state_episode and seeds
        # the next cycle from state_episode[T-1] = s_{T-1} (so cycles
        # overlap by one state, and each cycle advances T-1 transitions).
        # We use trajectory[-1] = s_{T-1} for seeding to match that
        # convention; final_carry_state is discarded.
        trajectory, _final_after_T, final_history, new_key = rollout_fn(state, shock_scale)
        next_seed = trajectory[-1]
        # Persist the history window alongside episode_state so recurrent
        # policies see continuous trajectories across cycles instead of
        # a constant-prefix rebuild at every rollout. For MLP (history_len=1)
        # final_history passes through as None.
        state = state._replace(
            episode_state=next_seed, key=new_key, history_state=final_history,
        )

        # 2. Build the minibatch dataset from the full trajectory.
        # Two layouts (chosen at trace time via sorted_within_batch):
        # - time-major [T*B, D]: reshape trajectory [T, B, D] directly.
        #   IID shuffling of all samples, used when sorted_within_batch=False.
        # - trajectory-major [B*T, D]: transpose to [B, T, D] first.
        #   Contiguous slices of length `batch_size` are single-trajectory
        #   temporal segments; used when sorted_within_batch=True.
        if history_len > 1:
            # Sequence path: windows already carry temporal coherence.
            # sorted_within_batch is a no-op here.
            dataset = build_history_windows(trajectory, history_len)
        elif sorted_within_batch:
            # [T, B, D] -> [B, T, D] -> [B*T, D]
            dataset = jnp.transpose(trajectory, (1, 0, 2)).reshape(-1, model.n_states)
        else:
            dataset = trajectory.reshape(-1, model.n_states)

        n_samples = dataset.shape[0]
        n_mbs_available = max(1, n_samples // batch_size)
        n_mbs = (
            min(n_minibatches_per_epoch, n_mbs_available)
            if n_minibatches_per_epoch is not None
            else n_mbs_available
        )

        # 3. Sweep: n_epochs × n_minibatches gradient updates.
        # Cycle-level metrics are aggregated across ALL minibatches in the
        # sweep, matching DEQN_MAO's run_cycle which reports
        # MSE_epoch_loss = epoch_loss / (N_episode_length * N_sim_batch).
        # Per-minibatch loss from grad_step is already a mean over its
        # batch, so averaging over minibatches gives an overall mean.
        # grad_norm is tracked as a max across the sweep (the spike is
        # what matters for stability diagnostics). Residuals are averaged
        # per equation.
        last_metrics = None
        loss_acc = jnp.array(0.0)
        grad_max = jnp.array(0.0)
        eq_acc: Optional[Dict[str, Array]] = None
        n_steps = 0
        for _ in range(n_epochs_per_rollout):
            perm_key, state_key = jax.random.split(state.key)
            state = state._replace(key=state_key)
            if sorted_within_batch and history_len == 1:
                # Shuffle the *order of contiguous chunks*, not samples.
                # Each chunk is dataset[b*batch_size : (b+1)*batch_size],
                # a trajectory segment of length batch_size.
                batch_perm = jax.random.permutation(perm_key, n_mbs_available)
            else:
                # Shuffle individual sample indices (IID minibatch).
                perm = jax.random.permutation(perm_key, n_samples)
            for mb_idx in range(n_mbs):
                if sorted_within_batch and history_len == 1:
                    b = batch_perm[mb_idx]
                    start = b * batch_size
                    minibatch = jax.lax.dynamic_slice_in_dim(dataset, start, batch_size)
                else:
                    idx = perm[mb_idx * batch_size:(mb_idx + 1) * batch_size]
                    minibatch = dataset[idx]
                state, last_metrics = grad_step(state, minibatch, lr_scale, shock_scale)
                loss_acc = loss_acc + last_metrics.loss
                grad_max = jnp.maximum(grad_max, last_metrics.grad_norm)
                if eq_acc is None:
                    eq_acc = {k: v * 0.0 for k, v in last_metrics.residuals.items()}
                for k, v in last_metrics.residuals.items():
                    eq_acc[k] = eq_acc[k] + v
                n_steps += 1

        # 4. Bump episode counter once per cycle.
        state = state._replace(episode=state.episode + 1)

        # 5. Build aggregated metrics.
        if n_steps == 0 or last_metrics is None:
            return state, last_metrics
        n_steps_f = float(n_steps)
        avg_loss = loss_acc / n_steps_f
        avg_eq = {k: v / n_steps_f for k, v in (eq_acc or {}).items()}
        aggregated = Metrics(loss=avg_loss, residuals=avg_eq, grad_norm=grad_max)
        return state, aggregated

    return cycle_step


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
        model, episode_length, history_len, ss_reset_frac,
        initialize_each_episode=initialize_each_episode,
    )

    if gradient_surgery == "pcgrad" and kind == OptimizerKind.STANDARD:
        grad_step = _make_grad_step_pcgrad(
            model, opt, mc_samples, quad_nodes, quad_weights,
            loss_reweight, reweight_alpha, use_target_network, compute_loss_fn,
        )
    elif kind == OptimizerKind.MAO:
        grad_step = _make_grad_step_mao(
            model, opt, mc_samples, quad_nodes, quad_weights,
            loss_reweight, reweight_alpha, use_target_network, compute_loss_fn,
            grad_clip,
        )
    elif kind == OptimizerKind.LBFGS:
        grad_step = _make_grad_step_lbfgs(
            model, opt, mc_samples, quad_nodes, quad_weights,
            loss_reweight, reweight_alpha, use_target_network, compute_loss_fn,
        )
    elif kind == OptimizerKind.GN:
        grad_step = _make_grad_step_gn(
            model, opt, mc_samples, batch_size, quad_nodes, quad_weights,
            loss_reweight, reweight_alpha, use_target_network, compute_loss_fn,
        )
    else:
        grad_step = _make_grad_step_standard(
            model, opt, mc_samples, quad_nodes, quad_weights,
            loss_reweight, reweight_alpha, use_target_network, compute_loss_fn,
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
        _bad_opts = {"mao", "lm", "gn", "lbfgs"}
        _opt_name = config.optimizer.name.lower()
        _is_pcgrad = (config.gradient_surgery == "pcgrad")
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


def _resolve_model_for_training(config) -> Tuple[ModelSpec, int]:
    """Load the model, validate sizes, apply constants override and setup_fn.

    Returns ``(model, n_equations)``. Done as one helper because the
    validation steps depend on the loaded model and we want to apply
    all model-side adaptations (constants override, setup_fn) before
    computing ``n_equations``.
    """
    from deqn_jax.models import load_model

    model = load_model(config.model)

    sim_batch_eff = config.sim_batch if config.sim_batch is not None else config.batch_size
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
        model = model._replace(
            constants={**model.constants, **config.constants}
        )
        if config.verbose:
            print(f"  Constants override: {dict(config.constants)}")

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
        )

        state = _resume_from_checkpoint(template_state, config.resume)
        start_episode = int(state.episode)
        total_for_schedule = config.episodes

        optimizer_changed = config.optimizer.name != orig_config.optimizer.name
        if optimizer_changed:
            new_opt, new_kind = create_optimizer(effective_opt_cfg)
            if new_kind == OptimizerKind.MAO and hasattr(new_opt, 'with_num_tasks'):
                new_opt = new_opt.with_num_tasks(n_equations)
            new_opt_state = new_opt.init(eqx.filter(state.params, eqx.is_array))
            state = state._replace(opt_state=new_opt_state)
            opt, kind = new_opt, new_kind
            if config.verbose:
                print(f"  Resumed from {config.resume} (episode {start_episode})")
                print(f"  Switched optimizer: {orig_config.optimizer.name} -> {config.optimizer.name}")
        else:
            opt, kind = create_optimizer(effective_opt_cfg)
            if kind == OptimizerKind.MAO and hasattr(opt, 'with_num_tasks'):
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
    )

    is_linear_plus_mlp = config.network.type == "linear_plus_mlp"
    if config.warm_start and is_linear_plus_mlp:
        if config.verbose:
            print("  Warm start skipped: linear_plus_mlp architecture starts at linear policy by construction.")
    elif config.warm_start:
        _hl = get_history_len(state.params)
        if _hl > 1:
            if model.steady_state_fn is not None:
                ss_state, ss_policy = model.steady_state_fn(model.constants)
                ws_key = jax.random.PRNGKey(0)
                noise = jax.random.uniform(ws_key, (256, model.n_states), minval=-0.2, maxval=0.2)
                sample_states = ss_state * (1 + noise)
                sample_history = make_constant_history(sample_states, _hl)
                targets = jnp.tile(ss_policy, (256, 1))
                def _ws_loss(params):
                    pred = jax.vmap(params)(sample_history)
                    return jnp.mean((pred - targets) ** 2)
                from deqn_jax.training.warm_start import _lbfgs_minimize
                final_params, n_iters, final_loss = _lbfgs_minimize(
                    _ws_loss, state.params, max_iter=100, tol=1e-6,
                )
                if config.verbose:
                    print(f"  Warm start (sequence net, constant-SS): loss={final_loss:.2e}, iters={n_iters}")
                state = state._replace(params=final_params)
        elif config.warm_start_dynare:
            from deqn_jax.training.warm_start import warm_start_from_dynare
            state = state._replace(
                params=warm_start_from_dynare(
                    state.params, model,
                    dynare_dir=config.warm_start_dynare,
                    verbose=config.verbose,
                )
            )
        else:
            from deqn_jax.training.warm_start import warm_start_network
            state = state._replace(
                params=warm_start_network(
                    state.params, model, verbose=config.verbose,
                    linearize=config.warm_start_linearize,
                )
            )

    return state, opt, kind, start_episode, total_for_schedule


def _build_custom_loss_fn(config, model: ModelSpec, history_len: int):
    """Build the wrapped loss function for non-default loss configurations.

    Returns the custom loss callable (or None if the default MSE
    `compute_loss` should be used as-is). Handles three layered cases:
    composite loss, state-barrier penalty, and Huber loss for the bare
    path.
    """
    from functools import partial

    custom_loss_fn = None
    if config.loss_type == "composite":
        from deqn_jax.training.composite_loss import make_composite_loss, prepare_composite_data
        from deqn_jax.training.linearize import linearize_model

        if config.verbose:
            print("  Building composite loss (linearize + ergodic cov)...")
        P, Q = linearize_model(model, verbose=config.verbose)

        comp_cfg = config.composite_loss
        comp_data = prepare_composite_data(
            model, P, Q,
            n_anchor_points=comp_cfg.n_anchor_points,
            anchor_sigma=comp_cfg.anchor_sigma,
            seed=config.seed,
            verbose=config.verbose,
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
        )
        if config.verbose:
            extras = []
            if config.loss_choice != "mse":
                extras.append(f"loss_choice={config.loss_choice} (δ={config.huber_delta})")
            if comp_cfg.jac_anchor_weight > 0:
                extras.append(f"sobolev-anchor w={comp_cfg.jac_anchor_weight}")
            extras_str = " · ".join(extras)
            print(f"  Composite loss ready.{(' · ' + extras_str) if extras_str else ''}")

    barrier_weight = config.barrier_weight
    if barrier_weight > 0 and custom_loss_fn is None and model.state_barrier_fn is not None:
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

    return custom_loss_fn


def _run_training_loop(
    config,
    model: ModelSpec,
    state: TrainState,
    opt: Any,
    kind: OptimizerKind,
    gradient_surgery: str,
    train_step: Callable,
    lr_schedule_fn: Optional[Callable],
    quad_nodes_jax: Optional[Array],
    quad_weights_jax: Optional[Array],
    history_len: int,
    custom_loss_fn: Optional[Callable],
    use_target: bool,
    n_equations: int,
    start_episode: int,
    logger,
) -> Optional[Tuple[Any, Dict[str, list]]]:
    """Run the per-episode train loop with all the runtime knobs.

    Encapsulates: mid-training optimizer switching, LR scheduling
    (stateful + stateless), curriculum and shock-mask scaling, NaN
    detection + checkpoint rollback, early stopping, grouped logging
    (scalars / histograms / model cycle_hook + scalar_diagnostics),
    and periodic + best-checkpoint persistence.
    """
    # Local imports kept inside the helper so trainer.py's top-level imports
    # don't grow further; these are only needed during training.
    from deqn_jax.config import OptimizerConfig

    # ---- Mid-training optimizer switch setup ----
    switch_episode = config.switch_episode
    switched = False
    if config.switch_optimizer and config.switch_episode is None:
        raise ValueError("--switch-optimizer requires --switch-episode")

    # ---- LR schedule: dynamic scaling via train_step argument ----
    # All train_step variants accept (state, lr_scale). The optimizer uses
    # lr=1.0 when a schedule is active; lr_scale carries the actual LR.
    # When no schedule, lr_scale=1.0 (no-op, XLA optimizes it away).
    current_lr = config.optimizer.learning_rate

    # ---- NaN recovery setup ----
    nan_rollback_enabled = config.checkpoint_dir is not None and config.checkpoint_every is not None
    nan_lr_reduction = 0.75  # reduce LR by 25% on NaN
    max_nan_rollbacks = 10  # max rollbacks before giving up
    nan_rollback_count = 0
    nan_lr_scale = 1.0      # cumulative LR reduction from NaN rollbacks
    last_good_state = None  # snapshot for rollback (updated at checkpoints)
    last_good_episode = start_episode

    # ---- Training loop ----
    total_episodes = config.episodes
    if start_episode >= total_episodes:
        print(f"WARNING: checkpoint episode {start_episode} >= config.episodes {total_episodes}. Nothing to do.")
        return None
    ep_width = len(str(total_episodes))

    history: Dict[str, list] = {"loss": [], "grad_norm": []}
    t_start = time.perf_counter()
    last_metrics = None
    best_loss = float("inf")
    patience_counter = 0

    # Best-checkpoint tracking (separate from early-stop `best_loss` because
    # we always want to preserve the best snapshot even without early stop).
    best_save_loss = float("inf")
    best_save_episode = start_episode
    # Grace period: don't save as "best" during curriculum ramp (shocks
    # are reduced → loss is artificially low). Falls back to log_every
    # when no curriculum is configured.
    best_save_grace = max(config.curriculum_episodes, config.log_every)

    for ep_num in range(start_episode + 1, total_episodes + 1):
        # Mid-training optimizer switch
        if (
            not switched
            and switch_episode is not None
            and config.switch_optimizer is not None
            and ep_num == switch_episode
        ):
            switch_lr = config.switch_lr or config.optimizer.learning_rate
            switch_cfg = OptimizerConfig(
                name=config.switch_optimizer,
                learning_rate=switch_lr,
                grad_clip=config.optimizer.grad_clip,
            )
            new_opt, new_kind = create_optimizer(switch_cfg)
            # Disable schedule after mid-training switch (uses constant LR)
            lr_schedule_fn = None
            if new_kind == OptimizerKind.MAO and hasattr(new_opt, 'with_num_tasks'):
                new_opt = new_opt.with_num_tasks(n_equations)
            new_opt_state = new_opt.init(eqx.filter(state.params, eqx.is_array))
            state = state._replace(opt_state=new_opt_state)
            opt, kind = new_opt, new_kind
            train_step = make_train_step(
                model, opt, config.episode_length, config.mc_samples, config.batch_size,
                loss_reweight=config.loss_reweight,
                reweight_alpha=config.reweight_alpha,
                kind=kind,
                gradient_surgery=gradient_surgery,
                grad_clip=config.optimizer.grad_clip,
                quad_nodes=quad_nodes_jax,
                quad_weights=quad_weights_jax,
                history_len=history_len,
                compute_loss_fn=custom_loss_fn,
                ss_reset_frac=config.ss_reset_frac,
                use_target_network=use_target,
                n_epochs_per_rollout=config.n_epochs_per_rollout,
                n_minibatches_per_epoch=config.n_minibatches_per_epoch,
                initialize_each_episode=config.initialize_each_episode,
                sorted_within_batch=config.sorted_within_batch,
            )
            switched = True
            # Reset early stopping after optimizer switch
            best_loss = float("inf")
            patience_counter = 0
            if config.verbose:
                print(f"  >> Switched to {config.switch_optimizer} (lr={switch_lr:.0e}) at episode {ep_num}")

        # Compute LR scale for this episode (Python-side, passed as dynamic arg).
        # Stateful schedules (ReduceLROnPlateau) consume the most recent loss;
        # stateless schedules (cosine) accept but ignore it.
        if lr_schedule_fn is not None:
            last_loss = history["loss"][-1] if history["loss"] else None
            try:
                current_lr = float(lr_schedule_fn(ep_num, last_loss)) * nan_lr_scale
            except TypeError:
                # optax schedules accept a single positional arg; fall back.
                current_lr = float(lr_schedule_fn(ep_num)) * nan_lr_scale
            lr_scale = jnp.array(current_lr)
        else:
            current_lr = config.optimizer.learning_rate * nan_lr_scale
            lr_scale = jnp.array(nan_lr_scale)

        # Curriculum: ramp shock_scale from start to 1.0
        if config.curriculum_episodes > 0 and ep_num < config.curriculum_episodes:
            t = ep_num / config.curriculum_episodes
            shock_scale_val = config.curriculum_start + (1.0 - config.curriculum_start) * t
        else:
            shock_scale_val = 1.0

        # Per-shock masking: shock_mask=[1,0,1,1,1] zeros shock 1.
        # Multiply into shock_scale so it becomes a vector [n_shocks].
        # Broadcasting in loss.py handles scalar vs vector transparently.
        if config.shock_mask is not None:
            shock_scale = jnp.array(shock_scale_val) * jnp.array(config.shock_mask)
        else:
            shock_scale = jnp.array(shock_scale_val)

        state, metrics = train_step(state, lr_scale, shock_scale)
        last_metrics = metrics

        # Periodic target network update (Polyak averaging or hard copy)
        if use_target and ep_num % config.target_update_every == 0:
            if config.target_tau >= 1.0:
                # Hard copy
                state = state._replace(target_params=state.params)
            else:
                # Polyak: target = tau * current + (1-tau) * target
                tau = config.target_tau
                new_target = jax.tree.map(
                    lambda p, t: tau * p + (1 - tau) * t,
                    eqx.filter(state.params, eqx.is_array),
                    eqx.filter(state.target_params, eqx.is_array),
                )
                state = state._replace(
                    target_params=eqx.combine(new_target, state.params)
                )

        loss_val = float(metrics.loss)
        grad_val = float(metrics.grad_norm)

        # ---- NaN detection + rollback ----
        if math.isnan(loss_val) or math.isinf(loss_val):
            if nan_rollback_enabled and last_good_state is not None and nan_rollback_count < max_nan_rollbacks:
                nan_rollback_count += 1
                nan_lr_scale *= nan_lr_reduction
                effective_lr = config.optimizer.learning_rate * nan_lr_scale
                if config.verbose:
                    print(f"  >> NaN at episode {ep_num}! "
                          f"Rolling back to ep {last_good_episode}, "
                          f"reducing LR to {effective_lr:.1e} "
                          f"(rollback {nan_rollback_count}/{max_nan_rollbacks})")
                state = last_good_state
                continue
            elif nan_rollback_count >= max_nan_rollbacks:
                if config.verbose:
                    print(f"  >> NaN at episode {ep_num} after {max_nan_rollbacks} rollbacks. Stopping.")
                break
            # No checkpoint to roll back to — just continue (NaN will propagate)

        # ---- Early stopping (only after optimizer switch, or if no switch configured) ----
        early_stop_active = (
            config.early_stop_patience is not None
            and (switched or config.switch_optimizer is None)
        )
        if early_stop_active and not math.isnan(loss_val):
            if loss_val < best_loss - config.early_stop_min_delta:
                best_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                if config.verbose:
                    print(f"  >> Early stopping at episode {ep_num}: "
                          f"no improvement for {config.early_stop_patience} episodes "
                          f"(best={best_loss:.2e})")
                break

        history["loss"].append(loss_val)
        history["grad_norm"].append(grad_val)

        # ---- Grouped logging ----
        if ep_num % config.log_every == 0 or ep_num == total_episodes:
            elapsed = time.perf_counter() - t_start
            eps_done = ep_num - start_episode
            ep_per_sec = eps_done / elapsed if elapsed > 0 else 0

            param_norm = float(optax.global_norm(eqx.filter(state.params, eqx.is_array)))

            log_dict = {
                "train/loss": loss_val,
                "train/grad_norm": grad_val,
                "train/param_norm": param_norm,
                "train/ep_per_sec": ep_per_sec,
            }
            log_dict["train/lr"] = current_lr
            if metrics.residuals:
                for k, v in metrics.residuals.items():
                    if k.startswith("aux_"):
                        log_dict[f"aux/{k[4:]}"] = float(v)
                    else:
                        log_dict[f"eq/{k}"] = float(v)
            # Log per-equation weights when adaptive reweighting is active
            if config.loss_reweight != "none" and model.equation_names:
                for i, name in enumerate(model.equation_names):
                    log_dict[f"weights/{name}"] = float(state.loss_weights[i])

            # State, policy, and definition histograms
            import numpy as np
            hist_dict: Dict[str, Any] = {}
            ep_states = state.episode_state  # [batch, n_states]

            # State variable histograms
            if model.state_names:
                for i, name in enumerate(model.state_names):
                    hist_dict[f"state/{name}"] = np.asarray(ep_states[:, i])

            # Policy output histograms
            # For sequence nets, approximate with constant history at current state
            if history_len > 1:
                ep_history = make_constant_history(ep_states, history_len)
                policy_out = jax.vmap(state.params)(ep_history)
            else:
                policy_out = jax.vmap(state.params)(ep_states)  # [batch, n_policies]
            if model.policy_names:
                for i, name in enumerate(model.policy_names):
                    hist_dict[f"policy/{name}"] = np.asarray(policy_out[:, i])

            # Definition histograms (derived economic quantities)
            if model.definitions_fn is not None:
                defs = jax.vmap(
                    lambda s, p: model.definitions_fn(s, p, model.constants)
                )(ep_states, policy_out)
                for name, vals in defs.items():
                    hist_dict[f"derived/{name}"] = np.asarray(vals)

                # Model-supplied scalar diagnostics (e.g. disaster's
                # eq2_diag / eq4_diag Phillips-curve decompositions).
                # Generic hook: any model can declare a
                # ``scalar_diagnostics_fn`` on its ModelSpec returning
                # a dict of pre-namespaced scalars to log. Failure is
                # tolerated to avoid killing training over a bad
                # diagnostic.
                if model.scalar_diagnostics_fn is not None:
                    if history_len > 1:
                        _diag_policy_fn = lambda s: state.params(
                            make_constant_history(s[None], history_len)[0]
                        )
                    else:
                        _diag_policy_fn = state.params
                    try:
                        diag = model.scalar_diagnostics_fn(
                            model, _diag_policy_fn, ep_states, policy_out, defs,
                        )
                        for dk, dv in diag.items():
                            log_dict[dk] = float(dv)
                    except Exception as exc:
                        import warnings
                        warnings.warn(
                            f"scalar_diagnostics_fn raised at ep {ep_num}: {exc}"
                        )

            logger.log_scalars(log_dict, step=ep_num)

            # Filter out arrays with NaN/Inf (early training can produce these)
            hist_dict = {k: v for k, v in hist_dict.items()
                         if np.isfinite(v).all() and v.size > 0}
            if hist_dict:
                logger.log_histograms(hist_dict, step=ep_num)

            # Model-provided cycle hook (plots, custom diagnostics).
            # Runs outside JIT in the Python-level log path; errors are
            # caught so training isn't killed by a bad plot.
            if model.cycle_hook is not None:
                try:
                    model.cycle_hook(state, model, ep_num)
                except Exception as exc:
                    import warnings
                    warnings.warn(f"cycle_hook raised at ep {ep_num}: {exc}")

        if config.verbose and ep_num % config.log_every == 0:
            elapsed = time.perf_counter() - t_start
            eps_done = ep_num - start_episode
            ep_per_sec = eps_done / elapsed if elapsed > 0 else 0
            residuals = metrics.residuals or {}

            # Summary line
            print(
                f"  [{ep_num:>{ep_width}}/{total_episodes}] "
                f"loss={loss_val:.2e} | grad={grad_val:.2e} | {ep_per_sec:.0f} ep/s"
            )

            # Residuals: inline for <=3 equations, columnar table for more
            if residuals:
                eq_items = [
                    (_strip_eq_prefix(k), float(v))
                    for k, v in residuals.items()
                    if not k.startswith("aux_")
                ]
                aux_items = [
                    (k[4:], float(v))
                    for k, v in residuals.items()
                    if k.startswith("aux_")
                ]
                if len(eq_items) <= 3:
                    print("    " + "  ".join(
                        f"{n}={v:.2e}" for n, v in eq_items
                    ))
                else:
                    _print_residual_table(eq_items)
                if aux_items:
                    print("    aux: " + "  ".join(
                        f"{n}={v:.2e}" for n, v in aux_items
                    ))

        # ---- Checkpointing with config snapshot + pruning ----
        if (
            config.checkpoint_dir is not None
            and config.checkpoint_every is not None
            and ep_num % config.checkpoint_every == 0
        ):
            _save_checkpoint(state, config.checkpoint_dir, ep_num, config=config)
            if config.max_checkpoints is not None:
                _prune_checkpoints(config.checkpoint_dir, config.max_checkpoints)
            # Snapshot for NaN rollback
            last_good_state = state
            last_good_episode = ep_num

        # ---- Save-best tracking ----
        # Always writes the best-so-far checkpoint on improvement (after
        # grace period). Independent of early_stop and of
        # checkpoint_every. The on-disk path is owned by
        # ``training.checkpointing``.
        if (
            config.save_best_checkpoint
            and config.checkpoint_dir is not None
            and ep_num > best_save_grace
            and not math.isnan(loss_val)
            and loss_val < best_save_loss
        ):
            best_save_loss = loss_val
            best_save_episode = ep_num
            _save_best_checkpoint(
                state, config.checkpoint_dir, ep_num, loss_val, config=config,
            )

    elapsed = time.perf_counter() - t_start

    if config.verbose and last_metrics is not None:
        _print_final(
            elapsed=elapsed,
            episodes=config.episodes,
            final_loss=float(last_metrics.loss),
            final_residuals=last_metrics.residuals,
        )
        if best_save_loss < float("inf"):
            print(
                f"Best checkpoint: {best_save_loss:.2e} at episode "
                f"{best_save_episode} → {_best_checkpoint_path(config.checkpoint_dir)}"
            )

    logger.close()
    return state.params, history


# ---------------------------------------------------------------------------
# Training entry points
# ---------------------------------------------------------------------------


def train_from_config(config) -> Tuple[Any, Dict[str, list]]:
    """Train from a TrainConfig object.

    This is the primary entry point for config-driven training.
    Supports checkpoint resume, mid-training optimizer switching,
    and grouped TensorBoard logging.

    Args:
        config: TrainConfig instance

    Returns:
        Tuple of (trained_params, history_dict)
    """
    _validate_train_config(config)
    model, n_equations = _resolve_model_for_training(config)

    fp64 = jnp.zeros(1).dtype == jnp.float64
    hidden_sizes = config.network.hidden_sizes
    key = jax.random.PRNGKey(config.seed)

    # ---- Build LR schedule helper for logging ----
    from deqn_jax.optimizers.registry import _build_lr_schedule

    # When a schedule is active, the optimizer is created with lr=1.0.
    # The actual LR is passed as a dynamic scalar to train_step each episode.
    has_schedule = config.optimizer.lr_schedule != "constant"
    if has_schedule:
        effective_opt_cfg = config.optimizer.model_copy(
            update={"learning_rate": 1.0, "lr_schedule": "constant"}
        )
    else:
        effective_opt_cfg = config.optimizer

    state, opt, kind, start_episode, total_for_schedule = _build_initial_state(
        config, model, key, n_equations, effective_opt_cfg,
    )

    # ---- Metric logger ----
    wandb_config = config.to_dict() if config.wandb_project else None
    logger = create_logger(
        tensorboard_dir=config.tensorboard_dir,
        wandb_project=config.wandb_project,
        wandb_config=wandb_config,
    )

    # ---- Print header ----
    if config.verbose:
        _print_header(
            model_spec=model,
            optimizer=config.optimizer.name,
            learning_rate=config.optimizer.learning_rate,
            hidden_sizes=hidden_sizes,
            n_params=_count_params(state.params),
            batch_size=config.batch_size,
            mc_samples=config.mc_samples,
            warm_start=config.warm_start,
            grad_clip=config.optimizer.grad_clip,
            loss_reweight=config.loss_reweight,
            fp64=fp64,
            lr_schedule=config.optimizer.lr_schedule,
            lr_warmup=config.optimizer.lr_warmup,
            lr_min_factor=config.optimizer.lr_min_factor,
            net_type=getattr(config.network, "type", "mlp") if config.network else "mlp",
            history_len=get_history_len(state.params),
        )

    # Build LR schedule function for computing per-episode LR (None if constant)
    lr_schedule_fn = None
    if has_schedule:
        lr_schedule_fn = _build_lr_schedule(config.optimizer, total_for_schedule)

    # ---- Pre-compute quadrature nodes (if using Gauss-Hermite) ----
    quad_nodes_jax = None
    quad_weights_jax = None
    exp_type = config.expectation_type
    if exp_type in ("quadrature", "gh", "gauss_hermite"):
        n_qp = config.n_quadrature_points
        quad = gauss_hermite_nd(n_qp, model.n_shocks)
        if quad is not None:
            quad_nodes_jax = jnp.array(quad[0])
            quad_weights_jax = jnp.array(quad[1])
            if config.verbose:
                print(f"  Quadrature: {n_qp}^{model.n_shocks} = {quad[0].shape[0]} nodes (Gauss-Hermite)")
        else:
            n_total = n_qp ** model.n_shocks
            if config.verbose:
                print(f"  Quadrature: {n_total} nodes exceeds limit, falling back to MC")

    # ---- Determine history length from network (Python-level, before JIT) ----
    history_len = get_history_len(state.params)

    # ---- Shock mask ----
    if config.shock_mask is not None and config.verbose:
        shock_names = model.shock_names if model.shock_names else tuple(f"shock_{i}" for i in range(model.n_shocks))
        active = [n for n, m in zip(shock_names, config.shock_mask) if m > 0]
        zeroed = [n for n, m in zip(shock_names, config.shock_mask) if m == 0]
        print(f"  Shock mask: active={active}, zeroed={zeroed}")

    custom_loss_fn = _build_custom_loss_fn(config, model, history_len)

    # ---- Target network setup ----
    use_target = config.target_update_every > 0
    if use_target:
        state = state._replace(target_params=state.params)
        if config.verbose:
            print(f"  Target network: update every {config.target_update_every} episodes"
                  f" (tau={config.target_tau})")

    # ---- Create JIT-compiled train step ----
    gradient_surgery = config.gradient_surgery
    train_step = make_train_step(
        model, opt, config.episode_length, config.mc_samples, config.batch_size,
        loss_reweight=config.loss_reweight,
        reweight_alpha=config.reweight_alpha,
        kind=kind,
        gradient_surgery=gradient_surgery,
        grad_clip=config.optimizer.grad_clip,
        quad_nodes=quad_nodes_jax,
        quad_weights=quad_weights_jax,
        history_len=history_len,
        compute_loss_fn=custom_loss_fn,
        ss_reset_frac=config.ss_reset_frac,
        use_target_network=use_target,
        n_epochs_per_rollout=config.n_epochs_per_rollout,
        n_minibatches_per_epoch=config.n_minibatches_per_epoch,
        initialize_each_episode=config.initialize_each_episode,
        sorted_within_batch=config.sorted_within_batch,
    )

    if config.verbose and kind == OptimizerKind.STANDARD and gradient_surgery != "pcgrad":
        # Compute and report the effective schedule so users can see what
        # the trainer is actually doing per outer iteration.
        ep_samples = config.episode_length * config.batch_size
        mbs_avail = max(1, ep_samples // config.batch_size)
        mbs_this_epoch = (
            min(config.n_minibatches_per_epoch, mbs_avail)
            if config.n_minibatches_per_epoch is not None
            else mbs_avail
        )
        updates_per_cycle = config.n_epochs_per_rollout * mbs_this_epoch
        print(
            f"  Schedule: 1 rollout ({config.episode_length}×{config.batch_size}="
            f"{ep_samples} states) → {config.n_epochs_per_rollout} epoch(s) × "
            f"{mbs_this_epoch} minibatch(es) of {config.batch_size} "
            f"= {updates_per_cycle} grad updates/cycle "
            f"({updates_per_cycle * config.episodes} total over "
            f"{config.episodes} cycles)"
        )

    return _run_training_loop(
        config=config,
        model=model,
        state=state,
        opt=opt,
        kind=kind,
        gradient_surgery=gradient_surgery,
        train_step=train_step,
        lr_schedule_fn=lr_schedule_fn,
        quad_nodes_jax=quad_nodes_jax,
        quad_weights_jax=quad_weights_jax,
        history_len=history_len,
        custom_loss_fn=custom_loss_fn,
        use_target=use_target,
        n_equations=n_equations,
        start_episode=start_episode,
        logger=logger,
    )


def train(
    model_name: str,
    episodes: int = 1000,
    hidden_sizes: Tuple[int, ...] = (64, 64),
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    episode_length: int = 100,
    mc_samples: int = 5,
    optimizer: str = "adam",
    warm_start: bool = False,
    seed: int = 42,
    log_every: int = 100,
    verbose: bool = True,
    grad_clip: Optional[float] = None,
    loss_weights: Optional[List[float]] = None,
    loss_reweight: str = "none",
    reweight_alpha: float = 0.9,
    tensorboard_dir: Optional[str] = None,
    wandb_project: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: Optional[int] = None,
) -> Tuple[Any, Dict[str, list]]:
    """Train DEQN model (backward-compatible wrapper).

    Builds a TrainConfig and delegates to train_from_config().

    Preserves the pre-sweep per-cycle training budget (one gradient step
    per cycle) to avoid surprising legacy callers. New code should prefer
    ``train_from_config(TrainConfig(...))`` which defaults to the full
    rollout+minibatch-sweep schedule matching reference DEQN.
    """
    from deqn_jax.config import NetworkConfig, OptimizerConfig, TrainConfig

    config = TrainConfig(
        model=model_name,
        episodes=episodes,
        batch_size=batch_size,
        episode_length=episode_length,
        mc_samples=mc_samples,
        seed=seed,
        network=NetworkConfig(hidden_sizes=hidden_sizes),
        optimizer=OptimizerConfig(
            name=optimizer,
            learning_rate=learning_rate,
            grad_clip=grad_clip,
        ),
        warm_start=warm_start,
        loss_weights=list(loss_weights) if loss_weights is not None else None,
        loss_reweight=loss_reweight,
        reweight_alpha=reweight_alpha,
        log_every=log_every,
        verbose=verbose,
        tensorboard_dir=tensorboard_dir,
        wandb_project=wandb_project,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        n_minibatches_per_epoch=1,
    )

    return train_from_config(config)
