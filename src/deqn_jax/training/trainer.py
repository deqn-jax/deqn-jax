"""Main training loop for DEQN-JAX.

Key design: single JIT boundary around entire train_step for maximum performance.
Three step variants dispatched at construction time (before JIT):

- STANDARD: normal jax.grad + opt.update(grads, state, params)
- MAO: jax.jacrev(per_eq_loss_vector) -> per-equation Jacobian -> mao.update(eq_jac, state, params)
- LBFGS: optax.lbfgs (GradientTransformationExtraArgs) -- needs value + value_fn for line search
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
import equinox as eqx
import optax

from deqn_jax.types import ModelSpec, TrainState, Metrics, ReweightState, make_reweight_state
from deqn_jax.networks import create_mlp
from deqn_jax.training.loss import compute_loss, eq_losses_to_array
from deqn_jax.training.episode import run_episode, sample_initial_states
from deqn_jax.metrics import create_logger
from deqn_jax.optimizers.registry import OptimizerKind, create_optimizer
from deqn_jax.optimizers.mao import _MAOFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_params(model: eqx.Module) -> int:
    """Count trainable parameters in an Equinox model."""
    params = eqx.filter(model, eqx.is_array)
    leaves = jax.tree_util.tree_leaves(params)
    return sum(x.size for x in leaves)


def _network_shape_str(model_spec: ModelSpec, hidden_sizes: Tuple[int, ...]) -> str:
    """Format network shape as e.g. '2 -> 64 -> 64 -> 1'."""
    sizes = [model_spec.n_states] + list(hidden_sizes) + [model_spec.n_policies]
    return " \u2192 ".join(str(s) for s in sizes)


def _format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def _print_header(
    model_spec: ModelSpec,
    optimizer: str,
    learning_rate: float,
    hidden_sizes: Tuple[int, ...],
    n_params: int,
    batch_size: int,
    mc_samples: int,
    warm_start: bool,
    grad_clip: Optional[float],
    loss_reweight: str,
    fp64: bool,
):
    """Print rich training header."""
    eq_names = list(model_spec.equation_names or [])
    if not eq_names:
        eq_names = ["(auto-discovered)"]

    precision = "float64" if fp64 else "float32"
    ws_str = "yes" if warm_start else "no"
    clip_str = str(grad_clip) if grad_clip else "none"
    reweight_str = loss_reweight if loss_reweight != "none" else "none"

    w = 60
    print("=" * w)
    print("DEQN-JAX Training")
    print("=" * w)
    print(f"  Model:           {model_spec.name}")
    print(f"  Optimizer:       {optimizer} (lr={learning_rate:.0e})")
    print(f"  Precision:       {precision}")
    print(f"  Network:         MLP [{_network_shape_str(model_spec, hidden_sizes)}]")
    print(f"  Parameters:      {n_params:,}")
    print(f"  Batch size:      {batch_size}")
    print(f"  MC samples:      {mc_samples}")
    print(f"  Warm start:      {ws_str}")
    if grad_clip:
        print(f"  Grad clip:       {clip_str}")
    if loss_reweight != "none":
        print(f"  Reweighting:     {reweight_str}")
    print("=" * w)


def _print_final(
    elapsed: float,
    episodes: int,
    final_loss: float,
    final_residuals: Optional[Dict[str, float]],
):
    """Print final training summary."""
    eps_per_sec = episodes / elapsed if elapsed > 0 else 0
    w = 60
    print("=" * w)
    print(f"Training complete in {_format_time(elapsed)} ({eps_per_sec:.0f} ep/s)")
    print(f"Final loss: {final_loss:.2e}")
    if final_residuals:
        for name, val in final_residuals.items():
            print(f"  {name}: {float(val):.2e}")
    print("=" * w)


def _save_checkpoint(state: TrainState, checkpoint_dir: str, episode: int, config=None):
    """Save training state checkpoint and optionally config snapshot."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_{episode:06d}.eqx")
    eqx.tree_serialise_leaves(path, state)
    # Save config once (first checkpoint only)
    if config is not None:
        cfg_path = os.path.join(checkpoint_dir, "config.yaml")
        if not os.path.exists(cfg_path):
            config.to_yaml(cfg_path)


def _prune_checkpoints(checkpoint_dir: str, max_keep: int):
    """Delete oldest checkpoints, keeping only the most recent max_keep."""
    import glob as glob_mod
    pattern = os.path.join(checkpoint_dir, "checkpoint_*.eqx")
    existing = sorted(glob_mod.glob(pattern))
    while len(existing) > max_keep:
        os.remove(existing.pop(0))


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
    if network_config is not None:
        hidden_sizes = network_config.hidden_sizes
        activation = network_config.activation
        activations = network_config.activations
        init = network_config.init

    # Create policy network
    policy_net = create_mlp(
        n_states=model.n_states,
        n_policies=model.n_policies,
        hidden_sizes=hidden_sizes,
        activation=activation,
        activations=activations,
        init=init,
        policy_lower=model.policy_lower,
        policy_upper=model.policy_upper,
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

    # Resolve MAO factory
    if kind == OptimizerKind.MAO:
        if isinstance(opt, _MAOFactory):
            opt = opt.with_num_tasks(n_equations)
        opt_state = opt.init(eqx.filter(policy_net, eqx.is_array))
    else:
        opt_state = opt.init(eqx.filter(policy_net, eqx.is_array))

    # Sample initial states
    init_states = sample_initial_states(model, state_key, batch_size)

    # Loss weights
    if loss_weights is not None:
        weights = jnp.array(loss_weights)
    else:
        weights = jnp.ones(n_equations)

    state = TrainState(
        params=policy_net,
        opt_state=opt_state,
        episode_state=init_states,
        key=key,
        step=0,
        episode=0,
        loss_weights=weights,
        reweight_state=make_reweight_state(n_equations),
    )

    return state, opt, kind


# ---------------------------------------------------------------------------
# Adaptive reweighting (pure functions for JIT)
# ---------------------------------------------------------------------------

def _update_weights_lr_annealing(
    eq_loss_arr: Array,
    reweight_state: ReweightState,
    alpha: float,
    n_eq: int,
) -> Tuple[Array, ReweightState]:
    """LR annealing: inverse EMA weighting, normalized to sum=n_eq."""
    new_running = alpha * reweight_state.running_max + (1.0 - alpha) * eq_loss_arr
    raw = 1.0 / (new_running + 1e-8)
    weights = raw / jnp.sum(raw) * n_eq
    new_rw = reweight_state._replace(running_max=new_running, prev_losses=eq_loss_arr)
    return weights, new_rw


def _update_weights_relobralo(
    eq_loss_arr: Array,
    reweight_state: ReweightState,
    alpha: float,
    n_eq: int,
) -> Tuple[Array, ReweightState]:
    """ReLoBRaLo: relative balancing with softmax of loss ratios."""
    init = jnp.where(reweight_state.initialized, reweight_state.init_losses, eq_loss_arr)
    prev = jnp.where(reweight_state.initialized, reweight_state.prev_losses, eq_loss_arr)

    eps = 1e-8
    w_t = jax.nn.softmax(eq_loss_arr / (prev + eps)) * n_eq
    w_0 = jax.nn.softmax(eq_loss_arr / (init + eps)) * n_eq
    weights = alpha * w_t + (1.0 - alpha) * w_0

    new_rw = reweight_state._replace(
        prev_losses=eq_loss_arr,
        init_losses=init,
        initialized=jnp.array(True),
    )
    return weights, new_rw


# ---------------------------------------------------------------------------
# Common episode + batch logic (shared by all step variants)
# ---------------------------------------------------------------------------

def _run_episode_and_sample(model, state, episode_length, batch_size):
    """Run episode and sample training batch. Returns (train_states, keys)."""
    key = state.key
    key, episode_key, loss_key, shuffle_key = jax.random.split(key, 4)

    trajectory, final_state = run_episode(
        model,
        state.params,
        state.episode_state,
        episode_key,
        episode_length,
    )

    all_states = trajectory.reshape(-1, model.n_states)
    n_states = all_states.shape[0]
    indices = jax.random.permutation(shuffle_key, n_states)[:batch_size]
    train_states = all_states[indices]

    return train_states, final_state, loss_key, key


def _update_reweighting(eq_losses, state, loss_reweight, reweight_alpha, n_eq):
    """Apply adaptive loss reweighting."""
    eq_loss_arr = eq_losses_to_array(eq_losses)

    if loss_reweight == "lr_annealing":
        new_weights, new_rw = _update_weights_lr_annealing(
            eq_loss_arr, state.reweight_state, reweight_alpha, n_eq,
        )
    elif loss_reweight == "relobralo":
        new_weights, new_rw = _update_weights_relobralo(
            eq_loss_arr, state.reweight_state, reweight_alpha, n_eq,
        )
    else:
        new_weights = state.loss_weights
        new_rw = state.reweight_state

    return new_weights, new_rw


# ---------------------------------------------------------------------------
# Three step variants
# ---------------------------------------------------------------------------

def _make_standard_step(
    model: ModelSpec,
    opt: Any,
    episode_length: int,
    mc_samples: int,
    batch_size: int,
    loss_reweight: str = "none",
    reweight_alpha: float = 0.9,
):
    """Standard train step: jax.grad + opt.update(grads, state, params)."""
    n_eq = len(model.equation_names) if model.equation_names else 1

    @jax.jit
    def train_step(state: TrainState) -> Tuple[TrainState, Metrics]:
        train_states, final_state, loss_key, key = _run_episode_and_sample(
            model, state, episode_length, batch_size,
        )

        def loss_fn(params):
            loss, eq_losses = compute_loss(
                model, params, train_states, loss_key, mc_samples,
                weights=state.loss_weights,
            )
            return loss, eq_losses

        (loss, eq_losses), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(state.params)

        params_arrays = eqx.filter(state.params, eqx.is_array)
        grads_arrays = eqx.filter(grads, eqx.is_array)

        updates, new_opt_state = opt.update(grads_arrays, state.opt_state, params_arrays)
        new_params_arrays = optax.apply_updates(params_arrays, updates)
        new_params = eqx.combine(new_params_arrays, state.params)

        grad_norm = optax.global_norm(grads_arrays)
        new_weights, new_rw = _update_reweighting(
            eq_losses, state, loss_reweight, reweight_alpha, n_eq,
        )

        new_state = TrainState(
            params=new_params,
            opt_state=new_opt_state,
            episode_state=final_state,
            key=key,
            step=state.step + 1,
            episode=state.episode + 1,
            loss_weights=new_weights,
            reweight_state=new_rw,
        )
        return new_state, Metrics(loss=loss, residuals=eq_losses, grad_norm=grad_norm)

    return train_step


def _make_mao_step(
    model: ModelSpec,
    mao_opt: Any,
    episode_length: int,
    mc_samples: int,
    batch_size: int,
    loss_reweight: str = "none",
    reweight_alpha: float = 0.9,
):
    """MAO train step: per-equation Jacobian -> mao.update(eq_jac, state, params)."""
    n_eq = len(model.equation_names) if model.equation_names else 1

    @jax.jit
    def train_step(state: TrainState) -> Tuple[TrainState, Metrics]:
        train_states, final_state, loss_key, key = _run_episode_and_sample(
            model, state, episode_length, batch_size,
        )

        params_arrays = eqx.filter(state.params, eqx.is_array)
        params_static = eqx.filter(state.params, lambda x: not eqx.is_array(x))

        # Per-equation loss vector: returns [n_eq] array
        def per_eq_loss_fn(p_arrays):
            full_params = eqx.combine(p_arrays, params_static)
            _, eq_losses = compute_loss(
                model, full_params, train_states, loss_key, mc_samples,
                weights=None,  # MAO handles weighting internally
            )
            return eq_losses_to_array(eq_losses)

        # Get per-equation Jacobian: pytree, each leaf [n_eq, *param_shape]
        eq_jac = jax.jacrev(per_eq_loss_fn)(params_arrays)

        # Also compute scalar loss + grad norm for metrics
        def total_loss_fn(params):
            loss, eq_losses = compute_loss(
                model, params, train_states, loss_key, mc_samples,
                weights=state.loss_weights,
            )
            return loss, eq_losses

        (loss, eq_losses), grads = eqx.filter_value_and_grad(total_loss_fn, has_aux=True)(
            state.params
        )
        grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_array))

        # MAO update
        updates, new_opt_state = mao_opt.update(eq_jac, state.opt_state, params_arrays)
        new_params_arrays = optax.apply_updates(params_arrays, updates)
        new_params = eqx.combine(new_params_arrays, state.params)

        new_weights, new_rw = _update_reweighting(
            eq_losses, state, loss_reweight, reweight_alpha, n_eq,
        )

        new_state = TrainState(
            params=new_params,
            opt_state=new_opt_state,
            episode_state=final_state,
            key=key,
            step=state.step + 1,
            episode=state.episode + 1,
            loss_weights=new_weights,
            reweight_state=new_rw,
        )
        return new_state, Metrics(loss=loss, residuals=eq_losses, grad_norm=grad_norm)

    return train_step


def _make_lbfgs_step(
    model: ModelSpec,
    opt: Any,
    episode_length: int,
    mc_samples: int,
    batch_size: int,
    loss_reweight: str = "none",
    reweight_alpha: float = 0.9,
):
    """L-BFGS train step: passes value + value_fn for line search."""
    n_eq = len(model.equation_names) if model.equation_names else 1

    @jax.jit
    def train_step(state: TrainState) -> Tuple[TrainState, Metrics]:
        train_states, final_state, loss_key, key = _run_episode_and_sample(
            model, state, episode_length, batch_size,
        )

        params_arrays = eqx.filter(state.params, eqx.is_array)
        params_static = eqx.filter(state.params, lambda x: not eqx.is_array(x))

        def loss_fn(params):
            loss, eq_losses = compute_loss(
                model, params, train_states, loss_key, mc_samples,
                weights=state.loss_weights,
            )
            return loss, eq_losses

        (loss, eq_losses), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(state.params)
        grads_arrays = eqx.filter(grads, eqx.is_array)
        grad_norm = optax.global_norm(grads_arrays)

        # Value function for line search (operates on param arrays only)
        def value_fn(p_arrays):
            full_params = eqx.combine(p_arrays, params_static)
            v, _ = compute_loss(
                model, full_params, train_states, loss_key, mc_samples,
                weights=state.loss_weights,
            )
            return v

        updates, new_opt_state = opt.update(
            grads_arrays,
            state.opt_state,
            params_arrays,
            value=loss,
            grad=grads_arrays,
            value_fn=value_fn,
        )
        new_params_arrays = optax.apply_updates(params_arrays, updates)
        new_params = eqx.combine(new_params_arrays, state.params)

        new_weights, new_rw = _update_reweighting(
            eq_losses, state, loss_reweight, reweight_alpha, n_eq,
        )

        new_state = TrainState(
            params=new_params,
            opt_state=new_opt_state,
            episode_state=final_state,
            key=key,
            step=state.step + 1,
            episode=state.episode + 1,
            loss_weights=new_weights,
            reweight_state=new_rw,
        )
        return new_state, Metrics(loss=loss, residuals=eq_losses, grad_norm=grad_norm)

    return train_step


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

    Returns:
        JIT-compiled train_step function
    """
    kwargs = dict(
        model=model,
        opt=opt if kind != OptimizerKind.MAO else None,
        episode_length=episode_length,
        mc_samples=mc_samples,
        batch_size=batch_size,
        loss_reweight=loss_reweight,
        reweight_alpha=reweight_alpha,
    )

    if kind == OptimizerKind.MAO:
        return _make_mao_step(
            model=model,
            mao_opt=opt,
            episode_length=episode_length,
            mc_samples=mc_samples,
            batch_size=batch_size,
            loss_reweight=loss_reweight,
            reweight_alpha=reweight_alpha,
        )
    elif kind == OptimizerKind.LBFGS:
        return _make_lbfgs_step(**kwargs)
    else:
        return _make_standard_step(**kwargs)


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
    from deqn_jax.models import load_model
    from deqn_jax.config import TrainConfig, OptimizerConfig

    model = load_model(config.model)
    n_equations = len(model.equation_names) if model.equation_names else 1

    if config.loss_weights is not None and len(config.loss_weights) != n_equations:
        raise ValueError(
            f"loss_weights has {len(config.loss_weights)} entries but model "
            f"has {n_equations} equations"
        )

    fp64 = jnp.zeros(1).dtype == jnp.float64
    hidden_sizes = config.network.hidden_sizes

    key = jax.random.PRNGKey(config.seed)

    # ---- Resume from checkpoint or create fresh state ----
    start_episode = 0

    if config.resume:
        # Load original config to reconstruct matching template for deserialization
        ckpt_dir = os.path.dirname(config.resume)
        orig_cfg_path = os.path.join(ckpt_dir, "config.yaml")
        if os.path.exists(orig_cfg_path):
            orig_config = TrainConfig.from_yaml(orig_cfg_path)
        else:
            orig_config = config  # assume same optimizer

        # Build template with ORIGINAL optimizer (matching checkpoint pytree structure)
        template_state, orig_opt, orig_kind = create_train_state(
            model,
            key,
            hidden_sizes=orig_config.network.hidden_sizes,
            batch_size=orig_config.batch_size,
            loss_weights=config.loss_weights,
            n_equations=n_equations,
            optimizer_config=orig_config.optimizer,
            network_config=orig_config.network,
        )

        state = eqx.tree_deserialise_leaves(config.resume, template_state)
        start_episode = int(state.episode)

        # Check if optimizer changed
        optimizer_changed = config.optimizer.name != orig_config.optimizer.name
        if optimizer_changed:
            new_opt, new_kind = create_optimizer(config.optimizer)
            if new_kind == OptimizerKind.MAO and isinstance(new_opt, _MAOFactory):
                new_opt = new_opt.with_num_tasks(n_equations)
            new_opt_state = new_opt.init(eqx.filter(state.params, eqx.is_array))
            state = state._replace(opt_state=new_opt_state)
            opt, kind = new_opt, new_kind
            if config.verbose:
                print(f"  Resumed from {config.resume} (episode {start_episode})")
                print(f"  Switched optimizer: {orig_config.optimizer.name} -> {config.optimizer.name}")
        else:
            opt, kind = orig_opt, orig_kind
            if config.verbose:
                print(f"  Resumed from {config.resume} (episode {start_episode})")
    else:
        state, opt, kind = create_train_state(
            model,
            key,
            hidden_sizes=hidden_sizes,
            batch_size=config.batch_size,
            loss_weights=config.loss_weights,
            n_equations=n_equations,
            optimizer_config=config.optimizer,
            network_config=config.network,
        )

        # Warm start from steady state (only for fresh training)
        if config.warm_start:
            from deqn_jax.training.warm_start import warm_start_network
            state = state._replace(
                params=warm_start_network(state.params, model, verbose=config.verbose)
            )

    # ---- Metric logger ----
    wandb_config = config.to_dict() if config.wandb_project else None
    logger = create_logger(
        tensorboard_dir=config.tensorboard_dir,
        wandb_project=config.wandb_project,
        wandb_config=wandb_config,
    )

    # ---- Print header ----
    n_params = _count_params(state.params)
    if config.verbose:
        _print_header(
            model_spec=model,
            optimizer=config.optimizer.name,
            learning_rate=config.optimizer.learning_rate,
            hidden_sizes=hidden_sizes,
            n_params=n_params,
            batch_size=config.batch_size,
            mc_samples=config.mc_samples,
            warm_start=config.warm_start,
            grad_clip=config.optimizer.grad_clip,
            loss_reweight=config.loss_reweight,
            fp64=fp64,
        )

    # ---- Create JIT-compiled train step ----
    train_step = make_train_step(
        model, opt, config.episode_length, config.mc_samples, config.batch_size,
        loss_reweight=config.loss_reweight,
        reweight_alpha=config.reweight_alpha,
        kind=kind,
    )

    # ---- Mid-training optimizer switch setup ----
    switch_episode = config.switch_episode
    switched = False
    if config.switch_optimizer and config.switch_episode is None:
        raise ValueError("--switch-optimizer requires --switch-episode")

    # ---- Training loop ----
    total_episodes = start_episode + config.episodes
    ep_width = len(str(total_episodes))

    history: Dict[str, list] = {"loss": [], "grad_norm": []}
    t_start = time.perf_counter()
    last_metrics = None

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
            if new_kind == OptimizerKind.MAO and isinstance(new_opt, _MAOFactory):
                new_opt = new_opt.with_num_tasks(n_equations)
            new_opt_state = new_opt.init(eqx.filter(state.params, eqx.is_array))
            state = state._replace(opt_state=new_opt_state)
            opt, kind = new_opt, new_kind
            train_step = make_train_step(
                model, opt, config.episode_length, config.mc_samples, config.batch_size,
                loss_reweight=config.loss_reweight,
                reweight_alpha=config.reweight_alpha,
                kind=kind,
            )
            switched = True
            if config.verbose:
                print(f"  >> Switched to {config.switch_optimizer} (lr={switch_lr:.0e}) at episode {ep_num}")

        state, metrics = train_step(state)
        last_metrics = metrics

        loss_val = float(metrics.loss)
        grad_val = float(metrics.grad_norm)
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
            if metrics.residuals:
                for k, v in metrics.residuals.items():
                    log_dict[f"eq/{k}"] = float(v)
            # Log per-equation weights when adaptive reweighting is active
            if config.loss_reweight != "none" and model.equation_names:
                for i, name in enumerate(model.equation_names):
                    log_dict[f"weights/{name}"] = float(state.loss_weights[i])
            logger.log_scalars(log_dict, step=ep_num)

        if config.verbose and ep_num % config.log_every == 0:
            elapsed = time.perf_counter() - t_start
            eps_done = ep_num - start_episode
            ep_per_sec = eps_done / elapsed if elapsed > 0 else 0
            residuals = metrics.residuals or {}
            residual_str = " | ".join(
                f"{k}={float(v):.2e}" for k, v in residuals.items()
            )
            sep = " | " if residual_str else ""
            print(
                f"  [{ep_num:>{ep_width}}/{total_episodes}] "
                f"loss={loss_val:.2e}{sep}{residual_str} | "
                f"grad={grad_val:.2e} | {ep_per_sec:.0f} ep/s"
            )

        # ---- Checkpointing with config snapshot + pruning ----
        if (
            config.checkpoint_dir is not None
            and config.checkpoint_every is not None
            and ep_num % config.checkpoint_every == 0
        ):
            _save_checkpoint(state, config.checkpoint_dir, ep_num, config=config)
            if config.max_checkpoints is not None:
                _prune_checkpoints(config.checkpoint_dir, config.max_checkpoints)

    elapsed = time.perf_counter() - t_start

    if config.verbose and last_metrics is not None:
        _print_final(
            elapsed=elapsed,
            episodes=config.episodes,
            final_loss=float(last_metrics.loss),
            final_residuals=last_metrics.residuals,
        )

    logger.close()
    return state.params, history


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
    """
    from deqn_jax.config import TrainConfig, OptimizerConfig, NetworkConfig

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
    )

    return train_from_config(config)
