"""Main training loop for DEQN-JAX.

Key design: single JIT boundary around entire train_step for maximum performance.
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
        # Discover from a dummy call if equation_names not set
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


def _save_checkpoint(state: TrainState, checkpoint_dir: str, episode: int):
    """Save training state checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_{episode:06d}.eqx")
    eqx.tree_serialise_leaves(path, state)


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
) -> Tuple[TrainState, optax.GradientTransformation]:
    """Initialize training state and optimizer.

    Args:
        model: Model specification
        key: PRNG key
        hidden_sizes: MLP hidden layer sizes
        learning_rate: Optimizer learning rate
        batch_size: Batch size for states
        optimizer: Optimizer name ("adam", "sgd", "adamw")
        grad_clip: Global gradient clipping norm (None = no clipping)
        loss_weights: Manual per-equation weights (None = uniform)
        n_equations: Number of equations (for reweight state init)

    Returns:
        Tuple of (TrainState, optax optimizer)
    """
    key, net_key, state_key = jax.random.split(key, 3)

    # Create policy network
    policy_net = create_mlp(
        n_states=model.n_states,
        n_policies=model.n_policies,
        hidden_sizes=hidden_sizes,
        policy_lower=model.policy_lower,
        policy_upper=model.policy_upper,
        key=net_key,
    )

    # Create base optimizer
    if optimizer == "sgd":
        base_opt = optax.sgd(learning_rate)
    elif optimizer == "adamw":
        base_opt = optax.adamw(learning_rate)
    else:
        base_opt = optax.adam(learning_rate)

    # Optionally chain with gradient clipping
    if grad_clip is not None:
        opt = optax.chain(optax.clip_by_global_norm(grad_clip), base_opt)
    else:
        opt = base_opt

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

    return state, opt


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
    # Softmax of ratios
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
# JIT-compiled train step
# ---------------------------------------------------------------------------

def make_train_step(
    model: ModelSpec,
    opt: optax.GradientTransformation,
    episode_length: int,
    mc_samples: int,
    batch_size: int,
    loss_reweight: str = "none",
    reweight_alpha: float = 0.9,
):
    """Create a JIT-compiled training step function.

    Uses closure to capture model and optimizer, avoiding hashability issues.

    Args:
        model: Model specification
        opt: Optax optimizer
        episode_length: Steps per episode
        mc_samples: MC samples for loss
        batch_size: Batch size
        loss_reweight: Adaptive strategy ("none", "lr_annealing", "relobralo")
        reweight_alpha: EMA decay for adaptive reweighting

    Returns:
        JIT-compiled train_step function
    """
    # Discover n_eq from equation_names or default to 1
    n_eq = len(model.equation_names) if model.equation_names else 1

    @jax.jit
    def train_step(state: TrainState) -> Tuple[TrainState, Metrics]:
        """Single training step (JIT-compiled).

        This is the hot path - everything is compiled into one XLA program.
        """
        key = state.key
        key, episode_key, loss_key, shuffle_key = jax.random.split(key, 4)

        # Run episode to collect trajectory
        trajectory, final_state = run_episode(
            model,
            state.params,
            state.episode_state,
            episode_key,
            episode_length,
        )

        # Flatten trajectory to [episode_length * batch, n_states]
        # trajectory is [episode_length, batch, n_states]
        all_states = trajectory.reshape(-1, model.n_states)

        # Shuffle and take batch
        n_states = all_states.shape[0]
        indices = jax.random.permutation(shuffle_key, n_states)[:batch_size]
        train_states = all_states[indices]

        # Loss and gradient (with current weights)
        def loss_fn(params):
            loss, eq_losses = compute_loss(
                model, params, train_states, loss_key, mc_samples,
                weights=state.loss_weights,
            )
            return loss, eq_losses

        (loss, eq_losses), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(state.params)

        # Update parameters
        params_arrays = eqx.filter(state.params, eqx.is_array)
        grads_arrays = eqx.filter(grads, eqx.is_array)

        updates, new_opt_state = opt.update(grads_arrays, state.opt_state, params_arrays)
        new_params_arrays = optax.apply_updates(params_arrays, updates)

        # Combine updated arrays with static parts
        new_params = eqx.combine(new_params_arrays, state.params)

        # Compute gradient norm for monitoring
        grad_norm = optax.global_norm(grads_arrays)

        # Update loss weights if adaptive reweighting is enabled
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

        metrics = Metrics(
            loss=loss,
            residuals=eq_losses,
            grad_norm=grad_norm,
        )

        return new_state, metrics

    return train_step


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

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
    """Train DEQN model.

    Args:
        model_name: Name of model to train ("brock_mirman", "disaster")
        episodes: Number of training episodes
        hidden_sizes: MLP hidden layer sizes
        learning_rate: Optimizer learning rate
        batch_size: Training batch size
        episode_length: Steps per episode
        mc_samples: Monte Carlo samples for expectations
        optimizer: Optimizer name
        warm_start: Initialize network from steady state using L-BFGS
        seed: Random seed
        log_every: Log frequency
        verbose: Print progress
        grad_clip: Global gradient clipping norm (None = no clipping)
        loss_weights: Manual per-equation weights (None = uniform)
        loss_reweight: Adaptive strategy ("none", "lr_annealing", "relobralo")
        reweight_alpha: EMA decay for adaptive reweighting
        tensorboard_dir: TensorBoard log directory (None = disabled)
        wandb_project: W&B project name (None = disabled)
        checkpoint_dir: Checkpoint save directory (None = disabled)
        checkpoint_every: Checkpoint interval in episodes (None = disabled)

    Returns:
        Tuple of (trained_params, history_dict)
    """
    # Import model
    if model_name == "brock_mirman":
        from deqn_jax.models.brock_mirman import MODEL
    elif model_name == "disaster":
        from deqn_jax.models.disaster import MODEL
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = MODEL

    # Discover n_equations
    n_equations = len(model.equation_names) if model.equation_names else 1

    # Validate loss_weights length
    if loss_weights is not None and len(loss_weights) != n_equations:
        raise ValueError(
            f"loss_weights has {len(loss_weights)} entries but model "
            f"has {n_equations} equations"
        )

    # Detect fp64
    fp64 = jnp.zeros(1).dtype == jnp.float64

    # Initialize state + optimizer
    key = jax.random.PRNGKey(seed)
    state, opt = create_train_state(
        model,
        key,
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        batch_size=batch_size,
        optimizer=optimizer,
        grad_clip=grad_clip,
        loss_weights=loss_weights,
        n_equations=n_equations,
    )

    # Warm start from steady state
    if warm_start:
        from deqn_jax.training.warm_start import warm_start_network
        state = state._replace(
            params=warm_start_network(state.params, model, verbose=verbose)
        )

    # Create metric logger
    wandb_config = dict(
        model=model_name,
        episodes=episodes,
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        batch_size=batch_size,
        mc_samples=mc_samples,
        optimizer=optimizer,
        grad_clip=grad_clip,
        loss_reweight=loss_reweight,
    ) if wandb_project else None

    logger = create_logger(
        tensorboard_dir=tensorboard_dir,
        wandb_project=wandb_project,
        wandb_config=wandb_config,
    )

    # Print header
    n_params = _count_params(state.params)
    if verbose:
        _print_header(
            model_spec=model,
            optimizer=optimizer,
            learning_rate=learning_rate,
            hidden_sizes=hidden_sizes,
            n_params=n_params,
            batch_size=batch_size,
            mc_samples=mc_samples,
            warm_start=warm_start,
            grad_clip=grad_clip,
            loss_reweight=loss_reweight,
            fp64=fp64,
        )

    # Create JIT-compiled train step
    train_step = make_train_step(
        model, opt, episode_length, mc_samples, batch_size,
        loss_reweight=loss_reweight,
        reweight_alpha=reweight_alpha,
    )

    # Training history
    history: Dict[str, list] = {
        "loss": [],
        "grad_norm": [],
    }

    # Training loop
    t_start = time.perf_counter()
    last_metrics = None

    for ep in range(episodes):
        state, metrics = train_step(state)
        last_metrics = metrics

        loss_val = float(metrics.loss)
        grad_val = float(metrics.grad_norm)
        history["loss"].append(loss_val)
        history["grad_norm"].append(grad_val)

        ep_num = ep + 1

        # Log to backends
        if ep_num % log_every == 0 or ep_num == episodes:
            log_dict = {"loss": loss_val, "grad_norm": grad_val}
            if metrics.residuals:
                for k, v in metrics.residuals.items():
                    log_dict[f"eq/{k}"] = float(v)
            logger.log_scalars(log_dict, step=ep_num)

        # Print progress
        if verbose and ep_num % log_every == 0:
            elapsed = time.perf_counter() - t_start
            eps = ep_num / elapsed if elapsed > 0 else 0
            residuals = metrics.residuals or {}
            residual_str = " | ".join(
                f"{k}={float(v):.2e}" for k, v in residuals.items()
            )
            sep = " | " if residual_str else ""
            print(
                f"  [{ep_num:>{len(str(episodes))}}/{episodes}] "
                f"loss={loss_val:.2e}{sep}{residual_str} | "
                f"grad={grad_val:.2e} | {eps:.0f} ep/s"
            )

        # Checkpointing
        if (
            checkpoint_dir is not None
            and checkpoint_every is not None
            and ep_num % checkpoint_every == 0
        ):
            _save_checkpoint(state, checkpoint_dir, ep_num)

    # Final summary
    elapsed = time.perf_counter() - t_start

    if verbose and last_metrics is not None:
        _print_final(
            elapsed=elapsed,
            episodes=episodes,
            final_loss=float(last_metrics.loss),
            final_residuals=last_metrics.residuals,
        )

    logger.close()

    return state.params, history
