"""Main training loop for DEQN-JAX.

Key design: single JIT boundary around entire train_step for maximum performance.
"""

from typing import Any, Callable, Dict, Optional, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
import equinox as eqx
import optax

from deqn_jax.types import ModelSpec, TrainState, Metrics
from deqn_jax.networks import create_mlp
from deqn_jax.training.loss import compute_loss
from deqn_jax.training.episode import run_episode, sample_initial_states


def create_train_state(
    model: ModelSpec,
    key: Array,
    hidden_sizes: Tuple[int, ...] = (64, 64),
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    optimizer: str = "adam",
) -> TrainState:
    """Initialize training state.

    Args:
        model: Model specification
        key: PRNG key
        hidden_sizes: MLP hidden layer sizes
        learning_rate: Optimizer learning rate
        batch_size: Batch size for states
        optimizer: Optimizer name ("adam", "sgd", "adamw")

    Returns:
        Initialized TrainState
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

    # Create optimizer
    if optimizer == "sgd":
        opt = optax.sgd(learning_rate)
    elif optimizer == "adamw":
        opt = optax.adamw(learning_rate)
    else:
        opt = optax.adam(learning_rate)

    opt_state = opt.init(eqx.filter(policy_net, eqx.is_array))

    # Sample initial states
    init_states = sample_initial_states(model, state_key, batch_size)

    return TrainState(
        params=policy_net,
        opt_state=opt_state,
        episode_state=init_states,
        key=key,
        step=0,
        episode=0,
    )


def make_train_step(
    model: ModelSpec,
    opt: optax.GradientTransformation,
    episode_length: int,
    mc_samples: int,
    batch_size: int,
):
    """Create a JIT-compiled training step function.

    Uses closure to capture model and optimizer, avoiding hashability issues.

    Args:
        model: Model specification
        opt: Optax optimizer
        episode_length: Steps per episode
        mc_samples: MC samples for loss
        batch_size: Batch size

    Returns:
        JIT-compiled train_step function
    """

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

        # Loss and gradient
        def loss_fn(params):
            loss, eq_losses = compute_loss(model, params, train_states, loss_key, mc_samples)
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

        new_state = TrainState(
            params=new_params,
            opt_state=new_opt_state,
            episode_state=final_state,
            key=key,
            step=state.step + 1,
            episode=state.episode + 1,
        )

        metrics = Metrics(
            loss=loss,
            residuals=eq_losses,
            grad_norm=grad_norm,
        )

        return new_state, metrics

    return train_step


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

    # Initialize
    key = jax.random.PRNGKey(seed)
    state = create_train_state(
        model,
        key,
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        batch_size=batch_size,
        optimizer=optimizer,
    )

    # Warm start from steady state
    if warm_start:
        from deqn_jax.training.warm_start import warm_start_network
        state = state._replace(
            params=warm_start_network(state.params, model, verbose=verbose)
        )

    # Create optimizer for JIT
    if optimizer == "sgd":
        opt = optax.sgd(learning_rate)
    elif optimizer == "adamw":
        opt = optax.adamw(learning_rate)
    else:
        opt = optax.adam(learning_rate)

    # Training history
    history = {
        "loss": [],
        "grad_norm": [],
    }

    if verbose:
        print(f"Training {model_name} for {episodes} episodes...")
        print(f"  Hidden sizes: {hidden_sizes}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {batch_size}")
        print(f"  MC samples: {mc_samples}")
        print()

    # Create JIT-compiled train step
    train_step = make_train_step(model, opt, episode_length, mc_samples, batch_size)

    # Training loop
    for ep in range(episodes):
        state, metrics = train_step(state)

        history["loss"].append(float(metrics.loss))
        history["grad_norm"].append(float(metrics.grad_norm))

        if verbose and (ep + 1) % log_every == 0:
            residuals = metrics.residuals or {}
            residual_str = ", ".join(
                f"{k}={float(v):.2e}" for k, v in residuals.items()
            )
            print(
                f"Episode {ep + 1:5d} | Loss: {float(metrics.loss):.4e} | "
                f"Grad: {float(metrics.grad_norm):.2e} | {residual_str}"
            )

    if verbose:
        print(f"\nFinal loss: {history['loss'][-1]:.4e}")

    return state.params, history
