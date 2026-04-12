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
from typing import Any, Callable, Dict, List, NamedTuple as _NamedTuple, Optional, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
import equinox as eqx
import optax

from deqn_jax.types import ModelSpec, TrainState, Metrics, ReweightState, make_reweight_state
from deqn_jax.networks import create_mlp, create_lstm, create_transformer
from deqn_jax.training.loss import compute_loss, compute_residuals, eq_losses_to_array, sample_antithetic_shocks, gauss_hermite_nd
from deqn_jax.training.episode import run_episode, run_episode_with_history, sample_initial_states
from deqn_jax.training.history import get_history_len, make_constant_history, build_history_windows
from deqn_jax.metrics import create_logger
from deqn_jax.optimizers.registry import OptimizerKind, create_optimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import re


def _strip_eq_prefix(name: str) -> str:
    """Strip 'eq1_', 'eq12_' numeric prefixes from equation names."""
    return re.sub(r"^eq\d+_", "", name)


def _print_residual_table(items: list, n_cols: int = 3):
    """Print residuals as an aligned multi-column table."""
    name_width = max(len(n) for n, _ in items)
    col_width = name_width + 11  # name + space + "1.23e-04" + padding
    rows = (len(items) + n_cols - 1) // n_cols
    for r in range(rows):
        parts = []
        for c in range(n_cols):
            idx = r + c * rows
            if idx < len(items):
                n, v = items[idx]
                parts.append(f"{n:<{name_width}} {v:>9.2e}")
        print("    " + "   ".join(parts))


def _eq4_diagnostics(
    model: ModelSpec,
    policy_fn,
    states: Array,
    policy_out: Array,
    defs: Dict[str, Array],
) -> Dict[str, float]:
    """Compute eq4 (wage_phillips_K) diagnostic quantities.

    Runs a zero-shock forward step to get next-period definitions,
    then decomposes the eq4 residual into its constituent terms.

    Returns scalar statistics (mean/std) for TensorBoard and console logging.
    """
    import numpy as np
    c = model.constants
    batch_size = states.shape[0]

    # Next state with zero shock (mean scenario)
    zero_shock = jnp.zeros((batch_size, model.n_shocks))
    next_states = model.step_fn(states, policy_out, zero_shock, c)
    next_policies = jax.vmap(policy_fn)(next_states)
    defs_n = jax.vmap(
        lambda s, p: model.definitions_fn(s, p, c)
    )(next_states, next_policies)

    # eq4 ratio base: pi_w_tilda' * mu_z_ss / pi_w'
    ratio_base = defs_n["pi_w_tilda"] * c["mu_z_ss"] / defs_n["pi_w"]
    exponent = c["lambda_w"] / (1 - c["lambda_w"]) * (1 + c["sigma_L"])
    eq4_ratio = ratio_base ** exponent

    # Three terms of eq4
    sigma_L = c["sigma_L"]
    # h is a policy variable — extract index from policy_names
    h_idx = list(model.policy_names).index("h") if "h" in model.policy_names else None
    if h_idx is not None:
        h = policy_out[:, h_idx]
        h_term = h ** (1 + sigma_L)
    else:
        h_term = jnp.zeros(batch_size)

    expect_term = c["beta"] * c["xi_w"] * eq4_ratio * defs_n["K_w"]
    K_w = defs["K_w"]
    eq4_rhs = h_term + expect_term
    # Log-space residual (matches equations.py)
    log_residual = jnp.log(jnp.maximum(eq4_rhs, 1e-8)) - jnp.log(jnp.maximum(K_w, 1e-8))

    # K_w_inner: check how many hit the soft floor zone
    xi_w = c["xi_w"]
    lambda_w = c["lambda_w"]
    K_w_inner_ratio = (defs["pi_w_tilda"] / defs["pi_w"] * c["mu_z_ss"]) ** (1 / (1 - lambda_w))
    K_w_inner = (1 - xi_w * K_w_inner_ratio) / (1 - xi_w)
    floor_frac = float(np.mean(np.asarray(K_w_inner) < 0.02))

    rb = np.asarray(ratio_base)
    lr = np.asarray(log_residual)
    return {
        "ratio_base_mean": float(np.mean(rb)),
        "ratio_base_std": float(np.std(rb)),
        "ratio_base_min": float(np.min(rb)),
        "ratio_base_max": float(np.max(rb)),
        "eq4_ratio_mean": float(np.mean(np.asarray(eq4_ratio))),
        "K_w_mean": float(np.mean(np.asarray(K_w))),
        "K_w_n_mean": float(np.mean(np.asarray(defs_n["K_w"]))),
        "h_term_mean": float(np.mean(np.asarray(h_term))),
        "expect_term_mean": float(np.mean(np.asarray(expect_term))),
        "log_residual_mean": float(np.mean(lr)),
        "log_residual_std": float(np.std(lr)),
        "K_w_inner_floor_frac": floor_frac,
        "exponent": float(exponent),
    }


def _eq2_diagnostics(
    model: ModelSpec,
    policy_fn,
    states: Array,
    policy_out: Array,
    defs: Dict[str, Array],
) -> Dict[str, float]:
    """Compute eq2 (price_phillips_K) diagnostic quantities.

    Mirrors _eq4_diagnostics() for eq4. Decomposes eq2 residual into
    constituent terms to diagnose convergence issues.

    Returns scalar statistics (mean/std) for TensorBoard and console logging.
    """
    import numpy as np
    c = model.constants
    batch_size = states.shape[0]

    # Next state with zero shock (mean scenario)
    zero_shock = jnp.zeros((batch_size, model.n_shocks))
    next_states = model.step_fn(states, policy_out, zero_shock, c)
    next_policies = jax.vmap(policy_fn)(next_states)
    defs_n = jax.vmap(
        lambda s, p: model.definitions_fn(s, p, c)
    )(next_states, next_policies)

    # eq2 ratio base: pi_tilda' / pi'
    ratio_base = defs_n["pi_tilda"] / next_policies[:, list(model.policy_names).index("pi")]
    exponent = c["lambda_f"] / (1 - c["lambda_f"])
    eq2_ratio = ratio_base ** exponent

    # Three terms of eq2
    lambda_z_idx = list(model.policy_names).index("lambda_z")
    lambda_z = policy_out[:, lambda_z_idx]
    lhs_term = lambda_z * c["lambda_f"] * defs["y_z"] * defs["s"]
    expect_term = c["beta"] * c["xi_p"] * eq2_ratio * defs_n["K_p"]
    K_p = defs["K_p"]
    eq2_rhs = lhs_term + expect_term
    # Log-space residual (matches equations.py)
    log_residual = jnp.log(jnp.maximum(eq2_rhs, 1e-8)) - jnp.log(jnp.maximum(K_p, 1e-8))

    # K_p_inner: check how many hit the soft floor zone
    xi_p = c["xi_p"]
    lambda_f = c["lambda_f"]
    K_p_inner_ratio = (defs["pi_tilda"] / policy_out[:, list(model.policy_names).index("pi")]) ** (1 / (1 - lambda_f))
    K_p_inner = (1 - xi_p * K_p_inner_ratio) / (1 - xi_p)
    floor_frac = float(np.mean(np.asarray(K_p_inner) < 0.02))

    rb = np.asarray(ratio_base)
    lr = np.asarray(log_residual)
    return {
        "ratio_base_mean": float(np.mean(rb)),
        "ratio_base_std": float(np.std(rb)),
        "ratio_base_min": float(np.min(rb)),
        "ratio_base_max": float(np.max(rb)),
        "eq2_ratio_mean": float(np.mean(np.asarray(eq2_ratio))),
        "K_p_mean": float(np.mean(np.asarray(K_p))),
        "K_p_n_mean": float(np.mean(np.asarray(defs_n["K_p"]))),
        "lhs_term_mean": float(np.mean(np.asarray(lhs_term))),
        "expect_term_mean": float(np.mean(np.asarray(expect_term))),
        "log_residual_mean": float(np.mean(lr)),
        "log_residual_std": float(np.std(lr)),
        "K_p_inner_floor_frac": floor_frac,
        "exponent": float(exponent),
    }


def _count_params(model: eqx.Module) -> int:
    """Count trainable parameters in an Equinox model."""
    params = eqx.filter(model, eqx.is_array)
    leaves = jax.tree_util.tree_leaves(params)
    return sum(x.size for x in leaves)


def _network_shape_str(model_spec: ModelSpec, hidden_sizes) -> str:
    """Format network shape as e.g. '2 -> 64 -> 64 -> 1'."""
    if isinstance(hidden_sizes, int):
        hidden_sizes = (hidden_sizes,)
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
    lr_schedule: str = "constant",
    lr_warmup: int = 0,
    lr_min_factor: float = 0.0,
    net_type: str = "mlp",
    history_len: int = 1,
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
    learning_rate = float(learning_rate)
    lr_str = f"lr={learning_rate:.0e}"
    if lr_schedule != "constant":
        lr_str += f", {lr_schedule}"
        if lr_warmup > 0:
            lr_str += f", warmup={lr_warmup}"
        lr_str += f", min={learning_rate * float(lr_min_factor):.0e}"
    print(f"  Optimizer:       {optimizer} ({lr_str})")
    print(f"  Precision:       {precision}")
    net_str = _network_shape_str(model_spec, hidden_sizes)
    if net_type != "mlp":
        net_str = f"{net_type.upper()} {net_str} (H={history_len})"
    print(f"  Network:         {net_str}")
    print(f"  Parameters:      {n_params:,}")
    print(f"  Batch size:      {batch_size}")
    print(f"  Expectations:    {mc_samples} MC samples")
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
        eq_items = [
            (_strip_eq_prefix(k), float(v))
            for k, v in final_residuals.items()
            if not k.startswith("aux_")
        ]
        aux_items = [
            (k[4:], float(v))
            for k, v in final_residuals.items()
            if k.startswith("aux_")
        ]
        if len(eq_items) <= 3:
            for n, v in eq_items:
                print(f"  {n}: {v:.2e}")
        else:
            _print_residual_table(eq_items)
        if aux_items:
            print("  aux: " + "  ".join(f"{n}={v:.2e}" for n, v in aux_items))
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

def _run_episode_and_sample(model, state, episode_length, batch_size, history_len=1,
                            ss_reset_frac: float = 0.0):
    """Run episode and sample training batch.

    For history_len=1 (MLP): returns [batch, n_states]
    For history_len>1 (LSTM/Transformer): returns [batch, H, n_states]

    history_len is captured in closure and resolves at trace time.

    ss_reset_frac: fraction of batch to reset to SS-neighborhood each episode.
    Prevents on-policy drift into degenerate basins.
    """
    key = state.key
    key, episode_key, loss_key, shuffle_key, reset_key = jax.random.split(key, 5)

    # Mix in fresh SS-neighborhood states to prevent trajectory drift
    ep_states = state.episode_state
    if ss_reset_frac > 0.0:
        batch_n = ep_states.shape[0]
        n_reset = int(ss_reset_frac * batch_n)
        if n_reset > 0:
            ss_state, _ = model.steady_state_fn(model.constants)
            noise = jax.random.uniform(reset_key, (n_reset, model.n_states), minval=-0.05, maxval=0.05)
            fresh = ss_state * (1 + noise)
            ep_states = ep_states.at[:n_reset].set(fresh)

    if history_len > 1:
        # Sequence path: run episode with history, build windows
        trajectory, final_state = run_episode_with_history(
            model,
            state.params,
            ep_states,
            episode_key,
            episode_length,
            history_len,
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
        )
        all_states = trajectory.reshape(-1, model.n_states)
        n_states = all_states.shape[0]
        indices = jax.random.permutation(shuffle_key, n_states)[:batch_size]
        train_batch = all_states[indices]  # [batch, D]

    return train_batch, final_state, loss_key, key


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
# Shared step preamble / postamble
# ---------------------------------------------------------------------------

class _StepContext(_NamedTuple):
    """Pre-computed values shared by all train step variants."""
    train_states: Array
    final_state: Any
    loss_key: Array
    key: Array
    target_fn: Any  # None or frozen params


def _prepare_step(model, state, episode_length, batch_size, history_len,
                  ss_reset_frac, use_target_network):
    """Common preamble for all train step variants.

    Runs episode, samples training batch, and resolves target network.
    Called inside JIT; use_target_network is a Python bool resolved at trace time.
    """
    train_states, final_state, loss_key, key = _run_episode_and_sample(
        model, state, episode_length, batch_size, history_len,
        ss_reset_frac=ss_reset_frac,
    )
    target_fn = state.target_params if use_target_network else None
    return _StepContext(train_states, final_state, loss_key, key, target_fn)


def _finalize_step(state, ctx, new_params, new_opt_state, loss, eq_losses,
                   grad_norm, loss_reweight, reweight_alpha, n_eq):
    """Common postamble: reweighting + TrainState construction + Metrics.

    Called inside JIT. All arguments are arrays/pytrees (JAX-compatible).
    """
    new_weights, new_rw = _update_reweighting(
        eq_losses, state, loss_reweight, reweight_alpha, n_eq,
    )
    new_state = TrainState(
        params=new_params,
        opt_state=new_opt_state,
        episode_state=ctx.final_state,
        key=ctx.key,
        step=state.step + 1,
        episode=state.episode + 1,
        loss_weights=new_weights,
        reweight_state=new_rw,
        target_params=state.target_params,
    )
    return new_state, Metrics(loss=loss, residuals=eq_losses, grad_norm=grad_norm)


# ---------------------------------------------------------------------------
# Five step variants
# ---------------------------------------------------------------------------

def _make_standard_step(
    model: ModelSpec,
    opt: Any,
    episode_length: int,
    mc_samples: int,
    batch_size: int,
    loss_reweight: str = "none",
    reweight_alpha: float = 0.9,
    quad_nodes: Optional[Array] = None,
    quad_weights: Optional[Array] = None,
    history_len: int = 1,
    compute_loss_fn: Optional[Callable] = None,
    ss_reset_frac: float = 0.0,
    use_target_network: bool = False,
):
    """Standard train step: jax.grad + opt.update(grads, state, params)."""
    n_eq = len(model.equation_names) if model.equation_names else 1
    _compute_loss = compute_loss_fn or compute_loss

    @jax.jit
    def train_step(state: TrainState, lr_scale: Array, shock_scale: Array = jnp.array(1.0)) -> Tuple[TrainState, Metrics]:
        ctx = _prepare_step(model, state, episode_length, batch_size,
                            history_len, ss_reset_frac, use_target_network)

        def loss_fn(params):
            loss, eq_losses = _compute_loss(
                model, params, ctx.train_states, ctx.loss_key, mc_samples,
                weights=state.loss_weights, shock_scale=shock_scale,
                quad_nodes=quad_nodes, quad_weights=quad_weights,
                target_policy_fn=ctx.target_fn,
            )
            return loss, eq_losses

        (loss, eq_losses), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(state.params)

        params_arrays = eqx.filter(state.params, eqx.is_array)
        grads_arrays = eqx.filter(grads, eqx.is_array)

        updates, new_opt_state = opt.update(grads_arrays, state.opt_state, params_arrays)
        updates = jax.tree.map(lambda u: lr_scale * u, updates)
        new_params_arrays = optax.apply_updates(params_arrays, updates)
        new_params = eqx.combine(new_params_arrays, state.params)

        grad_norm = optax.global_norm(grads_arrays)
        return _finalize_step(state, ctx, new_params, new_opt_state,
                              loss, eq_losses, grad_norm,
                              loss_reweight, reweight_alpha, n_eq)

    return train_step


def _make_pcgrad_step(
    model: ModelSpec,
    opt: Any,
    episode_length: int,
    mc_samples: int,
    batch_size: int,
    loss_reweight: str = "none",
    reweight_alpha: float = 0.9,
    quad_nodes: Optional[Array] = None,
    quad_weights: Optional[Array] = None,
    history_len: int = 1,
    compute_loss_fn: Optional[Callable] = None,
    ss_reset_frac: float = 0.0,
    use_target_network: bool = False,
):
    """PCGrad train step: per-equation gradients with conflict projection.

    When two equations have conflicting gradients (negative dot product),
    project out the conflicting component. This prevents equations from
    fighting over shared parameters (pi, q, etc.).

    Reference: Yu et al. "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020)
    """
    n_eq = len(model.equation_names) if model.equation_names else 1
    # PCGrad uses base compute_loss for per-equation Jacobian (no aux terms)
    # but can use composite loss for total loss logging
    _compute_loss_total = compute_loss_fn or compute_loss

    @jax.jit
    def train_step(state: TrainState, lr_scale: Array, shock_scale: Array = jnp.array(1.0)) -> Tuple[TrainState, Metrics]:
        ctx = _prepare_step(model, state, episode_length, batch_size,
                            history_len, ss_reset_frac, use_target_network)

        # Per-equation loss vector for jacrev (always base compute_loss)
        def eq_loss_vector(params):
            _, eq_losses = compute_loss(
                model, params, ctx.train_states, ctx.loss_key, mc_samples,
                weights=state.loss_weights, shock_scale=shock_scale,
                quad_nodes=quad_nodes, quad_weights=quad_weights,
                target_policy_fn=ctx.target_fn,
            )
            return eq_losses_to_array(eq_losses)

        # Also need total loss + eq_losses for logging
        def total_loss_fn(params):
            loss, eq_losses = _compute_loss_total(
                model, params, ctx.train_states, ctx.loss_key, mc_samples,
                weights=state.loss_weights, shock_scale=shock_scale,
                quad_nodes=quad_nodes, quad_weights=quad_weights,
                target_policy_fn=ctx.target_fn,
            )
            return loss, eq_losses

        # Compute per-equation gradients via jacrev: each leaf [n_eq, *param_shape]
        eq_jac = jax.jacrev(eq_loss_vector)(state.params)

        # Flatten per-equation gradients to [n_eq, n_params]
        params_arrays = eqx.filter(state.params, eqx.is_array)
        flat_params, unflatten_fn = jax.flatten_util.ravel_pytree(params_arrays)
        n_params = flat_params.shape[0]

        # Extract and flatten each equation's gradient
        eq_jac_arrays = eqx.filter(eq_jac, eqx.is_array)
        flat_eq_grads = jnp.stack([
            jax.flatten_util.ravel_pytree(
                jax.tree.map(lambda x: x[i], eq_jac_arrays)
            )[0]
            for i in range(n_eq)
        ])  # [n_eq, n_params]

        # PCGrad: vectorized conflict projection
        # Gram matrix of per-equation gradients
        gram = flat_eq_grads @ flat_eq_grads.T  # [n_eq, n_eq]
        norms_sq = jnp.diag(gram)  # [n_eq]

        # Conflict coefficients: project out negative dot product components
        # coeff_ij = dot(g_i, g_j) / ||g_j||^2 (only when dot < 0)
        coeffs = jnp.where(gram < 0, gram / (norms_sq[None, :] + 1e-8), 0.0)
        # Zero diagonal (don't project against self)
        coeffs = coeffs.at[jnp.diag_indices(n_eq)].set(0.0)

        # Project: g_i_new = g_i - sum_j coeff_ij * g_j
        projected = flat_eq_grads - coeffs @ flat_eq_grads  # [n_eq, n_params]

        # Sum projected gradients
        final_flat_grad = jnp.sum(projected, axis=0)  # [n_params]

        # Unflatten back to pytree matching params_arrays
        grads_arrays = unflatten_fn(final_flat_grad)

        # Standard optimizer update
        updates, new_opt_state = opt.update(grads_arrays, state.opt_state, params_arrays)
        updates = jax.tree.map(lambda u: lr_scale * u, updates)
        new_params_arrays = optax.apply_updates(params_arrays, updates)
        new_params = eqx.combine(new_params_arrays, state.params)

        # Get loss and eq_losses for logging
        loss, eq_losses = total_loss_fn(state.params)

        grad_norm = jnp.sqrt(jnp.sum(final_flat_grad ** 2))
        return _finalize_step(state, ctx, new_params, new_opt_state,
                              loss, eq_losses, grad_norm,
                              loss_reweight, reweight_alpha, n_eq)

    return train_step


def _make_mao_step(
    model: ModelSpec,
    mao_opt: Any,
    episode_length: int,
    mc_samples: int,
    batch_size: int,
    loss_reweight: str = "none",
    reweight_alpha: float = 0.9,
    grad_clip: Optional[float] = None,
    quad_nodes: Optional[Array] = None,
    quad_weights: Optional[Array] = None,
    history_len: int = 1,
    compute_loss_fn: Optional[Callable] = None,
    ss_reset_frac: float = 0.0,
    use_target_network: bool = False,
):
    """MAO train step: per-equation Jacobian -> mao.update(eq_jac, state, params)."""
    n_eq = len(model.equation_names) if model.equation_names else 1
    # MAO uses base compute_loss for per-equation Jacobian
    _compute_loss_total = compute_loss_fn or compute_loss

    @jax.jit
    def train_step(state: TrainState, lr_scale: Array, shock_scale: Array = jnp.array(1.0)) -> Tuple[TrainState, Metrics]:
        ctx = _prepare_step(model, state, episode_length, batch_size,
                            history_len, ss_reset_frac, use_target_network)

        params_arrays = eqx.filter(state.params, eqx.is_array)
        params_static = eqx.filter(state.params, lambda x: not eqx.is_array(x))

        # Per-equation loss vector: returns [n_eq] array
        def per_eq_loss_fn(p_arrays):
            full_params = eqx.combine(p_arrays, params_static)
            _, eq_losses = compute_loss(
                model, full_params, ctx.train_states, ctx.loss_key, mc_samples,
                weights=None,  # MAO handles weighting internally
                shock_scale=shock_scale,
                quad_nodes=quad_nodes, quad_weights=quad_weights,
                target_policy_fn=ctx.target_fn,
            )
            return eq_losses_to_array(eq_losses)

        # Get per-equation Jacobian: pytree, each leaf [n_eq, *param_shape]
        eq_jac = jax.jacrev(per_eq_loss_fn)(params_arrays)

        # Also compute scalar loss + grad norm for metrics
        def total_loss_fn(params):
            loss, eq_losses = _compute_loss_total(
                model, params, ctx.train_states, ctx.loss_key, mc_samples,
                weights=state.loss_weights, shock_scale=shock_scale,
                quad_nodes=quad_nodes, quad_weights=quad_weights,
                target_policy_fn=ctx.target_fn,
            )
            return loss, eq_losses

        (loss, eq_losses), grads = eqx.filter_value_and_grad(total_loss_fn, has_aux=True)(
            state.params
        )
        grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_array))

        # MAO update
        updates, new_opt_state = mao_opt.update(eq_jac, state.opt_state, params_arrays)

        # Grad clipping (MAO bypasses optax.chain, so clip here)
        if grad_clip is not None:
            update_norm = optax.global_norm(updates)
            clip_scale = jnp.minimum(1.0, grad_clip / (update_norm + 1e-8))
            updates = jax.tree.map(lambda u: clip_scale * u, updates)

        updates = jax.tree.map(lambda u: lr_scale * u, updates)
        new_params_arrays = optax.apply_updates(params_arrays, updates)
        new_params = eqx.combine(new_params_arrays, state.params)

        return _finalize_step(state, ctx, new_params, new_opt_state,
                              loss, eq_losses, grad_norm,
                              loss_reweight, reweight_alpha, n_eq)

    return train_step


def _make_lbfgs_step(
    model: ModelSpec,
    opt: Any,
    episode_length: int,
    mc_samples: int,
    batch_size: int,
    loss_reweight: str = "none",
    reweight_alpha: float = 0.9,
    quad_nodes: Optional[Array] = None,
    quad_weights: Optional[Array] = None,
    history_len: int = 1,
    compute_loss_fn: Optional[Callable] = None,
    ss_reset_frac: float = 0.0,
    use_target_network: bool = False,
):
    """L-BFGS train step: passes value + value_fn for line search."""
    n_eq = len(model.equation_names) if model.equation_names else 1
    _compute_loss = compute_loss_fn or compute_loss

    @jax.jit
    def train_step(state: TrainState, lr_scale: Array, shock_scale: Array = jnp.array(1.0)) -> Tuple[TrainState, Metrics]:
        ctx = _prepare_step(model, state, episode_length, batch_size,
                            history_len, ss_reset_frac, use_target_network)

        params_arrays = eqx.filter(state.params, eqx.is_array)
        params_static = eqx.filter(state.params, lambda x: not eqx.is_array(x))

        def loss_fn(params):
            loss, eq_losses = _compute_loss(
                model, params, ctx.train_states, ctx.loss_key, mc_samples,
                weights=state.loss_weights, shock_scale=shock_scale,
                quad_nodes=quad_nodes, quad_weights=quad_weights,
                target_policy_fn=ctx.target_fn,
            )
            return loss, eq_losses

        (loss, eq_losses), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(state.params)
        grads_arrays = eqx.filter(grads, eqx.is_array)
        grad_norm = optax.global_norm(grads_arrays)

        # Value function for line search (operates on param arrays only)
        def value_fn(p_arrays):
            full_params = eqx.combine(p_arrays, params_static)
            v, _ = _compute_loss(
                model, full_params, ctx.train_states, ctx.loss_key, mc_samples,
                weights=state.loss_weights, shock_scale=shock_scale,
                quad_nodes=quad_nodes, quad_weights=quad_weights,
                target_policy_fn=ctx.target_fn,
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
        updates = jax.tree.map(lambda u: lr_scale * u, updates)
        new_params_arrays = optax.apply_updates(params_arrays, updates)
        new_params = eqx.combine(new_params_arrays, state.params)

        return _finalize_step(state, ctx, new_params, new_opt_state,
                              loss, eq_losses, grad_norm,
                              loss_reweight, reweight_alpha, n_eq)

    return train_step


def _make_gn_step(
    model: ModelSpec,
    opt: Any,
    episode_length: int,
    mc_samples: int,
    batch_size: int,
    loss_reweight: str = "none",
    reweight_alpha: float = 0.9,
    quad_nodes: Optional[Array] = None,
    quad_weights: Optional[Array] = None,
    history_len: int = 1,
    compute_loss_fn: Optional[Callable] = None,
    ss_reset_frac: float = 0.0,
    use_target_network: bool = False,
):
    """Gauss-Newton / Levenberg-Marquardt train step.

    GN uses the residual Jacobian J directly: update = -(J^T J)^{-1} J^T r.
    This gives quadratic convergence near the solution for least-squares.
    """
    n_eq = len(model.equation_names) if model.equation_names else 1
    # GN uses base compute_loss for residuals; composite for logging only
    _compute_loss_log = compute_loss_fn or compute_loss
    use_quadrature = quad_nodes is not None and quad_weights is not None

    @jax.jit
    def train_step(state: TrainState, lr_scale: Array, shock_scale: Array = jnp.array(1.0)) -> Tuple[TrainState, Metrics]:
        ctx = _prepare_step(model, state, episode_length, batch_size,
                            history_len, ss_reset_frac, use_target_network)

        # Residual function: params -> [n_eq * batch] per-batch E_shock[r]

        def residual_fn(params):
            if use_quadrature:
                n_nodes = quad_nodes.shape[0]
                shocks = jnp.broadcast_to(
                    quad_nodes[:, None, :],
                    (n_nodes, batch_size, model.n_shocks),
                ) * shock_scale
                sample_weights = quad_weights
            else:
                shocks = sample_antithetic_shocks(
                    ctx.loss_key, mc_samples, batch_size, model.n_shocks, shock_scale,
                )
                n_samples = shocks.shape[0]
                sample_weights = jnp.ones(n_samples) / n_samples

            def sample_residuals(shock):
                return compute_residuals(model, params, ctx.train_states, shock,
                                         target_policy_fn=ctx.target_fn)

            all_residuals = jax.vmap(sample_residuals)(shocks)
            # all_residuals: Dict[str, [n_samples, batch]]
            # E_shock[r] per batch element per equation, then stack
            per_eq = []
            for r in all_residuals.values():
                # r: [n_samples, batch] -> weighted mean over samples -> [batch]
                mean_r = jnp.einsum('s,sb->b', sample_weights, r)
                per_eq.append(mean_r)
            return jnp.concatenate(per_eq)  # [n_eq * batch]

        # Also get loss + eq_losses for logging
        loss, eq_losses = _compute_loss_log(
            model, state.params, ctx.train_states, ctx.loss_key, mc_samples,
            weights=state.loss_weights, shock_scale=shock_scale,
            quad_nodes=quad_nodes, quad_weights=quad_weights,
            target_policy_fn=ctx.target_fn,
        )

        # GN update: optimizer handles Jacobian computation internally
        new_params, new_opt_state = opt.update(residual_fn, state.params, state.opt_state)

        # Compute grad norm from residual gradient for logging
        def scalar_loss(p):
            r = residual_fn(p)
            return jnp.sum(r ** 2)
        grad_norm = optax.global_norm(eqx.filter(jax.grad(scalar_loss)(state.params), eqx.is_array))

        return _finalize_step(state, ctx, new_params, new_opt_state,
                              loss, eq_losses, grad_norm,
                              loss_reweight, reweight_alpha, n_eq)

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
    gradient_surgery: str = "none",
    grad_clip: Optional[float] = None,
    quad_nodes: Optional[Array] = None,
    quad_weights: Optional[Array] = None,
    history_len: int = 1,
    compute_loss_fn: Optional[Callable] = None,
    ss_reset_frac: float = 0.0,
    use_target_network: bool = False,
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
    kwargs = dict(
        model=model,
        opt=opt if kind != OptimizerKind.MAO else None,
        episode_length=episode_length,
        mc_samples=mc_samples,
        batch_size=batch_size,
        loss_reweight=loss_reweight,
        reweight_alpha=reweight_alpha,
        quad_nodes=quad_nodes,
        quad_weights=quad_weights,
        history_len=history_len,
        compute_loss_fn=compute_loss_fn,
        ss_reset_frac=ss_reset_frac,
        use_target_network=use_target_network,
    )

    if gradient_surgery == "pcgrad" and kind == OptimizerKind.STANDARD:
        return _make_pcgrad_step(**kwargs)
    elif kind == OptimizerKind.MAO:
        return _make_mao_step(
            model=model,
            mao_opt=opt,
            episode_length=episode_length,
            mc_samples=mc_samples,
            batch_size=batch_size,
            loss_reweight=loss_reweight,
            reweight_alpha=reweight_alpha,
            grad_clip=grad_clip,
            quad_nodes=quad_nodes,
            quad_weights=quad_weights,
            history_len=history_len,
            compute_loss_fn=compute_loss_fn,
            ss_reset_frac=ss_reset_frac,
            use_target_network=use_target_network,
        )
    elif kind == OptimizerKind.LBFGS:
        return _make_lbfgs_step(**kwargs)
    elif kind == OptimizerKind.GN:
        return _make_gn_step(**kwargs)
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

    # ---- Build LR schedule helper for logging ----
    from deqn_jax.optimizers.registry import _build_lr_schedule

    # When a schedule is active, the optimizer is created with lr=1.0.
    # The actual LR is passed as a dynamic scalar to train_step each episode.
    has_schedule = config.optimizer.lr_schedule != "constant"
    if has_schedule:
        total_for_schedule = config.episodes  # overridden below for resume
        effective_opt_cfg = config.optimizer.model_copy(
            update={"learning_rate": 1.0, "lr_schedule": "constant"}
        )
    else:
        effective_opt_cfg = config.optimizer

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
        # Resume runs TO config.episodes, not an additional config.episodes
        total_episodes = config.episodes
        if has_schedule:
            total_for_schedule = total_episodes
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
            # Rebuild optimizer with effective_opt_cfg (lr=1.0 when schedule active)
            # to avoid double-LR: checkpoint optimizer has lr baked in, but
            # lr_scale already carries the scheduled LR from _build_lr_schedule.
            opt, kind = create_optimizer(effective_opt_cfg)
            if kind == OptimizerKind.MAO and hasattr(opt, 'with_num_tasks'):
                opt = opt.with_num_tasks(n_equations)
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
            optimizer_config=effective_opt_cfg,
            network_config=config.network,
        )

        # Warm start from steady state (only for fresh training)
        if config.warm_start:
            _hl = get_history_len(state.params)
            if _hl > 1:
                # Sequence net warm start: fit to constant-SS policy
                # using warm_start_to_function with history wrapping
                from deqn_jax.training.warm_start import warm_start_to_function
                if model.steady_state_fn is not None:
                    ss_state, ss_policy = model.steady_state_fn(model.constants)
                    ws_key = jax.random.PRNGKey(0)
                    noise = jax.random.uniform(ws_key, (256, model.n_states), minval=-0.2, maxval=0.2)
                    sample_states = ss_state * (1 + noise)
                    # Build constant history windows for each sample state
                    sample_history = make_constant_history(sample_states, _hl)  # [256, H, D]
                    target_fn = lambda h: ss_policy  # constant target
                    # Custom loss for sequence net
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
            lr_schedule=config.optimizer.lr_schedule,
            lr_warmup=config.optimizer.lr_warmup,
            lr_min_factor=config.optimizer.lr_min_factor,
            net_type=getattr(config.network, "type", "mlp") if config.network else "mlp",
            history_len=get_history_len(state.params),
        )

    # Build LR schedule function for computing per-episode LR (None if constant)
    lr_schedule_fn = None
    if has_schedule:
        if config.resume:
            total_for_schedule = config.episodes
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

    # ---- Build composite loss if configured ----
    custom_loss_fn = None
    if config.loss_type == "composite":
        from deqn_jax.training.linearize import linearize_model
        from deqn_jax.training.composite_loss import prepare_composite_data, make_composite_loss

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
            barrier_weight=comp_cfg.barrier_weight,
            newton_weight=comp_cfg.newton_weight,
            leverage_mult=comp_cfg.leverage_mult,
            aux_decay_floor=comp_cfg.aux_decay_floor,
            history_len=history_len,
        )
        if config.verbose:
            print("  Composite loss ready.")

    # ---- State barrier penalty ----
    barrier_weight = config.barrier_weight
    if barrier_weight > 0 and custom_loss_fn is None and model.state_barrier_fn is not None:
        from functools import partial
        custom_loss_fn = partial(compute_loss, barrier_weight=barrier_weight)
        if config.verbose:
            print(f"  State barrier: weight={barrier_weight}")

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
    )

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
        return
    ep_width = len(str(total_episodes))

    history: Dict[str, list] = {"loss": [], "grad_norm": []}
    t_start = time.perf_counter()
    last_metrics = None
    best_loss = float("inf")
    patience_counter = 0

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
            )
            switched = True
            # Reset early stopping after optimizer switch
            best_loss = float("inf")
            patience_counter = 0
            if config.verbose:
                print(f"  >> Switched to {config.switch_optimizer} (lr={switch_lr:.0e}) at episode {ep_num}")

        # Compute LR scale for this episode (Python-side, passed as dynamic arg)
        if lr_schedule_fn is not None:
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
            eq2_diag = None
            eq4_diag = None
            if model.definitions_fn is not None:
                defs = jax.vmap(
                    lambda s, p: model.definitions_fn(s, p, model.constants)
                )(ep_states, policy_out)
                for name, vals in defs.items():
                    hist_dict[f"derived/{name}"] = np.asarray(vals)

                # Wrap policy_fn for sequence nets (use constant history)
                if history_len > 1:
                    _diag_policy_fn = lambda s: state.params(make_constant_history(s[None], history_len)[0])
                else:
                    _diag_policy_fn = state.params

                # eq2 (price_phillips_K) diagnostics
                if "K_p" in defs and "pi_tilda" in defs and "s" in defs:
                    eq2_diag = _eq2_diagnostics(
                        model, _diag_policy_fn, ep_states, policy_out, defs,
                    )
                    for dk, dv in eq2_diag.items():
                        log_dict[f"eq2_diag/{dk}"] = float(dv)

                # eq4 (wage_phillips_K) diagnostics
                if "K_w" in defs and "pi_w_tilda" in defs and "pi_w" in defs:
                    eq4_diag = _eq4_diagnostics(
                        model, _diag_policy_fn, ep_states, policy_out, defs,
                    )
                    for dk, dv in eq4_diag.items():
                        log_dict[f"eq4_diag/{dk}"] = float(dv)

            # Log all scalars (including eq2/eq4 diagnostics)
            logger.log_scalars(log_dict, step=ep_num)

            # Filter out arrays with NaN/Inf (early training can produce these)
            hist_dict = {k: v for k, v in hist_dict.items()
                         if np.isfinite(v).all() and v.size > 0}
            if hist_dict:
                logger.log_histograms(hist_dict, step=ep_num)

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

            # eq2 diagnostic line (when available)
            if eq2_diag is not None:
                print(
                    f"    eq2: ratio={eq2_diag['ratio_base_mean']:.4f}±{eq2_diag['ratio_base_std']:.4f}"
                    f"  log_r={eq2_diag['log_residual_mean']:.3f}±{eq2_diag['log_residual_std']:.3f}"
                    f"  K_p={eq2_diag['K_p_mean']:.3f}"
                    f"  lhs={eq2_diag['lhs_term_mean']:.3f}"
                    f"  floor%={eq2_diag['K_p_inner_floor_frac']:.1%}"
                )

            # eq4 diagnostic line (when available)
            if eq4_diag is not None:
                print(
                    f"    eq4: ratio={eq4_diag['ratio_base_mean']:.4f}±{eq4_diag['ratio_base_std']:.4f}"
                    f"  log_r={eq4_diag['log_residual_mean']:.3f}±{eq4_diag['log_residual_std']:.3f}"
                    f"  K_w={eq4_diag['K_w_mean']:.3f}"
                    f"  h²={eq4_diag['h_term_mean']:.3f}"
                    f"  floor%={eq4_diag['K_w_inner_floor_frac']:.1%}"
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
            # Snapshot for NaN rollback
            last_good_state = state
            last_good_episode = ep_num

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
