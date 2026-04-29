"""Stable public API surface for DEQN-JAX.

This module re-exports the agent-facing entry points an external tool
(or a Claude agent stack) needs to:

  * **discover** what's available — models, optimizers, networks,
  * **register** a model programmatically (no edits to ``models/__init__.py``),
  * **configure** a training run via Pydantic models,
  * **train** a policy and get back history,
  * **evaluate** the trained policy with the standard ergodic Euler-error
    diagnostic and a structural stability check,
  * **inspect** impulse responses from a checkpoint.

Everything imported here is part of the stable interface — version
bumps that change anything in this module are breaking changes. The
underlying submodules (``deqn_jax.training.trainer``,
``deqn_jax.training.loss``, etc.) are *not* stable and may be
refactored at any time; agents should import from
``deqn_jax.api`` only.

Example::

    from deqn_jax.api import (
        ModelSpec,
        register_model,
        TrainConfig,
        NetworkConfig,
        OptimizerConfig,
        train_from_config,
        euler_equation_errors,
    )

    MODEL = ModelSpec(name="my_model", ...)  # see docs/site/REFERENCE.md
    register_model(MODEL, description="my custom model")

    cfg = TrainConfig(
        model="my_model",
        episodes=1000,
        network=NetworkConfig(hidden_sizes=(64, 64)),
        optimizer=OptimizerConfig(name="adam", learning_rate=1e-3),
    )
    state, history = train_from_config(cfg)
    diag = euler_equation_errors(state.params, load_model("my_model"))
    print_euler_errors(diag)
"""

from __future__ import annotations

# --- Configuration -----------------------------------------------------
from deqn_jax.config import (
    CompositeLossConfig,
    MomentMatchingConfig,
    NetworkConfig,
    OptimizerConfig,
    ReplayBufferConfig,
    TrainConfig,
    load_config,
)

# --- Evaluation / verification ----------------------------------------
from deqn_jax.evaluate import (
    euler_equation_errors,
    market_clearing_errors,
    print_euler_errors,
    print_moments,
    simulated_moments,
    stability_check,
)

# --- Impulse response functions ---------------------------------------
from deqn_jax.irf import (
    load_policy_from_checkpoint,
    print_irf_summary,
    run_girf,
    run_irf,
    save_irf_csv,
)

# --- Discovery ---------------------------------------------------------
# --- Registration ------------------------------------------------------
from deqn_jax.models import list_models, load_model, register_model

# --- Network factories (for advanced/custom training paths) ----------
# Most agents drive this through ``NetworkConfig.type`` and never import
# the classes directly. Exposed here for the rare case of constructing a
# policy net manually (e.g. when calling the low-level ``create_train_state``
# / ``make_train_step`` path for a custom outer loop).
from deqn_jax.networks import (
    MLP,
    LSTMPolicy,
    TransformerPolicy,
    create_linear_plus_mlp,
    create_lstm,
    create_mlp,
    create_transformer,
)
from deqn_jax.networks.kf_anchored_mlp import KfAnchoredMLP, create_kf_anchored_mlp
from deqn_jax.networks.linear_plus_mlp import LinearPlusMLP
from deqn_jax.optimizers.registry import list_optimizers

# --- Training entry points --------------------------------------------
from deqn_jax.training.trainer import (
    create_train_state,
    make_train_step,
    train,
    train_from_config,
)

# --- Core types --------------------------------------------------------
from deqn_jax.types import (
    Metrics,
    ModelSpec,
    ReweightState,
    TrainState,
    make_reweight_state,
)


def list_networks() -> list[str]:
    """Names of built-in network architectures (NetworkConfig.type values)."""
    return sorted(NetworkConfig.VALID_TYPES)


__all__ = [
    # Discovery
    "list_models",
    "list_optimizers",
    "list_networks",
    "load_model",
    # Registration
    "register_model",
    # Configuration
    "TrainConfig",
    "OptimizerConfig",
    "NetworkConfig",
    "CompositeLossConfig",
    "ReplayBufferConfig",
    "MomentMatchingConfig",
    "load_config",
    # Core types
    "ModelSpec",
    "TrainState",
    "ReweightState",
    "Metrics",
    "make_reweight_state",
    # Training
    "train",
    "train_from_config",
    "create_train_state",
    "make_train_step",
    # Evaluation
    "euler_equation_errors",
    "print_euler_errors",
    "stability_check",
    "simulated_moments",
    "print_moments",
    "market_clearing_errors",
    # IRF
    "run_irf",
    "run_girf",
    "load_policy_from_checkpoint",
    "save_irf_csv",
    "print_irf_summary",
    # Networks (advanced)
    "MLP",
    "LSTMPolicy",
    "TransformerPolicy",
    "LinearPlusMLP",
    "KfAnchoredMLP",
    "create_mlp",
    "create_lstm",
    "create_transformer",
    "create_linear_plus_mlp",
    "create_kf_anchored_mlp",
]
