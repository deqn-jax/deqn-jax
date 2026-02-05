"""Optimizer registry for DEQN-JAX.

Maps optimizer names to factory functions and optimizer kinds.
The kind determines which train_step variant is used.
"""

from enum import Enum
from typing import Any, Callable, Dict, List, Tuple

import optax


class OptimizerKind(str, Enum):
    """Determines which train_step variant runs."""

    STANDARD = "standard"  # adam, sgd, adamw, lion, muon, ngd, shampoo, kfac
    MAO = "mao"  # per-equation Jacobian
    LBFGS = "lbfgs"  # extra args for line search


# Registry: name -> (factory_fn(config) -> optimizer, kind)
_REGISTRY: Dict[str, Tuple[Callable, OptimizerKind]] = {}


def register_optimizer(
    name: str,
    kind: OptimizerKind = OptimizerKind.STANDARD,
):
    """Decorator to register an optimizer factory."""

    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = (fn, kind)
        return fn

    return decorator


def create_optimizer(config) -> Tuple[Any, OptimizerKind]:
    """Create optimizer from config.

    Args:
        config: OptimizerConfig with at least a ``name`` field.

    Returns:
        Tuple of (optimizer, OptimizerKind)
    """
    name = config.name
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown optimizer '{name}'. Available: {available}")

    factory, kind = _REGISTRY[name]
    opt = factory(config)

    # Chain grad clip for STANDARD optimizers
    if kind == OptimizerKind.STANDARD and config.grad_clip is not None:
        opt = optax.chain(optax.clip_by_global_norm(config.grad_clip), opt)

    return opt, kind


def list_optimizers() -> List[str]:
    """Return sorted list of registered optimizer names."""
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Built-in optax registrations
# ---------------------------------------------------------------------------

@register_optimizer("adam")
def _adam(config):
    return optax.adam(
        config.learning_rate,
        b1=config.beta1,
        b2=config.beta2,
        eps=config.epsilon,
    )


@register_optimizer("sgd")
def _sgd(config):
    return optax.sgd(config.learning_rate)


@register_optimizer("adamw")
def _adamw(config):
    return optax.adamw(
        config.learning_rate,
        b1=config.beta1,
        b2=config.beta2,
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )


@register_optimizer("lion")
def _lion(config):
    return optax.lion(
        config.learning_rate,
        b1=config.beta1,
        b2=config.beta2,
    )


@register_optimizer("muon")
def _muon(config):
    return optax.contrib.muon(
        learning_rate=config.learning_rate,
        ns_steps=config.ns_steps,
    )
