"""Optimizer registry for DEQN-JAX.

Maps optimizer names to factory functions and optimizer kinds.
The kind determines which train_step variant is used.
"""

import copy
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import optax


class OptimizerKind(str, Enum):
    """Determines which train_step variant runs."""

    STANDARD = "standard"  # adam, sgd, adamw, lion, muon, ngd, shampoo, kfac
    MAO = "mao"  # per-equation Jacobian
    LBFGS = "lbfgs"  # extra args for line search
    GN = "gn"  # Gauss-Newton / LM (needs residual_fn)


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


class ReduceLROnPlateau:
    """Keras-style loss-reactive LR decay.

    Mirrors DEQN-MAO's ``lr_scheduler: ReduceLROnPlateau`` with the same
    knob names (factor, patience, cooldown, min_delta, min_lr). Unlike
    the other schedules here, this one is stateful and must be called
    with the current loss at every cycle, not just the step index.

    Call signature: ``self(ep_num, loss=None) -> lr``. ``loss=None`` is
    treated as "no update this cycle" -- the scheduler returns the
    current LR without advancing state. Used so the schedule callable
    can be probed early in training (e.g. for logging) before a first
    loss is available.
    """

    def __init__(self, initial_lr, factor, patience, cooldown, min_delta, min_lr):
        self.initial_lr = float(initial_lr)
        self.factor = float(factor)
        self.patience = int(patience)
        self.cooldown = int(cooldown)
        self.min_delta = float(min_delta)
        self.min_lr = float(min_lr)

        self._lr = self.initial_lr
        self._best = float("inf")
        self._wait = 0
        self._cooldown_counter = 0

    def __call__(self, _ep_num=None, loss=None):
        if loss is None:
            return self._lr

        # In cooldown: don't change LR, just tick down.
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            self._best = min(self._best, loss)
            self._wait = 0
            return self._lr

        # Track improvement.
        if loss < self._best - self.min_delta:
            self._best = loss
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                new_lr = max(self._lr * self.factor, self.min_lr)
                self._lr = new_lr
                self._cooldown_counter = self.cooldown
                self._wait = 0
        return self._lr


def _build_lr_schedule(config, total_steps: int):
    """Build an LR schedule callable from config fields.

    Returns either a float (constant), a stateless optax schedule
    (cosine) keyed on step index, or a stateful ``ReduceLROnPlateau``
    instance. All three are invoked via ``fn(ep_num, loss)`` in the
    training loop; the stateless ones ignore ``loss``.
    """
    lr = float(config.learning_rate)
    schedule = getattr(config, "lr_schedule", "constant")
    warmup = int(getattr(config, "lr_warmup", 0))
    min_factor = float(getattr(config, "lr_min_factor", 0.0))

    if schedule == "constant":
        return lr

    if schedule == "cosine":
        end_value = lr * min_factor
        if warmup > 0:
            return optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=lr,
                warmup_steps=warmup,
                decay_steps=total_steps,
                end_value=end_value,
            )
        else:
            return optax.cosine_decay_schedule(
                init_value=lr,
                decay_steps=total_steps,
                alpha=min_factor,
            )

    if schedule == "reduce_on_plateau":
        return ReduceLROnPlateau(
            initial_lr=lr,
            factor=float(getattr(config, "lr_reduce_factor", 0.5)),
            patience=int(getattr(config, "lr_reduce_patience", 500)),
            cooldown=int(getattr(config, "lr_reduce_cooldown", 100)),
            min_delta=float(getattr(config, "lr_reduce_min_delta", 1e-6)),
            min_lr=lr * min_factor,
        )

    raise ValueError(
        f"Unknown lr_schedule '{schedule}'. "
        "Available: constant, cosine, reduce_on_plateau"
    )


def create_optimizer(
    config,
    total_steps: Optional[int] = None,
) -> Tuple[Any, OptimizerKind]:
    """Create optimizer from config.

    LR schedules are NOT applied here — they break XLA kernel fusion
    and cause 5-6x slowdowns. Instead, the training loop periodically
    recreates the optimizer with an updated constant LR. See
    ``_build_lr_schedule`` for computing schedule values and
    ``train_from_config`` for the periodic update logic.

    Args:
        config: OptimizerConfig with at least a ``name`` field.
        total_steps: Unused (kept for API compat). Schedule is handled
            by the training loop.

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


@register_optimizer("gn", kind=OptimizerKind.GN)
def _gn(config):
    from deqn_jax.optimizers.gauss_newton import gauss_newton
    return gauss_newton(
        learning_rate=config.learning_rate,
        damping=config.damping,
    )


@register_optimizer("lm", kind=OptimizerKind.GN)
def _lm(config):
    from deqn_jax.optimizers.gauss_newton import levenberg_marquardt
    return levenberg_marquardt(
        learning_rate=config.learning_rate,
        initial_damping=config.damping,
    )
