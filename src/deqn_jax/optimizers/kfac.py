"""K-FAC optimizer wrapper.

Attempts to use kfac-jax if installed; falls back to NGD with a warning.
Full kfac-jax integration (which requires a dedicated OptimizerKind) is
future work -- for now this provides NGD as a reasonable fallback.
"""

import warnings

from deqn_jax.optimizers.registry import register_optimizer, OptimizerKind
from deqn_jax.optimizers.ngd import ngd


@register_optimizer("kfac", kind=OptimizerKind.STANDARD)
def _kfac(config):
    try:
        import kfac_jax  # noqa: F401
        # Full kfac-jax integration is future work.
        # For now, fall through to NGD.
        warnings.warn(
            "kfac-jax is installed but full integration is not yet implemented. "
            "Using diagonal Fisher NGD as fallback.",
            stacklevel=2,
        )
    except ImportError:
        warnings.warn(
            "kfac-jax not installed (pip install kfac-jax). "
            "Using diagonal Fisher NGD as fallback.",
            stacklevel=2,
        )

    return ngd(
        learning_rate=config.learning_rate,
        damping=config.kfac_damping,
        decay=config.decay,
    )
