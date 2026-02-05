"""L-BFGS optimizer via optax.

Thin wrapper around optax.lbfgs() which is a GradientTransformationExtraArgs --
it needs ``value`` and ``value_fn`` passed to update() for line search.
"""

import optax

from deqn_jax.optimizers.registry import register_optimizer, OptimizerKind


@register_optimizer("lbfgs", kind=OptimizerKind.LBFGS)
def _lbfgs(config):
    return optax.lbfgs(
        learning_rate=config.learning_rate,
        memory_size=config.memory_size,
    )
