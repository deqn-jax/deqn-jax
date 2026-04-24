"""Optimizers for DEQN-JAX.

Includes:
- Standard optimizers via Optax (Adam, SGD, AdamW, Lion, Muon)
- Natural Gradient Descent (diagonal Fisher)
- Multi-Adaptive Optimizer (per-equation moments)
- Kronecker-factored Shampoo
- L-BFGS via optax
- Gauss-Newton / Levenberg-Marquardt for residual minimization

All standard/NGD/Shampoo/Lion/Muon/K-FAC optimizers are registered in the
registry and created via ``create_optimizer(config)``.
"""

# Import all optimizer modules to trigger @register_optimizer
from deqn_jax.optimizers import lbfgs as _lbfgs_mod  # noqa: F401
from deqn_jax.optimizers import mao as _mao_mod  # noqa: F401
from deqn_jax.optimizers import mao_kfac as _mao_kfac_mod  # noqa: F401
from deqn_jax.optimizers import ngd as _ngd_mod  # noqa: F401
from deqn_jax.optimizers import shampoo as _shampoo_mod  # noqa: F401
from deqn_jax.optimizers.gauss_newton import (
    GaussNewton,
    GaussNewtonState,
    LevenbergMarquardt,
    gauss_newton,
    levenberg_marquardt,
)
from deqn_jax.optimizers.mao import MAOState, MAOTransform, _MAOFactory
from deqn_jax.optimizers.mao_kfac import MAOKFACState, MAOKFACTransform, _MAOKFACFactory
from deqn_jax.optimizers.ngd import NGDState, ngd
from deqn_jax.optimizers.registry import (  # noqa: F401
    OptimizerKind,
    create_optimizer,
    list_optimizers,
    register_optimizer,
)
from deqn_jax.optimizers.shampoo import ShampooState, shampoo

__all__ = [
    # Registry
    "OptimizerKind",
    "create_optimizer",
    "list_optimizers",
    "register_optimizer",
    # Custom optimizers
    "MAOTransform",
    "MAOState",
    "MAOKFACTransform",
    "MAOKFACState",
    "ngd",
    "NGDState",
    "shampoo",
    "ShampooState",
    # GN/LM
    "GaussNewton",
    "GaussNewtonState",
    "LevenbergMarquardt",
    "gauss_newton",
    "levenberg_marquardt",
]
