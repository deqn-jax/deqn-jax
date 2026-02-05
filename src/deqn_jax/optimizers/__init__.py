"""Optimizers for DEQN-JAX.

Includes:
- Standard optimizers via Optax (Adam, SGD, AdamW)
- Gauss-Newton/Levenberg-Marquardt for residual minimization
- L-BFGS via jaxopt for warm-starting
"""

from deqn_jax.optimizers.gauss_newton import (
    GaussNewton,
    GaussNewtonState,
    LevenbergMarquardt,
    gauss_newton,
    levenberg_marquardt,
)

__all__ = [
    "GaussNewton",
    "GaussNewtonState",
    "LevenbergMarquardt",
    "gauss_newton",
    "levenberg_marquardt",
]
