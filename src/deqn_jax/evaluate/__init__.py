"""Evaluation suite for trained DEQN models.

Standard accuracy metrics (Azinovic et al. 2022, Den Haan & Marcet 1994): Euler
equation errors, market clearing, simulated moments, stability, and Dynare
cross-checks. Split into a package for readability; this module re-exports the
public surface so ``from deqn_jax.evaluate import ...`` keeps working.
"""

from deqn_jax.evaluate.cli import run_evaluate_cli
from deqn_jax.evaluate.diagnostics import (
    euler_equation_errors,
    market_clearing_errors,
    print_euler_errors,
    print_moments,
    simulated_moments,
    stability_check,
)
from deqn_jax.evaluate.dynare import (
    compare_to_dynare_ghx,
    compare_to_dynare_irfs,
    compare_to_dynare_moments,
    print_dynare_comparison,
)

__all__ = [
    "euler_equation_errors",
    "market_clearing_errors",
    "simulated_moments",
    "stability_check",
    "print_euler_errors",
    "print_moments",
    "compare_to_dynare_moments",
    "compare_to_dynare_ghx",
    "compare_to_dynare_irfs",
    "print_dynare_comparison",
    "run_evaluate_cli",
]
