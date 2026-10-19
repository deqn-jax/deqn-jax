"""6-agent OLG (Krueger-Kubler 2004) with equations derived from its Lagrangian.

Same economics, calibration, dynamics, steady state, diagnostics and
feasibility penalties as ``olg_analytic_6``. The five Euler equations are
not written by hand: ``equations.py`` supplies the period objective (the sum
of the six cohorts' utilities) and the prices the cohorts take as given,
and ``residuals_from_lagrangian`` derives the rest.
"""

from deqn_jax.models.olg_analytic_6 import MODEL as _HAND
from deqn_jax.models.olg_analytic_6_autodiff.equations import (
    EQUATION_NAMES,
    equations,
    objective,
    prices,
)

# Everything but the name and the equations is the hand-written model's.
MODEL = _HAND._replace(
    name="olg_analytic_6_autodiff",
    equation_names=EQUATION_NAMES,
    equations_fn=equations,
)

__all__ = ["MODEL", "objective", "prices"]
