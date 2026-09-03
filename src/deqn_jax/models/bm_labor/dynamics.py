"""State transition for Brock-Mirman with endogenous labor.

Identical accumulation law to ``brock_mirman``; only ``SPEC`` and
``definitions()`` differ, so the step is built from the shared factory.
``bm_labor_constrained`` and ``bm_labor_autodiff`` import this exact
``step`` object rather than rebuilding one.
"""

from deqn_jax.models.bm_labor.equations import definitions
from deqn_jax.models.bm_labor.variables import SPEC
from deqn_jax.models.brock_mirman.dynamics import make_step

step = make_step(SPEC, definitions)

__all__ = ["step"]
