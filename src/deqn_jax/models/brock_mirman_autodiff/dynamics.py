"""Dynamics for the autodiff variant -- identical to canonical brock_mirman.

The point of the autodiff variant is solely to synthesize the Euler
residual from primitives; the state transition law is still needed
(both to step the economy forward in simulation, and to reconstruct
K_{t+2} inside the synthesized residual). Reused verbatim from the
canonical model.
"""

from deqn_jax.models.brock_mirman.dynamics import step

__all__ = ["step"]
