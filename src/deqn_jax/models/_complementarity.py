"""Helpers for encoding KKT / complementarity conditions as
single-equation residuals.

Fischer-Burmeister function
---------------------------
Define f_FB(a, b) = sqrt(a^2 + b^2) - a - b.

f_FB(a, b) = 0  iff  (a >= 0, b >= 0, a*b = 0)

Use case: for a KKT condition with stationarity residual ``a``
("FOC with multiplier") and slack ``b`` (``constraint_rhs - variable``),
``f_FB(a, b) = 0`` enforces both sign constraints and the complementary-
slackness identity with a single smooth equation — exactly what a DEQN
loss function needs.

Numerical note
--------------
A small ``eps`` is added under the square root to keep gradients
well-defined at the origin (a=b=0). This matches the convention used in
the Azinovic-Gaegauf-Scheidegger reference notebooks and standard NCP
solvers.
"""

from typing import Union

import jax.numpy as jnp
from jax import Array


def fischer_burmeister(
    a: Union[Array, float],
    b: Union[Array, float],
    eps: float = 1e-13,
) -> Array:
    """Fischer-Burmeister residual for the NCP a >= 0, b >= 0, a*b = 0.

    Args:
        a: stationarity / FOC residual
        b: constraint slack
        eps: regularization inside the sqrt for differentiability at 0

    Returns:
        ``sqrt(a^2 + b^2 + eps) - a - b``. Zero iff the NCP is satisfied.
    """
    return jnp.sqrt(a * a + b * b + eps) - a - b
