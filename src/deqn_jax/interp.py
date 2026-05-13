"""Mechanistic interpretability primitives for DEQN networks.

Five pure functions for inspecting a trained ``LinearPlusMLP``:

1. ``branch_decompose`` — split the policy into Blanchard-Kahn
   linearization and MLP correction.
2. ``forward_with_activations`` — capture per-layer post-activations.
3. ``neuron_contributions`` — per-neuron attribution to downstream units.
4. ``linear_probe`` — regress concept scalars on hidden activations.
5. ``ablate_neuron`` — zero out a chosen post-activation and rerun.

Companion narrated notebook: ``examples/interp_brock_mirman.ipynb``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple  # noqa: F401

import equinox as eqx  # noqa: F401
from jax import Array  # noqa: F401

from deqn_jax.networks.linear_plus_mlp import LinearPlusMLP  # noqa: F401
from deqn_jax.networks.mlp import MLP  # noqa: F401

# ---------------------------------------------------------------------------
# Primitives — populated by subsequent tasks
# ---------------------------------------------------------------------------
