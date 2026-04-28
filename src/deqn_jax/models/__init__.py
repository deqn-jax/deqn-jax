"""Economic models for DEQN-JAX.

Each model subpackage exports a MODEL constant (ModelSpec).
To add a new model: create the subpackage, then add one line here.
"""

from typing import List, Tuple

from deqn_jax.models.bm_deterministic import MODEL as _bm_deterministic
from deqn_jax.models.bm_labor import MODEL as _bm_labor
from deqn_jax.models.bm_labor_autodiff import MODEL as _bm_labor_autodiff
from deqn_jax.models.brock_mirman import MODEL as _brock_mirman
from deqn_jax.models.brock_mirman_autodiff import MODEL as _brock_mirman_autodiff
from deqn_jax.models.brock_mirman_ez import MODEL as _brock_mirman_ez
from deqn_jax.models.disaster import MODEL as _disaster
from deqn_jax.models.irbc import MODEL as _irbc
from deqn_jax.models.olg_analytic_6 import MODEL as _olg_analytic_6
from deqn_jax.types import ModelSpec

_MODELS = {
    "bm_deterministic": _bm_deterministic,
    "brock_mirman": _brock_mirman,
    "brock_mirman_autodiff": _brock_mirman_autodiff,
    "brock_mirman_ez": _brock_mirman_ez,
    "bm_labor": _bm_labor,
    "bm_labor_autodiff": _bm_labor_autodiff,
    "olg_analytic_6": _olg_analytic_6,
    "irbc": _irbc,
    "disaster": _disaster,
}

_DESCRIPTIONS = {
    "bm_deterministic": "Deterministic Brock-Mirman (s* = alpha*beta closed form)",
    "brock_mirman": "Brock-Mirman (1972) optimal growth model",
    "brock_mirman_autodiff": "Brock-Mirman with Euler synthesized from Pi via jax.grad (autodiff POC)",
    "brock_mirman_ez": "Brock-Mirman with Epstein-Zin recursive utility (actor-critic demo)",
    "bm_labor": "Brock-Mirman (1972) with endogenous labor supply",
    "bm_labor_autodiff": "Brock-Mirman with labor, both FOCs from Pi via jax.grad (multi-policy autodiff)",
    "olg_analytic_6": "6-agent OLG with closed-form solution (Krueger-Kubler 2004)",
    "irbc": "2-country International RBC with irreversibility (Fischer-Burmeister)",
    "disaster": "NK-DSGE with financial frictions",
}


def load_model(name: str) -> ModelSpec:
    if name not in _MODELS:
        raise ValueError(f"Unknown model: {name!r}. Available: {list(_MODELS)}")
    return _MODELS[name]


def list_models() -> List[Tuple[str, str]]:
    return [(name, _DESCRIPTIONS.get(name, "")) for name in _MODELS]


__all__ = ["load_model", "list_models"]
