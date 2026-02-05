"""Economic models for DEQN-JAX.

Each model subpackage exports a MODEL constant (ModelSpec).
To add a new model: create the subpackage, then add one line here.
"""

from typing import List, Tuple

from deqn_jax.types import ModelSpec
from deqn_jax.models.brock_mirman import MODEL as _brock_mirman
from deqn_jax.models.disaster import MODEL as _disaster

_MODELS = {
    "brock_mirman": _brock_mirman,
    "disaster": _disaster,
}

_DESCRIPTIONS = {
    "brock_mirman": "Brock-Mirman (1972) optimal growth model",
    "disaster": "NK-DSGE with financial frictions",
}


def load_model(name: str) -> ModelSpec:
    if name not in _MODELS:
        raise ValueError(f"Unknown model: {name!r}. Available: {list(_MODELS)}")
    return _MODELS[name]


def list_models() -> List[Tuple[str, str]]:
    return [(name, _DESCRIPTIONS.get(name, "")) for name in _MODELS]


__all__ = ["load_model", "list_models"]
