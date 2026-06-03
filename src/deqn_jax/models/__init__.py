"""Economic models for DEQN-JAX.

Each model subpackage exports a ``MODEL: ModelSpec`` constant. Two
registration paths, both supported and orthogonal:

1. **In-tree** (this file's ``_MODELS`` / ``_DESCRIPTIONS`` dicts):
   add an import + entry. Right thing for models that ship with the
   library and are version-controlled here.
2. **Programmatic** (``register_model(spec, description=...)``): add
   at runtime from anywhere. Right thing for agent-codegen'd models
   in user projects, notebook prototypes, or pluggable extensions
   that don't want to fork the library.

Both paths feed the same ``_MODELS`` dict, so ``load_model`` and
``list_models`` see them identically.
"""

from typing import List, Optional, Tuple

from deqn_jax.models.bm_deterministic import MODEL as _bm_deterministic
from deqn_jax.models.bm_labor import MODEL as _bm_labor
from deqn_jax.models.bm_labor_autodiff import MODEL as _bm_labor_autodiff
from deqn_jax.models.bm_labor_constrained import MODEL as _bm_labor_constrained
from deqn_jax.models.brock_mirman import MODEL as _brock_mirman
from deqn_jax.models.brock_mirman_autodiff import MODEL as _brock_mirman_autodiff
from deqn_jax.models.disaster import MODEL as _disaster
from deqn_jax.models.irbc import MODEL as _irbc
from deqn_jax.models.olg_analytic_6 import MODEL as _olg_analytic_6
from deqn_jax.models.olg_lifecycle import MODEL as _olg_lifecycle
from deqn_jax.types import ModelSpec

_MODELS = {
    "bm_deterministic": _bm_deterministic,
    "brock_mirman": _brock_mirman,
    "brock_mirman_autodiff": _brock_mirman_autodiff,
    "bm_labor": _bm_labor,
    "bm_labor_autodiff": _bm_labor_autodiff,
    "bm_labor_constrained": _bm_labor_constrained,
    "olg_analytic_6": _olg_analytic_6,
    "olg_lifecycle": _olg_lifecycle,
    "irbc": _irbc,
    "disaster": _disaster,
}

_DESCRIPTIONS = {
    "bm_deterministic": "Deterministic Brock-Mirman (s* = alpha*beta closed form)",
    "brock_mirman": "Brock-Mirman (1972) optimal growth model",
    "brock_mirman_autodiff": "Brock-Mirman with Euler synthesized from Pi via jax.grad (autodiff POC)",
    "bm_labor": "Brock-Mirman (1972) with endogenous labor supply",
    "bm_labor_autodiff": "Brock-Mirman with labor, both FOCs from Pi via jax.grad (multi-policy autodiff)",
    "bm_labor_constrained": "Brock-Mirman with endogenous labor and an upper labor cap (Fischer-Burmeister)",
    "olg_analytic_6": "6-agent OLG with closed-form solution (Krueger-Kubler 2004)",
    "olg_lifecycle": "6-generation life-cycle OLG with borrowing constraints (Fischer-Burmeister, two-stage loss)",
    "irbc": "2-country International RBC with irreversibility (Fischer-Burmeister)",
    "disaster": "NK-DSGE with financial frictions",
}


def load_model(name: str) -> ModelSpec:
    """Return the registered model named ``name``.

    Raises ``ValueError`` with the available names if not found.
    """
    if name not in _MODELS:
        raise ValueError(f"Unknown model: {name!r}. Available: {list(_MODELS)}")
    return _MODELS[name]


def list_models() -> List[Tuple[str, str]]:
    """Return ``[(name, description), ...]`` for every registered model.

    Sees both in-tree and runtime-registered entries.
    """
    return [(name, _DESCRIPTIONS.get(name, "")) for name in _MODELS]


def register_model(
    spec: ModelSpec,
    *,
    description: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """Register a model at runtime so ``load_model(spec.name)`` finds it.

    Equivalent to adding an entry to ``_MODELS`` and ``_DESCRIPTIONS``
    in this file, but at import / instantiation time. Intended for:

      * agent-codegen'd models in user projects (no fork required),
      * notebook prototyping,
      * plugin packages that add models to deqn-jax via ``register_model``
        on import.

    Args:
        spec: A populated ``ModelSpec``. Its ``spec.name`` field is the
            registry key. The framework consumes this object via the same
            ``load_model`` lookup as in-tree models.
        description: Human-readable one-liner for ``list_models``. Defaults
            to an empty string if omitted.
        overwrite: If True, an existing entry with the same name is
            replaced. If False (default) and ``spec.name`` is already
            registered, raises ``ValueError`` so you don't shadow an
            in-tree model by accident.

    Raises:
        ValueError: If ``spec.name`` is empty, or already registered and
            ``overwrite=False``.
        TypeError: If ``spec`` is not a ``ModelSpec``.

    Example::

        from deqn_jax.api import ModelSpec, register_model

        MY_MODEL = ModelSpec(name="my_model", ...)
        register_model(MY_MODEL, description="My agent-built model")

        # Now usable everywhere:
        from deqn_jax.api import load_model, train_from_config, TrainConfig
        cfg = TrainConfig(model="my_model", ...)
        state, history = train_from_config(cfg)
    """
    if not isinstance(spec, ModelSpec):
        raise TypeError(
            f"register_model expects a ModelSpec; got {type(spec).__name__}"
        )
    if not spec.name:
        raise ValueError("ModelSpec.name must be a non-empty string")
    if spec.name in _MODELS and not overwrite:
        raise ValueError(
            f"Model {spec.name!r} is already registered. Pass overwrite=True "
            f"to replace it, or pick a different name. Currently registered: "
            f"{sorted(_MODELS)}"
        )
    _MODELS[spec.name] = spec
    _DESCRIPTIONS[spec.name] = description or ""


def unregister_model(name: str) -> None:
    """Remove a runtime-registered model. No-op if not present.

    Mainly useful in tests so a leaked ``register_model`` from one test
    case doesn't bleed into another. Note: in-tree models can also be
    removed this way, but doing so is rarely a good idea.
    """
    _MODELS.pop(name, None)
    _DESCRIPTIONS.pop(name, None)


__all__ = ["load_model", "list_models", "register_model", "unregister_model"]
