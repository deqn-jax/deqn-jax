"""Config base class + Pydantic-error reframing + scalar coercion helpers."""

from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
)

# ---------------------------------------------------------------------------
# Base class: wraps Pydantic ValidationError → ValueError / TypeError
# ---------------------------------------------------------------------------


class _ConfigBase(BaseModel):
    """Shared base that converts Pydantic ValidationError into the
    ValueError / TypeError that existing callers expect."""

    model_config = ConfigDict(extra="forbid")

    def __init__(self, **data: Any):
        try:
            super().__init__(**data)
        except ValidationError as exc:
            _reraise_validation_error(exc, type(self).__name__)

    @classmethod
    def model_validate(cls, obj: Any, **kwargs):
        try:
            return super().model_validate(obj, **kwargs)
        except ValidationError as exc:
            _reraise_validation_error(exc, cls.__name__)


def _reraise_validation_error(exc: ValidationError, cls_name: str):
    """Convert a Pydantic ValidationError into ValueError or TypeError.

    Preserves the original error messages that tests match against.
    """
    errors = exc.errors()
    if len(errors) == 1:
        err = errors[0]
        msg = err.get("msg", "")
        loc = ".".join(str(p) for p in err.get("loc", ()))
        err_type = err.get("type", "")

        # Our custom validators raise with "Value error, <message>"
        if msg.startswith("Value error, "):
            raise ValueError(msg[len("Value error, ") :]) from None

        # Bool rejection from our validators
        if "expected" in msg and "got bool" in msg:
            raise TypeError(msg) from None

        # Pydantic's built-in type errors
        if err_type in (
            "int_type",
            "int_parsing",
            "float_type",
            "float_parsing",
            "string_type",
            "bool_type",
            "tuple_type",
            "list_type",
        ):
            expected = _pydantic_type_to_name(err_type)
            inp = err.get("input")
            actual = type(inp).__name__
            raise TypeError(
                f"{cls_name}.{loc}: expected {expected}, got {actual} ({inp!r})"
            ) from None

        # extra_forbidden (unknown key via ConfigDict(extra='forbid'))
        if err_type == "extra_forbidden":
            raise ValueError(str(exc)) from None

    # Multiple errors or unrecognised pattern — use first error
    first = errors[0]
    msg = first.get("msg", str(exc))
    if msg.startswith("Value error, "):
        raise ValueError(msg[len("Value error, ") :]) from None
    raise ValueError(str(exc)) from None


def _pydantic_type_to_name(err_type: str) -> str:
    """Map Pydantic error type codes to human-readable type names."""
    mapping = {
        "int_type": "int",
        "int_parsing": "int",
        "float_type": "float",
        "float_parsing": "float",
        "string_type": "str",
        "bool_type": "bool",
        "tuple_type": "Tuple[int, ...]",
        "list_type": "List[float]",
    }
    return mapping.get(err_type, err_type)


# ---------------------------------------------------------------------------
# Shared coercion helpers (called from field_validator one-liners)
# ---------------------------------------------------------------------------


def _coerce_float(v: Any, prefix: str) -> Any:
    """Reject bool, coerce str→float, reject list/dict/tuple."""
    if isinstance(v, bool):
        raise TypeError(f"{prefix}: expected float, got bool ({v!r})")
    if isinstance(v, (list, dict, tuple)):
        raise TypeError(f"{prefix}: expected float, got {type(v).__name__} ({v!r})")
    if isinstance(v, str):
        try:
            return float(v)
        except (TypeError, ValueError):
            raise TypeError(f"{prefix}: expected float, got str ({v!r})") from None
    return v


def _coerce_int(v: Any, prefix: str) -> Any:
    """Reject bool, coerce str→int, coerce float→int."""
    if isinstance(v, bool):
        raise TypeError(f"{prefix}: expected int, got bool ({v!r})")
    if isinstance(v, str):
        try:
            return int(v)
        except (TypeError, ValueError):
            raise TypeError(f"{prefix}: expected int, got str ({v!r})") from None
    if isinstance(v, float):
        return int(v)
    return v


def _coerce_optional_float(v: Any, prefix: str) -> Any:
    """Like _coerce_float but passes None through."""
    if v is None:
        return v
    return _coerce_float(v, prefix)


def _coerce_optional_int(v: Any, prefix: str) -> Any:
    """Like _coerce_int but passes None through."""
    if v is None:
        return v
    return _coerce_int(v, prefix)
