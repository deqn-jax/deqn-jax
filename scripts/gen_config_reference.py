"""Generate docs/site/config_reference.md from Pydantic config introspection.

Usage: ``uv run python scripts/gen_config_reference.py``

Output overwrites ``docs/site/config_reference.md`` with one table per
config class (OptimizerConfig / NetworkConfig / CompositeLossConfig /
TrainConfig) listing every field, its type, default, and description
as declared via ``Field(description=...)``. Fields without an explicit
description fall back to ``—``; the rendered doc therefore *shows*
which fields still need a description, making this a progress bar
for the config documentation effort.

The generator is deliberately boring: no templating engine, no plugins,
just introspection + f-strings. Regenerate after any config change.
"""

from __future__ import annotations

import typing as _t
from pathlib import Path

from deqn_jax.config import (
    CompositeLossConfig,
    NetworkConfig,
    OptimizerConfig,
    TrainConfig,
)

SECTIONS = [
    ("TrainConfig", TrainConfig, "Top-level training configuration."),
    (
        "OptimizerConfig",
        OptimizerConfig,
        "Optimizer choice and hyperparameters; nested under ``optimizer:`` in YAML.",
    ),
    (
        "NetworkConfig",
        NetworkConfig,
        "Policy network architecture; nested under ``network:`` in YAML.",
    ),
    (
        "CompositeLossConfig",
        CompositeLossConfig,
        "Composite-loss weights (only active when ``loss_type: composite``); nested under ``composite_loss:`` in YAML.",
    ),
]


def _format_type(annotation: _t.Any) -> str:
    """Render a type annotation as a compact string."""
    try:
        origin = _t.get_origin(annotation)
        args = _t.get_args(annotation)
    except TypeError:
        origin = None
        args = ()

    if annotation is type(None):
        return "None"
    if origin is None:
        if hasattr(annotation, "__name__"):
            return annotation.__name__
        return str(annotation)
    if origin is _t.Union or (
        hasattr(annotation, "__class__")
        and annotation.__class__.__name__ == "UnionType"
    ):
        inner = ", ".join(_format_type(a) for a in args)
        return f"Union[{inner}]"
    origin_name = getattr(origin, "__name__", str(origin))
    if args:
        return f"{origin_name}[{', '.join(_format_type(a) for a in args)}]"
    return origin_name


def _format_default(default: _t.Any) -> str:
    if default is None:
        return "`None`"
    if callable(default) and default.__class__.__name__ == "PydanticUndefinedType":
        return "_required_"
    if isinstance(default, str):
        return f"`{default!r}`"
    if isinstance(default, (list, tuple)) and len(default) == 0:
        return "`[]`" if isinstance(default, list) else "`()`"
    return f"`{default!r}`"


def _escape_md(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def render_class(name: str, cls: _t.Any, subtitle: str) -> str:
    fields = cls.model_fields  # Pydantic v2 API: {field_name: FieldInfo}

    lines = [f"## `{name}`", "", subtitle, ""]
    lines += [
        "| Field | Type | Default | Description |",
        "|---|---|---|---|",
    ]

    for field_name, field in fields.items():
        ann = _format_type(field.annotation)
        default = _format_default(field.default if field.default is not ... else None)
        description = (field.description or "—").strip()
        lines.append(
            f"| `{field_name}` | `{_escape_md(ann)}` | {default} | {_escape_md(description)} |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    out_path = (
        Path(__file__).resolve().parent.parent / "docs" / "site" / "config_reference.md"
    )

    preface = """# Config reference

Every field on the four Pydantic config classes (``TrainConfig``, ``OptimizerConfig``, ``NetworkConfig``, ``CompositeLossConfig``) with its type, default, and a one-line description.

Generated from introspection by ``scripts/gen_config_reference.py`` — regenerate after any config change:

```bash
uv run python scripts/gen_config_reference.py
```

Fields with description ``—`` haven't had an explicit ``Field(description=...)`` added yet; the generator surfaces these as a TODO list for the docs effort. Start there when a user asks "what does X do."

For YAML / CLI usage patterns (override precedence, sampling conventions, checkpoint/resume rules, etc.) see [Running experiments](running_experiments.md). For building models with these configs, see [Implementing a model](models/implementing.md).

"""

    body_parts = [render_class(name, cls, subtitle) for name, cls, subtitle in SECTIONS]
    content = preface + "\n".join(body_parts)

    out_path.write_text(content)
    print(f"wrote {out_path}  ({len(content.splitlines())} lines)")


if __name__ == "__main__":
    main()
