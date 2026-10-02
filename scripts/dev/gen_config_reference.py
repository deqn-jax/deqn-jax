"""Generate docs/site/config_reference.md from Pydantic config introspection.

Usage: ``uv run python scripts/dev/gen_config_reference.py``

Output overwrites ``docs/site/config_reference.md`` with one table per
config class (``TrainConfig`` and the six blocks nested in it) listing
every field, its type, default, and description as declared via
``Field(description=...)``. Fields without an explicit description fall
back to ``—``, so the rendered doc shows which fields still need one.

The generator is deliberately boring: no templating engine, no plugins,
just introspection + f-strings. Regenerate after any config change.
"""

from __future__ import annotations

import typing as _t
from pathlib import Path

from pydantic import BaseModel

from deqn_jax.config import (
    CompositeLossConfig,
    CoverageConfig,
    MomentMatchingConfig,
    NetworkConfig,
    OptimizerConfig,
    ReplayBufferConfig,
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
    (
        "ReplayBufferConfig",
        ReplayBufferConfig,
        "Prioritized state-replay buffer (only active when ``enabled: true``); nested under ``replay_buffer:`` in YAML.",
    ),
    (
        "CoverageConfig",
        CoverageConfig,
        "EWM coverage sampling (only active when ``enabled: true``); nested under ``coverage:`` in YAML.",
    ),
    (
        "MomentMatchingConfig",
        MomentMatchingConfig,
        "Moment-matching auxiliary loss (only active when ``enabled: true``); nested under ``moment_matching:`` in YAML.",
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


def _format_default(field: _t.Any) -> str:
    if field.is_required():
        return "_required_"
    if field.default_factory is not None:
        default = field.default_factory()
        if isinstance(default, BaseModel):
            return f"`{type(default).__name__}()`"
    else:
        default = field.default
    if default is None:
        return "`None`"
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
        default = _format_default(field)
        description = (field.description or "—").strip()
        lines.append(
            f"| `{field_name}` | `{_escape_md(ann)}` | {default} | {_escape_md(description)} |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    out_path = (
        Path(__file__).resolve().parents[2] / "docs" / "site" / "config_reference.md"
    )

    preface = """# Config reference

Every field on the seven Pydantic config classes (``TrainConfig`` and its nested blocks ``OptimizerConfig``, ``NetworkConfig``, ``CompositeLossConfig``, ``ReplayBufferConfig``, ``CoverageConfig``, ``MomentMatchingConfig``) with its type, default, and a one-line description.

Generated from introspection by ``scripts/dev/gen_config_reference.py`` — regenerate after any config change:

```bash
uv run python scripts/dev/gen_config_reference.py
```

A field with description ``—`` has no explicit ``Field(description=...)`` yet.

For YAML / CLI usage patterns (override precedence, sampling conventions, checkpoint/resume rules, etc.) see [Running experiments](running_experiments.md). For building models with these configs, see [Implementing a model](models/implementing.md).

"""

    body_parts = [render_class(name, cls, subtitle) for name, cls, subtitle in SECTIONS]
    content = preface + "\n".join(body_parts)

    out_path.write_text(content)
    print(f"wrote {out_path}  ({len(content.splitlines())} lines)")


if __name__ == "__main__":
    main()
