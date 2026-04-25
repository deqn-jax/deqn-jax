"""Verify the hand-drawn module dependency graph in docs/site/architecture.md
against the actual import graph extracted by pydeps.

Usage: ``uv run python scripts/check_module_graph.py``

Exits 0 with a "no drift" report if the package-level edges in
DRAWN_EDGES below match the real import graph at depth 2.
Exits 1 with a list of missing / phantom edges if they don't.

When the architecture changes, edit DRAWN_EDGES to match the new
real graph AND update the mermaid in architecture.md.
"""

from __future__ import annotations

import json
import subprocess
import sys

DRAWN_EDGES = {
    ("deqn_jax.cli", "deqn_jax.training"),
    ("deqn_jax.cli", "deqn_jax.models"),
    ("deqn_jax.cli", "deqn_jax.evaluate"),
    ("deqn_jax.cli", "deqn_jax.irf"),
    ("deqn_jax.cli", "deqn_jax.config"),
    ("deqn_jax.cli", "deqn_jax.optimizers"),
    ("deqn_jax.benchmark", "deqn_jax.training"),
    ("deqn_jax.benchmark", "deqn_jax.models"),
    ("deqn_jax.training", "deqn_jax.config"),
    ("deqn_jax.training", "deqn_jax.types"),
    ("deqn_jax.training", "deqn_jax.metrics"),
    ("deqn_jax.training", "deqn_jax.networks"),
    ("deqn_jax.training", "deqn_jax.optimizers"),
    ("deqn_jax.training", "deqn_jax.models"),
    ("deqn_jax.models", "deqn_jax.types"),
    ("deqn_jax.models", "deqn_jax.training"),
    ("deqn_jax.networks", "deqn_jax.training"),
    ("deqn_jax.evaluate", "deqn_jax.training"),
    ("deqn_jax.evaluate", "deqn_jax.irf"),
    ("deqn_jax.irf", "deqn_jax.training"),
    ("deqn_jax.irf", "deqn_jax.models"),
    ("deqn_jax.irf", "deqn_jax.config"),
}


def _collapse(mod: str) -> str | None:
    """Collapse a module name to its top-level deqn_jax.<package>. Drop everything outside the package."""
    if not mod.startswith("deqn_jax"):
        return None
    parts = mod.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else parts[0]


def real_edges() -> set[tuple[str, str]]:
    """Run pydeps and return the package-level edge set, filtered to within-package."""
    out = subprocess.check_output(
        [
            "uv",
            "run",
            "pydeps",
            "src/deqn_jax",
            "--show-deps",
            "--no-output",
            "--noshow",
            "--max-bacon=4",
        ],
        text=True,
    )
    data = json.loads(out)
    edges: set[tuple[str, str]] = set()
    for mod, info in data.items():
        src = _collapse(mod)
        if src is None or src == "deqn_jax":
            continue
        for imp in info.get("imports", []):
            dst = _collapse(imp)
            if dst is None or dst == src or dst == "deqn_jax":
                continue
            edges.add((src, dst))
    return edges


def main() -> int:
    real = real_edges()
    missing = sorted(real - DRAWN_EDGES)
    phantom = sorted(DRAWN_EDGES - real)
    print(f"Real package-level edges: {len(real)}")
    print(f"Edges drawn in mermaid:   {len(DRAWN_EDGES)}")

    if not missing and not phantom:
        print("\nNo drift. Mermaid matches the real import graph.")
        return 0

    if missing:
        print("\nEdges in real graph but NOT drawn in mermaid:")
        for a, b in missing:
            print(f"  {a:30s} -> {b}")
    if phantom:
        print("\nEdges drawn but NOT in real graph (mermaid is stale):")
        for a, b in phantom:
            print(f"  {a:30s} -> {b}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
