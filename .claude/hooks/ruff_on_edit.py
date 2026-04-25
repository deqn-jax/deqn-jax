#!/usr/bin/env python3
"""PostToolUse hook: format + lint .py files just edited by Claude.

Runs on Edit / Write / NotebookEdit tool calls. Pipeline:

  1. ``ruff format`` -- reformats in place. Style drift is fixed
     silently, never blocks.
  2. ``ruff check --fix`` -- applies safe auto-fixes (import order,
     redundant ``as`` aliases, etc.) in place.
  3. ``ruff check`` -- reports anything still broken (real bugs:
     F401 unused, F821 undefined name, etc.). On non-zero exit, the
     hook exits with code 2 so Claude Code feeds the output back as
     tool feedback.

Non-.py files, missing files, or fully-clean files exit 0 silently.

Configured via .claude/settings.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run(*args: str) -> subprocess.CompletedProcess:
    """Run a `uv run -- ruff …` command, capturing output."""
    cmd = ["uv", "run", "--", "ruff", *args]
    return subprocess.run(cmd, capture_output=True, text=True)


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except Exception as exc:
        # Malformed stdin -- don't block, just log to stderr silently.
        print(f"ruff_on_edit: bad stdin: {exc}", file=sys.stderr)
        return 0

    tool_input = event.get("tool_input") or {}
    file_path = tool_input.get("file_path") or tool_input.get("notebook_path")
    if not file_path:
        return 0

    path = Path(file_path)
    if path.suffix != ".py":
        return 0
    if not path.is_file():
        # File was just deleted, or path points outside the project; skip.
        return 0

    # Step 1: reformat in place. Stays silent on drift; only surfaces
    # parse errors (which would also fail the next steps anyway).
    _run("format", str(path))

    # Step 2: auto-fix safe lint issues in place (I001 import order,
    # redundant aliases, trailing commas, etc.). Unsafe fixes (e.g.
    # deleting bindings ruff thinks are unused but might be re-exports)
    # are NOT applied -- those still surface in step 3.
    _run("check", "--fix", "--no-cache", str(path))

    # Step 3: report anything still broken. These are real bugs the
    # autofixer couldn't safely handle.
    result = _run("check", "--no-cache", str(path))
    if result.returncode == 0:
        return 0

    output = (result.stdout or "") + (result.stderr or "")
    print(
        f"ruff flagged issues in {path}:\n\n{output.strip()}\n",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
