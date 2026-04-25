#!/usr/bin/env python3
"""PostToolUse hook: run ruff on .py files just edited by Claude.

Runs on Edit / Write / NotebookEdit tool calls. If the edited file is a
.py file that fails ruff lint, the hook exits with code 2 so Claude Code
feeds the ruff output back to the agent as tool feedback. For non-.py
files, missing files, or clean files, exits 0 silently.

Configured via .claude/settings.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


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

    # Use the project's pinned ruff via `uv run` -- avoids uvx cold-start
    # cost and guarantees the same version CI / contributors would see.
    cmd = ["uv", "run", "--", "ruff", "check", "--no-cache", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return 0

    # Feed ruff output back to Claude as tool feedback (exit code 2).
    output = (result.stdout or "") + (result.stderr or "")
    print(
        f"ruff flagged issues in {path}:\n\n{output.strip()}\n",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
