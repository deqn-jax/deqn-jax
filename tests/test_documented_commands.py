"""The commands and imports the docs show must work against the code.

Pages read: ``README.md`` and ``docs/site/**/*.md``.

(a) Every ``deqn-jax <subcommand> ...`` in a fenced shell block or an inline
    code span parses with the real argparse parser (built from ``deqn_jax.cli``
    without running anything), and the models, configs, optimizers, ``--set``
    keys and shocks it names exist.
(b) Every import statement in a ``python`` block resolves.
(c) README's quick start runs top to bottom at smoke size in a tmp dir.

Placeholders are recognised explicitly (see ``PLACEHOLDER_PATHS`` and
``_is_placeholder``); nothing is skipped by file.
"""

import argparse
import ast
import contextlib
import importlib
import io
import pkgutil
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOC_FILES = [ROOT / "README.md", *sorted((ROOT / "docs" / "site").rglob("*.md"))]

FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})[ \t]*(?P<lang>[\w+-]*)")
SHELL_LANGS = {"bash", "sh", "shell", "console", "text", ""}
PYTHON_LANGS = {"python", "py"}
ELISION = {"...", "…"}
SHELL_OPERATORS = {"|", "||", "&&", ";", "&", ">", ">>", "2>&1"}
# Options of the top-level parser in deqn_jax.cli.main (argparse adds -h).
TOP_LEVEL_FLAGS = {"--version", "-V", "--help", "-h"}

# Paths the docs name that a reader creates, not the repository: each entry
# says where the docs tell the reader to make it.
PLACEHOLDER_PATHS = {
    "models/my_model": "the reader's model folder (REFERENCE.md, 'Path C')",
    "my_config.yaml": "the reader's YAML (getting-started/configuration.md)",
}


@dataclass(frozen=True)
class Command:
    """One documented ``deqn-jax`` invocation: the arguments after the program."""

    where: str
    args: tuple
    inline: bool

    def __str__(self):
        return f"{self.where}: deqn-jax {' '.join(self.args)}"


def _rel(path):
    return str(path.relative_to(ROOT))


def _fenced_blocks(path):
    """Yield ``(first_line_no, lang, body_lines)`` of every fenced block."""
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        indent, fence, lang = m["indent"], m["fence"], m["lang"]
        start, body = i + 1, []
        i += 1
        while i < len(lines):
            s = lines[i].strip()
            if s.startswith(fence) and not s.strip(fence[0]):
                break
            line = lines[i]
            body.append(
                line[len(indent) :] if line.startswith(indent) else line.lstrip()
            )
            i += 1
        yield start, lang, body
        i += 1


def _outside_fences(path):
    """Yield ``(line_no, line)`` for lines that are not inside a fenced block."""
    inside = None
    for no, line in enumerate(path.read_text().splitlines(), 1):
        m = FENCE.match(line)
        if inside is None and m:
            inside = m["fence"]
            continue
        if inside is not None:
            s = line.strip()
            if s.startswith(inside) and not s.strip(inside[0]):
                inside = None
            continue
        yield no, line


def _deqn_args(tokens):
    """Arguments after ``deqn-jax`` when it is the command being run, else None.

    Accepts a leading ``$`` prompt, ``VAR=value`` assignments and ``uv run``;
    ``cd deqn-jax`` or a diagram label is not a command. A pipe or redirect
    (``| tee run.log``) ends the arguments.
    """
    i = 0
    if tokens[:1] == ["$"]:
        i = 1
    while i < len(tokens) and re.fullmatch(r"[A-Z_][A-Z0-9_]*=\S*", tokens[i]):
        i += 1
    if tokens[i : i + 2] == ["uv", "run"]:
        i += 2
    if i >= len(tokens) or tokens[i] != "deqn-jax":
        return None
    args = []
    for token in tokens[i + 1 :]:
        if token in SHELL_OPERATORS:
            break
        if token not in ELISION:
            args.append(token)
    return tuple(args)


def documented_commands():
    out = []
    for path in DOC_FILES:
        for start, lang, body in _fenced_blocks(path):
            if lang not in SHELL_LANGS:
                continue
            joined, first = "", None
            for offset, line in enumerate(body):
                if first is None:
                    first = start + 1 + offset
                if line.rstrip().endswith("\\"):
                    joined += line.rstrip()[:-1] + " "
                    continue
                text, joined = joined + line, ""
                if "deqn-jax" in text:
                    args = _deqn_args(shlex.split(text, comments=True))
                    if args is not None:
                        out.append(Command(f"{_rel(path)}:{first}", args, False))
                first = None
        for no, line in _outside_fences(path):
            for span in re.findall(r"`([^`]+)`", line):
                if re.match(r"(uv run )?deqn-jax \S", span):
                    args = _deqn_args(shlex.split(span))
                    out.append(Command(f"{_rel(path)}:{no}", args, True))
    return out


COMMANDS = documented_commands()


@lru_cache(maxsize=1)
def cli_parser():
    """The ``deqn-jax`` parser, assembled the way ``deqn_jax.cli.main`` does."""
    import deqn_jax.cli as cli

    parser = argparse.ArgumentParser(prog="deqn-jax")
    subparsers = parser.add_subparsers(dest="command")
    for info in pkgutil.iter_modules(cli.__path__):
        module = importlib.import_module(f"deqn_jax.cli.{info.name}")
        if hasattr(module, "add_parser"):
            module.add_parser(subparsers)
    return parser, subparsers.choices


@lru_cache(maxsize=1)
def registry():
    from deqn_jax.models import list_models, load_model
    from deqn_jax.optimizers.registry import list_optimizers

    shocks = {
        name: tuple(load_model(name).shock_names or ()) for name, _ in list_models()
    }
    return shocks, set(list_optimizers())


def _is_placeholder(token):
    """A value of the reader's choosing: ``<checkpoint.eqx>``, or a metavar
    such as ``MODEL`` / ``CKPT`` in a usage line."""
    return (token.startswith("<") and token.endswith(">")) or bool(
        re.fullmatch(r"[A-Z][A-Z_]*", token)
    )


def _check_path(value, problems, what):
    if _is_placeholder(value) or value in PLACEHOLDER_PATHS:
        return
    if not (ROOT / value).exists():
        problems.append(f"{what} {value!r} does not exist in the repository")


def _check_config(value, problems):
    # A run writes its resolved config next to its checkpoints (checkpointing.py).
    if value.endswith("/config.yaml") and not value.startswith("configs/"):
        return
    _check_path(value, problems, "config")


def _model_of_checkpoint(path, shocks):
    """The registered model a checkpoint path names as a directory, if any."""
    named = [part for part in Path(path).parts[:-1] if part in shocks]
    return named[-1] if named else None


def _check_semantics(ns, problems):
    from deqn_jax.config import TrainConfig

    shocks, optimizers = registry()
    model = getattr(ns, "model", None)
    if ns.command in ("train", "info") and model and not _is_placeholder(model):
        if "/" in model:
            _check_path(model, problems, "model folder")
        elif model not in shocks:
            problems.append(f"model {model!r} is not registered")
    if ns.command == "verify":
        _check_path(ns.folder, problems, "model folder")
    if getattr(ns, "config", None):
        _check_config(ns.config, problems)
    for attr in ("checkpoint", "resume"):
        value = getattr(ns, attr, None)
        if value and not _is_placeholder(value) and not value.endswith(".eqx"):
            problems.append(f"{attr} {value!r} is not a .eqx checkpoint")
    if ns.command == "train" and ns.optimizer and ns.optimizer not in optimizers:
        problems.append(f"optimizer {ns.optimizer!r} is not registered")
    for item in getattr(ns, "set", None) or ():
        key, _, value = item.partition("=")
        try:
            TrainConfig().with_overrides({key: value})
        except ValueError as exc:  # what `train --set` raises on this pair
            problems.append(f"--set {item}: {exc}")
        if key == "optimizer.name" and value not in optimizers:
            problems.append(f"optimizer {value!r} is not registered")
    if ns.command == "irf" and ns.shocks:
        named = _model_of_checkpoint(ns.checkpoint, shocks)
        valid = set(shocks[named]) if named else set().union(*shocks.values())
        for shock in ns.shocks:
            if shock not in valid:
                owner = named or "any registered model"
                problems.append(f"shock {shock!r} is not a shock of {owner}")


def _synopsis_flags(args):
    """Option strings in a usage line such as ``[--horizon N] [--girf|--x]``."""
    flags = []
    for token in args:
        for part in token.strip("[]").split("|"):
            if part.startswith("-"):
                flags.append(part.split("=")[0])
    return flags


@pytest.mark.parametrize("cmd", COMMANDS, ids=str)
def test_documented_command_matches_cli(cmd):
    parser, subcommands = cli_parser()
    args = list(cmd.args)
    if len(args) == 1 and args[0] in TOP_LEVEL_FLAGS:
        return
    assert args and args[0] in subcommands, f"{cmd}: no subcommand {args[:1]}"
    sub = subcommands[args[0]]

    if cmd.inline and len(args) == 1:
        return  # the command named in prose, e.g. "`deqn-jax evaluate` exit code"
    if "--help" in args or "-h" in args:
        return  # argparse prints the help and exits before reading the rest
    if any(c in token for token in args for c in "[]|"):
        unknown = [
            f for f in _synopsis_flags(args) if f not in sub._option_string_actions
        ]
        assert not unknown, f"{cmd}: {args[0]} has no option {unknown}"
        return

    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        try:
            ns = parser.parse_args(args)
        except SystemExit:
            pytest.fail(f"{cmd}\n{err.getvalue().strip().splitlines()[-1]}")
    problems = []
    _check_semantics(ns, problems)
    assert not problems, f"{cmd}\n" + "\n".join(problems)


def test_commands_were_found():
    """The extractor sees the pages: a regex slip must not empty the check."""
    where = {c.where.split(":")[0] for c in COMMANDS}
    assert {"README.md", "docs/site/getting-started/quickstart.md"} <= where
    assert len(COMMANDS) > 50


# ---------------------------------------------------------------- (b) imports


def _import_statements(source):
    """Import statements of a python block.

    Blocks that show a signature or elide arguments (``ModelSpec(name: str,
    ...``, ``n_states=…``) are not valid Python; their imports are read line
    by line, joining parenthesised continuations.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        stmts, pending = [], ""
        for line in source.splitlines():
            if pending or re.match(r"(import|from)\s+\w", line):
                pending += line + "\n"
                if pending.count("(") == pending.count(")"):
                    stmts.extend(ast.parse(pending).body)
                    pending = ""
        return stmts
    return [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]


def documented_imports():
    out = []
    for path in DOC_FILES:
        for start, lang, body in _fenced_blocks(path):
            if lang in PYTHON_LANGS:
                for node in _import_statements("\n".join(body)):
                    where = f"{_rel(path)}:{start + node.lineno}"
                    out.append(pytest.param(ast.unparse(node), id=f"{where}"))
    return out


@pytest.mark.parametrize("statement", documented_imports())
def test_documented_import_resolves(statement):
    exec(compile(statement, "<docs>", "exec"), {})


# ------------------------------------------------------ (c) README quick start

# Smoke size, appended (argparse keeps the last value): the repo's convention
# (3 episodes, hidden=(16,), batch=16) for training, a short simulation for
# evaluate.
SMOKE_ARGS = {
    "train": ["-n", "3", "--hidden", "16", "--batch-size", "16"],
    "evaluate": ["-n", "200"],
}
# Documented in the quick start but not run here, with the reason.
NOT_RUN = {
    ("train", "--config", "configs/disaster.yaml"): "experimental arm; its lr "
    "warm-up is longer than a smoke run, so a 3-episode run cannot start",
}


def _quick_start_commands():
    lines = (ROOT / "README.md").read_text().splitlines()
    start = lines.index("## Quick start")
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("## "))
    return [
        c.args
        for c in COMMANDS
        if c.where.startswith("README.md:") and not c.inline
        if start < int(c.where.split(":")[1]) <= end
    ]


def test_readme_quick_start_runs(tmp_path):
    commands = [a for a in _quick_start_commands() if a not in NOT_RUN]
    assert [a[0] for a in commands] == ["train", "evaluate", "irf"]
    for args in commands:
        run = subprocess.run(
            [sys.executable, "-m", "deqn_jax.cli", *args, *SMOKE_ARGS.get(args[0], [])],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=600,
        )
        tail = "\n".join((run.stdout + run.stderr).strip().splitlines()[-15:])
        assert run.returncode == 0, f"deqn-jax {' '.join(args)} failed:\n{tail}"
