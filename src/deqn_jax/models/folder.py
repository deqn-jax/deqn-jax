"""A model kept in its own folder, outside the package.

    <folder>/
        model.md      the model in words and equations (optional, shown in the report)
        model.py      defines ``MODEL``, a ``ModelSpec``
        train.yaml    training settings (``TrainConfig`` fields; ``model`` is filled in)
        exact.py      optional: ``exact_policy(state, constants)``, a known solution
        runs/01/, runs/02/, ...   one per ``deqn-jax train <folder>``, each with a frozen
                      copy of the files above, ``config.yaml``, ``history.csv``,
                      ``checkpoints/``, and after ``deqn-jax verify`` the report

``model.py`` may set ``SHOCK_NODES_PER_DIM`` when its shocks take a few discrete
values (e.g. 2 for a shock that is +-1 with equal probability); the solution
check then integrates over exactly those values. The simulation still draws
normal shocks, so such a model's ``step_fn`` reads ``jnp.sign(shock)``.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable, Optional

from deqn_jax.types import ModelSpec

SNAPSHOT_FILES = ("model.md", "model.py", "train.yaml", "exact.py")


def is_model_folder(path: str | Path) -> bool:
    return Path(path).is_dir() and (Path(path) / "model.py").exists()


def folder_model_name(folder: Path) -> str:
    """Registry name of a folder's model: the folder name, made identifier-safe."""
    return re.sub(r"[^A-Za-z0-9_]", "_", Path(folder).resolve().name)


def _import(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_model_file(source: Path, name: str) -> tuple[ModelSpec, ModuleType]:
    """Import ``source/model.py`` and register its ``MODEL`` as ``name``."""
    from deqn_jax.models import register_model

    path = Path(source) / "model.py"
    module = _import(path, f"deqn_jax_folder_model_{name}")
    model = getattr(module, "MODEL", None)
    if not isinstance(model, ModelSpec):
        raise TypeError(f"{path} must define MODEL = ModelSpec(...)")
    model = model._replace(name=name)
    register_model(model, overwrite=True)
    return model, module


def load_exact_policy(source: Path) -> Optional[Callable]:
    """``exact_policy(state, constants)`` from ``source/exact.py``, or None."""
    path = Path(source) / "exact.py"
    if not path.exists():
        return None
    module = _import(path, f"deqn_jax_folder_exact_{abs(hash(path.resolve()))}")
    if not hasattr(module, "exact_policy"):
        raise TypeError(f"{path} must define exact_policy(state, constants)")
    return module.exact_policy


def run_dirs(folder: Path) -> list[Path]:
    runs = Path(folder) / "runs"
    if not runs.exists():
        return []
    numbered = (p for p in runs.iterdir() if p.is_dir() and p.name.isdigit())
    return sorted(numbered, key=lambda p: int(p.name))


def new_run_dir(folder: Path) -> Path:
    existing = run_dirs(folder)
    number = int(existing[-1].name) + 1 if existing else 1
    run = Path(folder) / "runs" / f"{number:02d}"
    run.mkdir(parents=True)
    return run


def find_run(folder: Path, run: Optional[str] = None) -> Path:
    """The run ``run`` (e.g. "03"), or the latest one."""
    if run is not None:
        path = Path(folder) / "runs" / run
        if not path.is_dir():
            raise FileNotFoundError(f"no run {path}")
        return path
    runs = run_dirs(folder)
    if not runs:
        raise FileNotFoundError(f"{folder} has no runs yet: deqn-jax train {folder}")
    return runs[-1]


def start_run(folder: Path) -> tuple[str, Path]:
    """Register the folder's model and open ``runs/NN`` with a frozen copy of its files."""
    import shutil

    folder = Path(folder)
    name = folder_model_name(folder)
    load_model_file(folder, name)  # a broken model.py fails before a run folder exists
    run = new_run_dir(folder)
    for filename in SNAPSHOT_FILES:
        if (folder / filename).exists():
            shutil.copy2(folder / filename, run / filename)
    (run / "checkpoints").mkdir()
    return name, run


def write_history(run: Path, history: dict) -> None:
    with (Path(run) / "history.csv").open("w") as fh:
        fh.write("episode,loss\n")
        for i, loss in enumerate(history.get("loss", [])):
            fh.write(f"{i},{float(loss)}\n")
