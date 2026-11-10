"""Metric-logging backends on the documented install (plain ``uv sync``).

Every ``configs/disaster*.yaml`` sets ``tensorboard_dir``, so TensorBoard
logging must work without any extra; W&B stays optional and, when absent,
``--wandb`` must stop with one line naming the extra, not a traceback.
"""

import sys
from importlib.metadata import requires

import pytest

from deqn_jax.cli import main
from deqn_jax.training.metrics import MissingDependencyError, create_logger


def test_tensorboardx_is_a_core_dependency():
    core = [r for r in requires("deqn-jax") or [] if "extra ==" not in r]
    assert any(r.lower().startswith("tensorboardx") for r in core)


def test_tensorboard_logger_writes_events(tmp_path):
    logger = create_logger(tensorboard_dir=str(tmp_path))
    logger.log_scalars({"loss": 1.0}, step=0)
    logger.close()
    assert any(p.name.startswith("events.out.tfevents") for p in tmp_path.iterdir())


def test_missing_wandb_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)  # import wandb -> ImportError
    with pytest.raises(MissingDependencyError, match=r"\[wandb\]"):
        create_logger(wandb_project="p")


def test_cli_missing_wandb_is_one_line_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "wandb", None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "deqn-jax",
            "train",
            "brock_mirman",
            "-n",
            "3",
            "--hidden",
            "16",
            "--batch-size",
            "16",
            "-q",
            "--wandb",
            "p",
            "--checkpoint-dir",
            str(tmp_path / "ck"),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    err = capsys.readouterr().err.strip()
    assert len(err.splitlines()) == 1
    assert "wandb" in err and "Traceback" not in err
