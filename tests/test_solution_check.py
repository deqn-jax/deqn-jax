"""The solution check passes a known exact solution and fails a slightly wrong one."""

import shutil
import sys
from pathlib import Path

import pytest

from deqn_jax.evaluate.solution import check_solution
from deqn_jax.models.folder import load_exact_policy, load_model_file

# Brock-Mirman with log utility and full depreciation: savings rate alpha * beta exactly.
FOLDER = Path(__file__).parent / "data" / "model_folder"


@pytest.fixture(scope="module")
def bm():
    model, _ = load_model_file(FOLDER, "model_folder_test")
    return model, load_exact_policy(FOLDER)


def test_exact_solution_is_solved(bm):
    model, exact = bm
    check = check_solution(lambda s: exact(s, model.constants), model, exact)
    assert check.solved
    # The expectation is taken by quadrature, so the exact policy leaves rounding only.
    assert check.errors.mean() < 1e-6


def test_two_percent_off_is_not_solved(bm):
    model, exact = bm
    check = check_solution(lambda s: 1.02 * exact(s, model.constants), model, exact)
    failed = {g.name for g in check.gates if g.passed is False}
    assert not check.solved
    assert failed == {
        "Average error, euler_k",
        "Worst 1% error, euler_k",
        "Matches the exact solution",
    }


@pytest.mark.slow
def test_model_folder_trains_and_verifies(tmp_path, monkeypatch):
    """`deqn-jax train <folder>` then `deqn-jax verify <folder>` solves the classic."""
    from deqn_jax.cli import main

    folder = tmp_path / "bm"
    shutil.copytree(FOLDER, folder)
    monkeypatch.setattr(sys, "argv", ["deqn-jax", "train", str(folder), "-q"])
    main()
    monkeypatch.setattr(sys, "argv", ["deqn-jax", "verify", str(folder)])
    with pytest.raises(SystemExit) as exit_:
        main()
    run = folder / "runs" / "01"
    assert exit_.value.code == 0
    assert (run / "model.py").read_text() == (FOLDER / "model.py").read_text()
    assert (run / "report.html").exists()
