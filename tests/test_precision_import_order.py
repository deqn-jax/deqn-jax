"""An fp64 run computes in float64 whatever was imported before x64 was switched on.

The suite switches x64 on in ``conftest.py`` before anything is imported, so
the import-order hazard cannot be reproduced in this process: every check runs
in a fresh interpreter started with x64 off. The cases that train or solve the
disaster steady state twice take ~20-30 s each and are marked ``slow``.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
BKPIN_CONFIG = REPO / "configs" / "disaster_gated_pcgrad_bkpin.yaml"

# Smoke size for the fp64 BK-pinned disaster arm; the warm-up must be shorter
# than the run for the cosine schedule to be legal.
SMOKE_OVERRIDES = {
    "episodes": 3,
    "batch_size": 16,
    "network.hidden_sizes": [16],
    "optimizer.lr_warmup": 1,
    "composite_loss.n_anchor_points": 16,
    "tensorboard_dir": None,
}


def _run_fresh(code: str, *args: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a new interpreter with x64 off at start-up."""
    env = {**os.environ, "JAX_ENABLE_X64": "0"}
    proc = subprocess.run(
        [sys.executable, *(["-c", code] if code else []), *args],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return proc


def _last_json(proc: subprocess.CompletedProcess):
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.slow
def test_api_imported_first_trains_fp64_bk_pin(tmp_path):
    code = f"""
import json
import deqn_jax.api  # loads every model package while x64 is still off
from deqn_jax.api import TrainConfig, train_from_config

cfg = TrainConfig.from_yaml({str(BKPIN_CONFIG)!r}).with_overrides(
    {{**{SMOKE_OVERRIDES!r}, "checkpoint_dir": {str(tmp_path / "ck")!r}}}
)
net, _ = train_from_config(cfg)
print(json.dumps({{k: str(getattr(net, k).dtype) for k in ("ss_state", "ss_policy", "P")}}))
"""
    dtypes = _last_json(_run_fresh(code))
    assert dtypes == {"ss_state": "float64", "ss_policy": "float64", "P": "float64"}


_STEADY_STATES = """
import json
import sys
import jax
{switch_before}
import deqn_jax.models
from deqn_jax.models import load_model
{switch_after}
risky = sys.modules["deqn_jax.models.disaster.steady_state"].risky_steady_state
model = load_model("disaster")
risky_constants = {{**model.constants, "p_disaster": 0.02}}
out = {{}}
for name, (s, p) in (
    ("deterministic", model.steady_state_fn(model.constants)),
    ("risky", risky(risky_constants)),
):
    out[name] = {{"dtypes": [str(s.dtype), str(p.dtype)],
                  "values": [s.tolist(), p.tolist()]}}
out["bounds"] = [str(model.policy_lower.dtype), str(model.policy_upper.dtype)]
print(json.dumps(out))
"""
_X64_ON = 'jax.config.update("jax_enable_x64", True)'


@pytest.mark.slow
def test_late_x64_switch_gives_the_fp64_steady_state():
    late = _last_json(
        _run_fresh(_STEADY_STATES.format(switch_before="", switch_after=_X64_ON))
    )
    early = _last_json(
        _run_fresh(_STEADY_STATES.format(switch_before=_X64_ON, switch_after=""))
    )
    assert late["bounds"] == ["float64", "float64"]
    for name in ("deterministic", "risky"):
        assert late[name]["dtypes"] == ["float64", "float64"], name
        for got, want in zip(late[name]["values"], early[name]["values"]):
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


@pytest.mark.slow
def test_cli_trains_fp64_bk_pin_config(tmp_path):
    raw = yaml.safe_load(BKPIN_CONFIG.read_text())
    raw.pop("tensorboard_dir")
    raw["batch_size"] = 16
    raw["network"]["hidden_sizes"] = [16]
    raw["optimizer"]["lr_warmup"] = 1
    raw["composite_loss"]["n_anchor_points"] = 16
    config = tmp_path / "fp64_bkpin.yaml"
    config.write_text(yaml.safe_dump(raw))
    ckpt = tmp_path / "ck"
    _run_fresh(
        "",
        "-m",
        "deqn_jax.cli",
        "train",
        "--config",
        str(config),
        "-n",
        "3",
        "-q",
        "--checkpoint-dir",
        str(ckpt),
    )
    assert (ckpt / "checkpoint_best.eqx").exists()
