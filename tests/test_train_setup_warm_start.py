"""A requested warm start that cannot run is never skipped silently.

Sequence networks (``lstm`` / ``transformer``) have only the constant-SS
warm start: a requested linearized or Dynare target is rejected at setup
instead of being replaced by the constant-SS fit, and a model without a
steady state gets the same notice as on the non-sequence path. The skip on
BK-anchored networks (graph @aleph/deqn, node #40) stays and is announced.
"""

import jax
import pytest

from deqn_jax.config import TrainConfig
from deqn_jax.training.state_init import (
    _build_initial_state,
    _resolve_model_for_training,
)


def _cfg(net: dict, **over) -> TrainConfig:
    base = {
        "model": "brock_mirman",
        "episodes": 3,
        "batch_size": 16,
        "seed": 0,
        "verbose": True,
        "warm_start": True,
        "network": {"hidden_sizes": [8], **net},
    }
    base.update(over)
    return TrainConfig.from_dict(base)


LSTM = {"type": "lstm", "history_len": 2}


def _build(cfg, model=None):
    resolved, n_eq = _resolve_model_for_training(cfg)
    model = resolved if model is None else model
    return _build_initial_state(cfg, model, jax.random.PRNGKey(0), n_eq, cfg.optimizer)


@pytest.mark.parametrize(
    "over",
    [{"warm_start_linearize": True}, {"warm_start_dynare": "dynare_out"}],
    ids=["linearize", "dynare"],
)
def test_sequence_net_rejects_warm_start_target_it_cannot_fit(over):
    cfg = _cfg(LSTM, **over)
    with pytest.raises(ValueError, match=next(iter(over))):
        _build(cfg)


def test_sequence_net_without_steady_state_announces_skip(capsys):
    cfg = _cfg(LSTM)
    model, _ = _resolve_model_for_training(cfg)
    _build(cfg, model._replace(steady_state_fn=None))
    assert "skipping warm start" in capsys.readouterr().out


def test_mlp_without_steady_state_announces_skip(capsys):
    cfg = _cfg({"type": "mlp"})
    model, _ = _resolve_model_for_training(cfg)
    _build(cfg, model._replace(steady_state_fn=None))
    assert "skipping warm start" in capsys.readouterr().out


def test_anchored_net_skip_is_announced(capsys):
    cfg = _cfg({"type": "linear_plus_mlp", "init_scale": 0.0})
    _build(cfg)
    assert "Warm start skipped" in capsys.readouterr().out
