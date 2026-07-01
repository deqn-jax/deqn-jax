"""Validator gates from the 2026-07 audit (batch B).

Silent-drop combos now rejected: pcgrad with non-STANDARD optimizers,
top-level barrier_weight under composite, grad_clip on LBFGS/GN paths,
GN-family on two-stage models. And the reverse direction: LBFGS was
wrongly blocked from composite/huber/weights — it differentiates
`compute_loss_fn or compute_loss` with weights and runs reweighting, so
those combos now validate.
"""

import pytest

from deqn_jax.config import TrainConfig
from deqn_jax.training.state_init import (
    _resolve_model_for_training,
    _validate_train_config,
)


def _cfg(**over):
    base = {
        "model": "brock_mirman",
        "verbose": False,
        "network": {"type": "mlp", "hidden_sizes": [8]},
    }
    base.update(over)
    return TrainConfig.from_dict(base)


def test_pcgrad_with_mao_rejected_unconditionally():
    # Previously passed validation with an otherwise-default config and
    # silently trained plain MAO.
    cfg = _cfg(optimizer={"name": "mao"}, gradient_surgery="pcgrad")
    with pytest.raises(ValueError, match="pcgrad"):
        _validate_train_config(cfg)


def test_pcgrad_with_lbfgs_rejected():
    cfg = _cfg(optimizer={"name": "lbfgs"}, gradient_surgery="pcgrad")
    with pytest.raises(ValueError, match="pcgrad"):
        _validate_train_config(cfg)


def test_barrier_with_composite_rejected():
    # Top-level barrier_weight is never forwarded by the composite loss.
    cfg = _cfg(loss_type="composite", barrier_weight=0.05)
    with pytest.raises(ValueError, match="barrier_weight"):
        _validate_train_config(cfg)


def test_grad_clip_with_lbfgs_rejected():
    cfg = _cfg(optimizer={"name": "lbfgs", "grad_clip": 1.0})
    with pytest.raises(ValueError, match="grad_clip"):
        _validate_train_config(cfg)


def test_grad_clip_with_gn_rejected():
    cfg = _cfg(optimizer={"name": "gn", "grad_clip": 1.0})
    with pytest.raises(ValueError, match="grad_clip"):
        _validate_train_config(cfg)


def test_grad_clip_with_adam_still_fine():
    cfg = _cfg(optimizer={"name": "adam", "grad_clip": 1.0})
    _validate_train_config(cfg)  # must not raise


def test_lbfgs_with_composite_now_validates():
    # LBFGS differentiates the custom loss fn (compute_loss_fn or
    # compute_loss) — composite genuinely reaches its gradient/line search.
    cfg = _cfg(loss_type="composite", optimizer={"name": "lbfgs"})
    _validate_train_config(cfg)  # must not raise


def test_lbfgs_with_huber_now_validates():
    cfg = _cfg(loss_choice="huber", optimizer={"name": "lbfgs"})
    _validate_train_config(cfg)  # must not raise


def test_gn_on_two_stage_model_rejected():
    # olg_lifecycle sets combine_fn (fb wrapping an expectation); the GN
    # residual vector would optimize the biased E[fb] objective.
    cfg = _cfg(model="olg_lifecycle", optimizer={"name": "gn"})
    with pytest.raises(ValueError, match="two-stage"):
        _resolve_model_for_training(cfg)


def test_lm_on_two_stage_model_rejected():
    cfg = _cfg(model="olg_lifecycle", optimizer={"name": "lm"})
    with pytest.raises(ValueError, match="two-stage"):
        _resolve_model_for_training(cfg)


def test_gn_on_standard_model_still_fine():
    cfg = _cfg(model="brock_mirman", optimizer={"name": "gn"})
    model, n_eq = _resolve_model_for_training(cfg)
    assert n_eq >= 1
