"""End-to-end smoke: irbc + Gauss-Hermite quadrature + coverage runs."""

import numpy as np

from deqn_jax.config import TrainConfig
from deqn_jax.training.trainer import train_from_config


def test_irbc_ewm_smoke_runs():
    cfg = TrainConfig.from_yaml("configs/irbc_ewm.yaml").with_overrides(
        {"episodes": 20, "log_every": 1, "verbose": False}
    )
    _params, history = train_from_config(cfg)
    losses = np.asarray(history["loss"])
    assert len(losses) >= 5
    assert np.all(np.isfinite(losses)), f"non-finite losses: {losses}"


def test_irbc_plain_smoke_runs():
    cfg = TrainConfig.from_yaml("configs/irbc_plain.yaml").with_overrides(
        {"episodes": 20, "log_every": 1, "verbose": False}
    )
    _params, history = train_from_config(cfg)
    losses = np.asarray(history["loss"])
    assert np.all(np.isfinite(losses))
