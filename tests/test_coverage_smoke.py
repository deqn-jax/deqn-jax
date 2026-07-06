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


def test_irbc_composite_coverage_composition_smoke_and_identity():
    """composite∘coverage (2026-07-06): three guards in one training trio.

    (a) the composition arm trains finite;
    (b) zero-weight pools collapse composite+coverage to EXACTLY the plain
        composite path (build-time pool skip; loss keys unused under
        quadrature, so the extra key split is inert);
    (c) live pools actually move the trajectory (coverage feeds gradients
        through the composite base term).
    """
    over = {"episodes": 12, "log_every": 1, "verbose": False}

    cfg_composite = TrainConfig.from_yaml("configs/irbc.yaml").with_overrides(over)
    _p, h_composite = train_from_config(cfg_composite)

    cfg_full = TrainConfig.from_yaml("configs/irbc_ewm_anchor.yaml").with_overrides(
        over
    )
    # coverage.* subkeys aren't flat-overridable (dict-valued fields);
    # model_copy the nested config for the zero-pool arm instead.
    cfg_zero = cfg_full.model_copy(
        update={
            "coverage": cfg_full.coverage.model_copy(
                update={"rho_stress": 0.0, "rho_local": 0.0}
            )
        }
    )
    _p, h_zero = train_from_config(cfg_zero)

    _p, h_full = train_from_config(cfg_full)

    a = np.asarray(h_composite["loss"])
    z = np.asarray(h_zero["loss"])
    f = np.asarray(h_full["loss"])
    assert np.all(np.isfinite(f)), f"non-finite composition losses: {f}"
    np.testing.assert_array_equal(z, a)  # (b) EXACT identity
    assert np.abs(f - a).max() > 1e-8  # (c) pools feed gradients
