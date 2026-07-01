"""Validation tests for CoverageConfig (EWM coverage sampling)."""

import pytest

from deqn_jax.config import CoverageConfig, TrainConfig


def test_defaults_disabled():
    c = CoverageConfig()
    assert c.enabled is False
    # paper's coverage-exact arm: path:stress:local = 1:0.5:0.25, H in {3,5}
    assert c.rho_base == 1.0 and c.rho_stress == 0.5 and c.rho_local == 0.25
    assert c.n_stress == 128 and c.n_local == 128
    assert c.rollout_horizon == 5
    assert c.local_sigma == 0.02
    assert c.stress_ranges == {}
    assert c.repair_ranges == {}


def test_bad_repair_range_rejected():
    with pytest.raises(Exception):
        CoverageConfig(repair_ranges={"k_0": (5.0, 0.2)})  # low > high


def test_extra_forbidden():
    with pytest.raises(Exception):
        CoverageConfig(notafield=1)


def test_negative_weight_rejected():
    with pytest.raises(Exception):
        CoverageConfig(rho_stress=-1.0)


def test_all_zero_weights_rejected():
    with pytest.raises(Exception):
        CoverageConfig(rho_base=0.0, rho_stress=0.0, rho_local=0.0)


def test_enabled_empty_stress_ranges_rejected():
    # rho_stress>0 but no stress box → cannot build the stress pool
    with pytest.raises(Exception):
        CoverageConfig(enabled=True, rho_stress=1.0, stress_ranges={})


def test_enabled_empty_pool_with_weight_rejected():
    # weighted pool with zero seeds → would NaN at jnp.mean over 0 rows
    with pytest.raises(Exception):
        CoverageConfig(
            enabled=True,
            rho_stress=1.0,
            n_stress=0,
            stress_ranges={"z_0": (-0.5, -0.2)},
        )


def test_bad_range_rejected():
    with pytest.raises(Exception):
        CoverageConfig(
            enabled=True,
            stress_ranges={"z_0": (0.2, -0.5)},  # low > high
        )


def test_valid_enabled_config():
    c = CoverageConfig(
        enabled=True,
        stress_ranges={"z_0": (-0.5, -0.2), "k_0": (1.05, 1.20)},
    )
    assert c.enabled is True
    assert c.stress_ranges["z_0"] == (-0.5, -0.2)


def test_trainconfig_has_coverage_default():
    cfg = TrainConfig()
    assert isinstance(cfg.coverage, CoverageConfig)
    assert cfg.coverage.enabled is False


def test_trainconfig_from_dict_coverage():
    cfg = TrainConfig.from_dict(
        {
            "model": "irbc",
            "coverage": {"enabled": True, "stress_ranges": {"z_0": [-0.5, -0.2]}},
        }
    )
    assert cfg.coverage.enabled is True
    assert cfg.coverage.stress_ranges["z_0"] == (-0.5, -0.2)


def test_trainconfig_from_dict_unknown_coverage_key():
    with pytest.raises(Exception):
        TrainConfig.from_dict({"coverage": {"nope": 1}})
