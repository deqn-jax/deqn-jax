"""The disaster network resolves every state index and constant it uses.

A shape prior that reads a state or a calibration constant the model does
not provide must fail naming what is missing. It used to fall back
silently: state index 5 for ``R_lag``, and defaults such as ``pi_ss=1.0``
or ``iota=0.0`` for constants absent from ``model.constants``.
"""

import glob
import json
import os

import jax
import jax.numpy as jnp
import pytest
import yaml

from deqn_jax.config import NetworkConfig
from deqn_jax.models import load_model
from deqn_jax.models.disaster.network import (
    DisasterPolicyNet,
    create_disaster_policy_net,
)
from deqn_jax.networks.factory import build_policy_net

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ZLB = {"use_zlb_feature": True}
QM = {"reparam_q_as_m": True}
PI = {"reparam_pi_as_kp_inner": True}
WT = {"reparam_wtilda_as_kw_inner": True}

CONSTANTS_NEEDED = [
    (ZLB, "R_lb"),
    (QM, "kappa"),
    (QM, "mu_z_ss"),
    (PI, "xi_p"),
    (PI, "lambda_f"),
    (PI, "iota"),
    (PI, "pi_ss"),
    (WT, "xi_w"),
    (WT, "lambda_w"),
    (WT, "sigma_L"),
    (WT, "iota_w"),
    (WT, "psi_L"),
    (WT, "pi_ss"),
    (WT, "mu_z_ss"),
]


@pytest.fixture(scope="module")
def model():
    return load_model("disaster")


def _build(model, **flags):
    return create_disaster_policy_net(
        model, hidden_sizes=(8,), key=jax.random.PRNGKey(0), **flags
    )


@pytest.mark.parametrize(
    "flags, constant",
    CONSTANTS_NEEDED,
    ids=[f"{next(iter(f))}-{c}" for f, c in CONSTANTS_NEEDED],
)
def test_missing_constant_is_named(model, flags, constant):
    constants = {k: v for k, v in model.constants.items() if k != constant}
    with pytest.raises(ValueError, match=constant):
        _build(model._replace(constants=constants), **flags)


def test_zlb_feature_without_r_lag_state_is_named(model):
    names = tuple("R_prev" if n == "R_lag" else n for n in model.state_names)
    with pytest.raises(ValueError, match="R_lag"):
        _build(model._replace(state_names=names), **ZLB)


def _direct(**kw):
    n_s, n_p = 13, 11
    return DisasterPolicyNet(
        n_states=n_s,
        n_policies=n_p,
        hidden_sizes=(8,),
        activation="tanh",
        P=jnp.zeros((n_p, n_s)),
        ss_state=jnp.ones(n_s),
        ss_policy=jnp.ones(n_p),
        key=jax.random.PRNGKey(0),
        **kw,
    )


@pytest.mark.parametrize(
    "kw, missing",
    [
        (dict(use_zlb_feature=True, r_lb=1.0), "r_lag_idx"),
        (dict(use_zlb_feature=True, r_lag_idx=5), "r_lb"),
    ],
)
def test_direct_construction_requires_the_zlb_inputs(kw, missing):
    with pytest.raises(ValueError, match=missing):
        _direct(**kw)


def _network_blocks():
    """Shipped disaster configs grouped by (model, network block): the net
    built depends on nothing else."""
    blocks = {}
    for path in sorted(glob.glob(os.path.join(REPO, "configs", "disaster*.yaml"))):
        cfg = yaml.safe_load(open(path))
        key = json.dumps([cfg["model"], cfg.get("network") or {}], sort_keys=True)
        blocks.setdefault(key, []).append(os.path.basename(path))
    return [(json.loads(k), v) for k, v in blocks.items()]


BLOCKS = _network_blocks()


@pytest.mark.parametrize("block, configs", BLOCKS, ids=[c[0] for _, c in BLOCKS])
def test_shipped_disaster_configs_construct(block, configs):
    model_name, network = block
    net_cfg = NetworkConfig(**{**network, "hidden_sizes": [8]})
    net = build_policy_net(load_model(model_name), jax.random.PRNGKey(0), (8,), net_cfg)
    assert isinstance(net, DisasterPolicyNet), configs
