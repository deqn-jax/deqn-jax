"""Field descriptions that enumerate a registry must list exactly its members.

The config descriptions are rendered into the public config reference
(``docs/site/config_reference.md``) and read by users choosing an option.
Each test below pins one enumeration to the code that defines the set, so a
registered or removed name fails here until the description follows.
"""

import inspect
import re

from deqn_jax.config import NetworkConfig, OptimizerConfig
from deqn_jax.networks import factory
from deqn_jax.networks.common import ACTIVATION_FNS, INIT_FNS
from deqn_jax.optimizers import list_optimizers


def _enumerated(description: str, marker: str) -> set:
    """Backticked names between ``marker`` and the end of that sentence."""
    head = description.split(marker, 1)[1]
    sentence = re.split(r"\.(?:\s|$)", head, maxsplit=1)[0]
    return set(re.findall(r"`([^`]+)`", sentence))


def _description(cls, field: str) -> str:
    return cls.model_fields[field].description


def test_optimizer_names_match_registry():
    listed = _enumerated(_description(OptimizerConfig, "name"), "Options:")
    assert listed == set(OptimizerConfig.VALID_NAMES) == set(list_optimizers())


def test_lr_schedules_match_validator():
    listed = _enumerated(_description(OptimizerConfig, "lr_schedule"), "LR schedule:")
    assert listed == set(OptimizerConfig.VALID_LR_SCHEDULES)


def test_network_types_match_factory():
    listed = _enumerated(_description(NetworkConfig, "type"), "Network architecture:")
    assert listed == set(NetworkConfig.VALID_TYPES)
    # ``mlp`` is the factory's fallback branch; every other type has its own.
    source = inspect.getsource(factory.build_policy_net)
    for net_type in NetworkConfig.VALID_TYPES - {"mlp"}:
        assert f'net_type == "{net_type}"' in source, net_type
    # The factory's module docstring tables the fields each type honors.
    for net_type in NetworkConfig.VALID_TYPES:
        assert f"``{net_type}``" in factory.__doc__, net_type


def test_activations_match_network_table():
    listed = _enumerated(
        _description(NetworkConfig, "activation"), "Per-layer activation:"
    )
    assert listed == set(NetworkConfig.VALID_ACTIVATIONS) == set(ACTIVATION_FNS)


def test_inits_match_network_table():
    listed = _enumerated(_description(NetworkConfig, "init"), "Weight init scheme:")
    assert listed == set(NetworkConfig.VALID_INITS) == set(INIT_FNS) | {"default"}
