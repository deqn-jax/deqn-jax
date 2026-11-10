"""create_train_state builds the same optimizer and state from the legacy
keyword path and from an OptimizerConfig, for every registered optimizer."""

import jax
import numpy as np
import pytest

from deqn_jax.config import OptimizerConfig
from deqn_jax.models import load_model
from deqn_jax.optimizers.registry import create_optimizer, list_optimizers
from deqn_jax.training.state_init import create_train_state


@pytest.mark.parametrize("name", list_optimizers())
def test_legacy_and_config_paths_agree(name):
    model = load_model("bm_labor")
    common = dict(hidden_sizes=(8,), batch_size=16, n_equations=2)
    key = jax.random.PRNGKey(0)

    s_legacy, _, k_legacy = create_train_state(
        model, key, optimizer=name, learning_rate=1e-3, **common
    )
    cfg = OptimizerConfig(name=name, learning_rate=1e-3)
    s_cfg, _, k_cfg = create_train_state(model, key, optimizer_config=cfg, **common)

    assert k_legacy == k_cfg == create_optimizer(cfg)[1]
    leaves_l, tree_l = jax.tree.flatten(s_legacy.opt_state)
    leaves_c, tree_c = jax.tree.flatten(s_cfg.opt_state)
    assert tree_l == tree_c
    for a, b in zip(leaves_l, leaves_c, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
