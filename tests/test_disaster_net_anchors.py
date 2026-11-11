"""The disaster network's BK anchors are fixed data, not parameters.

``P``, ``ss_state`` and ``ss_policy`` define the BK pin's promise:
``pi(s*) = pi*`` and ``dpi/ds(s*) = P`` for every parameter value. Held as
array leaves, they were handed to the optimizer with the weights; decoupled
weight decay (``adamw``, ``lion``) shrinks every leaf it is handed, gradient
or not, and moved them. They live outside the parameter tree instead.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import yaml

from deqn_jax.config import NetworkConfig, OptimizerConfig
from deqn_jax.models import load_model
from deqn_jax.models.disaster.network import create_disaster_policy_net
from deqn_jax.optimizers.registry import create_optimizer
from deqn_jax.training.checkpointing import load_policy_from_checkpoint
from deqn_jax.training.trainer import create_train_state

ANCHORS = ("P", "ss_state", "ss_policy")
LR = 1e-2
WD = 0.1


@pytest.fixture(scope="module")
def pinned():
    model = load_model("disaster")
    net = create_disaster_policy_net(
        model,
        hidden_sizes=(16,),
        init_scale=0.5,
        bk_pin=True,
        key=jax.random.PRNGKey(0),
    )
    return model, net


def _steps(net, opt_name, loss, n=3):
    opt, _ = create_optimizer(
        OptimizerConfig(name=opt_name, learning_rate=LR, weight_decay=WD)
    )
    opt_state = opt.init(eqx.filter(net, eqx.is_array))
    for _ in range(n):
        grads = eqx.filter(eqx.filter_grad(loss)(net), eqx.is_array)
        params = eqx.filter(net, eqx.is_array)
        updates, opt_state = opt.update(grads, opt_state, params)
        net = eqx.combine(optax.apply_updates(params, updates), net)
    return net


def test_parameter_leaves_are_the_mlp_alone(pinned):
    _, net = pinned
    assert len(jax.tree.leaves(eqx.filter(net, eqx.is_array))) == len(
        jax.tree.leaves(eqx.filter(net.mlp, eqx.is_array))
    )


@pytest.mark.parametrize("opt_name", ["adamw", "lion"])
def test_anchors_and_pin_survive_decaying_optimizer(pinned, opt_name):
    _, net = pinned
    s_off = net.ss_state[None, :] * 1.03
    trained = _steps(net, opt_name, lambda n: jnp.sum(n(s_off) ** 2))

    for name in ANCHORS:
        np.testing.assert_array_equal(
            np.asarray(getattr(trained, name)), np.asarray(getattr(net, name)), name
        )
    ss = net.ss_state
    np.testing.assert_allclose(
        np.asarray(trained(ss[None, :])[0]), np.asarray(net(ss[None, :])[0]), rtol=1e-10
    )
    J_trained = jax.jacobian(lambda s: trained(s[None, :])[0])(ss)
    J_net = jax.jacobian(lambda s: net(s[None, :])[0])(ss)
    np.testing.assert_allclose(np.asarray(J_trained), np.asarray(J_net), atol=1e-9)


# ``lion`` decays at optax's default rate; ``config.weight_decay`` is not passed.
@pytest.mark.parametrize("opt_name, rate", [("adamw", WD), ("lion", 1e-3)])
def test_weight_decay_still_shrinks_the_mlp(pinned, opt_name, rate):
    # Zero gradient: the update is pure decay, w <- w (1 - lr rate) per step.
    _, net = pinned
    trained = _steps(net, opt_name, lambda n: 0.0 * jnp.sum(n(n.ss_state[None, :])))
    before = jax.tree.leaves(eqx.filter(net.mlp, eqx.is_array))
    after = jax.tree.leaves(eqx.filter(trained.mlp, eqx.is_array))
    for b, a in zip(before, after):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b) * (1 - LR * rate) ** 3)


def test_checkpoint_round_trip(tmp_path):
    model = load_model("disaster")
    opt_cfg = OptimizerConfig(name="adamw", learning_rate=LR, weight_decay=WD)
    net_cfg = NetworkConfig(
        type="disaster_policy_net", hidden_sizes=[16], options={"bk_pin": True}
    )
    state, _, _ = create_train_state(
        model,
        jax.random.PRNGKey(0),
        hidden_sizes=(16,),
        batch_size=16,
        n_equations=len(model.equation_names),
        optimizer_config=opt_cfg,
        network_config=net_cfg,
    )
    # Move the weights off their init so a template-only load would show.
    mlp = jax.tree.map(lambda a: a + 0.1, eqx.filter(state.params.mlp, eqx.is_array))
    net = eqx.tree_at(lambda n: n.mlp, state.params, eqx.combine(mlp, state.params.mlp))
    ckpt = tmp_path / "checkpoint_000003.eqx"
    eqx.tree_serialise_leaves(str(ckpt), state._replace(params=net))
    cfg = {
        "model": "disaster",
        "batch_size": 16,
        "optimizer": opt_cfg.model_dump(),
        "network": net_cfg.model_dump(),
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))

    loaded, _ = load_policy_from_checkpoint(str(ckpt))
    states = net.ss_state[None, :] * jnp.linspace(0.97, 1.03, 5)[:, None]
    np.testing.assert_array_equal(np.asarray(loaded(states)), np.asarray(net(states)))
    for name in ANCHORS:
        np.testing.assert_array_equal(
            np.asarray(getattr(loaded, name)), np.asarray(getattr(net, name)), name
        )
