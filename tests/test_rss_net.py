"""RSS Phase-0 network architecture and checkpoint-transfer tests."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from deqn_jax.config import NetworkConfig
from deqn_jax.networks.factory import build_policy_net
from deqn_jax.networks.rss_net import (
    BASE_HIDDEN_SIZES,
    TF_WEIGHT_NAME_MAP,
    RSSMarketClearingNet,
    load_tf_weight_dict,
)
from deqn_jax.types import ModelSpec

STATE_NAMES = ("homo", "tau_12")
POLICY_NAMES = (
    "P_C_1_y",
    "K_1_y",
    "A_1_y",
    "P_C_2_y",
    "K_2_y",
    "A_2_y",
)
LABOR = (0.2, 0.8)


def _synthetic_weights(network, seed=17, dtype=np.float64):
    rng = np.random.default_rng(seed)
    groups = (
        ("base_nn/dense", network.base_layers[0]),
        ("base_nn/dense_1", network.base_layers[1]),
        ("base_nn/dense_2", network.base_layers[2]),
        ("ansatz_mlp/ansatz_hidden", network.ansatz_layers[0]),
        ("ansatz_mlp/ansatz_output", network.ansatz_layers[1]),
    )
    weights = {}
    for name, layer in groups:
        kernel_shape = tuple(reversed(layer.weight.shape))
        weights[f"{name}/kernel:0"] = rng.normal(scale=0.2, size=kernel_shape).astype(
            dtype
        )
        weights[f"{name}/bias:0"] = rng.normal(scale=0.1, size=layer.bias.shape).astype(
            dtype
        )
    return weights


def _numpy_reference(inputs, weights):
    weights = {name.removesuffix(":0"): value for name, value in weights.items()}

    def dense(value, name):
        return value @ weights[f"{name}/kernel"] + weights[f"{name}/bias"]

    base = 1.0 / (1.0 + np.exp(-dense(inputs, "base_nn/dense")))
    base = 1.0 / (1.0 + np.exp(-dense(base, "base_nn/dense_1")))
    base = dense(base, "base_nn/dense_2")
    ansatz = np.tanh(dense(inputs, "ansatz_mlp/ansatz_hidden"))
    ansatz = dense(ansatz, "ansatz_mlp/ansatz_output")
    output = ansatz + 3.0 * np.tanh(base)

    labor = np.asarray(LABOR)
    labor_mean = labor.mean()
    dynamic_labor = inputs[:, :1] * labor[None, :] + (1.0 - inputs[:, :1]) * labor_mean
    a_indices = np.array([2, 5])
    a_values = output[:, a_indices]
    projection = np.sum(a_values * dynamic_labor, axis=1, keepdims=True) / (
        np.sum(dynamic_labor**2, axis=1, keepdims=True) + 1e-15
    )
    output[:, a_indices] = 0.01 * (a_values - projection * dynamic_labor)
    output[:, [1, 4]] = 0.1 * np.logaddexp(0.0, output[:, [1, 4]])
    output[:, [0, 3]] = np.logaddexp(0.0, output[:, [0, 3]])
    return output


def _dummy_model():
    def equations(state, policy, next_state, next_policy, constants):
        del policy, next_state, next_policy, constants
        return {"zero": jnp.zeros(state.shape[0])}

    def step(state, policy, shock, constants):
        del policy, shock, constants
        return state

    return ModelSpec(
        name="rss_net_test",
        n_states=len(STATE_NAMES),
        n_policies=len(POLICY_NAMES),
        n_shocks=1,
        state_names=STATE_NAMES,
        policy_names=POLICY_NAMES,
        equation_names=("zero",),
        constants={"L": LABOR},
        equations_fn=equations,
        step_fn=step,
    )


def test_synthetic_checkpoint_matches_independent_numpy_forward():
    network = RSSMarketClearingNet(
        n_states=len(STATE_NAMES),
        policy_names=POLICY_NAMES,
        state_names=STATE_NAMES,
        labor_endowments=LABOR,
        base_hidden_sizes=(3, 4),
        ansatz_hidden_size=3,
        key=jax.random.PRNGKey(0),
    )
    network = jax.tree.map(lambda value: value.astype(jnp.float32), network)
    weights = _synthetic_weights(network, dtype=np.float32)
    assert set(name.removesuffix(":0") for name in weights) == set(TF_WEIGHT_NAME_MAP)
    assert TF_WEIGHT_NAME_MAP["base_nn/dense/kernel"] == (
        "base_layers",
        0,
        "weight",
    )
    network = load_tf_weight_dict(network, weights)

    inputs = np.array([[1.0, -0.2], [0.25, 0.4], [0.8, 1.1]], dtype=np.float32)
    expected = _numpy_reference(inputs.copy(), weights)
    actual = np.asarray(network(jnp.asarray(inputs)))
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(
        np.asarray(network(jnp.asarray(inputs[0]))),
        expected[0],
        rtol=3e-6,
        atol=3e-6,
    )

    dynamic_labor = inputs[:, :1] * np.asarray(LABOR)[None, :] + (
        1.0 - inputs[:, :1]
    ) * np.mean(LABOR)
    np.testing.assert_allclose(
        np.sum(actual[:, [2, 5]] * dynamic_labor, axis=1),
        0.0,
        atol=1e-7,
    )


def test_checkpoint_loader_rejects_missing_or_wrong_shape():
    network = RSSMarketClearingNet(
        n_states=len(STATE_NAMES),
        policy_names=POLICY_NAMES,
        state_names=STATE_NAMES,
        labor_endowments=LABOR,
        base_hidden_sizes=(3, 4),
        ansatz_hidden_size=3,
        key=jax.random.PRNGKey(0),
    )
    weights = _synthetic_weights(network)
    weights.pop("base_nn/dense/kernel:0")
    with pytest.raises(ValueError, match="missing RSS network weights"):
        load_tf_weight_dict(network, weights)

    weights = _synthetic_weights(network)
    weights["base_nn/dense/kernel:0"] = np.zeros((99, 99))
    with pytest.raises(ValueError, match="has shape"):
        load_tf_weight_dict(network, weights)


def test_factory_registers_fixed_checkpoint_architecture():
    config = NetworkConfig(
        type="rss_market_clearing_net", hidden_sizes=BASE_HIDDEN_SIZES
    )
    network = build_policy_net(_dummy_model(), jax.random.PRNGKey(0), (8,), config)

    assert isinstance(network, RSSMarketClearingNet)
    assert network.base_layers[0].weight.shape == (512, len(STATE_NAMES))
    assert network.base_layers[1].weight.shape == (512, 512)
    assert network.base_layers[2].weight.shape == (len(POLICY_NAMES), 512)
    assert network.ansatz_layers[0].weight.shape == (8, len(STATE_NAMES))


def test_factory_rejects_nonreference_widths():
    config = NetworkConfig(type="rss_market_clearing_net", hidden_sizes=(8, 8))
    with pytest.raises(ValueError, match="requires hidden_sizes"):
        build_policy_net(_dummy_model(), jax.random.PRNGKey(0), (8,), config)
