"""Phase-0 policy network for the RSS three-country trade model.

The architecture is kept separate from the economic model because its purpose
is checkpoint parity: an eight-unit tanh ansatz is added to a bounded correction
from a two-layer sigmoid MLP.  The final policy transform also enforces the
world-bond clearing projection used by the reference solution of Ravikumar,
Santacreu, and Sposi (2019).

Checkpoint kernels use the dense-layer convention ``[in, out]`` while Equinox
stores linear weights as ``[out, in]``.  ``load_tf_weight_dict`` is the single,
explicit conversion boundary between the two layouts.
"""

import math
import re
from collections.abc import Mapping, Sequence
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from deqn_jax.types import ModelSpec

BASE_HIDDEN_SIZES = (512, 512)
ANSATZ_HIDDEN_SIZE = 8
EPS_CORR = 3.0
K_MASK = 0.1
A_MASK = 0.01

# Canonical names written by the checkpoint exporter.  Keeping this map
# explicit prevents a layer-order coincidence from silently loading a kernel
# into the wrong part of the composite network.
TF_LAYER_NAME_MAP = {
    "base_nn/dense": ("base_layers", 0),
    "base_nn/dense_1": ("base_layers", 1),
    "base_nn/dense_2": ("base_layers", 2),
    "ansatz_mlp/ansatz_hidden": ("ansatz_layers", 0),
    "ansatz_mlp/ansatz_output": ("ansatz_layers", 1),
}
TF_WEIGHT_NAME_MAP = {
    f"{source_name}/{source_parameter}": (*destination, target_parameter)
    for source_name, destination in TF_LAYER_NAME_MAP.items()
    for source_parameter, target_parameter in (("kernel", "weight"), ("bias", "bias"))
}

_COUNTRY_POLICY = re.compile(r"^(?P<stem>[AK])_(?P<country>[1-9][0-9]*)(?:_y)?$")


def _glorot_linear(in_features: int, out_features: int, key: Array) -> eqx.nn.Linear:
    """Construct a zero-bias Glorot-uniform linear layer."""
    layer = eqx.nn.Linear(in_features, out_features, key=key)
    limit = math.sqrt(6.0 / (in_features + out_features))
    weight = jax.random.uniform(key, layer.weight.shape, minval=-limit, maxval=limit)
    return eqx.tree_at(
        lambda item: (item.weight, item.bias),
        layer,
        (weight, jnp.zeros_like(layer.bias)),
    )


def _zero_linear(in_features: int, out_features: int, key: Array) -> eqx.nn.Linear:
    """Construct a linear layer whose output is identically zero."""
    layer = eqx.nn.Linear(in_features, out_features, key=key)
    return eqx.tree_at(
        lambda item: (item.weight, item.bias),
        layer,
        (jnp.zeros_like(layer.weight), jnp.zeros_like(layer.bias)),
    )


def _country_indices(policy_names: Sequence[str], stem: str) -> tuple[int, ...]:
    """Find country-level ``A_i`` or ``K_i`` policies in country order."""
    found = []
    for index, name in enumerate(policy_names):
        match = _COUNTRY_POLICY.fullmatch(name)
        if match is not None and match.group("stem") == stem:
            found.append((int(match.group("country")), index))
    return tuple(index for _, index in sorted(found))


def _labor_tuple(labor) -> tuple[float, ...]:
    if isinstance(labor, Mapping):
        return tuple(float(labor[key]) for key in sorted(labor))
    values = np.asarray(labor).reshape(-1)
    return tuple(float(value) for value in values)


class RSSMarketClearingNet(eqx.Module):
    """Ansatz-plus-MLP policy with exact bond-market clearing.

    The raw composite output is

    ``ansatz(state) + 3 * tanh(base(state))``.

    Country-level next-capital outputs receive ``0.1 * softplus``.  Raw bond
    positions are projected onto ``sum_i A_i L_i = 0`` and scaled by ``0.01``.
    Every other Phase-0 policy receives the reference softplus activation.
    Reference hard policy clips live in the model accessor layer, not this net.
    """

    base_layers: tuple[eqx.nn.Linear, ...]
    ansatz_layers: tuple[eqx.nn.Linear, ...]
    a_indices: tuple[int, ...] = eqx.field(static=True)
    k_indices: tuple[int, ...] = eqx.field(static=True)
    softplus_indices: tuple[int, ...] = eqx.field(static=True)
    homo_index: Optional[int] = eqx.field(static=True)
    labor_endowments: tuple[float, ...] = eqx.field(static=True)
    eps_corr: float = eqx.field(static=True)
    k_mask: float = eqx.field(static=True)
    a_mask: float = eqx.field(static=True)
    # Checkpoint parity uses raw state inputs, but shared interp/viz tooling
    # expects every policy module to expose normalization metadata.
    input_shift: Optional[tuple] = eqx.field(static=True)
    input_scale: Optional[tuple] = eqx.field(static=True)

    def __init__(
        self,
        n_states: int,
        policy_names: Sequence[str],
        state_names: Sequence[str],
        labor_endowments: Sequence[float] | Mapping,
        *,
        key: Array,
        base_hidden_sizes: Sequence[int] = BASE_HIDDEN_SIZES,
        ansatz_hidden_size: int = ANSATZ_HIDDEN_SIZE,
    ):
        if len(base_hidden_sizes) != 2:
            raise ValueError(
                "rss_market_clearing_net requires exactly two base hidden layers"
            )
        if not policy_names:
            raise ValueError("rss_market_clearing_net requires model.policy_names")

        n_policies = len(policy_names)
        h0, h1 = (int(size) for size in base_hidden_sizes)
        keys = jax.random.split(key, 5)
        self.base_layers = (
            _glorot_linear(n_states, h0, keys[0]),
            _glorot_linear(h0, h1, keys[1]),
            _zero_linear(h1, n_policies, keys[2]),
        )
        self.ansatz_layers = (
            _glorot_linear(n_states, ansatz_hidden_size, keys[3]),
            _zero_linear(ansatz_hidden_size, n_policies, keys[4]),
        )

        self.a_indices = _country_indices(policy_names, "A")
        self.k_indices = _country_indices(policy_names, "K")
        if not self.a_indices or not self.k_indices:
            raise ValueError(
                "rss_market_clearing_net requires country policies A_i and K_i"
            )

        labor = _labor_tuple(labor_endowments)
        if len(labor) != len(self.a_indices):
            raise ValueError(
                f"labor endowments ({len(labor)}) must match bond policies "
                f"({len(self.a_indices)})"
            )
        if len(self.k_indices) != len(self.a_indices):
            raise ValueError("country K_i and A_i policy counts must match")

        special = set(self.a_indices) | set(self.k_indices)
        self.softplus_indices = tuple(
            index for index in range(n_policies) if index not in special
        )
        self.homo_index = next(
            (
                index
                for index, name in enumerate(state_names)
                if name.startswith("homo")
            ),
            None,
        )
        self.labor_endowments = labor
        self.eps_corr = EPS_CORR
        self.k_mask = K_MASK
        self.a_mask = A_MASK
        self.input_shift = None
        self.input_scale = None

    def _forward_single(self, state: Array) -> Array:
        base = jax.nn.sigmoid(self.base_layers[0](state))
        base = jax.nn.sigmoid(self.base_layers[1](base))
        base = self.base_layers[2](base)

        ansatz = jax.nn.tanh(self.ansatz_layers[0](state))
        ansatz = self.ansatz_layers[1](ansatz)
        output = ansatz + self.eps_corr * jnp.tanh(base)

        labor = jnp.asarray(self.labor_endowments, dtype=output.dtype)
        if self.homo_index is None:
            dynamic_labor = labor
        else:
            homotopy = state[self.homo_index]
            labor_mean = jnp.mean(labor)
            dynamic_labor = homotopy * labor + (1.0 - homotopy) * labor_mean

        a_values = output[jnp.asarray(self.a_indices)]
        projection = jnp.sum(a_values * dynamic_labor) / (
            jnp.sum(dynamic_labor**2) + 1e-15
        )
        cleared = self.a_mask * (a_values - projection * dynamic_labor)
        output = output.at[jnp.asarray(self.a_indices)].set(cleared)

        k_values = output[jnp.asarray(self.k_indices)]
        output = output.at[jnp.asarray(self.k_indices)].set(
            self.k_mask * jax.nn.softplus(k_values)
        )

        values = output[jnp.asarray(self.softplus_indices)]
        return output.at[jnp.asarray(self.softplus_indices)].set(
            jax.nn.softplus(values)
        )

    def __call__(self, state: Array) -> Array:
        if state.ndim == 1:
            return self._forward_single(state)
        return jax.vmap(self._forward_single)(state)


def create_rss_market_clearing_net(
    model: ModelSpec,
    *,
    key: Array,
) -> RSSMarketClearingNet:
    """Build the fixed Phase-0 RSS checkpoint-parity architecture."""
    labor = model.constants.get("L")
    if labor is None:
        raise ValueError("rss_market_clearing_net requires model.constants['L']")
    return RSSMarketClearingNet(
        n_states=model.n_states,
        policy_names=model.policy_names,
        state_names=model.state_names,
        labor_endowments=labor,
        key=key,
    )


def _canonical_weight_dict(weights: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Remove the optional Keras ``:0`` tensor suffix from exported names."""
    canonical = {}
    for name, value in weights.items():
        key = name[:-2] if name.endswith(":0") else name
        if key in canonical:
            raise ValueError(f"duplicate checkpoint weight after normalization: {key}")
        canonical[key] = value
    return canonical


def load_tf_weight_dict(
    network: RSSMarketClearingNet,
    weights: Mapping[str, np.ndarray],
) -> RSSMarketClearingNet:
    """Load a canonical dense-layer weight dictionary into ``network``.

    Kernels are transposed from ``[in, out]`` to Equinox's ``[out, in]``.
    Missing keys and shape mismatches fail loudly; unrelated metadata arrays
    in an ``npz`` export are ignored.
    """
    source = _canonical_weight_dict(weights)
    missing = sorted(set(TF_WEIGHT_NAME_MAP) - set(source))
    if missing:
        raise ValueError(f"checkpoint is missing RSS network weights: {missing}")

    base_layers = list(network.base_layers)
    ansatz_layers = list(network.ansatz_layers)
    destinations = {"base_layers": base_layers, "ansatz_layers": ansatz_layers}

    for source_name, (field_name, index) in TF_LAYER_NAME_MAP.items():
        layer = destinations[field_name][index]
        kernel = np.asarray(source[f"{source_name}/kernel"])
        bias = np.asarray(source[f"{source_name}/bias"])
        expected_kernel = tuple(reversed(layer.weight.shape))
        expected_bias = layer.bias.shape
        if kernel.shape != expected_kernel:
            raise ValueError(
                f"{source_name}/kernel has shape {kernel.shape}; "
                f"expected {expected_kernel}"
            )
        if bias.shape != expected_bias:
            raise ValueError(
                f"{source_name}/bias has shape {bias.shape}; expected {expected_bias}"
            )
        loaded = eqx.tree_at(
            lambda item: (item.weight, item.bias),
            layer,
            (
                jnp.asarray(kernel.T, dtype=layer.weight.dtype),
                jnp.asarray(bias, dtype=layer.bias.dtype),
            ),
        )
        destinations[field_name][index] = loaded

    return eqx.tree_at(
        lambda item: (item.base_layers, item.ansatz_layers),
        network,
        (tuple(base_layers), tuple(ansatz_layers)),
    )
