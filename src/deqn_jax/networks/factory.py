"""Policy-network factory: net_type dispatch for ``create_train_state``.

``build_policy_net`` handles the generic net types (mlp / linear_plus_mlp /
lstm / transformer) inline and the model-specific types (disaster_policy_net,
rss_market_clearing_net) via a lazy import. It is the ONE place
a ``NetworkConfig`` is turned into a module — checkpoint loaders must rebuild
their template through it with the full config (static fields such as the
disaster network's ``bk_pin`` change the forward graph and are not repaired
by leaf deserialization).

Each branch passes on the fields ``NetworkConfig.FIELDS_READ`` lists for its
type and no others; ``NetworkConfig`` rejects any other field set away from
its default, so a field that would not reach the network fails at config
time. ``transformer`` reads only the first entry of ``hidden_sizes``,
``rss_market_clearing_net`` requires exactly two.
"""

import jax.numpy as jnp

from deqn_jax.networks.linear_plus_mlp import create_linear_plus_mlp
from deqn_jax.networks.lstm import create_lstm
from deqn_jax.networks.mlp import create_mlp
from deqn_jax.networks.transformer import create_transformer
from deqn_jax.types import ModelSpec


def build_policy_net(model: ModelSpec, net_key, hidden_sizes, network_config):
    """Construct the policy network for the configured ``net_type``.

    Returns the Equinox policy module. ``net_key`` is the dedicated network
    PRNG subkey; ``hidden_sizes`` is the fallback when ``network_config`` is
    None (it is overridden by ``network_config.hidden_sizes`` otherwise).

    ``network_config=None`` builds a plain bounded MLP from the defaults
    below. Note those defaults are NOT identical to ``NetworkConfig()``'s:
    ``init`` is ``"xavier_normal"`` here versus ``"default"`` on the config.
    Callers that pass None (evaluation smokes, ``scripts/cert/gn_polish.py``)
    depend on the weights this produces, so the divergence is preserved.

    Every non-``mlp`` branch is only reachable when ``network_config`` is a
    ``NetworkConfig`` (``net_type`` is read off it), so those branches read
    the remaining fields off it directly.
    """
    # Extract network params from config or use defaults
    activation = "tanh"
    activations = None
    init = "xavier_normal"
    net_type = "mlp"
    history_len = 1
    num_heads = 4
    n_layers = 2
    init_scale = 0.0
    output_links = None
    if network_config is not None:
        hidden_sizes = network_config.hidden_sizes
        activation = network_config.activation
        activations = network_config.activations
        init = network_config.init
        net_type = network_config.type
        history_len = network_config.history_len
        num_heads = network_config.num_heads
        n_layers = network_config.n_layers
        init_scale = network_config.init_scale
        # output_links: explicit YAML setting wins, then
        # model.default_output_links, then None (the residual networks
        # default to all-linear, legacy behavior).
        output_links = network_config.output_links
    if output_links is None:
        output_links = model.default_output_links

    # Compute input normalization from steady state
    input_shift = None
    input_scale = None
    if model.steady_state_fn is not None:
        ss_state, _ = model.steady_state_fn(model.constants)
        input_shift = ss_state
        input_scale = jnp.maximum(jnp.abs(ss_state), 0.01)

    # Create policy network based on type
    if net_type == "lstm":
        policy_net = create_lstm(
            n_states=model.n_states,
            n_policies=model.n_policies,
            hidden_sizes=hidden_sizes,
            history_len=history_len,
            policy_lower=model.policy_lower,
            policy_upper=model.policy_upper,
            input_shift=input_shift,
            input_scale=input_scale,
            key=net_key,
        )
    elif net_type == "transformer":
        # For Transformer, hidden_sizes is a single value (hidden_dim)
        # Handle case where hidden_sizes is a single int (from --set override)
        if isinstance(hidden_sizes, int):
            hidden_dim = hidden_sizes
        else:
            hidden_dim = hidden_sizes[0] if hidden_sizes else 64
        policy_net = create_transformer(
            n_states=model.n_states,
            n_policies=model.n_policies,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            num_heads=num_heads,
            history_len=history_len,
            policy_lower=model.policy_lower,
            policy_upper=model.policy_upper,
            input_shift=input_shift,
            input_scale=input_scale,
            key=net_key,
        )
    elif net_type == "linear_plus_mlp":
        # Generic residual parameterization: policy = linear(state) + mlp(state).
        policy_net = create_linear_plus_mlp(
            model=model,
            hidden_sizes=hidden_sizes,
            activation=activation,
            init=init,
            init_scale=init_scale,
            input_shift=input_shift,
            input_scale=input_scale,
            output_links=output_links,
            key=net_key,
        )
    elif net_type == "disaster_policy_net":
        # The residual ansatz plus the disaster model's shape priors, which
        # arrive as the validated ``network.options`` block.
        from deqn_jax.models.disaster.network import create_disaster_policy_net

        policy_net = create_disaster_policy_net(
            model=model,
            hidden_sizes=hidden_sizes,
            activation=activation,
            init=init,
            init_scale=init_scale,
            input_shift=input_shift,
            input_scale=input_scale,
            **network_config.typed_options().model_dump(),
            output_links=output_links,
            key=net_key,
        )
    elif net_type == "rss_market_clearing_net":
        from deqn_jax.networks.rss_net import (
            BASE_HIDDEN_SIZES,
            create_rss_market_clearing_net,
        )

        if len(hidden_sizes) != 2:
            raise ValueError(
                "network.type='rss_market_clearing_net' takes exactly two hidden "
                f"layers (the reference checkpoint uses {BASE_HIDDEN_SIZES}); "
                f"got hidden_sizes={tuple(hidden_sizes)}"
            )
        policy_net = create_rss_market_clearing_net(
            model, base_hidden_sizes=tuple(hidden_sizes), key=net_key
        )
    else:
        policy_net = create_mlp(
            n_states=model.n_states,
            n_policies=model.n_policies,
            hidden_sizes=hidden_sizes,
            activation=activation,
            activations=activations,
            init=init,
            policy_lower=model.policy_lower,
            policy_upper=model.policy_upper,
            input_shift=input_shift,
            input_scale=input_scale,
            key=net_key,
        )

    return policy_net
