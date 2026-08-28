"""Policy-network factory: net_type dispatch for ``create_train_state``.

``build_policy_net`` handles the generic net types (mlp / linear_plus_mlp /
lstm / transformer) inline and the disaster-specific types
(disaster_policy_net / kf_anchored_mlp) via lazy imports. It is the ONE place
a ``NetworkConfig`` is turned into a module — checkpoint loaders must rebuild
their template through it with the full config (static fields such as
``bk_pin`` change the forward graph and are not repaired by leaf
deserialization).

Which ``NetworkConfig`` fields each branch actually honors
---------------------------------------------------------

Every branch honors ``type`` and ``hidden_sizes``. Beyond that the branches
differ, and fields a branch does not honor are **silently ignored** here:

===================== ==========================================================
``type``              additionally honored
===================== ==========================================================
``mlp``               ``activation``, ``activations`` (per-layer, overrides
                      ``activation``), ``init``, ``multi_head``,
                      ``skip_connections``
``lstm``              ``history_len`` only — ``activation``, ``activations``
                      and ``init`` are NOT forwarded to ``create_lstm``
``transformer``       ``history_len``, ``num_heads``, ``n_layers`` — again
                      ``activation`` / ``activations`` / ``init`` are NOT
                      forwarded to ``create_transformer``
``linear_plus_mlp``   ``activation``, ``init``, ``init_scale``,
                      ``output_links`` (falling back to
                      ``model.default_output_links``); ``activations`` is
                      NOT honored (the delta MLP is built with a single
                      activation repeated per layer)
``disaster_policy_net`` everything ``linear_plus_mlp`` honors, plus
                      ``kf_names``, ``use_zlb_feature``, ``zlb_feature_kind``,
                      ``bk_pin``, ``reparam_q_as_m``,
                      ``reparam_pi_as_kp_inner``, ``reparam_wtilda_as_kw_inner``
``kf_anchored_mlp``   ``activation``, ``init``, ``kf_names``
===================== ==========================================================

TODO: reject the ignored combinations (e.g. ``type: lstm`` with a non-default
``init``, or ``activations`` on any non-``mlp`` type) in
``deqn_jax.training.state_init._validate_train_config`` rather than dropping
them here, so a mis-set field fails loudly instead of not taking effect.
"""

import jax.numpy as jnp

from deqn_jax.networks.linear_plus_mlp import create_linear_plus_mlp
from deqn_jax.networks.lstm import create_lstm
from deqn_jax.networks.mlp import create_mlp
from deqn_jax.networks.transformer import create_transformer
from deqn_jax.types import ModelSpec


def build_policy_net(model: ModelSpec, net_key, hidden_sizes, network_config):
    """Construct the policy network for the configured ``net_type``.

    Pure relocation of ``create_train_state``'s net-construction block;
    returns the Equinox policy module. ``net_key`` is the dedicated network
    PRNG subkey; ``hidden_sizes`` is the fallback when ``network_config`` is
    None (it is overridden by ``network_config.hidden_sizes`` otherwise).

    ``network_config=None`` builds a plain bounded MLP from the defaults
    below. Note those defaults are NOT identical to ``NetworkConfig()``'s:
    ``init`` is ``"xavier_normal"`` here versus ``"default"`` on the config.
    Callers that pass None (evaluation smokes, ``scripts/gn_polish.py``)
    depend on the weights this produces, so the divergence is preserved.

    See the module docstring for which config fields each branch honors.
    Every non-``mlp`` branch is only reachable when ``network_config`` is a
    ``NetworkConfig`` (``net_type`` is read off it), so those branches read
    the remaining fields off it directly.
    """
    # Extract network params from config or use defaults
    activation = "tanh"
    activations = None
    init = "xavier_normal"
    multi_head = False
    skip_connections = False
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
        multi_head = network_config.multi_head
        skip_connections = network_config.skip_connections
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
        # Model-agnostic; for disaster-specific shape priors (K/F mask, ELB
        # feature, q-as-M reparam) use network.type='disaster_policy_net'.
        if model.steady_state_fn is None:
            raise ValueError(
                "network.type='linear_plus_mlp' requires model.steady_state_fn"
            )
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
        # Disaster-specific residual ansatz: linear_plus_mlp + the three
        # disaster shape priors (K/F gauge mask, ELB feature, q-as-M reparam).
        # Each prior is independently toggleable via NetworkConfig fields.
        from deqn_jax.models.disaster.network import create_disaster_policy_net

        if model.steady_state_fn is None:
            raise ValueError(
                "network.type='disaster_policy_net' requires model.steady_state_fn"
            )
        policy_net = create_disaster_policy_net(
            model=model,
            hidden_sizes=hidden_sizes,
            activation=activation,
            init=init,
            init_scale=init_scale,
            input_shift=input_shift,
            input_scale=input_scale,
            kf_names=network_config.kf_names,
            use_zlb_feature=network_config.use_zlb_feature,
            bk_pin=network_config.bk_pin,
            zlb_feature_kind=network_config.zlb_feature_kind,
            reparam_q_as_m=network_config.reparam_q_as_m,
            reparam_pi_as_kp_inner=network_config.reparam_pi_as_kp_inner,
            reparam_wtilda_as_kw_inner=network_config.reparam_wtilda_as_kw_inner,
            output_links=output_links,
            key=net_key,
        )
    elif net_type == "kf_anchored_mlp":
        # K/F gauge elimination: network outputs only non-K/F policies; K/F
        # values come from the model's Blanchard-Kahn linearization at each
        # state. See networks/kf_anchored_mlp.py for the rationale.
        from deqn_jax.networks.kf_anchored_mlp import create_kf_anchored_mlp

        policy_net = create_kf_anchored_mlp(
            model=model,
            hidden_sizes=hidden_sizes,
            activation=activation,
            init=init,
            kf_names=network_config.kf_names,
            input_shift=input_shift,
            input_scale=input_scale,
            key=net_key,
        )
    elif net_type == "rss_market_clearing_net":
        from deqn_jax.networks.rss_net import (
            BASE_HIDDEN_SIZES,
            create_rss_market_clearing_net,
        )

        if tuple(hidden_sizes) != BASE_HIDDEN_SIZES:
            raise ValueError(
                "network.type='rss_market_clearing_net' requires "
                f"hidden_sizes={BASE_HIDDEN_SIZES} for checkpoint parity"
            )
        policy_net = create_rss_market_clearing_net(model, key=net_key)
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
            multi_head=multi_head,
            skip_connections=skip_connections,
            input_shift=input_shift,
            input_scale=input_scale,
            key=net_key,
        )

    return policy_net
