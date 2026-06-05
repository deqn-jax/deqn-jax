"""Policy-network factory: net_type dispatch for ``create_train_state``.

Lifted verbatim from ``state_init.create_train_state`` so the state builder holds
no network-construction logic. ``build_policy_net`` handles the generic net types
(mlp / linear_plus_mlp / lstm / transformer) inline and the disaster-specific
types (disaster_policy_net / kf_anchored_mlp) via lazy imports, exactly as before.
Pure move -- byte-for-byte the same construction, just relocated.
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
    if network_config is not None:
        hidden_sizes = network_config.hidden_sizes
        activation = network_config.activation
        activations = network_config.activations
        init = network_config.init
        multi_head = getattr(network_config, "multi_head", False)
        skip_connections = getattr(network_config, "skip_connections", False)
        net_type = getattr(network_config, "type", "mlp")
        history_len = getattr(network_config, "history_len", 1)
        num_heads = getattr(network_config, "num_heads", 4)
        n_layers = getattr(network_config, "n_layers", 2)

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
        init_scale = getattr(network_config, "init_scale", 0.0)
        # output_links: explicit YAML setting wins, then model.default_output_links,
        # then None (factory defaults to all-linear, legacy behavior).
        output_links = getattr(network_config, "output_links", None)
        if output_links is None:
            output_links = getattr(model, "default_output_links", None)
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
        init_scale = getattr(network_config, "init_scale", 0.0)
        use_zlb_feature = getattr(network_config, "use_zlb_feature", False)
        zlb_feature_kind = getattr(network_config, "zlb_feature_kind", "raw")
        kf_names = getattr(network_config, "kf_names", ())
        reparam_q_as_m = getattr(network_config, "reparam_q_as_m", False)
        reparam_pi_as_kp_inner = getattr(
            network_config, "reparam_pi_as_kp_inner", False
        )
        reparam_wtilda_as_kw_inner = getattr(
            network_config, "reparam_wtilda_as_kw_inner", False
        )
        output_links = getattr(network_config, "output_links", None)
        if output_links is None:
            output_links = getattr(model, "default_output_links", None)
        policy_net = create_disaster_policy_net(
            model=model,
            hidden_sizes=hidden_sizes,
            activation=activation,
            init=init,
            init_scale=init_scale,
            input_shift=input_shift,
            input_scale=input_scale,
            kf_names=kf_names,
            use_zlb_feature=use_zlb_feature,
            zlb_feature_kind=zlb_feature_kind,
            reparam_q_as_m=reparam_q_as_m,
            reparam_pi_as_kp_inner=reparam_pi_as_kp_inner,
            reparam_wtilda_as_kw_inner=reparam_wtilda_as_kw_inner,
            output_links=output_links,
            key=net_key,
        )
    elif net_type == "kf_anchored_mlp":
        # K/F gauge elimination: network outputs only non-K/F policies; K/F
        # values come from the model's Blanchard-Kahn linearization at each
        # state. See networks/kf_anchored_mlp.py for the rationale.
        from deqn_jax.networks.kf_anchored_mlp import create_kf_anchored_mlp

        kf_names = getattr(network_config, "kf_names", ("F_p", "K_p", "F_w", "K_w"))
        policy_net = create_kf_anchored_mlp(
            model=model,
            hidden_sizes=hidden_sizes,
            activation=activation,
            init=init,
            kf_names=kf_names,
            input_shift=input_shift,
            input_scale=input_scale,
            key=net_key,
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
            multi_head=multi_head,
            skip_connections=skip_connections,
            input_shift=input_shift,
            input_scale=input_scale,
            key=net_key,
        )

    return policy_net
