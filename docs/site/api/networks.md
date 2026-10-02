# Networks

`NetworkConfig.type` selects one of six network types. Four are
general-purpose:

| `type` | Module | Use case |
| --- | --- | --- |
| `mlp` | `mlp.MLP` | Most models; default |
| `lstm` | `lstm.LSTMPolicy` | History-dependent policies, `history_len > 1` |
| `transformer` | `transformer.TransformerPolicy` | Same; multi-head attention over the history window |
| `linear_plus_mlp` | `linear_plus_mlp.LinearPlusMLP` | `policy = linear(state) + mlp(state)`; initialized at the BK linearization |

The other two are model-specific: `disaster_policy_net` (the `disaster`
model) and `rss_market_clearing_net` (the `rss_trade_ez_ref` model).

The general-purpose factories take `(n_states, n_policies, hidden_sizes, ...,
key)` and return an Equinox `eqx.Module`. Its `__call__(state) -> policy`
accepts both `[n_states]` and `[batch, n_states]` inputs, using `jax.vmap`
over a per-sample helper.

Output bounds are applied at the network output, per dimension:

- Finite `policy_upper[i]`: sigmoid scaled to `[lower, upper]`.
- `policy_upper[i] = jnp.inf`: `softplus(x) + lower`.

Shared utilities are in `networks/common.py`: `_normalize_input` (input
shift and scale, under `stop_gradient`), `_apply_bounds` (the
sigmoid/softplus dispatch) and `INIT_FNS` (init name to init function).

To add a network type, see [Adding a network](../networks/adding.md).

::: deqn_jax.networks.mlp

::: deqn_jax.networks.lstm

::: deqn_jax.networks.transformer

::: deqn_jax.networks.linear_plus_mlp
