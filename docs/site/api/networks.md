# Networks

Five built-in architectures, dispatched by `NetworkConfig.type`:

| `type` | Module | Use case |
| --- | --- | --- |
| `mlp` | `mlp.MLP`, `MultiHeadMLP`, `ResMLP` | Most models; default |
| `lstm` | `lstm.LSTMPolicy` | History-dependent policies, `history_len > 1` |
| `transformer` | `transformer.TransformerPolicy` | Same; multi-head attention over history window |
| `linear_plus_mlp` | `linear_plus_mlp.LinearPlusMLP` | `policy = linear(state) + mlp(state)`; init at the BK linearization |
| `kf_anchored_mlp` | `kf_anchored_mlp` | CMR-class disaster: K/F outputs pinned to BK anchor |

All factories take `(n_states, n_policies, hidden_sizes, ..., key)` and
return an Equinox `eqx.Module` whose `__call__(state) -> policy` works on
both `[n_states]` and `[batch, n_states]` inputs (via `jax.vmap` of a
per-sample helper).

Output bounds are enforced **at the network output**, per-dimension:

- Finite `policy_upper[i]` → sigmoid scaled to `[lower, upper]`.
- `policy_upper[i] = jnp.inf` → `softplus(x) + lower`.

Shared utilities live in `networks/common.py`: `_normalize_input`
(input shift/scale stop_gradient), `_apply_bounds` (the
sigmoid/softplus dispatch), `INIT_FNS` (the init-name → init-fn
table).

For adding a new network type, see [Adding a network](../networks/adding.md).

::: deqn_jax.networks.mlp

::: deqn_jax.networks.lstm

::: deqn_jax.networks.transformer

::: deqn_jax.networks.linear_plus_mlp
