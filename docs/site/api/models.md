# Models

Two registration paths share one `_MODELS` dict:

- **In-tree** — add an import + entry to `_MODELS` and `_DESCRIPTIONS`
  in [`src/deqn_jax/models/__init__.py`](https://github.com/deqn-jax/deqn-jax/blob/master/src/deqn_jax/models/__init__.py).
  Right thing for models that ship with the library.
- **Programmatic** — call `register_model(spec, description=...)` at
  runtime. Right thing for agent-codegen'd models in user projects,
  notebook prototypes, or external plugins. See
  [Adding a model](../REFERENCE.md#adding-a-model) for the contract.

Both paths feed `load_model(name)` and `list_models()` identically.

`VariableSpec` (in `variable_spec.py`) is a small helper that gives
named attribute access to state and policy arrays (`s.k`, `p.sav_rate`)
across both batched `[batch, n]` and unbatched `[n]` shapes — cleaner
than `state[:, 0]` everywhere, and traces through `jax.vmap` unchanged.
`make_init_state_fn` is a declarative builder for initial-state
samplers; supports `uniform`, `normal`, `lognormal`, `truncated_normal`,
`constant` per variable.

::: deqn_jax.models

::: deqn_jax.models.variable_spec
