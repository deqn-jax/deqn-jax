# Models

Two registration paths share one `_MODELS` dict:

- In-tree: add an import and an entry to `_MODELS` in
  [`src/deqn_jax/models/__init__.py`](https://github.com/deqn-jax/deqn-jax/blob/master/src/deqn_jax/models/__init__.py),
  and a `DESCRIPTION` string in your package's `variables.py` (`_DESCRIPTIONS`
  reads the `deqn-jax list` text from there). Use this for models that ship
  with the library.
- Programmatic: call `register_model(spec, description=...)` at runtime. Use
  this for generated models in user projects, notebook prototypes, or
  external plugins. See [Adding a model](../REFERENCE.md#adding-a-model) for
  the contract.

`load_model(name)` and `list_models()` treat both paths the same.

`VariableSpec` (in `variable_spec.py`) gives named attribute access to state
and policy arrays (`s.k`, `p.sav_rate`) for both batched `[batch, n]` and
unbatched `[n]` shapes. It replaces `state[:, 0]` indexing and traces through
`jax.vmap` unchanged. `make_init_state_fn` builds initial-state samplers
declaratively, with `uniform`, `normal`, `lognormal`, `truncated_normal` or
`constant` per variable.

::: deqn_jax.models

::: deqn_jax.models.variable_spec
