# Config

Configuration is a tree of Pydantic v2 models rooted at `TrainConfig`.
Constructing a `TrainConfig` validates every field; passing unknown keys
(typos) raises `ValueError` with did-you-mean suggestions. Sub-configs
(`OptimizerConfig`, `NetworkConfig`, `CompositeLossConfig`,
`ReplayBufferConfig`, `MomentMatchingConfig`) are constructed via
`default_factory`, so omitting a sub-block is safe.

For the field-by-field schema with defaults and ranges, see the
[Configuration schema](../REFERENCE.md#configuration-schema) section in
REFERENCE.md. This page is the auto-generated symbol-level reference.

YAML loading: `TrainConfig.from_yaml(path)`. CLI override priority:
`--set` overrides > CLI args > YAML > defaults. Round-trip via
`cfg.to_yaml(path)` (tuples are coerced to lists for `safe_load`
compatibility).

::: deqn_jax.config
