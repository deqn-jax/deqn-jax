# Config

Configuration is a tree of Pydantic v2 models rooted at `TrainConfig`.
Constructing a `TrainConfig` validates every field; unknown keys (typos) raise
`ValueError` with did-you-mean suggestions. The sub-configs
(`OptimizerConfig`, `NetworkConfig`, `CompositeLossConfig`,
`ReplayBufferConfig`, `MomentMatchingConfig`) are built by `default_factory`,
so a sub-block can be omitted.

The field-by-field schema with defaults and ranges is in
[Configuration schema](../REFERENCE.md#configuration-schema) in REFERENCE.md.
This page is the generated symbol-level reference.

Load YAML with `TrainConfig.from_yaml(path)` and write it back with
`cfg.to_yaml(path)` (tuples become lists so `safe_load` can read the file).
Override priority: `--set` overrides > CLI args > YAML > defaults.

::: deqn_jax.config
