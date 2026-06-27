# API reference

Auto-generated reference for the deqn-jax public surface, rendered with
mkdocstrings from source docstrings.

!!! tip "Building on deqn-jax? Read REFERENCE first"
    For the curated, type-signature-first contract -- the stable `deqn_jax.api`
    surface, the `ModelSpec` fields, the programmatic `register_model(...)`
    path, and the verification gates -- start with the
    [ModelSpec reference](../REFERENCE.md). Everything imported from anywhere
    other than `deqn_jax.api` is **internal** and may be refactored without
    notice.

This section documents:

- [Config](config.md) -- `TrainConfig`, `NetworkConfig`, `OptimizerConfig`, `CompositeLossConfig` (Pydantic v2).
- [Types](types.md) -- `ModelSpec`, `TrainState`, `Metrics` (NamedTuples).
- [Trainer](trainer.md) -- `train_from_config`, `train`, the train-step builders.
- [Loss](loss.md) -- MC residual loss with antithetic variates.
- [Models](models.md) -- `load_model`, `list_models`, `register_model`.
- [Networks](networks.md) -- MLP / LSTM / Transformer / LinearPlusMLP modules.
- [Optimizers](optimizers.md) -- the registry and `create_optimizer`.
