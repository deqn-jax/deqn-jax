# Trainer

The trainer is the orchestration layer. Three entry points, ordered by
abstraction:

1. **`train_from_config(config)`** — high-level. Pass a populated
   `TrainConfig`, get back `(state, history)`. Honors checkpointing,
   logging, early stopping, optimizer switching, warm start, replay
   buffer. This is what the CLI calls under the hood. **Use this from
   agent code.**
2. **`train(model_name, episodes, ...)`** — backward-compat wrapper that
   builds a `TrainConfig` from positional args and delegates.
3. **`create_train_state(...)` + `make_train_step(...)`** — low-level.
   Use when you need to drive the training loop yourself (custom outer
   loop, distributed setup, hand-coded LR schedule, …). The single
   `@jax.jit` boundary is around `train_step`.

The "rollout + minibatch sweep" cycle is shared across all five
optimizer families (STANDARD, PCGRAD, MAO, LBFGS, GN); only the
per-batch grad step differs (dispatched at construction time, before
JIT).

For the full surface and example call patterns, see
[Training entry points](../REFERENCE.md#training-entry-points) in
REFERENCE.md.

::: deqn_jax.training.trainer
