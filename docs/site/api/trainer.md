# Trainer

The trainer orchestrates a run. It has three entry points, from high level to
low level:

1. `train_from_config(config)`: pass a populated `TrainConfig`, get back
   `(policy_net, history)`, where `policy_net` is the trained network. Handles checkpointing, logging, early stopping,
   optimizer switching, warm start and the replay buffer. The CLI calls this,
   and it is the entry point to use from other programs.
2. `train(model_name, episodes, ...)`: a backward-compatible wrapper that
   builds a `TrainConfig` from its arguments and delegates.
3. `create_train_state(...)` + `make_train_step(...)`: use these to drive
   the training loop yourself (custom outer loop, distributed setup,
   hand-coded LR schedule, …). The step `make_train_step` returns is one
   JIT-compiled rollout followed by a sweep of JIT-compiled minibatch grad
   steps; the cycle itself is plain Python.

The rollout + minibatch-sweep cycle is shared by all five step variants
(STANDARD, PCGRAD, MAO, LBFGS, GN). Only the per-batch grad step differs, and
it is chosen at construction time, before JIT.

For the full surface and example calls, see
[Training entry points](../REFERENCE.md#training-entry-points) in REFERENCE.md.

::: deqn_jax.training.trainer
