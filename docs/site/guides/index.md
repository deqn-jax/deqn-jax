# Guides

Task-oriented and contributor docs that sit alongside the main story.

## Running and configuring

- [Running experiments](../running_experiments.md) -- the production training
  workflow, checkpointing, evaluation, and the ergodic Euler-error diagnostic.
- [CLI usage](../examples/cli.md) and [Python API usage](../examples/api.md) --
  copy-paste recipes for the `deqn-jax` command and `deqn_jax.api`.
- [Config field reference](../config_reference.md) -- every config field and its
  effect.

## For contributors

- [Reading guide](../reading_guide.md) -- a code-level narrative of the source
  for people about to modify it.
- [Module architecture](../architecture.md) -- the software-engineering view
  (module graph, JIT boundary, train-step variants). Contributor-only; if you
  are an economist evaluating the method, the [home page](../index.md) is the
  right starting point.
