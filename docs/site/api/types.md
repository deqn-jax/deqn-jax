# Types

The core types are `NamedTuple`s, so they are JAX pytrees. `ModelSpec` is the
user contract: the only object a model has to populate to be trainable.
`TrainState` carries everything mutable across a run (params, optimizer state,
episode state, key, step counter, replay state, …), so `train_step` can be a
pure function under `@jax.jit`.

The `ModelSpec` fields, with signatures and shape contracts, are listed in
[The user contract](../REFERENCE.md#the-user-contract-modelspec) in
REFERENCE.md. For a walkthrough of populating a `ModelSpec`, see
[Implementing a model](../models/implementing.md).

The other types (`ReweightState`, `ReplayState`, `EpisodeState`, `Metrics`)
are internal data plumbing. They appear in type signatures, but you rarely
construct them. The exception is `make_reweight_state(n_equations)`, used when
building a `TrainState` outside `create_train_state` (mainly in tests).

::: deqn_jax.types
