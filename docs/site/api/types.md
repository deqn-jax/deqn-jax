# Types

The framework's core types are `NamedTuple`s for JAX pytree
compatibility. `ModelSpec` is the **user contract** — the only object a
model needs to populate to be trainable. `TrainState` carries everything
mutable across a training run (params, opt state, episode state, key,
step counter, replay state, …) so `train_step` can be a pure function
suitable for `@jax.jit`.

For the `ModelSpec` field listing with signatures and shape contracts,
see [The user contract](../REFERENCE.md#the-user-contract-modelspec) in
REFERENCE.md. For the prose walkthrough on populating a `ModelSpec`,
see [Implementing a model](../models/implementing.md).

The remaining types (`ReweightState`, `ReplayState`, `EpisodeState`,
`Metrics`) are framework-internal data plumbing that you'll see in
type signatures but rarely construct yourself. The exception is
`make_reweight_state(n_equations)` — used when constructing a
`TrainState` outside `create_train_state` (rare; mainly in tests).

::: deqn_jax.types
