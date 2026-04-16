# Python API examples

Use DEQN-JAX as a library — useful when you want to script experiments,
plug a custom analysis on top of trained policies, or integrate with
notebook-style workflows.

## Train from a config object

```python
from deqn_jax.config import TrainConfig
from deqn_jax.training.trainer import train_from_config

config = TrainConfig.from_yaml("configs/disaster.yaml")
config = config.with_overrides({"episodes": 500})

params, history = train_from_config(config)
```

`history` is a dict of per-episode metrics (loss, per-equation residuals,
gradient norm). `params` is the trained Equinox model.

## Construct a config programmatically

```python
from deqn_jax.config import TrainConfig, NetworkConfig, OptimizerConfig

config = TrainConfig(
    model="brock_mirman",
    episodes=1000,
    network=NetworkConfig(type="mlp", hidden_sizes=(64, 64)),
    optimizer=OptimizerConfig(name="adam", learning_rate=1e-3),
)
```

## Inspect a model spec

```python
from deqn_jax.models import load_model

model = load_model("disaster")
print(model.state_names)
print(model.policy_names)
print(model.equation_names)
ss_state, ss_policy = model.steady_state_fn(model.constants)
```

## Evaluate residuals on a custom state batch

```python
import jax.numpy as jnp
from deqn_jax.training.loss import compute_residuals

states = jnp.array(...)   # [batch, n_states]
shock = jnp.zeros((states.shape[0], model.n_shocks))
residuals = compute_residuals(model, params, states, shock)
```

For full API details see the [API reference](../api/config.md).
