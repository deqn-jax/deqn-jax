# Configuration

DEQN-JAX uses a layered config with three sources merged in priority order:

```
--set overrides  >  CLI args  >  YAML file  >  defaults
```

## Minimal YAML

```yaml
model: brock_mirman
episodes: 1000
batch_size: 64

network:
  type: mlp
  hidden_sizes: [64, 64]

optimizer:
  name: adam
  learning_rate: 1e-3
```

Run with:

```bash
uv run deqn-jax train --config my_config.yaml
```

## Dot-notation overrides

```bash
uv run deqn-jax train --config my_config.yaml \
    --set optimizer.learning_rate=0.01 \
    --set network.hidden_sizes=128,128
```

## Per-run constants override

Override individual model constants without editing model code:

```yaml
constants:
  p_disaster: 0.02
  theta_disaster: 0.05
```

Useful for calibration sweeps and disaster-risk experiments. See
[disaster model](../models/disaster.md).

## Validation

Configs are Pydantic v2 models. Unknown keys are rejected with
did-you-mean suggestions. Type errors are reported with the offending
field path.

For the full schema, see the [Config API reference](../api/config.md).
