# CLI examples

The `deqn-jax` CLI exposes nine subcommands: `train`, `list`, `info`,
`check`, `evaluate`, `irf`, `optimizers`, `active-subspace`, `init-config`
(plus `--version`).

## Discovery

```bash
deqn-jax --version             # print the installed package version
deqn-jax list                  # show registered models
deqn-jax info disaster         # constants, state/policy names, equation count
deqn-jax optimizers            # list available optimizers
deqn-jax check                 # JAX device + version sanity check
deqn-jax init-config bm.yaml   # write a starter config you can edit
```

## Training

```bash
# Train from CLI flags
deqn-jax train brock_mirman -n 1000 -o adam --warm-start

# Train from YAML
deqn-jax train --config configs/disaster.yaml

# Override a single field at the command line
deqn-jax train --config configs/disaster.yaml --set episodes=500
```

## Evaluation & analysis

```bash
deqn-jax evaluate checkpoints/disaster/checkpoint_003000.eqx -n 2000
deqn-jax irf checkpoints/disaster/checkpoint_003000.eqx --shock eps
```

## Switching optimizers mid-training

```bash
# Phase 1: explore with Adam
deqn-jax train --config configs/disaster.yaml

# Phase 2: polish with NGD from a checkpoint
deqn-jax train --config configs/disaster.yaml \
    --resume checkpoints/disaster/checkpoint_003000.eqx \
    --set optimizer.name=ngd
```

See [Configuration](../getting-started/configuration.md) for the full
override syntax.
