# Quickstart

The 5-minute version: train the canonical Brock-Mirman model.

```bash
uv run deqn-jax train brock_mirman -n 1000 --warm-start
```

You should see loss drop several orders of magnitude in under a minute on
CPU.

## Disaster model (validated stack)

```bash
uv run deqn-jax train --config configs/disaster.yaml
```

This runs the validated `LinearPlusMLP` + composite loss stack on the
CMR-style NK-DSGE model. Expected wall-clock on a GB10: ~3k episodes per
minute post-JIT.

## Evaluate a checkpoint

```bash
uv run deqn-jax evaluate path/to/checkpoint.eqx -n 2000
```

## Impulse-response functions

```bash
uv run deqn-jax irf path/to/checkpoint.eqx --shock eps
```

## Resume with a different optimizer

```bash
# Train 3000 episodes with Adam
uv run deqn-jax train --config configs/disaster.yaml

# Continue from checkpoint with NGD
uv run deqn-jax train --config configs/disaster.yaml \
    --resume checkpoints/disaster/checkpoint_003000.eqx \
    --set optimizer.name=ngd
```

The trainer detects the optimizer change, re-initializes optimizer state,
and keeps the network weights.
