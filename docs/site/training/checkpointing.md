# Checkpointing & resume

## Configure

```yaml
checkpoint_dir: checkpoints/disaster
checkpoint_every: 250
max_checkpoints: 15        # rolling window; older ones are deleted
```

Each checkpoint is a serialized Equinox pytree of the full `TrainState`
(params + opt_state + episode counters + reweight state + target
params + aux params).

The original config is also written to `<checkpoint_dir>/config.yaml`
so resume can reconstruct the matching pytree template.

## Resume

```bash
deqn-jax train --config configs/disaster.yaml \
    --resume checkpoints/disaster/checkpoint_003000.eqx
```

`config.episodes` is the **target** total — resuming from episode 3000
with `episodes: 5000` runs 2000 more episodes.

## Switching optimizers on resume

```bash
deqn-jax train --config configs/disaster.yaml \
    --resume checkpoints/disaster/checkpoint_003000.eqx \
    --set optimizer.name=ngd
```

The trainer detects the optimizer change, re-initializes optimizer
state for the new method (with the right shape/structure), and keeps
the network weights. Useful for **Adam-then-second-order pipelines**:
rough exploration with Adam, polish with NGD or L-BFGS.

## What happens to schedules

If the original config had a cosine LR schedule, it's recomputed against
the **new** total episode count, not extrapolated. If the new optimizer
config has its own schedule settings, those apply from the resume point.
