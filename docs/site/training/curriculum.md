# Curriculum & warm start

## Shock curriculum

Training with full-magnitude shocks from step 0 often diverges. The
shock curriculum ramps shock magnitude from a small fraction up to 1.0
over the first N episodes:

```yaml
curriculum_episodes: 200
curriculum_start: 0.1     # start at 10% of full magnitude
```

After episode `curriculum_episodes`, shocks are at full scale.

## Warm start

Two flavours:

### `warm_start: true`

Fits the network to the **steady-state policy** via L-BFGS in
~10-50 steps. Effectively initializes the network to a constant
function.

```yaml
warm_start: true
```

For `network.type: linear_plus_mlp`, warm start is **automatically
skipped** — the residual architecture already starts at the linear
policy by construction.

### `warm_start_dynare: <path>`

Imports a Dynare-solved linear policy and fits the network to it. For
research workflows where Dynare is the ground truth.

## SS reset fraction

```yaml
ss_reset_frac: 0.15
```

A fraction of episode rollouts re-initialize from a noisy steady
state instead of continuing from where the previous episode ended.
Prevents trajectories from drifting permanently outside the training
support.
