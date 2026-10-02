# Curriculum & warm start

## Shock curriculum

Training with full-size shocks from step 0 often diverges. The shock
curriculum ramps the shock magnitude from a small fraction up to 1.0 over the
first N episodes:

```yaml
curriculum_episodes: 200
curriculum_start: 0.1     # start at 10% of full magnitude
```

After `curriculum_episodes` episodes, shocks are at full scale.

## Warm start

There are two variants.

### `warm_start: true`

Fits the network to the steady-state policy with L-BFGS in about 10-50 steps.
This initializes the network to a constant function.

```yaml
warm_start: true
```

For BK-anchored networks (`network.type: linear_plus_mlp` or
`disaster_policy_net`), the warm start is skipped automatically: these
networks start at the linear policy by construction.

### `warm_start_dynare: <path>`

Imports a Dynare-solved linear policy and fits the network to it, for research
workflows that treat Dynare as the ground truth. The path is used only when
`warm_start: true` is also set.

## SS reset fraction

```yaml
ss_reset_frac: 0.15
```

A fraction of episode rollouts restart from a noisy steady state instead of
continuing from where the previous episode ended. This keeps trajectories from
drifting permanently outside the training support.
