# Composite loss

The composite loss layers four supervised auxiliary terms on top of the
residual MSE. It exists because the bare residual loss is
**set-identifying** — many policies satisfy it equally well, including
degenerate self-referential fixed points.

## Terms

| Term                | Penalises                                                |
|---------------------|----------------------------------------------------------|
| `aux_anchor`        | $\lVert \pi_{\text{net}}(s) - \pi_{\text{lin}}(s)\rVert^2$ at fixed pre-sampled points near SS |
| `aux_jac`           | $\lVert \partial \pi_{\text{net}}/\partial s\,(s^*) - P\rVert^2$ |
| `aux_barrier_*`     | Box penalties on bounded states/policies                 |
| `aux_newton_*`      | Conditioning + residual of the Newton step at SS         |

All keys are prefixed with `aux_` so adaptive reweighting and per-equation
gradient surgery (PCGrad, MAO) ignore them.

## Why anchor

Without it, the network can drift to any point in the residual-loss
zero set. With it, the network is *supervised* toward the linearized
policy near SS, which uniquely identifies the equilibrium of interest.

The anchor weight stays active throughout training when
`aux_decay_floor: 1.0`. Lowering it lets the anchor term decay during
the curriculum ramp.

## Configure

```yaml
loss_type: composite
composite_loss:
  anchor_weight: 1.0
  jac_weight: 0.1
  barrier_weight: 0.01
  newton_weight: 0.01
  n_anchor_points: 128
  anchor_sigma: 1.0
  aux_decay_floor: 1.0
```

## Source

`src/deqn_jax/training/composite_loss.py`. Pre-computed linearization
data flows in via `prepare_composite_data(model, P, Q)` once before
training, then is reused inside the JIT boundary every step.
