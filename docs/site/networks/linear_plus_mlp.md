# LinearPlusMLP

A residual parameterization that wraps an MLP around a Blanchard-Kahn
first-order policy:

$$
\pi(s) = \pi^* + P\,(s - s^*) + \delta_{\text{MLP}}(s)
$$

- $\pi^*, s^*$: steady state policy / state
- $P$: linear policy rule from QZ decomposition of the linearized model
- $\delta_{\text{MLP}}$: nonlinear correction learned during training

With `init_scale=0.0`, the network *starts exactly at the linear policy*
and training only adds nonlinear refinements. This is the **architectural
prior** that makes the disaster model trainable.

## Why this matters

A plain MLP trained on the bare residual loss for the disaster model
finds degenerate self-referential fixed points — its ergodic
distribution sits ~15% off the true SS. This is a PINN identification
pathology: the residual loss is set-identifying, not point-identifying.

`LinearPlusMLP` solves this by *constraining* the policy to be a
correction to a known-good first-order solution. The composite loss then
keeps the correction small near the SS and lets it grow in the tails.

## YAML

```yaml
network:
  type: linear_plus_mlp
  hidden_sizes: [128, 128]
  activation: tanh
  init: xavier_normal
  init_scale: 0.0       # 0.0 = exact linear at init
```

## Source

`src/deqn_jax/networks/linear_plus_mlp.py`. Linearization uses the
`linearize_model` function in `src/deqn_jax/training/linearize.py`.
