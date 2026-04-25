# Built-in optimizers

DEQN-JAX wraps optax with a thin registry that dispatches to one of
five **train-step variants**.

| Optimizer            | Variant   | Notes                                       |
|----------------------|-----------|---------------------------------------------|
| `adam`               | STANDARD  | Default. Validated stack uses this.         |
| `sgd`                | STANDARD  | Plain SGD.                                  |
| `adamw`              | STANDARD  | Adam with decoupled weight decay.           |
| `lion`               | STANDARD  | Sign-momentum optimizer.                    |
| `muon`               | STANDARD  | Newton-Schulz orthogonal updates.           |
| `ngd`                | STANDARD  | Diagonal Fisher natural gradient.           |
| `shampoo`            | STANDARD  | Kronecker-factored second-order.            |
| `mao`                | MAO       | Multi-Adaptive Optimizer (per-equation).    |
| `mao_kfac`           | MAO       | MAO with K-FAC preconditioner.              |
| `lbfgs`              | LBFGS     | optax L-BFGS with line search.              |
| `gn`                 | GN        | Dense residual-Jacobian step.               |
| `ign`                | GN        | Matrix-free implicit GN via CG.             |
| `lm`                 | GN        | Damped Gauss-Newton.                        |

Use any with `--set optimizer.name=...`:

```bash
deqn-jax train --config configs/disaster.yaml --set optimizer.name=ngd
```

## PCGrad

Available as a gradient-surgery option independent of the optimizer:

```yaml
gradient_surgery: pcgrad
```

Per-equation gradients are computed and conflicting ones projected.
Currently only compatible with STANDARD-variant optimizers.

For implementation details see the
[Optimizers API reference](../api/optimizers.md).
