# DEQN-JAX

**Pure-JAX framework for solving economic equilibrium models with deep
equilibrium networks.**

Train a neural network to satisfy a dynamic model's equilibrium conditions
across the full state space, rather than solving point-by-point.

```
state  →  Network  →  policy  →  Equilibrium equations  →  Loss = Σ residuals²
```

![DEQN solver training loop](figures/deqn_solver_loop.svg)

## Status

Alpha (`v0.1.0`). API may change. Core plumbing is solid; 241 tests pass.
The package is intended to support multiple research papers — it is not
paper-specific.

## What's here

- **Models**: Brock-Mirman (canonical smoke test), CMR-style NK-DSGE with
  financial frictions and disaster risk.
- **Networks**: MLP, LSTM, Transformer, LinearPlusMLP (residual over
  Blanchard-Kahn linearization).
- **Optimizers**: Adam, SGD, AdamW, Lion, Muon, NGD, Shampoo, MAO,
  MAO-KFAC, L-BFGS, Gauss-Newton, Levenberg-Marquardt.
- **Loss**: Composite (anchor + Jacobian + barrier + Newton) layered on
  residual MSE.
- **Expectations**: Monte Carlo with antithetic variates or tensor-product
  Gauss-Hermite quadrature.

## Where to go next

- New here? → [Installation](getting-started/installation.md), then
  [Quickstart](getting-started/quickstart.md).
- Want to add a model? → [Adding a model](models/adding.md).
- Reading the source? → [docs/dev/reading_guide.md](https://github.com/mechanicpanic/deqn-jax/blob/master/docs/dev/reading_guide.md)
  is a code-level narrative for contributors.

## Citing

If you use DEQN-JAX in research, please cite the foundational DEQN
papers:

- Azinovic, M., Gaegauf, L., Scheidegger, S. (2022). *Deep Equilibrium Nets.*
  International Economic Review 63(4), 1471–1525.
- Scheidegger, S., Bilionis, I. (2019). *Machine learning for high-dimensional
  dynamic stochastic economies.* Journal of Computational Science 33, 68–82.
