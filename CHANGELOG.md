# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is `0.x.y` the public API is unstable and may change
in any minor version bump.

## [0.1.0] — Unreleased

### Added

- **Framework foundation**
  - `ModelSpec` — declarative economic model interface (variables, equations, dynamics, steady state).
  - `TrainState`, `Metrics`, `ReweightState` — `NamedTuple` state containers for JAX pytree compatibility.
  - Pydantic v2 configuration system with YAML + CLI overrides and strict validation.
  - CLI: `train`, `list`, `info`, `evaluate`, `irf`, `check`, `init-config`, `optimizers`.

- **Models**
  - `brock_mirman` — canonical RBC smoke test (2 states, 1 policy, analytical SS).
  - `disaster` — CMR NK-DSGE with financial frictions (13 states, 11 policies). Includes optional capital-destruction disaster block.
  - Risky steady state solver for disaster calibrations.

- **Networks**
  - `MLP`, `ResMLP` (Equinox) with sigmoid/softplus output bounding.
  - `LSTMPolicy`, `TransformerPolicy` for history-dependent policies.
  - `LinearPlusMLP` — residual parameterization over the Blanchard–Kahn linearization; architectural prior that structurally prevents convergence to degenerate fixed points on multi-equation models.

- **Training**
  - Single-JIT-boundary train step with 5 dispatched variants (STANDARD, PCGRAD, MAO, LBFGS, GN).
  - Monte Carlo loss with antithetic variates.
  - Gauss–Hermite quadrature for expectations.
  - Composite loss: anchor (supervised match to linear policy) + Jacobian matching at SS + barrier + Newton auxiliary terms.
  - Curriculum learning on shock magnitude (ramp 0→1).
  - Warm start from steady state (L-BFGS) or imported Dynare linearization.
  - Adaptive loss reweighting (`lr_annealing`, `relobralo`).
  - PCGrad gradient surgery for multi-equation conflicts.
  - Target networks (DQN-style) with Polyak averaging.
  - Checkpointing with resume, TensorBoard / W&B logging.

- **Optimizers**
  - Standard: `adam`, `sgd`, `adamw`, `lion`, `muon`.
  - Diagonal Fisher: `ngd`.
  - Kronecker-factored: `shampoo`.
  - Multi-adaptive (per-equation): `mao`, `mao_kfac`.
  - Second-order for least-squares: `gn` (Gauss–Newton), `lm` (Levenberg–Marquardt), `lbfgs`.

- **Evaluation tools**
  - Euler-equation error distribution (log10 residuals along simulated path).
  - Simulated ergodic moments vs. steady state.
  - Impulse response functions (all shocks, all variables).
  - Stability diagnostics.

- **Disaster model plumbing**
  - Mixture-expectation loss: residuals correctly averaged as
    `(1-p) · E[r | no disaster] + p · E[r | disaster]`, preserving the
    arithmetic-mean stochastic fixed point. Default `p_disaster=0`
    preserves baseline behaviour exactly for models that don't implement
    a disaster path.
  - `step_fn` accepts an optional `d_disaster` indicator scaling next-period
    capital by `exp(-θ)`.
  - Risky steady state solver (`risky_steady_state`).

### Known limits for 0.1.0

- `aiyagari` model exists in source but is not registered in the public
  model registry. Treat as internal/experimental.
- Capital-quality-dispersion and capital-immobilization disaster variants
  not yet implemented; only capital-destruction.
- Linearization and warm-start currently target the *deterministic* SS
  even for disaster calibrations. Using `risky_steady_state` as the
  anchoring target is a planned 0.2.0 improvement.
- Second-order optimizers (`gn`, `lm`) have limited smoke coverage on
  the disaster model.
- Per-equation Kronecker-factored `mao_kfac` is not yet integration-tested
  with `LinearPlusMLP` + composite loss.

### Notes

This is a research-framework alpha. The package is designed to support
multiple papers, not a single study. The `disaster` model is a testbed
for ongoing research on NK-DSGE with financial frictions, but the
framework itself is model-agnostic.
