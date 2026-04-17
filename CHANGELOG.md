# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is `0.x.y` the public API is unstable and may change
in any minor version bump.

## [0.2.0] — 2026-04-17

The disaster block now converges. Between v0.1.0 and v0.2.0 the disaster
model went from "trains only at p=0" to "trains to baseline-level
residuals (~1e-4) across p ∈ [0, 0.1]". The fix was three layered changes:
a disaster-aware Blanchard–Kahn linearization, an effective lower bound
on the Taylor-rule interest rate, and improved checkpoint tracking.

### Added

- **Per-run constants override**: `TrainConfig.constants: Dict[str, float]`
  patches `model.constants` at train-time. Enables calibration sweeps
  from YAML without editing model code.

- **Effective lower bound on R**: smooth-softfloor ZLB applied to the
  Taylor-rule output in `models/disaster/equations.py`. Configurable via
  `R_lb` (default 1.0 = zero nominal) and `R_lb_sharpness` (default 500,
  chosen so SS distortion is ~1e-7 — negligible). Eliminates the
  economically-nonsensical R<1 trajectories that destabilised disaster
  training in v0.1.0. Exposes `R_taylor` and `R_zlb_binding` in
  `definitions()` for post-hoc diagnostics.

- **Best-checkpoint tracking**: `TrainConfig.save_best_checkpoint`
  (default True) writes `checkpoint_best.eqx` + `checkpoint_best.meta`
  whenever running-min loss improves past a curriculum-aware grace
  period. Shippable artifact is now the best achievable snapshot, not
  the last episode.

- **`use_risky_steady_state` flag** on `TrainConfig` (default True):
  controls the auto-swap to `risky_steady_state` when `p_disaster > 0`.
  Set False to force deterministic SS as anchor for ablations.

- **Sweep configs** for the disaster model across p ∈ {0, 0.001, 0.005,
  0.02, 0.05, 0.1}: `disaster_p{…}_zlb.yaml` (validated stack) plus
  supporting ablation configs (`_detss`, `_riskylin`, `_anchor…`,
  `_kappaonly`, `_cmrlib`).

- **mkdocs-material documentation site** (`docs/site/`, `mkdocs.yml`)
  with getting-started, models, networks, optimizers, training topics,
  and auto-generated API reference via mkdocstrings.

- **Developer reading guide** (`docs/dev/reading_guide.md`):
  code-level narrative following one train step end-to-end; documents
  load-bearing constraints (single JIT boundary, `aux_` prefix, xi_p
  determinacy pinning, log-vs-ratio Jensen fix, target net
  `stop_gradient`, SS caching).

- **Training-step diagram spec** (`docs/figures/training_step_spec.md`):
  structural description for digitising the hand-drawn training-step
  diagram.

### Changed

- **`linearize.py` is disaster-aware**. When `p_disaster > 0`, both
  `G_vec` in `linearize_model` and `step_wrt_shock` in
  `compute_ergodic_covariance` use the disaster-mixture-expected step
  function, so the resulting `P` matrix and ergodic covariance match
  the law of motion that agents at the risky SS actually face. In v0.1.0
  these were computed with `d_disaster=0` regardless, producing an
  internally-inconsistent linearization that caused disaster-run
  divergence. At p=0.02 the expected-Jacobian P differs from the d=0
  P by up to ~3% on the Phillips-block rows.

- **Trainer** auto-swaps `model.steady_state_fn` to
  `disaster.risky_steady_state` when `model.name == "disaster"` and
  `p_disaster > 0` and `use_risky_steady_state` is True. Single entry
  point in `train_from_config`.

- **Stability check in `evaluate.py`** fixed. Two bugs: max-SS-deviation
  divided by max(|ss|, 1e-8), which produced billion-percent values
  for zero-SS states like `m_p`; bound-hit counter treated every policy
  as bound-hitting when the upper bound was infinite. Both corrected.

### Known limits for 0.2.0

- **Tail residuals concentrate at ZLB-binding states**. Typical mean
  Euler-residual is ~1%, but max can reach ~11% at states where the
  ZLB softfloor transitions from non-binding to binding (R_taylor
  near R_lb). This is the known-hard regime of occasionally-binding
  constraints for smooth-network approximations (see OccBin,
  Fernández-Villaverde et al.). Future work: treat ZLB regime-switch
  explicitly.

- **Calibration is the paper's working calibration**, not CMR (2014)'s
  Bayesian posterior modes. In particular κ=2 (vs CMR 10.78), ξ_p=0.6
  (vs 0.74), ξ_w=0.6 (vs 0.81), α_π=1.5 (vs 2.4), λ_w=1.2 (vs 1.05).
  Our calibration is uniformly more volatile/less sticky than CMR's
  estimates. Sweep configs for CMR-calibrated runs are included but
  not validated; a principled sensitivity analysis is v0.3.0 territory.

- **Disaster IRF** (forcing d=1 at a specific period to trace out
  disaster-response dynamics) is not exposed in the CLI. Gaussian IRFs
  work and show textbook responses.

- **Capital-quality-dispersion and capital-immobilization** disaster
  variants remain unimplemented (only capital destruction).

## [0.1.0] — 2026-04-15

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
