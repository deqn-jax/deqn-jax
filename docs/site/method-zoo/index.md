# Method Zoo

**DEQN-JAX is a toolkit, not a single algorithm.** A run is four orthogonal
choices -- *how you step the parameters*, *how you parameterize the decision
rule*, *how you take the expectation and score the residual*, and *how you
measure whether the answer is any good*. Every entry below is selected by one
config field and shares the same single-JIT train step, so you can mix and match
without touching framework code.

This page is a curated menu. The canonical, always-current lists come from the
registries themselves:

```bash
uv run deqn-jax optimizers   # the 13 registered optimizers, live
uv run deqn-jax list         # the registered models
```

!!! warning "Validated vs experimental -- read the badges"
    DEQN-JAX is **alpha**. Items tagged **(validated)** are exercised by the
    test suite and the [gallery](../gallery/index.md) on a working model. Items
    tagged **(experimental)** work but are lightly tested, model-specific, or
    (in the diagnostics cabinet) currently a research probe written up in
    `docs/dev/` rather than a packaged function. The combination validated
    across the gallery is small: **adam + MLP + MSE residual + antithetic MC (or
    Gauss-Hermite quadrature)**. Everything else is a research instrument -- a
    lead, not a turnkey recommendation.

---

## Cabinet 1 -- Optimizer zoo

The parameter-update rule. Selected with `--set optimizer.name=<name>`. Each
name maps to one of four **train-step variants** (how gradients are formed
before the update), dispatched once at construction, outside JIT.

| Optimizer | Variant | Status | When to reach for it |
|---|---|---|---|
| `adam` | STANDARD | validated | **The default.** Start here; only move if it stalls. |
| `adamw` | STANDARD | validated | Adam with decoupled weight decay -- mild regularization for a large net. |
| `sgd` | STANDARD | validated | Baselines and ablations; rarely the production choice. |
| `lion` | STANDARD | experimental | Sign-momentum; cheaper state than Adam. Try when memory-bound. |
| `muon` | STANDARD | experimental | Newton-Schulz orthogonalized updates; try when Adam plateaus on a deep MLP. |
| `ngd` | STANDARD | experimental | Diagonal-Fisher natural gradient; cheap curvature on stiff residual landscapes. |
| `shampoo` | STANDARD | experimental | Kronecker-factored second-order; for ill-conditioned losses. |
| `mao` | MAO | experimental | **Multi-equation models.** A separate Adam moment per equation so a loud equation can't drown a quiet one -- built for the 11-equation disaster system. |
| `mao_kfac` | MAO | experimental | MAO plus a shared-input Kronecker preconditioner. |
| `lbfgs` | LBFGS | experimental | Quasi-Newton with line search; near-deterministic residuals, and the warm-start engine. |
| `gn` | GN | experimental | Dense Gauss-Newton (H~=JtJ). Quadratic convergence *near* a solution -- a polish step. |
| `ign` | GN | experimental | Matrix-free implicit Gauss-Newton via conjugate gradients. |
| `lm` | GN | experimental | Levenberg-Marquardt: damped Gauss-Newton, the robust GN member. |

**Gradient surgery (orthogonal to the choice above).** PCGrad projects
conflicting per-equation gradients off each other before summing. It wraps any
STANDARD optimizer: `gradient_surgery: pcgrad` (experimental). Reach for it on
multi-equation models where equations pull the policy in competing directions.

> ML <-> econ: "optimizer" is just *how you solve for the approximation's
> coefficients* -- the inner solve of a projection method. Adam is the
> workhorse; the GN/LM family is the Newton-style polish from a deterministic
> solver.

---

## Cabinet 2 -- Network zoo

The decision-rule parameterization -- the role Chebyshev polynomials or splines
play in a projection method. Selected with `network.net_type`.

| Network | `net_type` | Status | When to reach for it |
|---|---|---|---|
| **MLP** | `mlp` | validated | The default basis. Start here for any Markov policy. |
| **LinearPlusMLP** | `linear_plus_mlp` | validated | **The canonical fix for degenerate basins.** Policy = Blanchard-Kahn linear rule + a zero-initialized MLP correction; at init the policy *is* the BK solution, so training can only improve on a correct first-order floor. Reach for it whenever a bare MLP collapses to a wrong, low-residual fixed point. |
| **LSTM** | `lstm` | experimental | History-dependent policies: a window of past states. |
| **Transformer** | `transformer` | experimental | Same history window, attention instead of recurrence. |
| **DisasterPolicyNet** | `disaster_policy_net` | experimental | LinearPlusMLP *plus* model-specific shape priors for CMR-style NK-DSGE (ZLB kink feature, Calvo reparameterizations, K/F gauge mask). The disaster superset -- not general-purpose. |
| **KfAnchoredMLP** | `kf_anchored_mlp` | legacy | An earlier, narrower gauge fix, **superseded by `disaster_policy_net`**. Kept for reproducibility; don't start new work on it. |

> The lineage that matters: `mlp` -> `linear_plus_mlp` (add a BK floor) ->
> `disaster_policy_net` (add model-specific priors). `kf_anchored_mlp` is an
> accidental earlier fork of the same gauge fix.

See [LinearPlusMLP](../networks/linear_plus_mlp.md) for the residual-ansatz math.

---

## Cabinet 3 -- Expectation & loss

Three orthogonal config axes. Mix freely (with the documented exclusions).

### (a) Expectation over shocks -- `expectation_type`

| Method | value | Status | When to reach for it |
|---|---|---|---|
| **Monte Carlo (antithetic)** | `mc` | validated | The default. Each eps paired with -eps for variance reduction; scales to many shock dimensions. |
| **Gauss-Hermite quadrature** | `gauss_hermite` | validated | Deterministic tensor-product nodes (cost `n_points^n_shocks`); a noise-free expectation with few shocks. The IRBC notebook uses this. |
| **Discrete Markov** | `discrete` | experimental | Exact enumeration over a finite chain (needs `model.transition_matrix` and `model.z_state_idx`). |

### (b) Residual aggregation -- `loss_choice`

| Aggregation | value | Status | When to reach for it |
|---|---|---|---|
| **MSE** | `mse` | validated | The default: square the shock-mean residual `(E[r])^2`. |
| **Huber** | `huber` | validated | Caps the gradient at +/-`huber_delta` when rare pathological states dominate. |
| **AiO (all-in-one)** | `aio` | experimental | Maliar-Maliar-Winant unbiased estimator; removes MSE's `Var(r-bar)/N` bias at small `mc_samples`. Requires `expectation_type=mc`, `mc_samples>=2`; per-eq losses can go transiently negative, so use `loss_reweight=none`. |

### (c) Loss structure -- `loss_type`

| Structure | value | Status | When to reach for it |
|---|---|---|---|
| **Plain residual** | `mse` | validated | Just the equilibrium residuals. The right default. |
| **Composite** | `composite` | experimental | Layers anchor + Jacobian-match + barrier + Newton auxiliary terms over the residual for stiff models. See [Composite loss](../training/composite_loss.md). |

Occasionally-binding constraints (irreversibility, borrowing limits, labor caps,
the ZLB) enter the residual as **Fischer-Burmeister complementarity** terms --
no special-casing the optimizer or the loss. **Two-stage / nested expectation**
(when a model defines `combine_fn`/`inside_fn`) is wired automatically
(experimental). **Adaptive reweighting** (`lr_annealing`, `relobralo`) balances
multi-equation losses; any term keyed with an `aux_` prefix is excluded from
reweighting and gradient surgery by construction.

> ML <-> econ: the "loss" is the Euler/FOC/market-clearing error; "taking the
> expectation" is the quadrature or Monte-Carlo integration over next-period
> shocks you'd do in any global solver.

---

## Cabinet 4 -- Diagnostic zoo

A low residual is **necessary but not sufficient.** These tools tell you whether
the solved policy is actually good -- several exist precisely because we caught
residual-minimization landing on wrong answers.

| Diagnostic | Where | Status | What it tells you |
|---|---|---|---|
| **errREE -- relative Euler errors** | `evaluate/diagnostics.py: euler_equation_errors` | validated | **The gold-standard accuracy metric** (Azinovic et al. 2022). The `log10|residual|` distribution on a long ergodic path. The number you quote. |
| **Market-clearing errors** | `evaluate/diagnostics.py: market_clearing_errors` | validated | Resource-constraint violation along the path -- feasibility independent of the Euler residual. |
| **Simulated moments** | `evaluate/diagnostics.py: simulated_moments` | validated | Ergodic means/stds vs a reference. Catches a *state-blind* policy. |
| **Stability check** | `evaluate/diagnostics.py: stability_check` | validated | Flags policies pinned to bounds, states drifting from SS, NaNs. A fast pass/fail gate. |
| **Dynare Jacobian match** | `evaluate/dynare.py` | validated | Frobenius distance between the network's policy slope at SS and the Dynare/BK matrix `P`. |
| **Active subspace / effective dimension** | `active_subspace.py` | experimental | Eigenanalysis of the policy-gradient covariance + a degeneracy detector. |
| **Ergodic replay buffer** | `training/replay.py` | experimental | A prioritized ring buffer so the policy doesn't forget rare-event branches (ZLB, disaster). A training mechanism, not a metric. |
| **Closed-loop spectral radius rho** | dev analysis (`docs/dev/disaster_stability_findings.md`) | research probe | rho of the closed-loop Jacobian at SS -- a local stability read. A documented analysis, **not yet a packaged function**. |
| **Bias floor -- MSE vs AiO** | dev analysis (`docs/dev/aio_loss_estimator.md`) | research probe | Estimates the MC bias floor with no ground truth. A write-up + probe, not shipped API. |
| **Anchor / selection diagnostic** | dev analysis (`docs/dev/disaster_stability_findings.md`) | research probe | How strongly anchor/Jacobian terms pin the solution to the BK branch. Exploratory. |

!!! note "The three research probes are leads, not features"
    Closed-loop rho, the bias floor, and the anchor/selection check live in
    `docs/dev/` as analyses on the disaster model. They informed real fixes but
    are not exposed as stable API. To use them today, read the dev note and run
    the analysis by hand.

---

## Putting it together

The validated *starting* recipe is humble -- `mlp` (or `linear_plus_mlp`),
`adam`, `mc` + `mse` -- and that's the right place to begin on a new model.
Reach into the zoo only when the default stalls, and let the diagnostic cabinet,
not the loss value, tell you whether you've actually arrived.
