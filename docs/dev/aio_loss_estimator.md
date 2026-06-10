# The AiO loss estimator: removing the MC bias in (E[r])²

**Status:** implemented (`loss_choice: aio`), validated statistically in
`tests/test_aio_loss.py`. Head-to-head experiment: `scripts/aio_head_to_head.py`,
results below.

## The problem

The DEQN loss aggregates equilibrium residuals as "average over shocks, then
square" (`training/loss.py`):

```
L̂(s) = ( (1/N) Σᵢ r(s, εᵢ) )²
```

This is the right *population* objective — equilibrium conditions are
`E_ε[r] = 0`, so the loss should be `(E[r])²`, not `E[r²]` (the latter is
minimized by nothing: it includes the irreducible shock variance of the
residual). But as a finite-sample **estimator**, the square of the sample
mean is biased upward:

```
E[L̂(s)] = (E[r])² + Var( (1/N) Σᵢ rᵢ )
         = (E[r])² + Var(r | s) / N        (iid case)
```

The bias is not symmetric noise that averages out over training steps — the
optimizer minimizes the *expected* loss, so it minimizes the bias term too.
The gradient of the bias term rewards policies that make the residual
**insensitive to the shock realization**, rather than zero in expectation.
That is a spurious smoothing force on the policy, strongest exactly where
the conditional variance `Var(r | s)` is largest: high-volatility, ZLB,
and disaster states.

### Interaction with antithetic variates

With antithetic pairs `(ε, -ε)` the bias term becomes the variance of the
pair-mean, which kills the **odd** (linear) part of `r(·)` in ε:

```
(r(ε) + r(-ε))/2 = E[r] + even-part fluctuations
```

So antithetic sampling shrinks the bias — by a lot for residuals nearly
linear in the shock — but cannot remove the contribution of the even part
(curvature). Models with strong precautionary curvature keep a bias floor.

### When the bias does NOT distort the optimum

Let `B(θ) = E_s[Var(r̄ | s)] ≥ 0` be the bias term as a function of policy
parameters. If the model admits a policy that zeroes the residual
**pointwise** (for every shock, not just in expectation), then `B(θ*) = 0`
at the optimum, and since `B ≥ 0` the argmin is unchanged — the biased loss
is merely a vertical shift near θ*. Brock–Mirman with log utility and
**full depreciation** (δ=1) is exactly this degenerate case: at
`sav_rate = αβ` the Euler residual is identically zero for every ε (income
and substitution effects cancel). The textbook test model is therefore
*structurally blind* to this bias.

Generic models (δ < 1, occasionally binding constraints, disaster risk) have
**irreducible shock variance at the optimum**: `E[r]=0` but `r(ε) ≠ 0`. There
the bias term has nonzero gradient at the true policy and shifts the argmin
by `O(Var/N)`.

## The fix: all-in-one (AiO) estimator

Maliar–Maliar–Winant (JME 2021, "Deep learning for solving dynamic economic
models", §AiO expectation). Split the N draws into two **independent**
groups, average each, and multiply:

```
L̂_aio(s) = r̄₁(s) · r̄₂(s),    r̄_g = (1/N_g) Σ_{i∈g} r(s, εᵢ)
```

Independence of the groups gives exactly

```
E[r̄₁ · r̄₂] = E[r̄₁] · E[r̄₂] = (E[r])²
```

with no variance term — unbiased for any N ≥ 2. The gradient is unbiased
too: `∇(r̄₁r̄₂) = r̄₂∇r̄₁ + r̄₁∇r̄₂` has expectation `2(E[r])∇(E[r]) = ∇(E[r])²`
(each factor independent of the other's gradient).

Implementation notes (`training/loss.py`):

- The two groups are drawn from **separate split keys**, each internally
  antithetic. A single antithetic stream cut in half would correlate the
  halves (mirrored pairs) and reintroduce bias — this is why the group
  split happens at sampling time, not at aggregation time.
- `mc_samples` is split `n1 = N - N//2`, `n2 = N//2`, so total residual
  evaluations (and JIT cost) match the mse estimator at equal `mc_samples`.
- The per-equation losses `mean_b(r̄₁ r̄₂)` can be **transiently negative**
  (sampling noise around a non-negative population value). This is correct,
  not a bug. It does mean `loss_reweight` schemes that assume positive
  losses (`lr_annealing`, `relobralo`) should not be combined with aio;
  the config description says so.
- Quadrature and discrete-chain expectations weight nodes exactly — there
  is no stochastic aggregation bias for AiO to remove, so
  `loss_choice='aio'` with those paths is rejected (config-level and again
  at `compute_loss` level for direct callers).

### Two-stage (combine_fn) path

For models whose residual wraps the expectation in a nonlinearity
(`r = c(E[g])`, e.g. Fischer–Burmeister on an intertemporal Euler), the mse
path squares `c(ĝ)` where `ĝ` is the sample mean of the inside terms. Two
distinct biases stack:

1. **Squaring bias** `Var(c(ĝ))` — same mechanism as above.
2. **Jensen bias** `E[c(ĝ)] ≠ c(E[g])` from the nonlinear combine — `O(1/N)`.

AiO applies `combine_fn` to each group's inside-mean separately and takes
the product `c(ĝ₁)·c(ĝ₂)`. This removes (1) exactly:
`E[c(ĝ₁)c(ĝ₂)] = (E[c(ĝ)])²` by independence. Bias (2) remains `O(1/N)` on
both estimators (now with group size N/2, so the constant is ~2× the mse
path's). For a *linear* combine, AiO is exactly unbiased — that case is the
wiring test in `tests/test_aio_loss.py::test_aio_two_stage_unbiased`.

## Variance trade-off

Unbiasedness is not free: near convergence, `E[r] ≈ 0` and the mse
estimator's value concentrates at the (positive) bias floor, while the aio
estimator fluctuates around zero with both signs. Per-step gradient variance
is comparable (both are dominated by the same `∇r̄` factors), but the aio
loss *curve* is noisier to read. Use the eval-time quadrature loss for
monitoring, not the training loss.

## When to use

- **mse (default):** quadrature/discrete expectations (exact anyway), or
  models near the degenerate pointwise-zero case, or when comparing against
  legacy runs.
- **aio:** MC expectations with small `mc_samples` on models with
  irreducible shock variance at the optimum — multi-shock models where
  quadrature is unaffordable (the disaster model: 5 shocks, MC-only) are
  the motivating case.

## Experiment: Brock–Mirman head-to-head (`scripts/aio_head_to_head.py`)

Design (canonical calibration δ=0.1 unless noted):

- **Ground truth** by time iteration on a (log k, z) grid with Gauss–Hermite
  quadrature, vectorized bisection on the Euler equation per grid point.
  Solver validated against the δ=1 closed form (`sav_rate ≡ αβ`).
- **Part A (training-free bias floor):** evaluate both estimators *at the
  true policy* on ergodic states. Theory: mse mean = bias floor ∝ 1/N;
  aio mean ≈ (E[r])² ≈ 0 (up to solver interpolation error, measured via
  the 64-node quadrature loss at the same policy).
- **Part B (training):** reference recipe (configs/brock_mirman.yaml),
  `loss_choice ∈ {mse, aio}` × `mc_samples ∈ {2, 8}` × seeds {0,1,2};
  report mean/max `|sav_net − sav_true|` on the training rect and the
  ergodic set.
- **Part C (degenerate-case sanity, δ=1):** both estimators must reach
  `sav_rate = αβ`; theory predicts *no* aio advantage here.

### Results (run 2026-06-10, fp64, 4000 keys, 5 seeds/cell)

**Ground truth.** EGM solver converges to 1e-12; validated to **3.3e-13**
against the δ=1 closed form (after switching to log-c interpolation in
(log k, z) — exactly bilinear at δ=1; a direct bilinear on c carries
~1e-4 e^z-curvature error, and a k'-grid floor above the lowest optimal
k' silently poisons the whole EGM fixed point). Quadrature loss at the
solved policy: (E[r])² ≈ 1.7e-10.

**Part A — the bias is exactly as derived.** At the true policy,
ergodic states:

| N | mse bias | aio bias (±se) |
|---|----------|----------------|
| 2 | 1.84e-09 | −1.2e-08 (±4.4e-08) |
| 4 | 9.22e-10 | −0.9e-12 (±1.0e-12) |
| 8 | 4.61e-10 | −2.3e-12 (±0.5e-12) |
| 16 | 2.29e-10 | −2.0e-12 (±0.3e-12) |
| 32 | 1.14e-10 | −1.7e-12 (±0.2e-12) |

mse bias halves exactly per doubling of N (pure 1/N) and at N=2 is
**10.7× the true loss** — at small mc_samples, most of what the
optimizer sees at the optimum is bias. AiO is unbiased to ~1% of the
exact loss (the residual ~−2e-12 offset is GH64 quadrature error on the
piecewise-bilinear interpolated policy — it shows up identically in the
extrapolated mse intercept). **Caveat: aio at N=2 is unbiased but very
noisy** (single-draw groups lose antithetic pairing; the product of two
raw draws is heavy-tailed). Use N≥4 so each group is an antithetic pair.

**Distortion probes — the bias barely moves the argmin on this model.**
Measuring the biased objective's slope `B'` and the true loss curvature
`L''` along policy perturbation directions at the optimum (N=2):

- level shift `s+d`: B' ≈ 2e-10, L'' ≈ 4.6e-2 → argmin shift **4e-9**
  (the bias is a near-constant vertical offset in this direction);
- z-tilt `s+d·z`: B' ≈ 5.7e-8 (asymmetric — negative tilt is rewarded),
  L'' ≈ 1.3e-3 → argmin shift **4.4e-5** tilt units ≈ 1e-5 sav units
  over the ergodic z-range.

So the predicted training-visible distortion is ~1e-5 — two orders below
the reference recipe's achievable optimization error (~2e-3).

**Part B — null, as the probes predict.** 20 runs (mse/aio × N∈{2,8} ×
5 seeds, 20001 episodes): conditional on convergence, mean ergodic
|Δsav| ∈ [1.6e-3, 4.1e-3] across all four cells, no significant
separation (n=3/cell). AiO does no harm: same convergence quality, and
its transiently negative losses do not destabilize Adam. Estimator-
independent side observation: 2/5 seeds per cell stall at |Δsav|≈0.3
under fp64; an fp32 control at the reference recipe (N=5) converges
5/5 (different PRNG stream, so precision-vs-luck is not cleanly
attributed) — pre-existing recipe brittleness, not an estimator effect.
Notably the fp32-converged runs are *less* accurate (SS sav error up to
1.5e-2 vs 2e-3 under fp64).

**Part C — degenerate case behaves as theorized.** δ=1 (pointwise-zero
residual at the optimum): mse 5.7e-2 vs aio 5.4e-2 mean |Δsav| — no
difference, as predicted.

### Conclusion

The mse estimator's bias is real, exactly Var(r̄)/N, and dominates the
loss signal near the optimum at small N — but on Brock–Mirman it is
benign: locally almost a constant offset, with argmin distortion ~1e-5,
invisible under optimization error. The motivating target remains
models where the policy strongly modulates residual shock-sensitivity
and conditional variance is large: the disaster model (5 shocks,
MC-only, jump exposure, ZLB curvature) and FB-constrained models.

**Cheap next step, no ground truth needed:** at any fixed policy,
`mean(mse loss) − mean(aio loss)` over keys estimates the bias floor
Var(r̄)/N directly. Running that diagnostic at a trained disaster-model
policy measures whether the disaster loss is bias-dominated at its
operating mc_samples — before investing in any retraining.

### Disaster verdict (2026-06-10, `scripts/aio_bias_floor_diagnostic.py`)

Run at a fresh canonical checkpoint (configs/disaster.yaml, p_disaster=0,
trained on the DGX, best composite loss 1.0e-1 @ ep 1449; diagnostic on
512 of its own ergodic states, 2000 keys/cell):

- **Quadrature is adequate at the operating q=3** (3^5=243 nodes): the
  GH2→GH5 ladder moves per-equation values by ≲1e-4 *relative*. The
  canonical recipe has no stochastic bias at all (quadrature path) and
  negligible truncation. Estimator error is NOT the disaster model's
  problem at this operating point.
- **Under MC the bias floor is visible but currently subdominant**:
  at N=2 the worst equations (eq4b Kw-recursion, eq7 investment Euler)
  carry bias ≈ 4.8e-5 against levels ~1.3-4.5e-3 (ratio ~0.01-0.04;
  TOTAL ratio 0.4%). Forward-looking implication: an MC-trained
  disaster run at N=2 hits bias-dominance once those equations reach
  (E[r])² ~ 5e-5 — only ~25× below current levels. If MC is ever needed
  (e.g. shock count growth making quadrature unaffordable), use aio.
- The five deterministic identities (eq1, eq2a/b is partly stochastic,
  eq2a, eq4a, eq9) have exactly zero estimator spread, as expected —
  their residuals don't depend on the shock.
- **Caveat:** p_disaster=0 and no ZLB binding in this run. Disaster
  jumps + ZLB kinks raise residual curvature, which degrades *both*
  GH3 truncation and the MC bias floor — re-run the same diagnostic at
  a p>0/ZLB checkpoint before trusting q=3 there.

Conclusion for the disaster convergence stall: the expectation
estimator is exonerated at the canonical operating point; the stall
(~1e-3 per-eq squared residuals) must come from elsewhere
(optimization landscape, architecture, fixed-point selection).

Artifacts: `docs/dev/figures/aio_head_to_head_results.json`,
`aio_bias_floor.png`, `aio_trained_policy_error.png`.

## References

- Maliar, L., Maliar, S., Winant, P. (2021). Deep learning for solving
  dynamic economic models. *Journal of Monetary Economics*, 122, 76–101.
  (All-in-one expectation operator.)
- Judd, K., Maliar, L., Maliar, S. (2011). Numerically stable and accurate
  stochastic simulation approaches for solving dynamic economic models.
  *Quantitative Economics* 2(2). (Background on simulation-based accuracy.)
