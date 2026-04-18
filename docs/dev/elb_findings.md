# ELB findings — internal note

*For the disaster-paper collaboration. Not shipped with the package;
gitignored `docs/dev/*` is appropriate if needed.*

## TL;DR

1. The CMR Taylor rule was **unconstrained** below zero before v0.2.0. Disaster training couldn't converge because trajectories drifted into deeply negative nominal rates (R_lag ≤ 0.95 in the failed runs) — a regime the rest of the calibration isn't designed for.
2. Adding a soft-floor on R at R_lb = 1.0 (sharpness = 500) fixes training: all p ∈ [0, 0.1] converge to ~1e-4 residual loss, matching baseline.
3. The cure introduces a *new* phenomenon: tail residuals now concentrate at ZLB-binding states (80% of worst-10 residuals, Spearman correlation +0.498 with R_zlb_binding). This is the known kink-approximation limit from the occasionally-binding-constraints literature.
4. The ELB-binding ergodic moments match theory: +2.2% precautionary capital, −4-6% on Calvo aggregators, R_lag min exactly at 1.0.
5. Default R_lb = 1.0 is the textbook ZLB. Realistic central-bank behaviour (SNB held −0.75% for seven years; ECB, BoJ, Riksbank etc. similar) suggests R_lb ≈ 0.998 per quarter is empirically honest.

## Where the ELB lives in the code

- `models/disaster/equations.py` — `definitions()` computes `R_taylor` per the Taylor rule, then applies `_soft_floor(R_taylor, R_lb, sharpness=R_lb_sharpness)`.
- `models/disaster/variables.py` — `R_lb`, `R_lb_sharpness` added to `CONSTANTS`.
- No other code knows or needs to know about the ELB. Linearization uses `definitions` via the standard path.

The soft-floor is:

$$
R = R_{lb} + \frac{\log(1 + e^{k(R_{\text{taylor}} - R_{lb})})}{k}
$$

With `k = 500`, SS distortion is below 1e-7. Gradient-preserving everywhere, kink-sharpness tunable via `R_lb_sharpness`.

## How training changed

Residual loss on the validated LinearPlusMLP + composite stack, 3000 episodes, all else equal:

| p_disaster | Pre-ZLB (v0.1.0 era) | With ZLB (v0.2.0) |
|------------|---------------------:|------------------:|
| 0 (baseline) | ~5e-4 | **1.14e-4** |
| 0.001 | 4.51 💀 | **1.10e-4** |
| 0.005 | 0.00168* | **1.09e-4** |
| 0.02 | 0.287 | **1.10e-4** |
| 0.05 | 0.463 | **1.20e-4** |
| 0.1 | 0.050 | **1.65e-4** |

\* with deterministic-SS workaround; with risky-SS anchor (intended path), was 1.13

Best-checkpoint tracking (save-best enabled in v0.2.0) adds marginal gains — final-to-best gap is tiny (≤ 0.1e-4), i.e. runs are *stable* all the way through 3000 episodes. Not true before ZLB: runs found good solutions around ep 1500 then got destabilised by a rare gradient event.

## What the ELB does to the solution

`eval` on `checkpoints/disaster_p10_zlb/checkpoint_best.eqx` (2000 simulated periods):

| Variable | SS | Ergodic mean | Dev | Interpretation |
|---|---:|---:|---:|---|
| `k_lag` | 27.35 | 27.95 | **+2.2%** | precautionary capital buffer |
| `K_p` | 4.83 | 4.60 | **-4.8%** | Calvo price aggregator deflates |
| `K_w` | 2.21 | 2.07 | **-6.1%** | wage aggregator deflates more |
| `F_p, F_w` | 4.78, 0.89 | 4.59, 0.85 | -4.0% | forward-looking sums |
| `R_lag` min | — | **1.0000** | — | ZLB binds on the tail |

~16.4% of simulated states have the ZLB active (`R_zlb_binding > 1e-4`). This is a legitimate disaster-risk equilibrium with the ELB properly priced in — agents accumulate buffer capital because they expect monetary policy to be unable to offset future shocks when the floor binds.

## The residual tail is a kink-approximation artefact

**Diagnostic** (post-hoc on 2500 simulated states from the trained p=0.1 policy):

- Mean |residual| ≈ 0.7%, median 0.8%, p99 6.5%, p99.9 8.7%, **max 11%**.
- Worst-10 states: 80% are ZLB-binding (vs 16.4% base rate — 5× enrichment).
- Spearman correlation between |R_zlb_binding| and max residual: **+0.498**.
- Worst states cluster where capital, consumption, wages are all ~80-90th percentile AND inflation/rate are 10-20th percentile — the textbook "liquidity trap" region.
- Dominant equations at the tail: `eq7_investment_euler` (60% of worst-10) and `eq2b/eq4b_Kw_recursion` (30%).

**Mechanism**: the softfloor is smooth (softplus) but sharp (sharpness=500 gives a transition band of ~0.002 in R_taylor). That's effectively a kink: below R_lb + 0.002, `∂R/∂R_taylor ≈ 0`; above, ≈ 1. A smooth `LinearPlusMLP` policy can only approximate a kink up to its finite representational resolution.

Investment Euler feels it strongly because it references R via `λ_z_next`; Kw recursion feels it because wage dynamics are regime-dependent near ZLB.

This is **the documented pathology of smooth-network approximation of kinked policies** in the occasionally-binding-constraints literature (Guerrieri & Iacoviello OccBin, Fernández-Villaverde et al. on ZLB-bound DSGE). Our 11% max-residual is comparable to or better than what this literature reports for ZLB-bound models solved via smooth global methods.

## The three candidate fixes for v0.3.0

Ordered by cost-to-test × theoretical rigour:

### (a) Regime feature — cheapest empirical probe

Add a derived input $\text{zlb\_proximity} = R_{\text{taylor}}(s)/R_{lb} - 1$ as a 14th input to the network. Lets the network condition its policy on regime explicitly. ~30-50 line patch across `networks/`, `compute_residuals`, and anchor-point construction in `composite_loss`.

**Hypothesis**: the tail residual is "the policy *could* represent this if the network knew which regime it's in." If true, residuals drop substantially with essentially zero architectural change.

### (b) ReLU activation — zero code change

YAML flip: `activation: tanh` → `activation: relu`. ReLU networks are piecewise-linear and natively represent kinks. Original DEQN (Scheidegger-Bilionis 2019) used ReLU.

**Hypothesis**: tanh is too smooth for the ZLB kink, ReLU's piecewise-linearity captures it naturally.

**Risk**: dead-neuron pathology; tanh has been our validated stack for the whole disaster block. May require re-tuning init scale.

### (c) KKT formulation — principled, invasive

Add a Lagrange multiplier $\mu \geq 0$ as a 12th policy output. Replace the softfloor with:

$$
R = R_{\text{taylor}} + \mu, \qquad \mu \geq 0, \qquad \mu \cdot (R - R_{lb}) = 0
$$

Network learns μ jointly; complementary-slackness becomes a new residual equation. Matches OccBin convention; gives an *economically meaningful* shadow-price of the ZLB constraint you can plot and interpret.

**Cost**: 12 policies instead of 11, new equation, modifications to `_solve_steady_state` (analytical `_build_state` with μ=0 at SS), risky-SS solver similar. ~1-2 weeks of work.

**Payoff**: correct formulation, paper-defensible, smooth policy surface in μ (no kink), and the network's μ output becomes a diagnostic for "how stuck is monetary policy."

## What to try first

Recommendation: **(b) then (a) then (c)**.

- (b) is a YAML flip. 3 minutes. Probably informative either way.
- (a) is one day of coding.
- (c) is real research scoped for a full v0.3.0 release.

## Open empirical questions (v0.2.1 sensitivity studies)

- **R_lb_sharpness sweep** at fixed p=0.1: sharpness ∈ {50, 100, 250, 500, 1000}. Plot tail residuals vs sharpness. Identify where the SS-distortion / kink-sharpness trade-off is Pareto-optimal.
- **R_lb level sweep**: R_lb ∈ {1.0, 0.998, 0.995, 0.99}. Does matching SNB-style empirical ELB change anything? Likely only marginal — the ZLB region is broad relative to the ±0.5% floor difference.
- **Disaster IRF with ELB binding**: force a disaster at t=1, simulate 40 periods, check whether R hits the floor during recovery (Fisherian debt-deflation amplification is the key disaster-plus-ZLB mechanism in Gourio-adjacent models).

## Literature anchors (for the paper's ELB section)

- Guerrieri & Iacoviello 2015, JME — OccBin, the standard piecewise-linear method for occasionally-binding constraints.
- Fernández-Villaverde, Gordon, Guerrón-Quintana, Rubio-Ramírez 2015, JEDC — ZLB in DSGE with global solution methods.
- Gourio 2012, AER — disaster-risk DSGE; original literature on how disaster risk interacts with monetary policy.
- Wachter 2013, JF — disaster-risk asset pricing; canonical calibration source.
- Christiano, Motto, Rostagno 2014, AER — the base model we extend.

## Status of the ZLB work

- ✅ Soft-floor implementation landed in v0.2.0 (commit `fdfbf2b`).
- ✅ Baseline + 5 disaster calibrations all converge.
- ✅ Ergodic moments show theory-consistent disaster-risk-plus-ELB pricing.
- ✅ Stability check correctness fix (commit `78a9572`).
- 🟡 Tail residuals at ZLB-binding states — known limitation, understood, documented.
- 🟡 Default R_lb = 1.0 vs empirical ~0.998 — defensible as pedagogical/conservative, worth sensitivity analysis.
- 🔲 Kink-aware architecture (regime feature / ReLU / KKT) — v0.3.0 territory, experiments started.

*Document started: 2026-04-17. Update as v0.3.0 experiments conclude.*
