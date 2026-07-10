# Disaster certification experiment — night report, 2026-07-06/07

**Question:** does the kink-aware treatment (gated anchor and/or ELB-targeted coverage) give the
disasterless model the certificates irbc earned on 2026-07-06 — closed-loop stability
(ρ(SS) < 1), steady-state level consistency, dying drift, and seed agreement?

**Status:** three of four arms complete; the composed arm was running at the time of writing —
its section is marked and will be updated. Everything below is measured, fp64, with the in-repo
probe (`scripts/disaster_ss_probe.py`); raw checkpoints under `runs/disaster_cert/` on the DGX.

## Design

Four arms × 3 seeds × the full 3000-episode production recipe (`configs/disaster.yaml`
derivatives; GPU container, ~16 min/run, coverage arms ~3× slower):

| arm | config | lever |
|---|---|---|
| baseline | `disaster.yaml` | BK anchor as shipped (teaches the no-floor linearization everywhere) |
| gated | `disaster_gated.yaml` | + `composite_loss.anchor_gate`: anchor muted where the linearized Taylor rate ≤ floor (21/128 anchor points, matching the audit's ~12% floor mass) |
| elbcov | `disaster_elbcov.yaml` | + EWM coverage: stress seeds in the floor region (m_p/R_lag/pi_lag box), rolled 5 steps, repaired; ρ_stress = 0.5 |
| gated+elbcov | `disaster_gated_elbcov.yaml` | both |

Certificates per (arm, seed), all at the final checkpoint: ρ(SS) (spectral radius of the
closed-loop map at SS), max |policy(SS) − policy\*| / |policy\*|, zero-shock drift from SS at
horizons 5–100. Knobs fixed a priori; seeds 0–2 fixed a priori; failures reported.

**Context (the bar):** the model's math is verified clean (2026-07-06 tex audit); the failure
under study is training-side. Pre-experiment baseline at HEAD (1500 eps): ρ ≈ 1.1–1.22, SS
levels off up to 13.8%, worst errors in the monetary block — consistent with the ELB-conflict
hypothesis (the anchor teaches a linearization that is blind to a kink the economy visits ~12%
of quarters).

## FINAL RESULT (consistent protocol) — supersedes the per-arm narratives below

**Protocol correction, and it is the headline.** The per-arm sections below were written live
as runs finished and MIX CHECKPOINT CONVENTIONS: early baseline probes used
`checkpoint_best.eqx`, later arms `checkpoint_003000.eqx`. The table here is the single-command,
single-convention record (final checkpoints, all arms, fp64;
`runs/disaster_cert/probe_final.json`):

| arm (final ckpt, 3 seeds) | ρ(SS) per seed | pass ρ<1 | max SS err per seed |
|---|---|---|---|
| baseline | 1.057 / 1.021 / 1.023 | 0/3 | 1.1% / 2.8% / 0.7% |
| gated | 1.049 / 1.064 / 1.052 | 0/3 | 2.3% / 2.5% / 2.6% |
| elbcov (1 seed) | 1.060 | 0/1 | 7.4% |
| gated+elbcov | 1.064 / 1.148 / 1.293 | 0/3 | 0.9% / 1.3% / 6.2% |
| gated+drift | 1.044 / 1.107 / 1.033 | 0/3 | 1.5% / 2.0% / 1.6% |
| gated+rsob w=1 (day 2) | 1.052 / 1.278 / 1.122 | 0/3 | 1.6% / 34.3% / 2.2% |
| gated+drift+rsob (day 3) | 1.238 / 1.268 / 1.204 | 0/3 | 9.1% / 4.0% / 20.5% |
| gated+rsob w=25 (day 3) | 1.020 / **0.989**† / 1.105 | 1/3† | 2.9% / **49.7%**† / 1.8% |
| gated+pcgrad (day 3) | **0.987** / 1.027 / 1.022 | **1/3** | **0.29%** / 4.1% / 7.0% |

† rsob25 s1's ρ<1 is stability of the WRONG economy: the rest point is displaced 49.7%
(drift@100 converges to the displaced point). It fails the SS-consistency certificate and
does not count as a pass of the stack. pcgrad s0's corrected reading (see "Referee
corrections" below): max learned-block eigenvalue 0.975 — the program's only learned
spectrum inside the unit circle near the truth — converging to a fixed point displaced
0.83% from the true SS. The probe's ρ column has a floor at 0.987 (the mu_ups exogenous
root); values at that floor mean "learned block below the exogenous ceiling," not a
learned 0.987.

**What actually survives, ranked by robustness:**

1. **15/15 runs, five treatments, zero crossings of ρ(SS) < 1.** The ~1.02–1.06 mildly
   unstable attractor is reached by the *baseline itself* at final checkpoints; treatments at
   best match it and at worst (v1 coverage) scatter upward. Selection on this model is not a
   sampling, anchor-placement, or (at this calibration) direct-spectral-pressure problem —
   at minimum it needs a longer drift horizon; possibly it is a genuine conflict between the
   residual objective and the stability certificate.
2. **Checkpoint selection is a larger certificate lever than any loss treatment tried
   tonight.** Best-by-loss checkpoints are systematically certificate-worse than final ones
   (baseline s0: ρ 1.22 at best vs 1.057 at final). `save_best_checkpoint` selects AGAINST
   the certificates. Immediate cheap fix: certificate-aware checkpoint selection (probe ρ and
   SS error at every save; keep the certificate-best).
3. **The drift term moves what it sees** (within-run, convention-immune): drift@20 halved
   (23–27% vs 40–50% gate-only). Its horizon (T=20) sees non-normal transient, not the
   asymptotic radius — successor calibration T=50–100, weight 5, target 0.97.
4. **The π-wall measure-migration is real** (training-log evidence): v1 ELB coverage sent the
   on-policy sampler into a 2500-episode deflationary transient (bound-attractor,
   zombie-paths family), and composed arms inherit elevated seed variance from it.
5. The earlier "gate causally moves ρ 1.22→1.05" claim was a convention artifact (best vs
   final checkpoints). What the gate demonstrably did NOT do is hurt anything; its 21/128
   down-weighting matches the audit's floor mass and its arm is the tightest-variance treated
   row — but the baseline's own final checkpoints land in the same basin.

**The method lesson, self-inflicted and kept:** three hours of live per-arm narrative built a
causal story that one consistent-protocol table dissolved. Certificates need frozen conventions
*before* the experiment — the probe script now defaults to final checkpoints for exactly this
reason.

## Day 2 (07-07 afternoon): the objective indicted, the war measured

The night ended with "the next lever must change what the loss can see." Day 2 sharpened that
into an elimination proof and then a measurement.

**Elimination ladder (why the objective, not the walk):** Huber reweighs the same residual
values — same argmin family, no new selection information. Second-order methods change the
walk, not the map (and the sweep_so precedent already showed no basin escape). μP/init is
eliminated by the strongest evidence we own: `init_scale: 0.0` **starts training AT the
linearized solution** and the optimizer walks *out* to ρ≈1.05 — when you start at the answer
and leave, the objective is indicted, not the starting point.

**Residual-Sobolev, weight 1 (the first "change what the loss sees" arm):** implemented as
`aux_res_sobolev` — directional derivatives of the per-state *expected* residual via JVPs
along fixed ergodic directions; the true policy zeroes E[r] on a neighborhood, impostors keep
values small with finite gradients. The impostor signature is REAL and measured: aux floor
4.7e-3 at the trained basin. But at weight 1.0 it is ~7% of the loss — a regularizer, not a
selector. Certificates above (s0 1.052 = unchanged basin; s1 1.278 with 34% SS displacement;
s2 1.122): 0/3, median ρ *above* the gate-only arm — at this dose the term adds gradient
noise without changing selection. Escalation arm `disaster_gated_rsob25` (weight 25,
8 directions — knobs fixed a priori before these probes) is queued.

**The gradient-conflict diagnostic (the afternoon's headline).** If the ρ≈1.05 basin were a
joint zero of all 11 equations, per-equation gradients there would all be ≈0. Measured at
`disaster_gated_s0/checkpoint_003000` (64 ergodic states, 3-node GH quadrature, fp64):

- total gradient norm **3.6** — the basin is not calm;
- per-equation gradient norms 0.07–1.49 (largest: investment Euler 1.49, wage-Phillips
  recursion 0.96, entrepreneur contract 0.83);
- pairwise cosines: **price-Phillips block vs wage-Phillips block −0.89 to −0.92**
  (eq1×eq3 −0.89, eq2b×eq3 −0.92, eq3×eq4a −0.92), with side conflicts against the bond
  Euler and BGG contract at −0.30 to −0.44.

**The basin is a COMPROMISE POINT, not a solution** — a frozen tug-of-war between the price-
and wage-setting blocks. The scalar MSE sums these opposing pulls to near-zero and calls it
converged. This is the sharpest evidence yet for the night's conclusion: the residual
objective *aggregation* (sum over equations) is what cannot see the difference between truce
and truth.

**Spec-let 6, implemented (commit eec06c5):** per-equation gradient surgery that carries the
certified aux stack. PCGrad with a composite loss now projects only the 11 core equation
gradients against each other (at the base mean-over-equations scale) and adds the exact
auxiliary gradient grad(total) − grad(base) unprojected — anchor, gate, Jacobian, barriers,
Newton, drift, rsob all reach the update. Two exact invariants are unit-tested: with no
conflicts the step equals the plain composite gradient step (delta vs `disaster_gated` is the
surgery alone), and under total conflict only the aux gradient moves. A/B smoke on the real
model shows early transients within the family's normal envelope (baseline control was
*larger*). Arm `disaster_gated_pcgrad` ×3 seeds queued behind rsob25.

**Day-2 queue** (DONE-resumable container relaunch after drift_rsob completes):
`disaster_gated_rsob` s2 → `disaster_gated_drift_rsob` ×3 → `disaster_gated_rsob25` ×3 →
`disaster_gated_pcgrad` ×3. Rows to be appended to the table above at the frozen convention.

**Day-3 addendum (07-09): drift+rsob is additive noise.** The stacked arm (drift T=20 +
rsob w=1, both underdosed) came back 0/3 at ρ = 1.238/1.268/1.204 with SS errors up to 20% —
*worse than either term alone and worse than gate-only*. Two weak selection pressures do not
compose into a strong one; they compose into variance. This sharpens the escalation logic:
the live question is whether a *dominant* dose (rsob25) or a *different aggregation*
(pcgrad surgery) moves the attractor, not whether more mild auxiliaries help. Both arms
launched 07-09 (container log `logs/cert_container_day3.log`).

## Day-3 endgame (07-10): the first crossing — and it's the surgery

> **[Superseded in part by "Referee corrections" below — written before the adversarial
> review. "Fully certified" and "contracts to the true rest point" are retracted there;
> the crossing itself survives in corrected form.]**

27/27 runs, program complete (`runs/disaster_cert/probe_day3.json`). The two escalation arms
answer the night's question — *can anything move the attractor?* — in opposite ways:

1. **`disaster_gated_pcgrad` s0 is the first fully-certified disaster solution in the
   program**: ρ(SS) = 0.987 with the steady state reproduced to 0.29% and drift@100 = 0.82%.
   Two independent certificates (spectral radius at SS; 100-period closed-loop rollout) agree
   the loop contracts to the *true* rest point. For scale: every one of the other 26 runs has
   drift@100 between 49% and 529%. Seeds 1–2 land basin-typical (1.027/1.022) — the surgery
   does not abolish the lottery (it removes the *compromise* pressure, not the multiplicity),
   but it is the only treatment in three days that ever crossed. Mechanism consistency: the
   diagnostic said the basin is a price-vs-wage compromise point (cosines −0.9); removing
   conflicting gradient components is exactly the intervention that should sometimes let the
   optimizer leave — and it did.
2. **`disaster_gated_rsob25` prices the naive stability trade**: its ρ<1 seed (s1, 0.989)
   sits at a rest point displaced 49.7% from the model's SS — the dominant Sobolev dose
   selected a *stable wrong economy*. s0/s2 (1.020/1.105, small SS error) show the dose also
   drags ρ down when it stays home, but not through 1. The derivative signal selects for
   flatness, not for *the truth's* flatness — it needed the anchor to hold the rest point,
   and at weight 25 it overpowers exactly that anchor.

**Program verdict, 27 runs, seven treatments:** selection on this model is an *aggregation*
problem before it is a sampling or dose problem. The only crossing came from changing how the
11 equations' disagreements combine into one direction (PCGrad), precisely as the
compromise-point diagnostic predicted. Immediate next levers, in order of cheapness: (a)
certificate-aware checkpoint selection on the pcgrad arm (s1/s2 may pass mid-training — the
best-vs-final lesson says the walk visits better policies than it keeps); (b) pcgrad × more
seeds (is s0 a 1/3 or a 1/10 event?); (c) pcgrad + rsob at *moderate* weight (surgery removes
the tug-of-war, Sobolev then selects among the survivors — composition, as on irbc).

## Referee corrections (07-10): what the winner actually is

An external adversarial static review (Codex, prompt designed to refute) produced 8 findings.
Two attacked the headline directly; both were **confirmed** by decisive checks the same day,
and both leave a corrected result standing. Checks: full eigendecomposition with an exact
autonomous/learned block split of the closed-loop Jacobian, Newton solve for the learned
fixed point ŝ = T(ŝ), and 1000-period zero-shock rollouts (fp64, final checkpoints).

**Correction 1 — the probe's ρ has a floor at 0.987, and the winner's headline number was
that floor, not a learned quantity.** The closed-loop Jacobian splits exactly into an
autonomous exogenous block — identical across all checkpoints, AR roots
{0.809, **0.98699** (mu_ups × soft-clip derivative), 0.940, 0.146, 0} — and the learned 8×8
endogenous block. The probe reports the max over both, so no run can ever read below 0.987,
and "ρ(SS)=0.987" means only "learned block at or below the ceiling." The corrected metric is
the **max learned-block eigenvalue**:

| checkpoint | max learned |λ| at s* | at ŝ | ŝ displacement | ρ(ŝ) stable? |
|---|---|---|---|---|
| pcgrad s0 | **0.9750** | **0.9754** | **0.83%** (L_lag) | yes |
| baseline s1 (the night's "0.987") | 1.0209 | 1.0248 | 10.2% | no → 529% attractor |
| gated s0 | 1.0492 | 1.0586 | 3.1% | no → 529% attractor |
| rsob25 s1 (by its probe value 0.9893 > floor) | 0.989 (learned) | — | 49.7% | yes (wrong economy) |

The crossing is therefore REAL and learned — 0.975 with genuine margin — and was *masked*,
not manufactured, by the exogenous ceiling. The night table's dismissal of baseline s1's
0.987 as "the exogenous root" is likewise confirmed (its learned block is 1.021).

**Correction 2 — "contracts to the true rest point" is retracted.** The drift trajectory
(which starts exactly at s\*) converges to the *learned* fixed point ŝ: Newton residual
7e-15, displacement **0.827%** from the true SS (worst dim: leverage lag; then q_lag 0.36%),
and the 1000-period rollout lands on ŝ to 7e-14. So pcgrad s0 is **a locally stable economy
0.83% away from the truth** — the rsob25-s1 phenomenon at 60× smaller displacement — not the
true equilibrium. The corrected hierarchy: it is the only run in the program whose learned
dynamics have a stable fixed point anywhere near the truth (baseline/gated fixed points are
3–10% displaced AND unstable; their trajectories escape to the 529% soft-clip attractor).

**Corrected claim, in one sentence:** the surgery arm produced the program's first
*stable-near-truth* solution (learned spectrum 0.975, rest point 0.83% off, policy levels
0.29% off at s\*) — "fully certified true equilibrium" is withdrawn, and closing the last
0.83% is now a concrete, bounded target rather than an existence question.

**Standing (unresolved) findings from the review:** causal attribution to PCGrad remains
unestablished — 1/3 vs 0/24, one-sided Fisher p ≈ 0.11, with Adam/clip norm-confounding
uncontrolled; the referee's demanded controls (preregistered matched seed pairs; a sham
standard arm rescaled to the surgery's realized update norm; LR/clip sweeps) are the required
next experiment before any mechanism claim. The certified calibration is the **disasterless**
one (p_disaster = 0) — as this report's opening says, but the word "disaster solution" must
carry that qualifier everywhere. The operator is the one-shot simultaneous PCGrad variant,
not the paper's sequential procedure (docstring corrected; for >2 tasks it is best read as
data-dependent gradient reweighting).

**Provenance corrections:** 27 DONE markers; 25 runs have final checkpoints — `disaster_elbcov`
s1/s2 carry DONE but no checkpoint files (night-of-07-07 cut/restart casualties) and appear in
no table. "Seven treatments" undercounted: nine arms. The sibling drift range is 130–529%
plus rsob25 s1's 49.7% (which is its displaced rest point, not dying drift). Validator now
rejects pcgrad×composite with non-unit uniform weights (a hole the review found in the
no-conflict equivalence). Certificate definition upgraded for successors: report the max
*learned-block* eigenvalue at ŝ, ‖ŝ − s\*‖, policy levels at s\*, and long-horizon
convergence — the probe script should compute ŝ and the block split natively.

## Risky steady state (07-10): the 0.83% is error, not economics

The last open reading of pcgrad s0's displaced rest point was the charitable one: the network
is trained on the *stochastic* model, whose true zero-shock rest point is the **risky steady
state** — legitimately displaced from the deterministic s\* by precautionary effects the
certainty-equivalent anchor cannot express. Measured (`scripts/disaster_risky_ss.py`):
CRW-style risky SS — rest point under zero realized shocks with equations holding in
expectation over future shocks, first-order (BK) rules for future behavior, 3⁵ Gauss-Hermite
nodes, damped Newton in fp64. The deterministic re-solve of the same system provides the
baseline, isolating pure risk from the machinery floor (soft-clip + linear-future-rule wedge,
≤0.05%/dim).

Result: **the pure Gaussian risky-SS shift at this calibration is ≤0.10% per state**
(L_lag +0.100%, q_lag −0.057%, all else <0.03%; on the policy side the Calvo pricing
recursions K_w/F_w/K_p/F_p carry the largest shifts, +0.02–0.07% — risk bites where the
curvature lives). The network's rest point is displaced **8× more than that in the OPPOSITE
direction** (L_lag −0.827% vs +0.100%; ratios −2 to −11 across endogenous states).

Conclusions: (1) pcgrad s0's 0.83% displacement is genuine approximation error, not a risk
premium — the charitable reading is dead by measurement. (2) The certainty-equivalent anchor
is nearly free at this calibration (the correction it forbids is ~0.1%), so anchoring hard at
s\* is validated. (3) Referencing certificates to the deterministic s\* is justified here.
(4) The reusable artifact is the solver itself: at spec-let 4's real disaster calibration
(p = 1%, θ = 15%) the risky SS should move materially, and the interesting question becomes
whether a DEQN net can capture a disaster premium no linear anchor can express. Scope:
Gaussian risk only (p_disaster = 0), first-order future rules (policy-curvature part of the
true risky SS not captured), 3-node GH per dim.

## Results (live per-arm narratives — superseded above, kept for the record)

### Baseline: the lottery, quantified (3/3 complete)

| seed | ρ(SS) | max SS err | drift@100 |
|---|---|---|---|
| 0 | 1.220 | 4.17% | 50% |
| 1 | 0.987 | 2.01% | 15% |
| 2 | 1.225 | 0.88% | 529% |

**1/3 stable; median ρ = 1.22; no two seeds fail the same way.** Seed 1's ρ = 0.987 is the
exogenous ρ_μ_ups root — i.e. an endogenously stable draw — yet its levels are 2% off and it
drifts to a displaced rest point. Seed 2 is near-perfect *at* the SS and violently unstable
*around* it. Levels and stability are separable failures, visible in one table. Additional
recurring observation: `checkpoint_best` (lowest training loss) is consistently *worse* on
certificates than the final checkpoint — training loss is not a certificate, measured again.

### Gated anchor: the lottery closes — on the wrong number (3/3 complete)

| seed | ρ(SS) | max SS err | drift@100 |
|---|---|---|---|
| 0 | 1.049 | 2.26% | 47% |
| 1 | 1.064 | 2.54% | 51% |
| 2 | 1.052 | 2.59% | 50% |

Three findings in one small table:

1. **Causal, matched-seed stability shift**: seed 0 differs from baseline seed 0 by the gate
   alone; ρ moves 1.220 → 1.049. The anchor's floor-region lies were genuinely destabilizing.
2. **Variance collapse**: baseline seeds scatter (1.22 / 0.99 / 1.22); gated seeds agree to two
   decimals — *including* seed 2, whose training had the wildest gradient spikes of the night.
   This is the same lottery-abolition signature the anchor produced on irbc.
3. **The shared basin sits above 1** (ρ ≈ 1.055). Muting the bad teacher makes the outcome
   deterministic but leaves the floor region taught by nobody; the residual loss alone there is
   exactly the self-confirming trap. Determinism without correctness.

### ELB coverage: the training measure chases the policy — then comes back (1 seed)

**Correction over the first draft of this section:** the training loss sat at ~4×10⁵ from
episode 500 through ~2500 and this was initially read as divergence — seeds 1–2 were cut on
that reading (a premature call, on the record: the DONE markers say so). In the final ~500
episodes the run *recovered*: loss 4×10⁵ → 918 → **0.025 — the best final loss of any arm
tonight**. The mid-run diagnosis remains valid and is the night's most instructive result:

- **The stress pool is innocent.** The checkpoint's residuals *on its own stress landings* are
  fine (total 0.017; worst equation 7.6e-2).
- **The landings tell the story:** every rolled landing has `pi_lag` pinned at exactly
  **0.95 — the policy's own lower bound**. Under the coverage-trained policy, rolling five
  steps from any floor-region seed deflates the economy to the π-wall.
- Therefore the 4×10⁵ is the **base pool** — the policy's own on-policy trajectory. Coverage
  taught floor-region behavior; the policy's simulated economy then *migrated to the floor*;
  the on-policy training measure followed it into a deflationary spiral pinned at the π bound.

**This is a known villain in a new costume.** The 2026-06-10 zombie-paths finding was 55/64
training paths absorbed at the soft-clip ceiling; tonight the absorption is at the π lower
bound, with coverage pressure as the driver. General form: **on this model, policy bounds are
attractors of the on-policy training measure** — any training pressure that pushes the policy
toward a bound can drag the sampling distribution with it, and the loss explodes on the
migrated measure. On irbc this could not happen: its kink is off-path, so coverage taught a
region the base measure never followed it into. The disaster model's kink is on-path — the
measure *can* follow. The eventual escape (final 500 episodes) shows the spiral is a long
transient rather than a terminal absorption under this calibration — but 2500 of 3000 episodes
were spent inside it.

**Final certificates (the punchline):** despite the best final loss of the night,

| seed | final loss | ρ(SS) | max SS err | drift@100 |
|---|---|---|---|---|
| 0 | **0.025** | 1.060 | **7.41%** (λ_z +7.4, π −6.2) | 53% |

Best loss, worst treated-arm levels — the most dramatic loss-is-not-a-certificate instance of
the night, in an experiment that had already demonstrated it twice.

### Gated + coverage (composed): s0 shone; s1–2 reopened the lottery (3/3 complete)

| seed | ρ(SS) | max SS err | drift@5 | drift@100 |
|---|---|---|---|---|
| 0 | 1.064 | **0.89%** (λ_z −0.9, i +0.8) | 4.4% | 529% |
| 1 | 1.148 | 1.31% | — | 529% |
| 2 | 1.293 | 6.15% | — | 529% |

**Amendment (seeds 1–2, post-first-draft):** the composed arm does NOT inherit the gate's
variance collapse. With the v1 coverage block on top, seed spread returns
(1.06 / 1.15 / 1.29 vs gate-only 1.049 / 1.052 / 1.064) — the measure-migration transient
interacts with the seed, and s2 lands nearly at baseline-grade instability. s0's headline
("best levels of the night") stands but was partly seed luck. **The gate-only arm remains the
best-behaved treatment of the experiment.**

What survives from the s0-era reading, and what doesn't:

1. **Still true — the gate tames the coverage transient on s0**: same π-wall spiral entry
   (2.7×10⁵ at ep 400), escape by ep 900 vs ~2000 episodes trapped ungated.
2. **Retracted — "the levers compose on levels"**: s0's sub-1% was not reproduced (1.31%,
   6.15% on s1/s2). Levels under composition are seed-dependent.
3. **Revised — the "shared basin ±0.008" was an artifact of which runs had finished**: it
   holds exactly for the gate-only arm (1.049–1.064) and the first seeds of the coverage
   arms, but composed s1/s2 sit at 1.15 and 1.29. The invariant across all **fourteen** runs
   of the experiment is weaker and more important: **no run, under any treatment, has crossed
   ρ(SS) < 1** (sole exception: baseline s1's 0.987 — the exogenous root — which still fails
   the drift certificate). The attractor is never crossed by anchor-placement or sampling
   interventions; where those interventions misbehave (v1 coverage), outcomes scatter *upward*
   only.

## Verdict so far, against the hypothesis

The ELB-conflict mechanism is **half-confirmed, precisely**:

- Confirmed: the anchor's floor-region teaching is causally destabilizing (matched-seed ρ
  1.22 → 1.05) and is a source of the seed lottery (variance collapse when removed).
- Refuted (v1 form): "just add residual pressure at the floor" — the paper-faithful coverage
  design that worked on irbc's off-path kink destabilizes the *training measure* when the kink
  is on-path (a 2500-episode bound-attractor transient), and even after recovering to the best
  loss of the night, its certificates are the worst of the treated arms.

**The pattern above the arms — now confirmed five times:** every treated run of the night
lands at ρ(SS) = 1.057 ± 0.008 — gated (1.049/1.064/1.052), coverage (1.060), composed
(1.064). The treatments decisively control the outcome's *variance* (lottery abolished), its
*levels* (composed: sub-1%), and its *transients* (gate cuts the π-wall spiral ~4×) — and do
not move the attractor's *location* at all. That is the signature of the loss surface itself
preferring one specific mildly-unstable solution: the disaster-model analogue of irbc's
identification finding, a solution the residual objective barely distinguishes from the true
one. **Conclusion: on this model, selection is not a sampling problem and not an
anchor-placement problem. The next lever must change what the loss can see** — regime-2 anchor
*content* at the floor (true piecewise dynamics as targets, not silence), or preconditioned /
reparametrized residuals (the identification program, spec-let 3). The night's dissection:
variance ← gate; levels ← composition; transient ← gate; attractor location ← none of the
above, by construction of five independent measurements.

## Successor design (spec for the next cycle)

1. **Gentler coverage**: ρ_stress 0.5 → ~0.1; seeds on `m_p` only (pure monetary shock, no
   compound deflation seeds); shorter rollout (H = 2–3); consider freezing the base measure
   (`initialize_each_episode: true`) for coverage arms so the sampler cannot chase the policy.
2. **A true teacher at the floor, not just pressure**: the gate creates a taught-by-nobody
   region; the principled filler is regime-2 *content* — anchor targets from floor-consistent
   dynamics (OccBin-style piecewise local model) rather than residual-only signal. This is the
   full spec-let 1 design the gate approximated.
3. **Bound-attractor guard**: a diagnostic that flags when the trajectory pool's quantiles
   touch policy bounds (the zombie-paths/π-wall family deserves an alarm, not an autopsy).
4. Keep: the certification protocol itself (this table format, 3+ seeds, final checkpoints,
   a-priori knobs) — it resolved three distinct mechanisms in one night.

## Reproduce

```bash
# training (DGX, NGC container):
LAUNCHER=scripts/cert_sweep_container.py ./scripts/run_sweep_in_container.sh
# certificates:
JAX_ENABLE_X64=1 uv run python scripts/disaster_ss_probe.py \
    --runs-dir runs/disaster_cert --json-out runs/disaster_cert/probe.json
```

Artifacts: `runs/disaster_cert/<arm>_s<seed>/` (checkpoints + config), sweep log
`logs/cert_container.log`. Implementation: gate 5585bf8, configs/probe same commit; baseline
HEAD probe in `runs/disaster_head_probe`.
