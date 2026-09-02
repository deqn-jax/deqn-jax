# The Selection Program — a chronicle (2026-07-06 → 2026-07-14, codas 2026-08-28, 2026-09-02)

**What this is.** The complete record of five days of research on one question that grew out
of another: *can deqn-jax produce a certified solution of the disaster model — and when it
couldn't, why not?* The answer changed shape four times, each time under adversarial
pressure, and ended somewhere nobody guessed at the start: a measured statement about what
residual objectives can and cannot see, three engineered exits, and the first evidence that
one of them works. Every claim below carries a receipt (commit, file, log, or command).
Superseded claims are kept and marked — the corrections are themselves results.

**How to use it.** §1–2 are context. §3–14 are chronological; §18 is the closing verdict
(07-14); §19 is the 08-28 coda (coverage measure question closed). §15 is the artifact index — if this file is the only thing you have, §15 rebuilds
everything else. §16 is what remains open. The companion narratives are `docs/how_it_all_works.md` (framework, for outsiders)
and `docs/dev/disaster_cert_report_2026_07_07.md` (the certification experiment's live
record, with all amendments in place).

---

## 1. Cast

- **Framework:** deqn-jax — JAX/Equinox DEQN trainer (Azinovic–Maliar–Maliar family):
  policy network trained so equilibrium residuals vanish on states the policy itself
  visits. On-policy by construction; that fact is the villain of this story.
- **Models:** `brock_mirman` (analytic benchmark), `irbc` (4-state international RBC with
  an off-path irreversibility kink), `disaster` (13-state NK-DSGE with financial frictions,
  Calvo price/wage blocks, BGG entrepreneur contract, effective-lower-bound softplus floor;
  shipped calibration is **disasterless**: p_disaster = 0).
- **Machines:** laptop (nothing heavy — thermal rule of 07-06), DGX Spark GB10
  (`anna@130.223.169.108`, repo `~/projects/deqn-jax`; host venv = CPU fp64 for probes and
  tests; NGC JAX container = GPU for training, `scripts/run_sweep_in_container.sh`).
- **Certification stack** (the program's instrument, sharpened twice along the way):
  ergodic residuals → held-out stress grid → closed-loop spectral radius at SS →
  SS-level consistency → (added 07-10) learned-block spectrum at the *solved* fixed point
  ŝ, ‖ŝ − s\*‖, and per-equation residuals at ŝ.

## 2. Prologue — what was known going in (before 07-06)

- **Zombie paths** (fixed b22259e): fixed-prefix ss_reset had left 55/64 training paths
  absorbed at the soft-clip ceiling; final checkpoints were good all along.
- **Disaster checkpoints don't reproduce the SS** → IRFs-from-SS are drift, not impulse
  responses (2026-06-27 finding). Ergodic-good ≠ SS-consistent.
- **House audit v2** (docs/dev/disaster_house_audit.md, untracked): π-gap mechanism =
  Taylor rule vs miscalibrated y_ss (indexation hypothesis refuted by blind refereeing);
  model determinate at shipped calibration (boundary ξ_p ≈ 0.79); ELB crossed 1.18σ from
  SS, ~12% of quarters — strongest confirmed finding; recal config exists
  (`configs/disaster_recal.yaml`: y_ss 3.006808, R_ss 1.0116421, opt-in, doubles ELB
  exposure).
- **Model tex vindicated** (07-06/07): all 11 equations re-derived by hand + cross-checked;
  math clean; 5 prose-slop errata fixed (b6fbe6f). ELB floor traced to commit fdfbf2b
  (2026-04-17) — added as a training guardrail, absent from the original model. The
  original calibration (docs/disaster.tex, from Alex): p = 1%, θ = 15% (Barro-consistent),
  σ_p carrying a 100× percent-typo, and a never-implemented credit-crisis dispersion
  channel (σ_ω 0.268 → 0.54) — spec-let 4 is its restoration.
- **EWM coverage feature** shipped + validated on irbc (spec 2026-06-29): `coverage:`
  config block = stress/local pools rolled through the exact transition.

## 3. Day 0 (07-06) — irbc solved; the doc; the referee habit

- **Composition result (3ed43c7, addendum 9829e35):** coverage∘composite on irbc =
  **5/5 seeds stable at ρ = 0.9808 with zero seed variance**, SS-exact, best stress
  residual (+2.3× over anchor-alone). Coverage-alone: real stress gains (~1.1 decades) but
  ρ 0/5; anchor-alone beats coverage-alone off-path 13× on this box. Artifact:
  `docs/superpowers/specs/data/ewm_composition_table_2026_07_06.json`. **irbc is closed:
  anchor abolishes the seed lottery, coverage adds off-path margin.**
- "How This All Works" written and adversarially refereed by a non-Anthropic model
  (read-only); three factual errors fixed including stale single-JIT folklore. Companion
  plain-words version committed (0642600). Evaluator-overhaul spec committed (4d25cd0).
- Method habit established: **external adversarial refereeing of every major claim.**

## 4. Night 1 (07-06 → 07-07) — the certification sweep

**Design:** 4 arms × 3 seeds × 3000 episodes, fp64 quadrature: `disaster` (baseline),
`disaster_gated` (kink-aware anchor: sigmoid gate mutes anchor points where the linearized
Taylor rate sits below the floor — 21/128 points down-weighted, matching the audit's floor
mass), `disaster_elbcov` (ELB-targeted coverage), `disaster_gated_elbcov`. Infrastructure:
NGC GPU container, ~16 min/run composite. Gated anchor shipped in 5585bf8.

**Live-narrative failures, confessed and kept:** (a) elbcov s1/s2 cut prematurely (a 4e5
plateau read as divergence; the run later recovered) — those two runs ended with DONE
markers but **no checkpoint files**; (b) three hours of per-arm narrative built a causal
story ("gate moves ρ 1.22 → 1.05") that the consistent-protocol table dissolved — it was a
best-vs-final **checkpoint-convention artifact**. The probe script's default is final
checkpoints (`checkpoint_003000.eqx`) because of this night.

**Consistent-protocol table (final checkpoints, fp64):**

| arm | ρ(SS) per seed | pass | max SS err per seed |
|---|---|---|---|
| baseline | 1.057 / 1.021 / 1.023 | 0/3 | 1.1% / 2.8% / 0.7% |
| gated | 1.049 / 1.064 / 1.052 | 0/3 | 2.3% / 2.5% / 2.6% |
| elbcov (1 usable seed) | 1.060 | 0/1 | 7.4% |
| gated+elbcov | 1.064 / 1.148 / 1.293 | 0/3 | 0.9% / 1.3% / 6.2% |

**Findings that survived:** zero crossings of ρ < 1; the ~1.02–1.06 basin is reached by the
baseline itself; the gate controls *variance* (lottery abolished) but not the attractor's
location; v1 ELB coverage sent the on-policy sampler into a 2500-episode deflationary
π-wall transient (**measure migration**: bound-attractor, zombie-paths family) — when the
kink is on-path, paper-style coverage destabilizes the training measure; **best-by-loss
checkpoints are systematically certificate-worse than final ones** (baseline s0: ρ 1.22 at
best vs 1.057 at final) — `save_best_checkpoint` selects against the certificates.

## 5. Day 2 (07-07) — drift, Sobolev, the diagnostic, the surgery

- **aux_drift** (certificate-in-the-loop): deterministic closed-loop rollouts from
  ergodic-shaped SS probes, hinge on mean per-period log growth. Result (arm
  gated+drift, final conv): ρ 1.044/1.107/1.033, 0/3 — but drift@20 (its own horizon)
  halved. It moves what it sees; T=20 sees the non-normal transient, not the asymptote.
- **aux_res_sobolev** ("Simon's Sobolev", decoded): penalize directional derivatives of the
  per-state *expected* residual via JVPs along fixed ergodic directions — the true policy
  zeroes E[r] on a neighborhood; impostors keep values small with finite gradients. Exact
  unit tests (toy impostor value (c−1)² to 6 digits). **Impostor signature measured: aux
  floor 4.68e-3 at the trained basin** — but at weight 1 it is ~7% of the loss.
- **Elimination ladder:** Huber = reweighs the same values (no new information);
  second-order optimization = changes the walk, not the map (sweep_so precedent); μP/init
  eliminated by the strongest evidence owned — `init_scale: 0` starts training AT the
  linearized solution and the optimizer walks OUT to ρ ≈ 1.05. **[SUPERSEDED 09-02, §20:
  the arms did not start at the linearization — the warm start had already moved ρ(SS) to
  1.14. The elimination is retracted.]**
- **The gradient-conflict diagnostic** (gated s0 final ckpt, 64 ergodic states, 3-node GH,
  fp64): total gradient norm **3.607**; per-equation norms 0.073–1.485 (largest: investment
  Euler 1.485, wage-Phillips recursion 0.963, entrepreneur contract 0.828); pairwise
  cosines **price-Phillips vs wage-Phillips −0.89 … −0.92** (eq1×eq3 −0.89, eq2b×eq3 −0.92,
  eq3×eq4a −0.92), side conflicts −0.30…−0.44 vs bond Euler/BGG. **The basin is a
  compromise point — a frozen tug-of-war the scalar mean sums to "converged".**
- **Aux-compatible PCGrad** (spec-let 6, eec06c5): with a composite loss, project only the
  11 core equation gradients (one-shot simultaneous Gram projection, at the base
  mean-over-equations scale) and add the full auxiliary gradient as grad(total) −
  grad(base), unprojected. Exact invariants unit-tested: no-conflict ⇒ identical to the
  standard composite step; total-conflict ⇒ aux gradient only. Validator gate relaxed
  (MAO/GN/LM and coverage×pcgrad stay rejected; non-unit uniform weights rejected later —
  referee hole). A/B 12-episode smoke: pcgrad transients within family envelope (control
  was larger).

## 6. Interlude — the blackout

The DGX dropped off the network for ~36 h (07-08 → 07-09). The container queue completed
unattended (DONE-marker resumable design). Lesson embodied, not just stated: all
long-running work must survive the operator vanishing.

## 7. Day 3 (07-09) — the Sobolev arms report

- gated+rsob w=1: ρ 1.052 / 1.278 / 1.122, 0/3, SS err up to 34.3% — **underdosed
  regularizer = noise**, median worse than gate-only.
- gated+drift+rsob: ρ 1.238 / 1.268 / 1.204, 0/3, SS err to 20.5% — **two weak selection
  pressures compose into variance, not selection.**
- Escalation arms launched (knobs a priori): `disaster_gated_rsob25` (w=25, 8 directions)
  and `disaster_gated_pcgrad` ×3.

## 8. Day 4 morning (07-10) — endgame, first claim

27/27 trained (25 with final checkpoints). At the frozen convention:

| arm | ρ(SS) per seed | max SS err |
|---|---|---|
| gated+rsob25 | 1.020 / 0.989† / 1.105 | 2.9% / **49.7%**† / 1.8% |
| gated+pcgrad | **0.987** / 1.027 / 1.022 | **0.29%** / 4.1% / 7.0% |

† rsob25 s1: ρ<1 around a rest point displaced 49.7% — dominant Sobolev dose bought
stability by *relocating the economy* (overpowered the anchor).

pcgrad s0: ρ(SS)=0.987, SS err 0.29%, **drift@100 = 0.82%** (every other run: 49–529%).
Claimed then as "first fully certified solution". Program verdict: **selection is an
aggregation problem, not a sampling or dose problem** — the only crossing came from
changing how the equations' disagreements combine (as the compromise-point diagnostic
predicted).

## 9. The referee cycle (07-10) — both attacks confirmed, result upgraded by correction

External adversarial review (Codex, prompt engineered to refute; 8 findings). The two
decisive ones were **confirmed by same-day measurement** and the claim was rewritten:

1. **The probe's ρ has a floor at 0.98699** — the closed-loop Jacobian splits exactly into
   a fixed autonomous block (AR roots {0.809, **0.98699** = mu_ups root × soft-clip
   derivative, 0.940, 0.146, 0} — identical in every checkpoint) and the learned 8×8
   endogenous block. Corrected metric = **max learned-block eigenvalue**: pcgrad s0
   **0.9750** at s\* / 0.9754 at ŝ; baseline s1 1.0209/1.0248; gated s0 1.0492/1.0586. The
   crossing is real and learned — masked, not manufactured, by the exogenous ceiling.
2. **"Contracts to the true rest point" retracted** — the drift trajectory (starting AT s\*)
   converges (7e-14 by t=1000) to the *learned* fixed point ŝ = T(ŝ), displaced **0.827%**
   (leverage lag worst, q_lag 0.36%; Newton residual 7e-15). One-step |T(s\*)−s\*| =
   2.87e-3. Corrected claim: **a locally stable economy 0.83% from the truth** — uniquely
   stable-near-truth in the program (baseline/gated fixed points: 3–10% displaced AND
   unstable → 529% soft-clip attractor).

Standing referee findings: causal attribution (1/3 vs 0/24, Fisher p ≈ 0.11; Adam/clip
norm-confound) needs preregistered matched pairs + sham norm-matched control; the operator
is the one-shot PCGrad *variant*, not the paper's sequential procedure; always say
**disasterless** calibration; provenance fixes (25-vs-27, drift ranges, committed
manifests). Code fixes: non-unit-uniform-weights gate + docstring (1e4c4d6).

## 10. The risky steady state (07-10) — the 0.83% decomposed

Question: is the displacement *correct economics* (the risky SS the certainty-equivalent
anchor cannot express) or error? Built `scripts/disaster_risky_ss.py` — CRW-style risky SS
(rest under zero realized shocks, equations in expectation over future shocks, first-order
future rules, 3⁵ GH quadrature, damped Newton fp64; deterministic re-solve of the same
system isolates the machinery floor, ≤0.05%/dim).

- **Shipped calibration (Gaussian risk only):** pure risky shift ≤ 0.10%/state (L_lag
  +0.100%, q_lag −0.057%; Calvo recursions carry the policy-side shift). The network's
  displacement is **8× larger with OPPOSITE sign** (L_lag −0.827% vs +0.100%; ratios −2 to
  −11). **The 0.83% is approximation error, not a risk premium.** Corollaries: the CE
  anchor is nearly free here; s\*-referenced certificates justified.
- **Alex's calibration (p=1%, θ=15%, Bernoulli mixture, 3-system decomposition —
  machinery/disaster/gaussian, Newton ≤ 4e-15):** the disaster premium is real and
  coherent — disaster risk *taxes capital*: k, i, c, w̃ down 0.12–0.14%, leverage +0.22%,
  λ_z +0.12%, π/R up ~0.07%, **Calvo recursions K_p/K_w/F_p/F_w +0.41–0.66%** (the big
  movers); linear in θ. At this calibration the risk correction ≈ the whole pcgrad-s0
  error budget ⇒ **spec-let 4 training must anchor to the risky SS**; the model's built-in
  flat-next-policy heuristic sits 0.42% from the CRW point (the rules wedge is as large as
  the shift) ⇒ use the CRW solver as the anchor target.

## 11. The EWM dissection (07-10)

Read in full (arXiv 2606.23463, public). Four legs: (1) coverage mixture over
ordinary/rare/stressed/counterfactual states rolled through the exact transition — **our
`coverage:` feature is a faithful implementation** (their Eq. 7), tested on disaster: 0
crossings, measure migration when the kink is on-path; (2) learned continuation surrogate
(the "world model") — amortization + optimization stabilization, untested by us, low value
where quadrature is cheap; (3) action-conditioned continuations — moot under exact
quadrature; (4) **certification = held-out exact residuals only** — no spectral radius, no
SS reproduction, no fixed-point consistency anywhere. Theory: convergence is asymptotic;
footnote 8 ("any loss with that zero set identifies the same policy") assumes exact,
unique zeros; **no treatment of identification under approximate equilibrium**. Every
wrong economy this program caught would pass EWM's certificate. Positioning available:
complementary, not competing — EWM answers "are residuals small everywhere that matters";
this program answers "is the economy they select the right one" and supplies the missing
certificate + the only intervention that moved selection.

## 12. The sharpened taxonomy (07-10 evening) — the maintainer's correction

Three possible pathologies: (i) optimization trap; (ii) **aggregation/identification
trap** — ∇L = (2/n)·J_Rᵀ R ≈ 0 with R ≠ 0 (least-squares compromise; stability directions
near the residual operator's null space); (iii) genuine equilibrium multiplicity (two
policies, all equations zero, different stability). The program's language ("impostor
economy", "selection", "stable wrong economy") had borrowed (iii)'s connotations; the
evidence supports (ii). Decisive same-day measurement — per-equation E-residuals AT each
learned fixed point (never previously computed): pcgrad s0 **3.8e-3** (investment Euler —
the same equation with the largest conflict gradient), gated s0 2.8e-2, rsob25 s1
**0.16**. Nobody rests at a joint zero; rsob25's "stable wrong economy" retracted → stable
displaced *simulator* rest point, not an economy. Additional relabelings: init-walk-out ≠
multiplicity proof; the basin is an attractor of the coupled learning dynamics (grad 3.6 ≠
0 — not even frozen-measure stationary; E-stability reading); the pcgrad crossing is an
update-geometry effect. Verdict adopted verbatim in the report (108ece7).

## 13. Experiment 4 (07-10 night) — the wedge measured, multiplicity dead

`scripts/gn_polish.py` (be5a016): Levenberg–Marquardt on the stacked 704-residual vector
(64 frozen ergodic-shaped states × 11 equations, min-norm steps, fp64, per-iteration
certificates), from BOTH basins:

| start | ‖R‖∞ floor | ρ_learned end | SS err | ŝ displacement | E-resid at ŝ |
|---|---|---|---|---|---|
| pcgrad s0 (0.975 / 0.29%) | 1.82e-2 | 1.173 | 6.0% | 11.2% | 7.9e-2 |
| gated s0 (1.049 / 2.9%) | 1.63e-2 | 1.244 | 8.5% | 9.2% | 5.5e-2 |

(a) **No joint zero within capacity near either basin** — the identification wedge is
measured (~1.7e-2 on this grid), not conjectured. (b) **Residual descent monotonically
destroys the certificates** — every accepted step from the stable point traded ρ and
SS-truth for residuals. Stability is **negatively priced** along R's descent directions.
(c) **No multiplicity** — both basins converge toward the SAME unstable displaced family
(ρ ≈ 1.2, the old best-by-loss neighborhood). One wrong least-squares attractor. pcgrad
s0's stability was held *against* the objective by the anchor. **Selection must be
structural.**

## 14. The three exits (launched 07-10 night; 6adadd8, 35049f0)

1. **Capacity:** `disaster_gated_pcgrad_wide` (256×256, 4× params) ×3 training in the
   container; plus fast lane — distill pcgrad s0 into a 256² net (`gn_polish.py --widen
   256`) and re-polish. Decides whether the 1.7e-2 wedge is a 128×128 artifact.
2. **Constrained polish:** LM on residuals augmented with SS-consistency rows and a
   smooth spectral-growth penalty (sharpened softplus on ‖J^K u‖ probes, K=30 — note the
   probes measure *transient* growth, stricter than asymptotic ρ; a plain softplus would
   have left a permanent 0.69 residual and hijacked the objective — caught pre-launch).
   Weights a priori: w_ss = w_ρ = 10. **Early result (iter 5): everything improves at
   once** — total residual 0.99 → 0.13, ρ_learned 0.975 → **0.910** (deeper than any
   training arm ever reached), SS err 0.29% → 0.13%. Where unconstrained descent paid for
   residuals with certificates, the constrained geometry finds directions that buy both —
   direct evidence of the near-null ρ-moving directions of experiment 3.
3. **Selection by construction:** `network.bk_pin` on DisasterPolicyNet — the MLP delta's
   value and tangent at s\* are subtracted structurally (JVP at s\*), so **π(s\*) = π\* and
   dπ/ds(s\*) = P hold exactly for every parameter value**; training cannot unlearn
   Blanchard–Kahn; the residual objective only shapes second-order-and-beyond deviations.
   Four exact tests (`tests/test_bk_pin.py`, incl. brutalize-all-MLP-weights invariance).
   Arm `disaster_gated_pcgrad_bkpin` ×3 training. Smoke: gradient norms ~2e-3 at episode 8
   vs 1e2–1e8 transients on every other arm — the pin removes the early wandering.

## 15. Artifact index

**Committed docs:** `docs/dev/disaster_cert_report_2026_07_07.md` (the experiment's full
record: final table, referee corrections, taxonomy, experiment 4, risky SS, exits);
`docs/dev/handoff_2026_07_06.md` (spec-lets 1–4 + statuses); this chronicle;
`docs/dev/how_it_all_works_companion.md`; `docs/superpowers/specs/…evaluator-overhaul…` and
EWM spec + `specs/data/ewm_composition_table_2026_07_06.json`; `docs/disaster_corrected.tex`
(+ errata item 12).
**Untracked, sign-off gated:** `docs/how_it_all_works.md`, `docs/dev/disaster_house_audit.md`,
`docs/dev/codex_accuracy_gap_brief.md`.
**Scripts (all whitelisted in .gitignore):** `disaster_ss_probe.py` (certificates; final-
checkpoint default), `gn_polish.py` (LM polish; constrained rows; distill), `disaster_risky_ss.py`
(CRW risky SS + disaster mixture + decomposition), `cert_sweep_container.py` (11 arms ×3,
DONE-resumable), `run_sweep_in_container.sh`, `ewm_stress_table.py`, `export_ergodic_viz.py`.
**Configs:** `disaster{,_gated,_elbcov,_gated_elbcov,_gated_drift,_gated_rsob,_gated_drift_rsob,
_gated_rsob25,_gated_pcgrad,_gated_pcgrad_wide,_gated_pcgrad_bkpin,_recal}.yaml`;
`irbc_plain/irbc_ewm/irbc_ewm_anchor.yaml`.
**Feature commits (this program):** gated anchor 5585bf8; composition 3ed43c7 (+9829e35);
pcgrad surgery eec06c5; referee code fixes 1e4c4d6; risky-SS solver f94bd71 + mixture
4a40446; gn_polish be5a016; bk_pin + constrained/widen + arms 6adadd8. Report milestones:
fada158, 91335aa, b22350b, 4b8b04b, 108ece7, 7b2a5d1, 35049f0.
**DGX artifacts:** `runs/disaster_cert/<arm>_s<seed>/` (checkpoints, configs, DONE),
`probe_final.json`, `probe_day3.json`, `polished_gn*.eqx`; logs
`cert_container*.log`, `gn_polish*.log`, `gn_polish_exits.log`; one-off analysis
`/home/anna/referee_checks.py`, `referee_blocks.py`, `resid_at_shat.py`.
**Reproduce (certificates):** `JAX_ENABLE_X64=1 uv run python scripts/disaster_ss_probe.py
--runs-dir runs/disaster_cert --arms <arms> --seeds 0,1,2`.
**Memory (Claude-side, `~/.claude/projects/...deqn-jax/memory/`):**
`disaster_cert_program_result.md` (the program), `fable_last_day_plan.md` (the arc),
`disaster_not_solvable_simon_paper.md` (EWM dissection), MEMORY.md index.

## 16. Open threads, ranked

1. ~~**Read the three exits** when they land~~ **DONE 07-14 — see §18: bkpin 3/3 + stress
   grid passed; the embargo lifted at the shipped calibration.**
2. **Experiment 3** (designed, not run): singular spectrum of J_R + the ρ-moving
   near-null direction — the direct underidentification quantifier. The constrained
   polish's iter-5 behavior already corroborates it indirectly.
3. **Referee's causal controls** for the PCGrad mechanism claim: preregistered matched
   seed pairs, sham norm-matched arm, LR/clip sweep.
4. **Spec-let 4 with risky-SS anchoring:** train at p=1%/θ=15% anchored to the CRW risky
   SS; certify ‖ŝ − s_rss‖ + learned-block spectrum. The question: can DEQN capture a
   disaster premium no certainty-equivalent method can express?
5. **Certificate-aware checkpoint selection** (cheap): probe ρ/SS at every save; the walk
   visits better policies than it keeps.
6. **E-stability theory for deep equilibrium solvers** (the paper-shaped hole): conditions
   under which residual-SGD learning dynamics are E-stable at the RE solution. This
   program is its empirical first draft.
7. Evaluator overhaul (spec 4d25cd0); sign-off queue for the three untracked docs;
   history rewrite decision (alias mentions in pushed history — tip is clean).

## 17. Method lessons (paid for)

1. **A small training loss is not a certificate** — best-by-loss is certificate-worst;
   residual descent can be certificate-destroying (experiment 4 made this a theorem-shaped
   measurement).
2. **Freeze conventions before the experiment** — the best-vs-final checkpoint mixing
   built a false causal story that one consistent table dissolved.
3. **Referee everything, then referee the referee** — Codex's attack survived two decisive
   checks and improved the result; the maintainer's taxonomy critique out-refereed the
   referee and redirected the program.
4. **One variable per arm, knobs a priori** — the escalation arms (rsob25, pcgrad) were
   designed before their predecessors' results were read.
5. **Live narratives lie; tables decide.** Keep the live text, mark it superseded — the
   corrections are results.
6. **Structure beats penalties** — the week's arc in one line: every *soft* selection
   pressure (weights, doses, penalties) either underdosed or overpowered something it
   needed; the interventions that worked changed *structure* (gate, surgery geometry,
   pin).

## 18. The verdict (07-14) — the embargo lifts

The day-4 container finished bkpin s1/s2; both were probed at the frozen convention the
same day, plus the held-out ELB stress grid on all three seeds (full tables and receipts:
cert report, "Certification verdict (07-14)").

The signature is *identical* across seeds: SS error at machine precision (the pin's
guarantee, now verified 3/3), learned-block ρ(s\*) = 0.976851 to six digits on every
seed — the pin fixes the policy tangent at s\*, so the closed-loop linearization is
seed-invariant by construction, and the seed lottery that defined this program's first
week is abolished at the certificate level — ŝ displaced 0.0525% (all three), max
|E[r]| at ŝ 4.3–6.4e-4, trajectories converging to ŝ at 1e-14 by t=1000. The stress
grid (ELB corner, 512 held-out points, fp64) has bkpin 3–5× below the pcgrad reference
per-equation, with pcgrad s1/s2 unstable outright.

**The pre-registered embargo criterion (§14, cert report Exit 3) is met: 3/3 seeds with
the s0 signature plus a held-out stress-grid certificate. The word is no longer
embargoed, for exactly this claim: the disaster model at the shipped (disasterless,
p_disaster = 0) calibration is SOLVED by selection-by-construction (`network.bk_pin`),
per the full certification stack.** What the claim does not cover: the real calibration
(p = 1%, θ = 15%), where the certification target must move to the CRW risky steady
state (§10) — the successor program's opening experiment (§16 thread 4).

Program epitaph, one line: residuals do not select; structure does — and when the
structure is right, the residuals finally agree with it.

## 19. Coda (08-28) — the coverage measure question, closed

After the EWM reference implementation became readable, a fidelity audit found our
coverage port's one divergence: box-seeded stress (SS-slice) vs the paper's path-seeded
stress. We implemented the paper's measure (`stress_seed_mode: path`, e27cc0f) and ran
the registered test on the elbcov arm. Prediction refuted in the inverting direction:
path seeding removes the *recovery* from measure migration (box: 1 clean / 1 recovered
/ 1 stuck; path: 1 clean / 0 recovered / 2 stuck), certificates worse (median ρ 1.215
vs 1.148). Mechanism: SS-anchored seeds resist a drifting path; path-anchored seeds
follow it. Night-1's verdict stands unnarrowed: coverage destabilizes on-path-kink
models regardless of seed measure. Cert report "EWM measure experiment (08-28)".
Method lesson #7: **audit-driven "fixes" get a registered prediction and an arm before
they get a verdict — this one would have been a confident wrong footnote otherwise.**

## 20. Coda (09-02) — the arms never started where we said they did

A whole-library review found that the constant-SS warm start (`warm_start: true` on every
disaster config) ran on the BK-anchored `disaster_policy_net` — the skip in the trainer
tested for `linear_plus_mlp` only. Fitting an anchored net to a *constant* policy trains
the delta to cancel the linear slope: on the shipped recipe the warm start alone takes
ρ(SS) from the 0.98699 floor to **1.14** and moves the policy 9–16% off π_BK on the ergodic
set, before the first residual gradient. All 24 non-pin certification arms started there;
the pin arms were insulated at first order. Measured numbers in the record stand (final
checkpoints); the causal story of §5 — "starts AT the linearization and walks OUT" — is
inverted: the anchor pulled a broken start *back* to ρ≈1.05. Fixed structurally (skip on
any net carrying `P`), pinned by `tests/test_warm_start_anchored_nets.py`; cert report
"Warm-start contamination (09-02)". Not yet re-run: baseline ×3 with the fix decides
whether night-1's table is a warm-start artifact.

Method lesson #8: **probe episode 0.** Every certificate was measured on final checkpoints
and none on the initial state; the one number that would have caught this (ρ at s\* before
training) was never in the table. Certify the start of a run the way you certify its end.
