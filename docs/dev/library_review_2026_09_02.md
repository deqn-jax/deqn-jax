# Library review (2026-09-02) — solver theory and infrastructure cruft

**What this is.** A whole-library review in two tracks, written for the maintainer, who
asked for the solver's theoretical decisions in plain words and for the infrastructure cruft
named. Track A (theory) was read end-to-end by one reviewer against the code at master
`f047650` plus the two open PRs. Track B (cruft) was swept package-by-package by seven
read-only agents; every item marked ✓ was re-verified by the reviewer with the grep shown,
unmarked items are agent-reported with their own receipts and should be re-checked before
deletion. Every claim carries a `file:line`, a command, or a number that can be reproduced.

**Read first: the one finding that changes the record.** The disaster certification arms did
not start where the record says they did. See A3 and PR #3.

---

## 0. Act on this, in order

1. **Merge PR #3** (warm start ran on the anchored disaster net; all 24 non-pin arms started
   at ρ(SS) = 1.14). Then re-run the baseline arm ×3 seeds with the fix — that decides whether
   the night-1 table is a warm-start artifact.
2. **Flip two defaults** that contradict the program's own method lessons:
   `save_best_checkpoint` (True → False; best-by-loss is certificate-worst, chronicle §17.1)
   and `aux_decay_floor` (0.2 → 1.0; 44/44 configs override it).
3. **Ship the July audit's C-remainder**: expectation-based, unit-normalized evaluator. Every
   reported "Euler error" for a single-stage model is a realized-shock residual, not an
   expected one, and equations are graded in different units (see A7).
4. **Cruft batch 1** (zero-risk deletes, ~1,500 lines, §B1) and the `--set coverage.*` fix.
   *Shipped on `chore/cruft-batch-1` (2026-09-02): everything in §B1 marked ✓ except the
   aiyagari package (Simon's course model, left for the landing decision), the scripts
   (research reproducibility, batch 3), the config ladders and the CLI long flags
   (user-facing, batch 3). The three ruff ignores stay: the sweep's "zero occurrences" claim
   was wrong (39 hits). `--set coverage.*` fixed by deriving the nested-block dispatch from
   `TrainConfig.model_fields`.*
5. Decide the museum policy for dead research branches (§A9) before cruft batch 3 (§B3).

---

## A. Track A — the solver's theoretical decisions

### A1. What the solver does, in one page

The model is a set of equilibrium conditions `r(s, π(s), s', π(s')) = 0` that must hold in
conditional expectation over next period's shock. The unknown is the policy π, a function of
the state. The solver approximates π by a neural network and pushes the expected residuals
toward zero on states the economy actually visits.

One training cycle (`training/cycle.py`):

1. **Rollout.** Simulate the economy under the current policy for `episode_length` steps
   from the last cycle's end states (`run_episode`, exact transition, Gaussian shocks scaled
   by the curriculum). On disaster, 15% of paths are re-seeded near the steady state first
   (`ss_reset_frac`, see A4).
2. **Dataset.** The visited states become the training set for this cycle.
3. **Gradient sweep.** For each minibatch: compute the expected residual per state and per
   equation (`compute_loss`), square it, average over the batch, average over equations, add
   auxiliary terms, take one optimizer step on the network weights.

The expectation over next period's shock is either a Gauss–Hermite or monomial rule (exact
for smooth integrands; used by irbc and disaster) or antithetic Monte Carlo (used by the OLG
family). The gradient flows through *both* today's policy and tomorrow's policy (no target
network by default), so this is the joint "all-in-one" objective of Maliar–Maliar–Winant,
not time iteration.

The architecture used for anything certified is `π(s) = clip(π_BK(s) + δ_θ(s))`: the
Dynare-style first-order solution (Blanchard–Kahn, `training/linearize.py`) plus a network
correction that starts at exactly zero (`init_scale: 0`). On disaster, three further devices
sit on top: the Calvo auxiliary outputs are held to the linear policy (K/F mask), an anchor
loss pulls the network toward the linear policy on a fixed cloud near the steady state, and
(in the certified arm) the correction's value and slope at the steady state are projected
out structurally (`bk_pin`).

Certification is not the training loss. It is: held-out stress residuals → learned-block
spectral radius at the steady state → the solved fixed point ŝ = T(ŝ) and its distance from
s\* → per-equation residuals at ŝ → long-horizon convergence, on final checkpoints, fp64,
three seeds (`scripts/disaster_ss_probe.py`).

### A2. Decision ledger

Verdicts: **sound** (right choice, adequately documented); **sound, undeclared** (right
choice, but the claims that depend on it don't say so); **questionable** (defensible, but
the evidence for it is weaker than the record implies); **wrong** (a bug or a false claim).

| # | decision | in plain words | why it is there | verdict | receipt |
|---|---|---|---|---|---|
| 1 | On-policy training measure | train only on states the current policy visits | the whole scaling bet: pay for the sheet, not the cube | sound; the known failure mode (self-confirming solutions) is what the rest of the stack addresses | `cycle.py:612-670`; EWM paper; companion §2 |
| 2 | Mean-then-square, `(E[r])²` | average the residual over shocks first, then square | the equations hold in expectation; squaring first would penalize shock variance the policy cannot remove | sound | `loss.py:399-505` |
| 3 | Cross-equation mean, unit weights | the scalar loss is the plain average of 11 (or 5, or 1) per-equation losses | one learning rate transfers across models | **questionable**: equations are in different units, so the average silently weights them by unit scale; this is the "aggregation trap" of chronicle §12 in its rawest form | A7 |
| 4 | Gradient through next-period policy | no target network; tomorrow's policy is the same network and gets gradient | AMM/MMW convention; joint objective | sound; the target-network option exists and no shipped config uses it (dead branch, §A9) | `loss.py:140-260`; configs table |
| 5 | Antithetic MC / GH / monomial expectation | quadrature where the shock dimension allows, antithetic MC otherwise | variance and cost | sound; note the OLG family trains under MC with `mse` aggregation, whose bias is Var/N (the `aio` estimator exists and is unused) | `loss.py:37-71, 312-345` |
| 6 | Two-stage `inside_fn`/`combine_fn` | for FB-style complementarity conditions, take the expectation *inside* the nonlinearity | `E[fb] ≠ fb(E)` | sound | `loss.py:461-485` |
| 7 | Disaster as a discrete mixture in the residual | residual = (1−p)·r(no disaster) + p·r(disaster) | Bernoulli mixture over the disaster indicator | sound for single-stage residuals (linear in the branch); would be wrong for a two-stage model, none exists at p>0 | `loss.py:246-260` |
| 8 | `π = clip(π_BK + δ)`, `init_scale 0` | start at the first-order solution, learn only the correction | inherits the linearization's local correctness; kills the collapse pathology | sound | `linear_plus_mlp.py:1-42` |
| 9 | Certainty-equivalent linearization as anchor target | π_BK is the zero-risk first-order policy | it is what Blanchard–Kahn gives | sound at p=0 (risky shift ≤0.10%/state measured); **wrong target at p=1%** where the risk correction ≈ the whole error budget | cert report "risky SS (07-10)" |
| 10 | Composite anchor + Jacobian terms, weight 1.0 / 0.1, never decaying | penalize distance from π_BK on a fixed 128-point ergodic-shaped cloud, throughout training | selection: the residual objective alone lands in a wrong basin | **sound, undeclared**: the trained object is the minimizer of residual + λ‖π−π_BK‖² on the ergodic set, i.e. a Tikhonov-regularized solution toward Dynare. No claim in the record states λ; the "curvature" measurement (net−linear slope 2.26) shows the net does leave the linearization, but how much of the certified policy is anchor vs residual has never been reported | A6 |
| 11 | K/F gauge mask, on by default | four Calvo auxiliary outputs are exactly linear forever | gauge near-degeneracy in the Calvo recursions | sound at p=0; **blocks spec-let 4**: at p=1% the Calvo recursions carry the largest risk shift (+0.41–0.66%), which the mask forbids the network from expressing | `network.py:112-116` (config default); cert report risky-SS decomposition |
| 12 | Kink-aware anchor gate | mute anchor points where the linear Taylor rate is below the floor | the linearization is the wrong local model there | sound | `composite_loss.py:121-135` |
| 13 | `bk_pin` | subtract the correction's value and tangent at s\* | selection by construction; training cannot unlearn Blanchard–Kahn | sound as construction; **the ρ(s\*) and SS-error legs are then donated by the pin**, not earned (seed-invariant to six digits *by construction*); the earned legs are the stress grid and residuals at ŝ | chronicle §18; A5 |
| 14 | Pin + gate + PCGrad stacked in the certified arm | three selection devices at once | each was added in sequence | **questionable**: no pin-alone arm exists, so the verdict attributes to the pin what may need all three | configs table |
| 15 | One-shot PCGrad | remove pairwise conflicting gradient components in one pass | the compromise-point diagnostic | sound as a variant, documented as such; causal claim unproven (1/3 vs 0/24, p≈0.11) | `pcgrad.py:1-25`; chronicle §9 |
| 16 | Soft clip inside the disaster transition | states are softly clipped inside T during training | numerical guardrail (zombie paths) | **sound, undeclared**: the certified closed loop is `T_clipped`, not `T`; the raw-ρ floor 0.98699 is a product of the clip derivative; no certificate was ever run with the clip removed | `dynamics.py:126-135`; A8 |
| 17 | `ss_reset_frac 0.15` on disaster | each cycle, 15% of paths restart within ±5% of s\* | keeps transients near the steady state in the training set | **sound, undeclared**: the training measure is not the ergodic set but ergodic ∪ SS-transients; it is itself a selection device and no claim says so | `cycle.py:477-507` |
| 18 | `save_best_checkpoint: True` default | keep the lowest-loss checkpoint | habit | **wrong** given the program's own result: best-by-loss is certificate-worst; the frozen convention is final checkpoints | `config/train.py:272-273`; chronicle §17.1 |
| 19 | Warm start on anchored nets | L-BFGS fit of the whole policy to a *constant* SS policy before episode 1 | intended for bare MLPs | **wrong**: ran on every disaster arm; see A3 | PR #3 |
| 20 | Evaluator residual = single realized draw | `deqn-jax evaluate` reports `r(s, ε)` for one ε per period, not `E[r]` | historical | **wrong**: reported errors are inflated by shock variance and graded across unit systems (July audit major #2, unshipped) | A7 |
| 21 | Curriculum shock ramp, cosine LR, warmup | ramp shocks 0→1, decay LR | standard | sound | `trainer.py` |
| 22 | Coverage (EWM) as a fixed-weight mixture of pools | residual imposed on base + stress + local pools | the EWM prescription | sound; measure question closed 08-28 (box vs path) | chronicle §19 |
| 23 | World arm (continuation surrogate) | small net Ŵ ≈ E[inside] at a Polyak target, fitted per cycle on path anchors | amortize the expectation | sound as built; validated only by unit tests; PR #1 | spec 2026-08-28 |

### A3. Finding W1 — the arms never started where the record says (fixed, PR #3)

Every disaster config sets `warm_start: true`. The trainer skipped the warm start only for
`network.type == "linear_plus_mlp"`; the disaster arms use `disaster_policy_net`. So
`warm_start_network(linearize=False)` ran (DGX logs: "Warm starting from steady state...")
and L-BFGS-fitted the whole anchored policy to a **constant** steady-state policy on ±20%
uniform states — which trains the correction δ to cancel the Blanchard–Kahn slope.

Measured on the shipped recipe (CPU, fp64):

| net state | raw ρ(SS) | max \|J(s\*) − P\| | max rel dev from π_BK on 1σ ergodic cloud |
|---|---|---|---|
| at init | 0.98699 (exogenous floor) | 0 | 0 |
| after warm start, no pin | **1.1406** | 2.19 | 9–16% per policy (F/K heads 0 by mask) |
| after warm start, `bk_pin` | 0.98699 | 0 | 1e-3 (1% cloud) |

Consequences: the 24 non-pin certification arms started *unstable* and 10–16% off the
linearization; the ~1.02–1.06 basin is where the anchor pulled a broken start *back to*. The
chronicle's "init_scale 0 starts AT the linearized solution and walks OUT" (§5) and the
"μP/init eliminated" step are retracted (marked in place). Measured certificates stand (final
checkpoints). The pin arms are insulated at first order. Not yet re-run: baseline ×3 with the
fix. Method lesson #8 added to the chronicle: probe episode 0.

### A4. Finding W2 — three selection devices are in the training measure and none is declared

The certified policy is trained on a measure that is not the ergodic set: 15% of paths are
re-seeded at s\* every cycle (row 17), the transition is soft-clipped (row 16), and the
anchor cloud is a separate 128-point sample that receives its own loss term every step (row
10). Each is defensible. Together they mean that the sentence "trained on the ergodic set,
certified at the steady state" describes neither the measure nor the objective. The record
should say: *trained on ergodic ∪ SS-transients under `T_clipped`, minimizing residual +
1.0·anchor + 0.1·Jacobian on a fixed cloud*. That sentence is true and still a result.

### A5. Finding W3 — which certificate legs are earned

Under `bk_pin`, π(s\*) = π\* and dπ/ds(s\*) = P hold for every weight vector. Therefore the
SS-error leg is identically zero and the learned-block ρ(s\*) is the closed-loop spectrum of
the *linearization*, 0.976851, on every seed — the chronicle's "seed-invariant to six
digits" is the pin's algebra, not training. What training earns: the stress-grid residuals
(3–5× below the pcgrad reference) and the per-equation residuals at ŝ (4–6e-4), plus ŝ's
displacement of 0.0525% (≈ the machinery floor). The cert-report verdict is correct as
stated; the dashboard-style summary "3/3 stable" should carry the word *donated* on the ρ
leg. A pin-alone arm (no gate, no PCGrad) would settle row 14 and is cheap.

### A6. Finding W4 — the anchor never decays and its weight is not in any claim

`aux_decay_floor: 1.0` in 44/44 configs; `anchor_weight 1.0`, `jac_weight 0.1`, cloud of
128 points at `anchor_sigma 1.0` (one ergodic standard deviation, i.e. the typical set). The
trained policy is thus the minimizer of residual + ‖π − π_BK‖² over the whole typical set at
weight 1, with the residual's own scale ~1e-3–1e-5. The reviewer could not find, in code or
docs, the value of the anchor term at the end of any certified run — i.e. how far the
certified policy sits from Dynare's. Two cheap additions: log `aux_anchor` in the
certificate table, and run one anchor-annealed arm (`aux_decay_floor 0.2`) to see whether the
certificate holds when the anchor releases. If it does not, the honest description is
"Dynare's solution plus a regularized correction", which is still publishable and much
easier to defend.

### A7. Finding W5 — units, and the evaluator (the unshipped C-remainder)

`compute_loss` averages per-equation losses with unit weights. The disaster equations are in
λ-units, i-units, price-Phillips units; irbc's in Lagrange-multiplier units; the July audit
measured ~3 decades of spread. The scalar objective therefore weights equations by an
accident of units, and the "compromise point" the program diagnosed (gradient cosines −0.9
between price and wage Phillips blocks) is partly a units artifact. Two fixes, one cheap: (a)
normalize each residual by its steady-state scale inside the model (Maliar's unit-free Euler
error convention) — a model-side change with a bit-identical guard on the current arms; (b)
the evaluator: `euler_equation_errors` reports a *single realized draw* `r(s, ε)` per period
for single-stage models (`evaluate/diagnostics.py:100-135`), so every reported log10 error
mixes approximation error with shock variance, and the grade thresholds are applied across
unit systems. Both were audit majors on 2026-07-02 and both are still open. PR #2's evaluator
change (Codex) touches the two-stage branch only and must not regress the GH default for
MC-trained checkpoints (review comment posted).

### A8. Finding W6 — the closed loop that is certified contains the guardrail

`soft_clip_state` is inside the disaster transition during training and inside the probe's
`step_fn`. Margins are ≥2 from s\*, so at s\* the clip is identity to 1e-6 and the certificate
at s\* is unaffected. But the raw-ρ floor 0.98699 is `mu_ups root × clip derivative`, and the
long-horizon and stress legs are evaluated under `T_clipped`. One probe run with the clip
removed (a flag on `dynamics.step`) would either confirm the certificate is a property of the
model or show it is a property of the guardrail. Never run.

### A9. Dead research branches (theory-side cruft)

Features that exist in the training path, were tried, and are used by no shipped config.
Each has a tombstone in the chronicle; the code is what remains. The reviewer's
recommendation is a museum policy: delete the code, keep the chronicle entry as the record,
tag the last commit that had it. Keeping them costs validator complexity (every new feature
must be gated against each), config surface, and reader confusion.

| branch | what it was | result | used by a shipped config? |
|---|---|---|---|
| `drift_weight` (certificate-in-the-loop) | penalize closed-loop growth over 20 steps | moves what it sees, not ρ; 0/3 | `disaster_gated_drift*` only (cert record) |
| `res_sobolev_weight` | penalize directional derivatives of E[r] | underdosed = noise; overdosed = relocates the economy | `disaster_gated_rsob*` only (cert record) |
| `jac_anchor_weight` (Sobolev anchor) | Jacobian match at every anchor point | never in a cert arm | one 300-episode config |
| target network (`target_update_every`) | frozen tomorrow's policy | never shipped | none |
| `loss_reweight` lr_annealing / relobralo | adaptive per-equation weights | never shipped | none |
| `loss_choice` huber / aio | robust / unbiased aggregation | aio finding recorded; neither shipped | one 300-episode config (huber) |
| `reparam_q_as_m`, `reparam_pi_as_kp_inner`, `reparam_wtilda_as_kw_inner` | output-space reparametrizations for Calvo/investment asymptotes | ~490 lines; `disaster_calvo` only, no sweep, no doc, wage-side forward pass untested | one unreferenced config |
| `use_zlb_feature` (+ `kink` variant) | ELB regime input | verdict not in the tracked record | one config, plots script only |
| `output_links: log` | multiplicative-around-SS outputs | `disaster_log` only | one config, no reference |
| sequence nets (LSTM, transformer, `history_len`) | recurrent policies | half-wired (cycle off-by-one, July audit #6) | none tracked |
| replay buffer | prioritized state replay | never shipped | none |
| moment matching | Dynare-moment aux loss | needs fixtures that are not in the repo | none |
| optimizer zoo: `mao_kfac`, `lion`, `adamw`, `sgd`, `ReduceLROnPlateau` | — | test-only or nowhere | none (ngd/mao/gn/lbfgs/shampoo/muon: April sweep artifacts only) |
| `KfAnchoredMLP` | superseded fork of LinearPlusMLP | documented as legacy | one May sweep script |
| warm start ×4 variants | constant-SS, linearized, Dynare, to-function | see A3; `to_function` has no caller | disaster (now a no-op) |

Not dead, keep: coverage (both modes, measured), `bk_pin`, anchor gate, PCGrad, world arm
(PR #1), LM (used by `gn_polish.py`, which rolls its own copy — unify).

---

## B. Track B — infrastructure cruft

Seven sweeps (config, training, optimizers, models, networks, evaluate/irf/cli,
scripts/tests/configs/CI). ✓ = reviewer re-verified. Full agent reports with per-item
receipts are in the session transcript; this section keeps the actionable subset.

### B1. Batch 1 — zero-risk deletes and one-line fixes (~1,500 lines)

| item | proof | action |
|---|---|---|
| ✓ `src/deqn_jax/benchmark.py` (198) | no `[project.scripts]` entry, no importer; refs are `check_module_graph.py`, an old audit note, a generated SVG | delete |
| ✓ `src/deqn_jax/models/aiyagari/` (257) | not in `_MODELS`; `grep aiyagari models/__init__.py` → 0 | move out of `src/` or register + smoke test |
| ✓ `networks/viz.py` (576) + `tests/test_networks_viz.py` (235) + disaster registration hook | only importers are its own test and the hook | delete, or declare it a dev tool and document |
| ✓ `warm_start_to_function`, `simulate_trajectory`, `episode.simulate_step` (pass-through), `loss.compute_loss_for_grad`, `loss.make_loss_fn` | zero callers outside `__init__` re-exports | delete (~125 lines) |
| ✓ `TrainState.aux_opt_state` | zero readers/writers (`aux_params` is now used by PR #1 — keep that one) | delete field |
| ✓ `OptimizerConfig.block_size`, `ReplayBufferConfig.eviction` | never read; Shampoo builds unblocked factors; ring is FIFO | delete + their tests |
| ✓ `pyproject.toml`: `treescope` (core dep, imported nowhere), `orbax-checkpoint`, `penzai`, the `dev` *extra* (competes with the `dev` group), five sdist `exclude` globs matching nothing, ruff ignores `E741 E731 E402` (0 occurrences) | greps in the sweep | delete |
| ✓ `mkdocs.yml:57` polyfill.io script | the domain was hijacked in 2024; MathJax 3 needs no polyfill | delete the line |
| ✓ CI "evaluate (short)" step can never run | smoke train has no `--checkpoint-dir` → `checkpoints/` never exists → always "skipping" | pass `--checkpoint-dir` or delete the step |
| ✓ CI syncs `--all-extras`, which includes `cuda` (multi-GB NVIDIA wheels on ubuntu) | `test.yml:27-28` | `--all-extras --no-extra cuda` |
| ✓ `tests/test_convergence.py`: 7 tests train 200–500 episodes at 64×64 | `grep episodes=` | mark module `slow`; annotate the aarch64 flake in place |
| ✓ 15 configs with zero references (`disaster_calvo`, `_p*_riskylin` ×5, `_p*_detss` ×3, `_p005_anchor0*` ×2, `_p02_cmrlib`, `_p02_kappaonly`, `_p10_zlb_huber_sob`; `_p1` needs a word-boundary re-check) | per-stem grep | move to `configs/archive/` (gitignored) |
| two `p_disaster` ladders (`_riskylin` ×5, `_zlb` ×6) differ in one key | single-hunk diffs | one config + `--set constants.p_disaster=` |
| `.claude/ralph_*.md` ×4 (April), stale one-shot permission literals, duplicated `pytest` allow | sweep | delete / prune |
| `scripts/make_plots.py`, `disaster_anchor_diagnostic.py`, second-order sweep bundle (`sweep_disaster_second_order.py` + fetcher + base config + ralph prompt), `fetch_sweep_results.sh` | unreferenced ≥83 days | un-whitelist together with their configs |
| `evaluate/cli.py --output` CSVs, `irf.py warmup` param, `girf_*.csv` with no reader | sweep | delete / route to `irf_*.csv` |
| 16 long CLI flags reachable only via `--set` and documented nowhere | sweep | drop or document |
| docstrings that lie: "single-JIT train step" (`state_init.py:7,179,200`), "three shape priors" (`disaster/network.py:11`), `make_composite_loss` usage example passes a config as a float, `stability_check`/`market_clearing_errors` return keys, `irf.py` usage line, `composite` rejection list in `train.py:78-81`, `factory.py` "pure move" header | sweep | fix |

### B2. Batch 2 — deduplication, behavior-preserving, bit-identical-guarded (~500 lines)

*Shipped 2026-09-03 as five parallel lanes (PRs #6 optimizers, #7 config, #8 evaluate/irf,
#9 networks, #10 models/steady-state), each in its own worktree with a bit-identical guard
against untouched master and a cold review before merge. Guards held exactly on every lane.
Two real defects surfaced on the way: `network.skip_connections` / `multi_head` had crashed
on every model with a steady state (never usable, now fixed, #9) and `interp.ablate_neuron`
carried the mixed-links NaN hazard (#9). The five-variant train-step helpers now exist for
real (`optimizers/_step_common.py`, #6), with a recorded-reference guard that is exact on
the recording platform and skips elsewhere — a cross-platform bit-identical assertion is not
a thing (CI differs by jax version and by x64 leaking from earlier tests). The DGX was
unreachable for the whole batch; all suites ran on the laptop CPU. Left for batch 3 by
design: the `ndim == 3` helper in loss.py, the six rollout copies' training-side docstring,
`make_composite_loss`'s unused `model` parameter.*

| item | proof | action |
|---|---|---|
| ✓ five train-step variants repeat the same 12-line `compute_loss(...)` call ×10 and the same 16-line finalize tail ×5; the `_prepare_step`/`_finalize_step` helpers that `docs/session_log.md:55` and the world-arm spec say exist **do not exist** (`grep -rn src → 0`) | `standard.py:58-102`, `pcgrad.py`, `mao.py`, `lbfgs.py`, `gauss_newton.py` | write the helpers for real; fix the two docs |
| ✓ nested-config block dispatch written three times (`io.py` ×2, `train.py:from_dict`) + a fourth in `irf.py`; they have diverged: **`--set coverage.enabled=true` raises "Unknown keys"** | reproduced | one loop over `TrainConfig.model_fields`; fixes `--set coverage.*` |
| hard clip with `inf→1e10` ×4; `output_links` assembly duplicated verbatim in `DisasterPolicyNet`; `states[:, -1, :] if ndim == 3` ×7; hand-rolled input normalization in `ResMLP`/`MultiHeadMLP` diverging from `common._normalize_input` | sweep | helpers in `networks/common.py` / `training/history.py` |
| disaster `_build_state` copy-pasted between the deterministic and risky SS solvers; L-BFGS loop duplicated between `steady_state.py` and `warm_start.py` | sweep | one helper each |
| six hand-rolled simulation loops in `evaluate/diagnostics.py` (×3), `irf.py` (×2), `cli.py`; only `training/episode.py` routes through `shocks.simulation_step`, whose docstring claims the others do too | sweep | one parameterized rollout |
| `state_init.py` is 325/759 lines of validation; `composite_loss.py` hosts the builder for every non-composite loss (and self-imports) | sweep | `training/validation.py`, `training/loss_builder.py` |
| `_matrix_power_neg_quarter` duplicated (shampoo, mao_kfac); Shampoo ignores `config.decay`; `precond_update_freq` gates only the cheap part | sweep | fix or retire with the zoo |

### B3. Batch 3 — decisions for the maintainer

- Sequence nets (`lstm`, `transformer`, `history_len`, ~530 lines + guards): no tracked
  config, known off-by-one. Delete, or keep and label "experimental, no shipped config".
- `KfAnchoredMLP` + `sweep_disaster_kf_validation.py` + its config (~640 lines): the only
  users of each other. Delete together (no local `.eqx` depends on it; DGX unverified).
- The three disaster reparams (~490 lines) and `disaster_calvo`: retire, or ship one arm and
  a wage-side forward-pass test.
- `ResMLP`, `MultiHeadMLP` (~200 lines): archive-configs only.
- Optimizer zoo: `mao_kfac` (261 lines, archive configs only), `lion`/`adamw`/`sgd`
  (nowhere), `ReduceLROnPlateau` (+4 config fields, tests only). Keep `ngd`/`mao`/`gn`/
  `lbfgs`/`shampoo`/`muon` only if the April second-order sweep is meant to be re-runnable.
- Dynare stack (`evaluate/dynare.py`, `dynare_io.py`, 11 skipped tests, `moment_matching`,
  `warm_start_dynare`): never runs in CI because `dynare/` is gitignored. Commit a small
  fixture set or accept it as unverified and say so in the docs.
- `active-subspace` CLI (694 lines): undocumented; keep the library functions, drop or
  document the command.
- The tested-only public surface: three `plots/` helpers, `VariableSpec.pack_*`/`get_*_idx`,
  15 of 18 `optimizers.__all__` re-exports, `DESCRIPTION` in 11 `variables.py` files
  (re-typed by hand in `models/__init__.py`).

### B4. Numbers

| sweep | deletable now | deletable after a decision |
|---|---|---|
| config | ~120 | ~150 (Field-constraint conversion) |
| training | ~215 | ~490 move |
| optimizers | ~290 | ~260 (`mao_kfac`) |
| models | ~150 | ~800 (reparams, aiyagari) |
| networks | ~250 | ~1,450 (viz, sequence nets, kf, Res/MultiHead) |
| evaluate/irf/cli | ~470 | ~150 (loop collapse) + Dynare decision |
| scripts/tests/configs/CI | ~2,000 (821 in configs) | ladders → `--set` (~570) |

Order of magnitude: a quarter of the 24k-line library is either dead, duplicated, or a
research branch whose verdict is already in the chronicle.

---

## C. What the reviewer did not do

- No agent findings were acted on; the only code change from this review is PR #3.
- Track A did not re-derive any model equation (verified clean 07-06/07 and by Codex).
- No training run; the DGX GPU driver is down. The four cheap experiments this review asks
  for (baseline ×3 with the fix; pin-alone arm; anchor-annealed arm; unclipped probe) are all
  container jobs of the existing sweep launcher.
