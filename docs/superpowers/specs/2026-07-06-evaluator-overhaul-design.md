# Evaluator overhaul: expectation-based errREE + per-equation unit normalization

**Status:** SPEC — awaiting sign-off (C-remainder of the 2026-07-02 slop audit; changes every
reported accuracy number, which is why it ships separately from batches A/B).
**Scope:** `evaluate/diagnostics.py` (primary), model `equations.py` metadata (one new optional
export), notebooks/evidence tables (follow-up, not this change).

## Problem (audit major #2, verified 2026-07-02)

`euler_equation_errors` is the repo's headline accuracy metric ("errREE decades"). Two defects:

1. **Single-draw realized residuals.** The simulation loop computes each period's residual with
   the *same realized shock* that advances the state. The equilibrium condition is `E[r] = 0` —
   a perfect policy has `r ≠ 0` on any single draw. The reported number is therefore floored by
   the residual's *shock variance*, not the policy's accuracy: beyond some training quality,
   errREE stops improving no matter how good the policy gets. The evidence harness already
   works around this externally (`_expected_errREE`); the discrete-chain branch inside
   `euler_equation_errors` already does it correctly (exact enumeration over the transition row)
   — the Gaussian-shock path is the only one still measuring noise.
2. **Fixed grade thresholds applied across mixed units.** `print_euler_errors` grades
   log10 |r| against fixed cutoffs, but residual units differ per equation and per model
   (irbc: λ-units vs i-units ≈ 3 decades of spread; disaster mixes wedge/price/quantity units).
   A "Good" on one equation and a "POOR" on another can describe the same relative accuracy.
   `brock_mirman/equations.py`'s docstring promises a unit normalization that `evaluate/` never
   implements.

Both defects push in the same direction: **published accuracy tables are provisional** (this is
flagged in `docs/how_it_all_works.md` §3.3 and was independently found by codex review, #6).

## Design

### D1. Expectation-based residuals (the estimator change)

At each visited state `s_t` (path simulation unchanged — the *measure* is still the ergodic
path), compute the residual as an expectation over next-period shocks, mirroring the trainer's
own operator hierarchy:

- **quadrature** when `n_points^n_shocks` is affordable (reuse `gauss_hermite_nd`; same limit
  logic as training) — exact for practical purposes; default for every current model
  (brock_mirman 1 shock, irbc 3, disaster 5 → 3^5 = 243 nodes, fine);
- **antithetic MC** fallback (N configurable, default 64) when quadrature is unaffordable;
- **discrete-chain enumeration** kept as-is (already correct);
- disaster branch: expectation must mix the Bernoulli disaster indicator into the node set when
  `p_disaster > 0` (two-point mixture over quadrature grids), not draw it — otherwise the
  disaster branch re-enters through the noise door.

API: `euler_equation_errors(..., expectation: str = "auto")` with `"auto" | "quadrature" |
"mc" | "realized"`. **`"realized"` keeps the old estimator available** for comparison and for
measuring the noise floor itself; the report labels which estimator produced it. Default flips
to expectation-based — this is the number-changing decision that needs sign-off.

### D2. Per-equation unit normalization (the reporting change)

Models declare their residual scales; the evaluator stops guessing:

- New optional export in each model's `equations.py`:
  `RESIDUAL_SCALES: dict[str, Callable(state, policy, defs, constants) -> Array]` (or a constant
  1.0), mapping each residual into a **dimensionless relative error** (Euler residuals ÷
  marginal-utility level, resource constraints ÷ output, FB residuals ÷ typical investment
  scale). Declared per model because only the model knows its units — the framework cannot
  infer them.
- `ModelSpec` gains `residual_scales_fn: Optional[...] = None`. When absent, the evaluator
  reports **raw units and says so** (no letter grades — grades only apply to normalized
  residuals). This makes the current mixed-unit grading impossible rather than default.
- brock_mirman first (its docstring already specifies the intended normalization), then irbc
  (λ/i units — the 3-decade spread test case), then disaster.

### D3. What does NOT change

- The simulated path / burn-in logic, `simulated_moments`, `stability_check`, CLI surface.
- Training loss (already mean-then-square; audit-verified clean).
- No backfill of old notebooks in this change; re-running them is the follow-up that produces
  the new public tables (and re-measures the stale ~0.85-decade gap claim, memory
  `accuracy_gap_vs_simon`).

## Tasks (TDD order)

1. `expectation="realized"` refactor: extract the residual-at-state computation; existing
   behavior bit-preserved (regression test on brock_mirman, fixed seed).
2. Quadrature path + test: on brock_mirman, expectation-errREE of the ANALYTIC policy must sit
   at machine precision (the analytic policy has E[r]=0 exactly; the realized estimator provably
   cannot reach it — that contrast IS the test), and beat realized-errREE on a trained
   checkpoint by a visible margin.
3. Antithetic-MC path + test: agrees with quadrature within MC tolerance on brock_mirman.
4. Disaster two-point mixture + test: p_disaster>0 changes the expectation continuously
   (no crash, no silent exclusion; extends `tests/test_disaster_eval_smoke.py`).
5. `RESIDUAL_SCALES` for brock_mirman + ModelSpec plumbing + test: grades invariant under a
   pure unit rescaling of an equation (multiply a residual definition by 100 in a test model →
   normalized report unchanged, raw report shifts 2 decades).
6. irbc scales + the 3-decade-spread regression shrinks accordingly.
7. Docs: config/API reference entry; `how_it_all_works.md` §3.3 caveat updated to "fixed" with
   the receipt.

## Acceptance

- Suite green; new tests cover all four estimator branches.
- `evaluate` CLI on a brock_mirman checkpoint prints: estimator label, per-equation normalized
  decades (where scales declared), grades only on normalized values.
- One before/after table (brock_mirman + irbc) checked into the spec as the RESULT section.

## Risks / notes

- Every published accuracy number shifts (mostly *down* = better, since noise floors vanish;
  normalized grades can shift either way). Requires the same-day notebook re-run or an explicit
  "tables predate the evaluator fix" banner — maintainer's call on sequencing.
- disaster expectation at 3^5 nodes × long paths is slower; mitigate with node-count knob and
  the MC fallback.
- Scales are model-authored → they can be wrong; the unit-rescaling invariance test only guards
  the plumbing. Referee pass on the declared scales is cheap and worth it.

## Bundled C-remainder items (same sign-off, independent diffs)

- **bm_labor gamma SS:** `steady_state()` ignores `gamma`; a constants override `gamma≠1`
  silently poisons SS/warm-start/anchors. Fix: general-gamma solve or hard `raise` on
  `gamma != 1`. (Audit minor; one function + test.)
- **warm-start additive noise:** `warm_start.py` uses multiplicative noise ⇒ zero variation on
  zero-SS dims (z in brock_mirman, m_p in disaster) ⇒ those input slopes stay at random init.
  Fix: additive noise scaled by per-dim ergodic sd (fallback: |ss| or 1.0). (One line + test.)
