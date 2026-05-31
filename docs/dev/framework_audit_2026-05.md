# DEQN-JAX Framework Audit — 2026-05-15

Scope: framework health, Brock-Mirman SS bug, new-model authoring. Disaster
model *internals* out of scope; disaster *coupling into the framework* in scope.
Method: 13 subsystem auditors → adversarial verification (fresh re-read,
default-to-refute) → completeness critic. Severities below are **post-verification**.

Verification coverage: 12/13 units adversarially verified. The `types.py/misc`
unit is **finder-only (unverified)** — flagged inline.

---

## Headline

The library is **healthier than the "2000-line monolith / 22 files reference
disaster" headline suggests.** Two big refactors you already did (composite_loss
de-disaster-ification via `composite_aux_fn`; ModelSpec discrete-chain hooks)
genuinely landed and are clean. All 8 *registered* models satisfy the ModelSpec
contract (SS residual ≤ 3e-5). The real problems are **concentrated**, not diffuse:

1. **A ship-breaker in the uncommitted diff** — tracked configs point at an untracked module.
2. **Disaster coupling that's still real** — but it's ~6 specific leaks, not 22 files of rot.
3. **Brock-Mirman is NOT broken** — the "three-way SS mismatch" is a calibration mixup. Goal #2 is essentially already solved; it just needs documenting + a regression test.

---

## P0 — Ship-breaker (fix before anything else)

### trainer-01 [critical→medium, partially-confirmed] Fresh checkout ImportErrors
`trainer.py:213-253`. The uncommitted diff adds a `disaster_policy_net` branch
importing `create_disaster_policy_net` from `models/disaster/network.py` — **which
is untracked** (it's in your `??` list). The same diff flips `disaster*.yaml` +
~26 configs to use it. So a clean checkout (or anyone else) hits ImportError, and
the disaster configs are broken until that file is committed.
**Fix:** commit `models/disaster/network.py` atomically with the config flips, or
hold the flips until it lands. Verify by `git stash -u && train disaster`.
> This is the concrete form of the "giant uncommitted refactor" — the refactor
> itself is a *good* decomposition per the auditor; it's just not atomically tracked.

---

## P1 — Disaster coupling (your central concern): the real map

The coupling is **6 genuine leaks**, all the same shape (framework code hardcodes
the magic key `'p_disaster'` / kwarg `d_disaster` / disaster field names) and all
fixable by the same move you already used twice: promote to a declared ModelSpec hook.

| ID | Sev | File | Leak |
|----|-----|------|------|
| config-01 | high | `config.py:685-713` | 8 disaster-only fields on generic `NetworkConfig` (`use_zlb_feature`, `kf_names`, `zlb_feature_kind`, …) |
| trainer-02 | high | `trainer.py:213-270` | `create_train_state` hardcodes disaster net-type branches + literal K/F var names |
| cli-01 | high | `irf.py:463` | generic `irf` CLI defaults to disaster's 5 shock names for *every* model |
| eval-04 / coupling-01 | medium | `evaluate.py:112-225`, `loss.py:170-228`, `linearize.py:68-241` | `p_disaster` two-point mixture hardcoded by name into 4 generic numerical paths |
| loss-03 / coupling-04 | medium | `composite_loss.py:219,331`, `config.py:389-408` | `leverage_mult`/`newton_weight`/`barrier_weight` knobs + their economic framing still in generic config (computation moved, *weights* didn't) |
| coupling-03 | low | `evaluate.py:321-360` | `market_clearing_errors` hardcodes disaster's resource-constraint accounting |

**The unifying fix:** add one generic `mixture_fn` / `rare_event_fn` hook on
ModelSpec (mirroring `composite_aux_fn`), move the disaster-only `NetworkConfig`
fields into a model-scoped `network.extra: dict`, and have CLI/benchmark default
shock names from `model.shock_names`. After that, the only place that knows the
word "disaster" is `models/disaster/`.

### Correctness sub-issue surfaced here
- **loss-01 / coupling-02 [→low]** Discrete-Markov branch (gated on `transition_matrix`)
  and disaster mixture (gated on `p_disaster>0`) can **both fire and double-count**
  the expectation if a model ever sets both. No guard today. Cheap fix: reject the
  combination in config validation, or gate the discrete branch on resolved
  `expectation_type` instead of sniffing model fields.

---

## P1 — Goal #2: Brock-Mirman SS "bug" — RESOLVED (it's not a bug)

**Two independent units agree: the shipped `brock_mirman/steady_state.py` is
correct and self-consistent.** `k* = ((1/β − 1 + δ)/α)^(1/(α−1)) = 6.367`,
`sav_rate* = δ·k*/y* = 0.327`, deterministic Euler residual ≈ 0.

The "three-way mismatch" in project memory (`sim 4.0 / 0.18 / 14`) is a **calibration
mixup**, not a solver bug:
- `0.18` = the full-depreciation (δ=1) closed form `k*=(αβ)^{1/(1-α)}` — a *different model*.
- `14` = partial-δ formula evaluated with an off-canonical β.
- The 2026-04-30 fixture/notebook silently ran **β=0.96** while the closed-form check used the canonical β.

**Action (not a fix — a close-out):**
1. Close task #110 / `known_issue_bm_ss_discrepancy.md` as "not a bug — calibration mixup."
2. Add a docstring table in `steady_state.py` enumerating the three k* values and which convention each assumes (bm-ss-04).
3. Add the regression test that's currently missing (tests-03): pin `k_ss` to closed form at rtol 1e-6 + a simulation-consistency test. **Right now the bug is pinned by no test, so any future change has no safety net.**

Related real issue uncovered while verifying this:
- **steadystate-01 [high, confirmed]** The *numerical* SS fallback
  (`training/steady_state.py:58-119`) minimizes residuals over the full `[state;policy]`
  vector with no fixed-point selection and **no pinning of exogenous states** — it can
  silently converge to a wrong, under-determined SS. brock_mirman uses the analytical
  path so it's unaffected, but any new model relying on the numerical fallback is at risk.

---

## P2 — Monoliths (decompose behavior-preserving)

- **evaluate.py (1022)** — 4 simulation primitives each hand-roll their own JIT'd
  step closures instead of reusing `training/shocks.py:simulation_step` (which
  `episode.py` already uses). Worse, **eval-01 [high, confirmed]: `simulated_moments`
  and `stability_check` silently skip the disaster branch** while `euler_equation_errors`
  was patched to include it — so two diagnostics report from a distribution that never
  visits disaster states. Fix routes all four through one `eval_simulation_step`.
- **config.py (1663)** — config-02: adding a nested config block requires editing
  ~7-8 hand-synced sites; derive the nested-config set programmatically from
  `TrainConfig.model_fields`.
- **trainer.py (1790)** — the variant dispatch is actually *clean* post-refactor;
  the only structural problem is trainer-02 (disaster branches in `create_train_state`).

---

## P2 — Goal #3: new-model authoring

- **newmodel-01 [medium]** Hook surface is good (`transition_matrix`, `composite_aux_fn`,
  `setup_fn`, … all duck-typed), and there's a real guide at
  `docs/site/models/implementing.md`. But authoring is gated by scattered network/SS
  requirements and **CLI/benchmark model lists are hardcoded literals** (bench-01,
  cli-02/04) so a newly-registered model doesn't appear automatically.
- **models-02 [→low] / tests-01 [high]** `EQUATION_NAMES` order vs `equations()` dict
  insertion order must agree but nothing checks it — a silent per-equation mislabel.
  And **there is no parametrized contract test over registered models**, so a broken
  new ModelSpec passes CI silently. This is the single highest-value missing test.
- **register_model() is unreachable from the CLI** (cli-02) — the "no fork required"
  promise needs a `--register-module MOD` flag.

---

## P3 — Quick wins / dead code (mostly trivial effort)

- **models-03 / bm-ez-03** `models/brock_mirman_ez/` is an **empty package** (only `__pycache__`), registered nowhere → delete.
- **opt-03** `GaussNewton.solve_method` documents 3 options, **implements none** → implement or delete the param.
- **opt-01** Shampoo **silently ignores `config.decay` and `config.block_size`** → wire `beta=config.decay` in.
- **opt-06** `muon` is registered + config-listed but has **no training test**.
- **config-06** `activation` docstring advertises `sigmoid` but `VALID_ACTIVATIONS` rejects it → confusing error.
- **cli-03** Dead `if (...): pass` no-op with mis-precedenced condition in `run_train`.
- **cli-06** Duplicated comment block in `api.py:101-111`.
- **networks-06** `kf_anchored_mlp.py` imports `_to_tuple` only to silence ruff (`_ = _to_tuple`) — dead.
- **coupling-05** Dead `n_samples = K` in `loss.py:322`.
- **reweighting-01** `ReweightState.running_max` is an EMA, not a max — misnamed/misleading.
- **types-04** Dead-code sweep result: `interp.py / active_subspace*.py / dynare_io.py` are **cleanly isolated, NOT dead** (reachable via `active-subspace` CLI / notebooks). No removal needed; optionally relocate `interp.py` to `examples/`.

---

## P3 — Network duplication (known: LinearPlusMLP vs KfAnchoredMLP)

- **networks-02 [→medium]** The residual "BK-linear + MLP-delta + output_links"
  forward is **triplicated** across `LinearPlusMLP`, `KfAnchoredMLP`, and disaster's
  `DisasterPolicyNet (923 lines)`. Extract shared helpers into `networks/residual_ansatz.py`.
- **networks-03 [→medium]** Generic `viz.py` imports the disaster model directly
  (`from deqn_jax.models.disaster.network import DisasterPolicyNet`) — invert via a
  renderer-registration hook.
- **networks-05** `ResMLP` / `MultiHeadMLP` are unreachable from config (plan-mode only).

---

## Test suite (goal-blocking gaps)

| ID | Sev | Gap |
|----|-----|-----|
| tests-01 | high | No contract test over registered models — broken ModelSpec passes silently |
| tests-03 | high | Brock-Mirman SS bug pinned by no test — fixing it has no regression net |
| tests-02 / tests-06 | med | Generic output-link/moment features tested ONLY through disaster (so framework refactors are blocked by disaster + pay full SS solve). Switch to brock_mirman. |
| tests-04 | med | Convergence tests assert hard thresholds on un-seeded chaotic trajectories (your bifurcation pain). Seed + use ratio assertions. |
| tests-05 | med | Zero-coverage generic modules: `cli`, `metrics`, `checkpointing`, `reporting`, `reweighting` math, `steady_state` |
| tests-07 | low | `@pytest.mark.slow` unregistered → `-m 'not slow'` doesn't work |

---

## Unverified (verifier was killed mid-run) — treat as finder-only

types.py/misc unit, NOT adversarially checked:
- **types-01** ModelSpec is a 28-field god-object (8 required, ~20 optional, 6+ disaster-driven); required/optional split is invisible. Suggest grouping optional hooks into a `ModelHooks` sub-struct.
- **types-02/03/05** discrete-chain fields used by zero shipped models; `getattr(model, '<declared_field>', None)` defensive access on real fields (typo-unsafe); field docstrings bake disaster into the generic contract.
- **types-06** TrainState has 5 always-None advanced-feature slots (same sprawl).

---

## Suggested sequencing

1. **Commit the untracked disaster network module** (trainer-01) — unblocks everyone, 5 min.
2. **Close out Brock-Mirman** (docstring table + regression test tests-03) — claims goal #2.
3. **Add the model-contract test** (tests-01) — makes every later refactor safe.
4. **Disaster decoupling pass** (config-01, trainer-02, cli-01, coupling-01) — one `mixture_fn`/`network.extra` move clears most of P1.
5. **Quick-win sweep** (the P3 trivials) — cheap, high signal-to-noise.
6. **Monolith decomposition** (evaluate.py first; it has a correctness bug eval-01) — behavior-preserving, verified by seeded bit-identical curves.
