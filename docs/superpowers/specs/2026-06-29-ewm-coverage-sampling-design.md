# EWM Coverage Sampling — Design (v1, irbc)

**Date:** 2026-06-29
**Branch:** `ewm-coverage` (private/local; do not push without explicit ask)
**Status:** approved design; revised after adversarial review (3-lens workflow) and a conceptual
re-review against the paper's §3.4 coverage construction + Appendix "IRBC coverage-measure
specification" (2026-06-30: stress box rescaled to the model's stationary law, repair clip added,
ρ/H aligned to the paper's coverage-exact arm, multi-seed protocol adopted); pre-implementation

## Motivation

Scheidegger & Schaab (2026), *Equilibrium World Models* (arXiv:2606.23463), diagnose the standard
on-policy DEQN solver — which is what deqn-jax is — as producing **self-confirming solutions**: the
equilibrium residual is imposed only on states the solver's own training measure visits, so the
solution is accurate there but untested off it (after rare shocks, near binding constraints, in
tails, along counterfactuals). Signatures: low on-measure residual + large off-measure residual,
strong seed dependence, closed-loop instability.

Their fix ("EWM") *changes the computational representation, not the economics*: impose the **same**
exact residual on a broader **coverage measure** = base ∪ stressed ∪ locally-perturbed states, where
off-path seeds are rolled forward through the **exact** model transition. On small-shock models the
learned continuation surrogate is inert; **coverage sampling alone** is the lever.

deqn-jax's `irbc` model is the paper's headline economy **minus its disaster block**: the paper's
IRBC carries a rare persistent disaster in the state (`d∈{0,1}`, 30% TFP collapse, δ 0.01→0.06,
asymmetric country volatilities); ours is the course calibration — irreversibility yes, disaster
no, symmetric σ. Their headline failure is disaster-region certification (0/10 seeds verified);
ours is closed-loop **instability** of the plain recipe (ρ(SS)=1.23 — simulations drift out of the
training rect, ARC violated O(1), negative investment). Both are coverage-gap symptoms, but this
work is an *adjacent test*, not a replication: does coverage repair the instability/selection
failure? Direct support from the paper: its surrogate-free **`DEQN-coverage-exact` control** —
exact quadrature + coverage measure, i.e. *exactly this v1 arm* — reached 8/10 (N=2) and 9/10
(N=4) verified seeds vs **0/10** for the pathwise baseline, and in the Brock–Mirman lab coverage
alone captured essentially all of the disaster-region gain. Our current irbc remedy is a
BK-anchored composite loss (ρ(SS)=0.981), which the paper classes as an "ex-ante engineering fix
that treats the symptom"; coverage is the structural alternative.

> **Note on irbc's training measure.** `configs/irbc.yaml` uses `episode_length: 1` +
> `initialize_each_episode: true`, so the base batch is a fresh uniform draw from the init rect
> (`k∈[0.9,1.1]`, `z∈[-0.05,0.05]`) each cycle — a fixed, policy-independent **init-rect** measure,
> not a simulated ergodic set. We therefore call pool 1 the **base** pool (not "ergodic"). Coverage
> here broadens enforcement *beyond a fixed init rect*; the mechanism is identical to the paper's,
> the label differs.

## Goal

Port EWM **coverage-only, ρ-weighted** sampling into deqn-jax as a config-gated training measure,
validated on `irbc`. The deliverable is the machinery + the experiment that produces the comparison;
the *outcome* of the experiment is reported, not a merge gate.

### Definition of Done (engineering — follows from what is built)

1. `CoverageConfig` + `training/coverage.py` implemented; coverage installs as a `compute_loss_fn`
   wrapper inside `_build_custom_loss_fn`.
2. **Bit-identical when disabled**: with `coverage.enabled=False`, one train step is *exactly* equal
   (params, opt_state, loss) to the baseline path — exact equality, same PRNG seed (see Testing).
3. All validators in place and tested (mutual exclusions, empty-pool guard, name check).
4. Unit + integration tests green; `uv run pytest tests/ -v` passes.
5. `configs/irbc_plain.yaml` and `configs/irbc_ewm.yaml` checked in; the three-way comparison table
   is produced by a reproducible evaluation path.

### Research finding (reported, NOT an acceptance gate)

**Multi-seed protocol (the paper's, scaled down).** The paper runs ten seeds per arm and reports
medians + verified counts, because distinct seeds settle into distinct self-confirming basins — a
single-run comparison can mislead in either direction (their pathwise arm's failure is structural:
3× the episode budget does not repair it). We run **5–10 seeds per arm** (`--set seed=<s>`) and
report per-seed ρ(SS), the count passing ρ(SS)<1, and median/p90 stress-region residuals:

| recipe | config | metric | note |
|---|---|---|---|
| plain MLP + MSE | `irbc_plain.yaml` | ρ(SS) per seed (documented single-seed: 1.23) | self-confirming, unstable baseline |
| LinearPlusMLP + BK-anchor | `irbc.yaml` | ρ(SS) per seed (documented: 0.981) | engineering fix (symptom) |
| plain MLP + coverage | `irbc_ewm.yaml` | **measured, N seeds** | structural fix (the bet) |

The hypothesis (drawn from the paper's coverage-exact arm: 8/10, 9/10 verified vs 0/10 pathwise)
is ρ(SS)<1 for most seeds and a materially lower stress-region residual. A correct implementation
that yields ρ(SS)≥1 is a valid *negative result*, not an engineering failure.

**RESULT (2026-07-02, 5 seeds/arm, 4001 episodes, DGX; fixed policy-independent eval sets,
27-node GH expectation):** a split verdict.

| metric (median over 5 seeds) | irbc_plain | irbc_ewm | delta |
|---|---|---|---|
| stress-region max(fb_0,fb_1,arc) mean (E[r])² | 4.79e-2 | **3.11e-4** | **~154× lower (2.2 decades)** |
| base (init-rect) total residual | ~1.5e-4..1.7e-3 | same range | no on-measure cost |
| ρ(SS) | 1.11 [1.09, 1.29] | 1.15 [1.11, 1.26] | **unchanged; 0/5 pass both arms** |

1. **The coverage-gap/certification claim fully replicates**: the plain arm shows the textbook
   self-confirming signature (base residual ~1e-4, stress residual ~5e-2 — 500× worse exactly
   where irreversibility binds), and coverage closes it by >2 decades at zero on-measure cost —
   stronger than the paper's ~1-decade BM-lab figure.
2. **The ρ(SS)<1 bet is a clean negative**: coverage does NOT repair closed-loop instability at
   the SS on this (no-disaster) irbc. The selection/stability failure is evidently a *different*
   pathology from the coverage gap — consistent with the BK-anchor recipe (ρ=0.981) remaining the
   working fix for selection, while coverage is the fix for off-path certification. Note the
   paper's "verified stationarity" is a training-loop fixed-point criterion, not a
   closed-loop-dynamics one, so this does not contradict the paper's IRBC result; it sharpens the
   distinction between the two notions of "verified."

**Protocol rule (adopted from the paper):** coverage design parameters (box, ρ's, H, σ_local) are
fixed a priori by the coverage criterion — enough mass on the rare/post-shock region, local radius
a small fraction of the ergodic spread — and are **never tuned against the reported metrics**.

**Criterion note:** the paper's per-seed pass criterion is "verified stationarity" — the trained
policy is time-invariant under continued training (sup-norm change < 10⁻³ on a fixed held-out
coverage-distributed set) plus a disaster-region residual threshold. That is a *training-loop
fixed-point* check; our ρ(SS) is a *closed-loop dynamics* check. We keep ρ(SS) as the headline
(it connects to the documented 1.23 / 0.981 numbers) and may add the policy-stationarity check as
a cheap secondary diagnostic.

**Stress-region metric (quantified).** On a fixed held-out stress grid (the stress box sampled once
with a pinned seed, rolled through the exact Γ), report the mean expected squared residual
`mean (E[r])²` **per equation** (at minimum `fb_0`, `fb_1`, `arc`) for plain vs coverage. Headline
number: percent reduction in the max-over-{fb_0,fb_1,arc} stress residual, coverage vs plain.

## Non-goals / v1 scope restrictions (enforced by validators)

- **No** continuation surrogate / action-conditioned continuation / distributional encoder (EWM
  depth ③ — deferred to a later `disaster` cycle; irbc's expectation is cheap quadrature).
- v1 coverage requires **STANDARD optimizer + `loss_type: mse`** only. Reject at config validation:
  - `coverage.enabled` with `loss_type == "composite"` (mutual exclusion);
  - `coverage.enabled` with `optimizer.name ∈ {mao, lm, gn, ign, lbfgs}` or
    `gradient_surgery == "pcgrad"` (these differentiate the per-equation/residual vector, not the
    pooled scalar — coverage pools would be silently dropped from the gradient);
  - `coverage.enabled` with `barrier_weight > 0`, `loss_choice != "mse"`, or
    `moment_matching.enabled` (the coverage wrapper wraps plain `compute_loss`; these layered terms
    would otherwise be silently dropped).
- v1 coverage is MLP-only: reject `network.history_len > 1` (sequence nets need [B,H,D] pool
  shapes — follow-up, mirroring the replay-buffer gate).
- v1 coverage rejects `replay_buffer.enabled` (the buffer concatenates old-policy states into the
  batch, muddying the "base pool" semantics; interaction untested).
- Only `irbc` validated this cycle. `disaster` reuse is later and needs a model-specific stress-seed
  map (its 8 endogenous states are not free coordinates — see Risks).

## Architecture

Coverage is an **alternative training measure** for the existing residual loss. Same equations, same
network, same optimizer, same `compute_loss`. The one new idea is a **loss wrapper** evaluating the
existing `compute_loss` on three state pools and taking a **mixture-weighted** sum:

```
L = ρ_base · L(base) + ρ_stress · L(stress) + ρ_local · L(local),   ρ's normalized to sum to 1
```

Normalizing ρ to a probability mixture is faithful to the paper (μ_K is a mixture measure) and keeps
`total` on the single-pool scale, avoiding an implicit LR shift. Default weights follow the paper's
coverage-exact arm: path:stress:local = **1 : 0.5 : 0.25**. Pools with zero weight are skipped at
build time (Python-level, pre-JIT), so setting ρ_stress=ρ_local=0 collapses the wrapper to *exactly*
the plain `compute_loss` — the paper's κ=0 ⇒ DEQN identity, and a unit test.

**Repair (from the paper's replicable spec):** every generated coverage state is **clipped to a
feasible box** (`repair_ranges`, e.g. z within ±4 stationary sd) *before* the residual is evaluated —
"the repair bounds the simulation only, never the residual"; no penalty term. This is the paper's
own mitigation for rollouts under a bad interim policy landing at extreme states whose residuals
would dominate the loss.

It installs exactly where `composite_loss` does — as the `compute_loss_fn` built inside
`_build_custom_loss_fn` (trainer.py) and passed through to `make_train_step`. When
`coverage.enabled=False`, the wrapper is simply not installed → the train step is bit-identical to
baseline.

### Why a wrapper (not editing `compute_loss`)

`compute_loss` (loss.py:264) takes the state batch as an **explicit argument** and only builds
*shocks* internally — the state measure is fully external and swappable. A wrapper touches **zero**
lines of `compute_loss` and the five train-step variants, and (because v1 is full-batch on irbc:
`batch_size=128`, `n_minibatches=1`, `n_epochs=1`) avoids any minibatch-shuffle weighting problem.

### Stop-gradient is mandatory (not analogous)

The baseline ergodic batch is detached **by the JIT boundary**: `rollout_fn` and `grad_step` are
separate `@jax.jit` calls, so the trajectory enters the differentiated loss as a constant. Coverage's
`roll_states` runs **inside the grad tape**, so its pool states must be wrapped in
`lax.stop_gradient` explicitly. Gradient then flows only through `policy_fn` re-evaluated at the
fixed pool states — exactly the DEQN/EWM convention (sampling is not differentiated).

## Components (each independently testable)

### 1. `src/deqn_jax/config/coverage.py` — `CoverageConfig`

Pydantic v2 `_ConfigBase` subclass, `model_config = ConfigDict(extra="forbid")`, mirroring
`ReplayBufferConfig` (config/replay.py) end-to-end.

| field | type | default | meaning |
|---|---|---|---|
| `enabled` | bool | `False` | master switch |
| `rho_base` | float | `1.0` | mixture weight on the base (init-rect) pool |
| `rho_stress` | float | `0.5` | mixture weight on the stress pool (paper's coverage-exact arm: 0.5 relative to path) |
| `rho_local` | float | `0.25` | mixture weight on the local pool (paper: 0.25; normalized internally) |
| `n_stress` | int | `128` | stress seeds drawn per step |
| `n_local` | int | `128` | local perturbations per step |
| `rollout_horizon` | int | `5` | H: steps to roll stress seeds through exact Γ (paper: "typically 3 or 5") |
| `local_sigma` | float | `0.02` | Gaussian std for local perturbations (state units; scalar — see Risks for the per-dim caveat) |
| `stress_ranges` | dict[str, tuple[float, float]] | `{}` | per-state-name uniform box for stress seeds; names are **state names** (resolved later against `model.state_names`). Empty ⇒ error when enabled with `rho_stress>0`. |
| `repair_ranges` | dict[str, tuple[float, float]] | `{}` | per-state-name feasible box; stress landings + local perturbations are `jnp.clip`ped into it before the residual (paper's repair step). Empty ⇒ no clipping. |

Validators (copy replay.py structure; these need **no model handle**):
- `@field_validator(..., mode="before")` int/float coercers from `config/_base.py`.
- `@model_validator(mode="after")`:
  - weights ≥ 0 and not all zero;
  - **empty-pool guard**: `rho_stress>0 ⇒ n_stress>0`; `rho_local>0 ⇒ n_local>0` (a weighted empty
    pool → `jnp.mean` over 0 rows → NaN);
  - `rho_stress>0 ⇒ rollout_horizon≥1 and stress_ranges non-empty`;
  - `n_stress, n_local, rollout_horizon ≥ 0`; `local_sigma ≥ 0`;
  - each `stress_ranges` and `repair_ranges` entry is `[low ≤ high]`.

Name-vs-model validation is **not** here (Pydantic has no model handle) — see §3.

### 2. `src/deqn_jax/training/coverage.py` — rollout helpers + wrapper

- `sample_stress_seeds(model, key, n, stress_ranges) -> seeds [n, n_states]`
  Uniform draw in the `stress_ranges` box, addressing dims by `model.state_names.index(name)`.
  Dimensions **absent** from `stress_ranges` are filled from the steady state
  (`model.steady_state_fn`), yielding complete seed vectors. (Names are pre-validated in §3, so an
  unknown name never reaches here.)

- `roll_states(model, policy_fn, seeds, key, horizon, shock_scale, lo=None, hi=None) -> states [n*horizon, n_states]`
  Rolls `seeds` forward `horizon` steps through the exact transition by reusing the
  `simulation_step`/`run_episode` path (which takes `policy_fn`, enforces the 2-D `[B, n_shocks]`
  shock contract, and auto-handles any disaster Bernoulli channel). **The rollout uses sampled
  (Monte-Carlo) shocks even when the loss uses quadrature** — state generation (sampling the
  reachable set) and the expectation `E[r²]` (evaluated at a fixed state) are different objects and
  do not conflict; both use standard-normal innovations and `step_fn` applies σ internally.
  Returns the **projected landings** `s_1 … s_H` (the states *after* transition; the raw off-manifold
  seed `s_0` is **excluded** so the imposed measure is on the reachable set), **clipped to the
  repair box** (`lo`/`hi` vectors, ±inf where unspecified) and `lax.stop_gradient`'d, shape
  `[n*H, n_states]`. (Implementation note: `run_episode`'s trajectory holds pre-transition states,
  so landings = `trajectory[1:]` ++ `final_state`.)

- `make_local_pool(states, key, n, sigma, lo=None, hi=None) -> states [n, n_states]`
  Sample `n` rows of the base `states` (with replacement, via a **split** subkey), add
  `Normal(0, sigma)` per state dim (second subkey), clip to the repair box, `lax.stop_gradient`'d.

- `make_coverage_loss(base_compute_loss, model, cfg) -> compute_loss_fn`
  Returns a fn whose signature **mirrors `compute_loss`/`composite_loss_fn` exactly**:
  ```python
  def coverage_loss_fn(model_, policy_fn, states, key,
                       mc_samples=5, weights=None, shock_scale=1.0,
                       quad_nodes=None, quad_weights=None,
                       target_policy_fn=None):  # -> (total, eq_losses)
  ```
  Body:
  1. **Split the key**: `k_base, k_seed, k_roll, k_local = jax.random.split(key, 4)`.
  2. `base = states`; **if `rho_stress>0`**: `stress = roll_states(model_, policy_fn, sample_stress_seeds(...), k_roll, cfg.rollout_horizon, shock_scale, lo, hi)`; **if `rho_local>0`**: `local = make_local_pool(states, k_local, cfg.n_local, cfg.local_sigma, lo, hi)`. Zero-weight pools are skipped at build time (Python-level `if`, pre-JIT) — with both off, the wrapper is *numerically identical* to plain `compute_loss` (κ=0 identity; unit-tested). `lo`/`hi` are the repair-box vectors built once from `cfg.repair_ranges`.
  3. Call `base_compute_loss` on **each** included pool, **forwarding all expectation kwargs**
     (`mc_samples, weights, shock_scale, quad_nodes, quad_weights, target_policy_fn`), passing
     `k_base` as the loss key — so all pools use the **identical** 27-node quadrature operator.
  4. Normalize `ρ` over the **included** pools; `total = Σ ρ_i · L_i / Σ ρ_i`.
  5. Return `(total, eq_losses)` where `eq_losses` = the **base pool's** per-equation dict (the
     trainable equations) **plus** diagnostic scalars `aux_cov_base`, `aux_cov_stress`,
     `aux_cov_local`. The `aux_` prefix keeps them out of reweighting/gradient-surgery
     (`eq_losses_to_array` strips `aux_`) and the metrics path auto-logs `aux_`-keys as `aux/<name>`
     (reporting.py) — **no metrics.py change needed**; they must be float-able scalars and must not
     key-collide with base `aux_` entries (irbc sets none).

### 3. Config wiring (corrected causality)

- New module `config/coverage.py` (§1); export in `config/__init__.py` `__all__`.
- `TrainConfig.coverage: CoverageConfig = Field(default_factory=CoverageConfig, ...)` in
  `config/train.py`, beside `replay_buffer`; wire `from_dict` (pop sub-dict, `_check_unknown_keys`,
  construct `CoverageConfig(**coverage_dict)`).
- **Install the wrapper inside `_build_custom_loss_fn`** (trainer.py:375 — it already receives
  `config`, hence `config.coverage`). When `config.coverage.enabled`, wrap the base `compute_loss`
  with `make_coverage_loss`; otherwise return the existing result unchanged. The result already flows
  out as `compute_loss_fn` into `make_train_step`. **Do NOT add a `coverage` arg to
  `make_train_step`** (that conflation was a review finding — `make_train_step` only receives a
  prebuilt `compute_loss_fn`; `replay_cfg` is a separate cycle-level path).
- **Model-aware validations** (run after the model is loaded, before JIT):
  - In `_resolve_model_for_training` (state_init.py:385-441, where shock_mask/loss_weights are
    already checked against the model): assert `set(stress_ranges) ∪ set(repair_ranges) ⊆
    set(model.state_names)`, erroring on unknown names (no silent "absent dim" fallback).
  - In `_validate_train_config` (state_init.py:296): add the mutual-exclusion / optimizer / layered-
    term gates listed in §Non-goals, mirroring the existing composite gate — including
    `network.history_len > 1` (NotImplementedError, mirroring the replay gate) and
    `replay_buffer.enabled` (ValueError).

### 4. Configs — two files, coverage as the only delta

- `configs/irbc_plain.yaml` — **reconstructed historical plain recipe** that fails: `network.type:
  mlp`, `loss_type: mse`, **no** `composite_loss` block, but **same** `expectation_type:
  gauss_hermite`, `n_quadrature_points: 3`, `batch_size: 128`, `episode_length: 1`,
  `initialize_each_episode: true`, optimizer adam as `irbc.yaml`. Reproduce/re-measure ρ(SS)≈1.23.
- `configs/irbc_ewm.yaml` — `irbc_plain.yaml` **+** a `coverage:` block (the only delta), e.g.:
  ```yaml
  coverage:
    enabled: true
    rho_base: 1.0
    rho_stress: 0.5      # paper's coverage-exact arm: path:stress:local = 1:0.5:0.25
    rho_local: 0.25
    n_stress: 128
    n_local: 128
    rollout_horizon: 5   # paper: H typically 3 or 5
    local_sigma: 0.02
    stress_ranges:
      z_0: [-0.18, -0.05]  # lower tail: ~1-4 stationary sd down (stationary sd(z) ≈ 0.045
      z_1: [-0.18, -0.05]  # with two 0.01 shocks at rho_z=0.95); NOT deeper — the paper
      k_0: [1.05, 1.20]    # clips coverage to ±4 ergodic sd, so seeds beyond that are
      k_1: [1.05, 1.20]    # unreachable states, not "stress". High k ⇒ i>=0 binds.
    repair_ranges:
      k_0: [0.2, 5.0]      # feasible box; landings/perturbations clipped here before
      k_1: [0.2, 5.0]      # the residual (paper's repair step — wide, a bound not a box)
      z_0: [-0.2, 0.2]     # ≈ ±4.4 stationary sd
      z_1: [-0.2, 0.2]
  ```
  irbc state names are exactly `('k_0','k_1','z_0','z_1')` (all free coordinates → box+SS-fill is
  valid). Stress dims are drawn **independently** per dimension, so asymmetric-country stress
  (one economy deep in recession, the other mild) is covered automatically — that is where the
  risk-sharing/ARC block is really tested.

## Data flow

```
cycle_step: build base batch (UNCHANGED; for irbc = init-rect draw)
   → grad_step calls compute_loss_fn = coverage_loss_fn  (installed in _build_custom_loss_fn)
       → split key → {k_base, k_seed, k_roll, k_local}
       → base   = batch states
         stress = roll_states(sample_stress_seeds(box))   [MC shocks, s_1..s_H, repair-clip, stop_gradient]
         local  = base + N(0, sigma)                       [repair-clip, stop_gradient]
         (zero-weight pools skipped at build time)
       → L_i = base_compute_loss(pool_i, k_base, +all quad/mc/weights/target kwargs)   for each pool
       → total = Σ (ρ_i/Σρ) · L_i   over included pools
       → grad wrt params  (flows only through policy_fn at fixed pool states)
```

No JIT-boundary change, no new pytree state, no `TrainState` change.

## Testing

- **Unit — config:** `CoverageConfig` validation/coercion: weights, counts, ranges, empty-pool
  guard, `extra="forbid"`; mirror replay-config tests.
- **Unit — name validation:** a `stress_ranges` key not in `model.state_names` raises at
  resolution (not silently filled).
- **Unit — rollout:** `roll_states` returns shape `[n*H, n_states]`, excludes the raw seed; gradient
  of a scalar of `roll_states(...)` w.r.t. params is zero through the state path (stop-gradient
  holds); `sample_stress_seeds` respects the box and SS-fills absent dims; `make_local_pool` shape +
  stop-gradient + tile/subsample behavior.
- **Unit — wrapper:** all included pools receive the quadrature kwargs (assert quad branch taken,
  not MC); mixture weights normalize; `aux_cov_*` present and scalar; repair clip applied.
- **Unit — κ=0 identity:** with `rho_stress=rho_local=0` (enabled), the wrapper's total equals
  plain `compute_loss` **exactly** under quadrature (the paper's κ=0 ⇒ DEQN collapse; catches
  normalization bugs structurally).
- **Bit-identical guard (concrete mechanism):** build two train steps from the same config except
  `coverage.enabled` (False) present-vs-absent; run one `grad_step` each with the **same** PRNG seed;
  assert `loss == loss` exactly and `jax.tree.map(np.testing.assert_array_equal, …)` over params and
  opt_state leaves. (Pattern: `tests/test_replay_smoke.py`.)
- **Validators:** coverage + composite / coverage + {mao,lm,gn,ign,lbfgs,pcgrad} / coverage +
  barrier/huber/moment each raise at config validation.
- **Integration smoke:** few-episode `irbc_ewm` train runs end-to-end, finite loss, all pools
  non-empty, `aux_cov_*` logged.
- **Numerical validation (separate, run on DGX — not a unit test):** `irbc_plain` and `irbc_ewm`
  trained over **5–10 seeds each** (`--set seed=<s>`); compute ρ(SS) per seed via the **same
  evaluation entry point that produced the existing ρ(SS)=0.981 figure** (name it in the plan), and
  the per-equation stress-grid residual metric defined in §Goal. Emit the three-way table with
  medians + pass counts, per the multi-seed protocol in §Goal.

## Risks / open questions

- **Stop-gradient through `run_episode`.** Reusing the rollout inside the grad tape must not retain a
  differentiable path — covered by the rollout gradient unit test.
- **Rollout-induced extreme residuals — mitigated by repair.** Rolling the deliberately-unstable
  plain policy from off-manifold seeds can push `k` to extreme (finite, softplus-floored) values;
  the paper's answer, adopted here, is the `repair_ranges` clip before the residual. irbc
  admissibility is otherwise architectural (softplus-positive policies; `c` pinned by `lam>0`).
- **Scalar `local_sigma` is cruder than the paper's local component** (they use multiplicative
  ~10% jitter on capital and additive 0.1·innovation-sd on TFP — per-dimension scales). A scalar in
  state units is acceptable for irbc where k~1 and z~0.05 share an order of magnitude; a per-dim
  dict is a v2 upgrade if a target model has heterogeneous state scales.
- **ρ weights fixed a priori** at the paper's 1:0.5:0.25; per the adopted protocol they are not
  tuned against the reported metrics. `aux_cov_*` per-pool logging makes imbalance visible.
- **Adaptive reweighting** (if ever enabled) sees **base-pool** per-equation losses only and applies
  them to all pools — acceptable for v1 (irbc uses fixed weights).

## Out of scope (future)

- EWM depth ③ (continuation surrogate) for `disaster`, where the 5-shock + Bernoulli expectation is
  the bottleneck. **`disaster`'s 8 endogenous states are not free coordinates** — a box+SS-fill would
  produce mutually-inconsistent seeds; it needs a model-specific stress-seed map. (The box+SS-fill
  approach is valid only where every state dimension is a free coordinate, as in irbc.)
- Composing coverage with the BK-anchor composite loss.
- Adaptive coverage (reach `κ` scheduling), action-conditioned continuations, JEPA encoder.
  (Note: the paper holds κ **fixed** across its warm-started IRBC stages — the staged homotopy
  there advances surrogate fidelity, not coverage — so a single fixed-κ run matches their IRBC
  coverage protocol; no schedule is needed for v1.)
- Minibatched coverage with per-sample pool weights (needed only if a target model is not full-batch).
- **On-policy base pool.** The paper's path component is genuine on-policy trajectories (64 tracks ×
  64 periods from broad LogUnif/Gaussian seeds); our irbc recipe's base is a one-step init-rect
  draw. The wrapper is agnostic (it covers whatever batch it is given), so if coverage-over-init-rect
  fails, switching the base to real trajectories (`episode_length: 64`, `sim_batch: 64`) is the
  next lever — a config change, not a code change.
- Per-dimension `local_sigma` (dict-by-state-name, mirroring the paper's per-dim local scales).
- Policy-stationarity diagnostic (the paper's "verified stationarity" criterion) as a cheap
  secondary convergence check.
