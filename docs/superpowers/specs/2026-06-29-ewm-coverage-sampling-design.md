# EWM Coverage Sampling — Design (v1, irbc)

**Date:** 2026-06-29
**Branch:** `ewm-coverage` (private/local; do not push without explicit ask)
**Status:** approved design, revised after adversarial review (3-lens workflow), pre-implementation

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

deqn-jax's `irbc` model is the paper's headline economy and exhibits the failure: the plain recipe
trains to small on-measure residuals but lands on a closed-loop **unstable** policy (ρ(SS)=1.23 —
simulations drift out of the training rect, ARC violated O(1), negative investment). Our current
remedy is a BK-anchored composite loss (ρ(SS)=0.981), which the paper would class as an "ex-ante
engineering fix that treats the symptom." This work ports **coverage sampling** as the structural
alternative.

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

Run the three recipes and report measured ρ(SS) and the stress-region residual delta:

| recipe | config | ρ(SS) | note |
|---|---|---|---|
| plain MLP + MSE | `irbc_plain.yaml` | ~1.23 (re-measure) | self-confirming, unstable baseline |
| LinearPlusMLP + BK-anchor | `irbc.yaml` | 0.981 (known) | engineering fix (symptom) |
| plain MLP + coverage | `irbc_ewm.yaml` | **measured** | structural fix (the bet) |

The hypothesis (drawn from the paper) is ρ(SS)<1 and a materially lower stress-region residual for
the coverage recipe. A correct implementation that yields ρ(SS)≥1 is a valid *negative result*, not
an engineering failure.

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
`total` on the single-pool scale, avoiding an implicit LR shift (raw 1/1/1 would be ~3× baseline).

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
| `rho_stress` | float | `1.0` | mixture weight on the stress pool |
| `rho_local` | float | `1.0` | mixture weight on the local pool (normalized internally) |
| `n_stress` | int | `128` | stress seeds drawn per step |
| `n_local` | int | `128` | local perturbations per step |
| `rollout_horizon` | int | `8` | H: steps to roll stress seeds through exact Γ |
| `local_sigma` | float | `0.01` | Gaussian std for local perturbations (state units) |
| `stress_ranges` | dict[str, tuple[float, float]] | `{}` | per-state-name uniform box for stress seeds; names are **state names** (resolved later against `model.state_names`). Empty ⇒ error when enabled. |

Validators (copy replay.py structure; these need **no model handle**):
- `@field_validator(..., mode="before")` int/float coercers from `config/_base.py`.
- `@model_validator(mode="after")`:
  - weights ≥ 0 and not all zero;
  - **empty-pool guard**: `rho_stress>0 ⇒ n_stress>0`; `rho_local>0 ⇒ n_local>0` (a weighted empty
    pool → `jnp.mean` over 0 rows → NaN);
  - `rho_stress>0 ⇒ rollout_horizon≥1 and stress_ranges non-empty`;
  - `n_stress, n_local, rollout_horizon ≥ 0`; `local_sigma ≥ 0`;
  - each `stress_ranges` entry is `[low ≤ high]`.

Name-vs-model validation is **not** here (Pydantic has no model handle) — see §3.

### 2. `src/deqn_jax/training/coverage.py` — rollout helpers + wrapper

- `sample_stress_seeds(model, key, n, stress_ranges) -> seeds [n, n_states]`
  Uniform draw in the `stress_ranges` box, addressing dims by `model.state_names.index(name)`.
  Dimensions **absent** from `stress_ranges` are filled from the steady state
  (`model.steady_state_fn`), yielding complete seed vectors. (Names are pre-validated in §3, so an
  unknown name never reaches here.)

- `roll_states(model, policy_fn, seeds, key, horizon, shock_scale) -> states [n*horizon, n_states]`
  Rolls `seeds` forward `horizon` steps through the exact transition by reusing the
  `simulation_step`/`run_episode` path (which takes `policy_fn`, enforces the 2-D `[B, n_shocks]`
  shock contract, and auto-handles any disaster Bernoulli channel). **The rollout uses sampled
  (Monte-Carlo) shocks even when the loss uses quadrature** — state generation (sampling the
  reachable set) and the expectation `E[r²]` (evaluated at a fixed state) are different objects and
  do not conflict; both use standard-normal innovations and `step_fn` applies σ internally.
  Returns the **projected landings** `s_1 … s_H` (the states *after* transition; the raw off-manifold
  seed `s_0` is **excluded** so the imposed measure is on the reachable set), shape `[n*H, n_states]`,
  `lax.stop_gradient`'d. (Implementation note: collect the post-transition `next_state` per scan
  step, or drop `trajectory[0]` and append the final landing.)

- `make_local_pool(states, key, n, sigma) -> states [n, n_states]`
  Take `n` rows of the base `states` (tile if `n > batch`, subsample if `n < batch`), add
  `Normal(0, sigma)` per state dim, `lax.stop_gradient`'d.

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
  2. `base = states`; `stress = roll_states(model_, policy_fn, sample_stress_seeds(model_, k_seed, cfg.n_stress, cfg.stress_ranges), k_roll, cfg.rollout_horizon, shock_scale)`; `local = make_local_pool(states, k_local, cfg.n_local, cfg.local_sigma)`.
  3. Call `base_compute_loss` on **each** pool, **forwarding all expectation kwargs**
     (`mc_samples, weights, shock_scale, quad_nodes, quad_weights, target_policy_fn`), passing
     `k_base` as the loss key — so all three pools use the **identical** 27-node quadrature operator.
  4. Normalize `ρ = (ρ_base, ρ_stress, ρ_local) / sum(ρ)`; `total = Σ ρ_i · L_i`.
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
    already checked against the model): assert `set(stress_ranges) ⊆ set(model.state_names)`, erroring
    on unknown names (no silent "absent dim" fallback).
  - In `_validate_train_config` (state_init.py:296): add the mutual-exclusion / optimizer / layered-
    term gates listed in §Non-goals, mirroring the existing composite gate.

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
    rho_stress: 1.0
    rho_local: 1.0
    n_stress: 128
    n_local: 128
    rollout_horizon: 8
    local_sigma: 0.01
    stress_ranges:
      z_0: [-0.5, -0.2]   # deep recession TFP
      z_1: [-0.5, -0.2]
      k_0: [1.05, 1.20]   # high capital ⇒ desired disinvestment ⇒ i>=0 binds
      k_1: [1.05, 1.20]
  ```
  irbc state names are exactly `('k_0','k_1','z_0','z_1')` (all free coordinates → box+SS-fill is
  valid).

## Data flow

```
cycle_step: build base batch (UNCHANGED; for irbc = init-rect draw)
   → grad_step calls compute_loss_fn = coverage_loss_fn  (installed in _build_custom_loss_fn)
       → split key → {k_base, k_seed, k_roll, k_local}
       → base   = batch states
         stress = roll_states(sample_stress_seeds(box))      [MC shocks, stop_gradient, s_1..s_H]
         local  = base + N(0, sigma)                          [stop_gradient]
       → L_i = base_compute_loss(pool_i, k_base, +all quad/mc/weights/target kwargs)   for each pool
       → total = Σ (ρ_i/Σρ) · L_i
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
- **Unit — wrapper:** all three pools receive the quadrature kwargs (assert quad branch taken, not
  MC) ; mixture weights normalize; `aux_cov_*` present and scalar.
- **Bit-identical guard (concrete mechanism):** build two train steps from the same config except
  `coverage.enabled` (False) present-vs-absent; run one `grad_step` each with the **same** PRNG seed;
  assert `loss == loss` exactly and `jax.tree.map(np.testing.assert_array_equal, …)` over params and
  opt_state leaves. (Pattern: `tests/test_replay_smoke.py`.)
- **Validators:** coverage + composite / coverage + {mao,lm,gn,ign,lbfgs,pcgrad} / coverage +
  barrier/huber/moment each raise at config validation.
- **Integration smoke:** few-episode `irbc_ewm` train runs end-to-end, finite loss, all pools
  non-empty, `aux_cov_*` logged.
- **Numerical validation (separate, run on DGX — not a unit test):** full `irbc_plain` and
  `irbc_ewm` trains; compute ρ(SS) via the **same evaluation entry point that produced the existing
  ρ(SS)=0.981 figure** (name it in the plan), and the per-equation stress-grid residual metric
  defined in §Goal. Emit the three-way table.

## Risks / open questions

- **Stop-gradient through `run_episode`.** Reusing the rollout inside the grad tape must not retain a
  differentiable path — covered by the rollout gradient unit test.
- **Rollout-induced extreme residuals.** Rolling the deliberately-unstable plain policy H=8 steps
  from off-manifold seeds can push `k` to large (finite, softplus-floored) values whose residuals
  dominate. irbc admissibility is architectural (softplus-positive policies; `c` pinned by `lam>0`),
  so no NaN — but `rollout_horizon` and stress-box depth are the tuning knobs; a per-pool residual
  clip is a future option (huber is forbidden in v1 to keep the baseline comparison clean).
- **ρ tuning.** Normalized mixture weights start at equal shares; `aux_cov_*` per-pool logging makes
  imbalance visible.
- **Adaptive reweighting** (if ever enabled) sees **base-pool** per-equation losses only and applies
  them to all pools — acceptable for v1 (irbc uses fixed weights).

## Out of scope (future)

- EWM depth ③ (continuation surrogate) for `disaster`, where the 5-shock + Bernoulli expectation is
  the bottleneck. **`disaster`'s 8 endogenous states are not free coordinates** — a box+SS-fill would
  produce mutually-inconsistent seeds; it needs a model-specific stress-seed map. (The box+SS-fill
  approach is valid only where every state dimension is a free coordinate, as in irbc.)
- Composing coverage with the BK-anchor composite loss.
- Adaptive coverage (reach `κ` scheduling), action-conditioned continuations, JEPA encoder.
- Minibatched coverage with per-sample pool weights (needed only if a target model is not full-batch).
