# EWM world arm: continuation surrogate for deqn-jax — design spec (v1, 2026-08-28)

**Status:** design; implementer TBD (Codex track 2 or the maintainer's agent); reviewer = the
maintainer's agent. **Priority: above the RSS port's EWM configs** — the coverage-only
implementation (`training/coverage.py`, spec 2026-06-29) is half of Equilibrium World Models;
this spec adds the other half.

**Reference:** Scheidegger & Schaab (2026), *Equilibrium World Models*, arXiv:2606.23463 —
the "coverage + surrogate" arm (their Section on the world model / Algorithm block). This
document describes the mechanism from the paper; the implementation is deqn-jax's own.

## 1. What is missing and why it matters
EWM = (a) a **coverage measure** (states the economy can reach but rarely visits, rolled
through the exact transition) and (b) a **continuation surrogate** `Ŵ_ψ(x) ≈ Q(x) = E[inside]`
that stands in for the expectation *inside the policy update*, trained on sparse exact anchors.
deqn-jax has (a). Without (b), every policy step pays the full quadrature at every state — for
models with many shock dimensions (the 12-shock RSS trade model: 24 monomial nodes; the
36-node reference) the expectation *is* the cost. (b) is the speed lever and the reason the
paper counts two budgets, `B_policy` (exact evaluations) and `B_world` (surrogate evaluations).

## 2. Why it drops in cleanly
deqn-jax's two-stage loss hooks already are the paper's decomposition:
`inside_fn(state, policy, next_state, next_policy)` = the continuation integrand (per shock
node), `E[inside]` = `Q(x)`, and `combine_fn(state, policy, expectations)` = the residual given
`Q`. The world arm replaces `expectations = E_nodes[inside_fn]` by `expectations = Ŵ_ψ(x)` in the
policy update — **nothing else in the loss changes**, and models without the hooks
(`inside_fn is None`) are rejected by the validator, not silently degraded. Models that use
the standard `equations_fn` path but want the surrogate must first factor their expectation
into the hooks (irbc: Euler + ARC integrands; disaster: eq 1/2b/3/4b/5/6/7/8 integrands).

## 3. Mechanism (per episode, after the rollout and coverage batch are built)
1. **Anchors**: `S = anchor_frac × |batch|` states sampled from (path ∪ coverage), at least
   one minibatch. Paper stages `anchor_frac` 0.1 → 0.2 → 0.4; default schedule
   `[0.1, 0.2, 0.4]` over thirds of training, overridable.
2. **Target policy**: Polyak `θ̄ ← τ θ̄ + (1−τ) θ`, `τ` default 0.97 (0.99 also fine; 0.90 diverges
   — record as a config-validator warning below 0.95).
3. **Exact targets**: `Q_tar = E_nodes[inside_fn(·; θ̄)]` on the anchors, `stop_gradient`.
   Count `B_policy += |S| × n_nodes`.
4. **World update**: `epochs_w` (default 12) Adam steps on `ψ` minimizing
   `mean‖Ŵ_ψ(anchors) − Q_tar‖²` (inputs standardized by running mean/std of the episode's
   states; output head `softplus` so `Ŵ > 0` like `Q` when the integrands are positive —
   make positivity a per-key flag from the model: `inside_positive: Tuple[bool]`).
   Count `B_world += |S| × epochs_w`.
5. **Policy update**: minibatch steps on the full batch with
   `combine_fn(x, π_θ(x), Ŵ_ψ(x))`; gradients flow through `π_θ` in `combine_fn` only
   (`Ŵ` is treated as a fixed function: `stop_gradient` on its output).
6. **Coverage residuals**: option `exact_in_coverage: true` (default) scores coverage rows with
   the exact expectation even in the surrogate arm (the reference notes surrogate-scored
   coverage diverged in every variant it tried); `false` uses `Ŵ` everywhere.

## 4. Config
```yaml
surrogate:
  enabled: false          # off = byte-identical to today
  width: 64               # Ŵ hidden width (two hidden layers, tanh)
  anchor_frac: [0.1, 0.2, 0.4]   # staged over training thirds; scalar allowed
  polyak_tau: 0.97
  epochs_w: 12
  lr_w: null              # null = policy lr
  exact_in_coverage: true
```
Validators (`state_init._validate_train_config`): requires `model.inside_fn/combine_fn`;
requires `coverage.enabled` OR an explicit `allow_without_coverage: true` (the surrogate
without coverage is the paper's ablation, allowed but named); rejected with MAO/GN/LM/IGN
(they differentiate a different object); `polyak_tau < 0.95` warns.

## 5. Implementation shape (extend, don't fork)
- `training/surrogate.py`: `SurrogateState` (params ψ, Adam state, running input stats,
  budgets), `make_surrogate_net`, `fit_world(...)`, `policy_loss_with_surrogate(...)` built by
  wrapping the existing two-stage `compute_loss` path: the wrapper substitutes the
  expectation dict. `TrainState` gains an optional `surrogate_state` subtree (**checkpoint
  loader template must forward it** — the 07-11/07-17 template-completeness bug class;
  add the roundtrip test in `tests/test_checkpoint_loader_template.py`).
- `training/trainer.py`: one new `OptimizerKind`/step variant `STANDARD_SURROGATE` sharing
  `optimizers/_step_common.py`'s `make_loss_call`/`finalize_step`; the world update runs
  OUTSIDE the policy JIT (its own jitted fn) once per episode; Polyak update on the
  array leaves via `eqx.filter`.
- Logging: `aux_world_fit` (Ŵ MSE on anchors), `aux_world_audit` (Ŵ vs exact Q on a fixed
  held-out grid, every `log_every`), `B_policy`, `B_world` as scalars.

## 6. Tests
- Exactness: with `anchor_frac = 1.0`, `epochs_w` large, and a wide Ŵ, the surrogate policy
  gradient must approach the exact gradient on a toy (brock_mirman two-stage refactor or
  olg_lifecycle); assert cosine > 0.99 after fitting.
- Identity: `surrogate.enabled=false` ⇒ bit-identical loss and update to today (regression
  guard on irbc and olg_lifecycle).
- Validators: hooks missing → rejected; MAO/GN combos → rejected; τ warning.
- Loader roundtrip with `surrogate_state` present.
- Budgets: counted values equal the analytic formula for a 3-episode smoke.

## 7. Validation (what "done" means — certificates, not loss)
1. **irbc** (our closed model): surrogate ∘ coverage ∘ composite reproduces the 5/5 stability
   at ρ = 0.9808 within seed noise, with `B_policy` reduced by the anchor fraction (report the
   two budgets), and the held-out stress residual within 2× of the exact-expectation arm.
2. **Ŵ audit**: on the held-out grids, relative error of Ŵ vs exact Q < 5% median at the end of
   training (the paper's audit leg).
3. **A non-TF parity target for the mechanism itself**: port the paper's own small
   demonstration model — two-country IRBC with a rare persistent disaster (a two-state
   chain crossed with the Gaussian shocks; 5-dim state) — as `irbc_disaster` and reproduce the
   paper's reported disaster-region held-out residual (10-seed median 1.4e-3, tolerance
   1.5e-2) with our implementation. This model is small, JAX-native in spirit, and makes the
   world arm testable against a published number rather than against anyone's code.
4. Only then: the RSS trade model's EWM config uses `surrogate.enabled: true`.

## 8. Delivery (own PR track, independent of the RSS port)
`ewm/world-arm-core` (§5 + §6) → `ewm/world-arm-irbc` (§7.1–7.2) → `ewm/irbc-disaster-model`
(§7.3) → RSS EWM config PR consumes it. Same PR discipline as the RSS spec §11 (branches
from pushed master, CI + reviewer, no direct master commits).
