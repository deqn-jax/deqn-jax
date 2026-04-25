# Training-cycle diagram — structural spec

Hand this to Claude.ai (Figma / tldraw / pure-JS-canvas) alongside the
original hand-drawn diagram so the digital recreation reflects the
**current** code, not the historical one. The original drawing is the
visual language; this file is the ground truth for content.

We ship **two artefacts**:

- **Conceptual SVG** — `docs/site/figures/deqn_conceptual.svg`,
  embedded in `what_is_deqn.md`. Pure four-level nested ladder
  (CYCLE ⊃ {SIM, TRAIN} ⊃ STEP/EPOCH ⊃ BATCH ⊃ action). No file
  paths, no JIT boundary, no variant fan-out — just the method.
- **High-level (code-level) SVG** — `docs/site/figures/deqn_solver_loop.svg`,
  embedded in `reading_guide.md`. Setup region + cycle loop + JIT box,
  with file-path annotations on every node so contributors can navigate
  from diagram label to source. This is the single code-level diagram;
  the originally-planned "detailed companion" was dropped after audit
  (the things it would add — mixture branch, composite aux cluster,
  variant fan-out, tensor-shape cascade — are already covered by prose
  in `composite_loss.md`, `architecture.md §4`, and the optimizer-
  family docs).

Both share the same visual language (JetBrains Mono, palette swap
header, same pytree/JIT/loop visual primitives).

---

The diagram has two regions: **setup** (one-time, top half, no JIT) and
the **cycle loop** (per-cycle, bottom half, with the JIT boundary).
A "cycle" = one rollout + N minibatch gradient steps. Cycles are the
outer iteration; episodes / rollouts happen inside them.

## Region 1 — Setup (one-time, eager)

Linear top-down chain. No branching except where noted. All file
references are to `src/deqn_jax/`.

```text
  ┌──────────────────────────────┐
  │ TrainConfig (YAML/CLI)       │   config.py — Pydantic v2
  └─────────────┬────────────────┘
                │
  ┌─────────────▼────────────────────┐
  │ train_from_config(config)        │   training/trainer.py
  │   • validate fp64 toggle         │
  │   • validate composite ↔ opt     │
  │   • validate ep_len/sim_batch/   │
  │     shock_mask combinations      │
  └─────────────┬────────────────────┘
                │
  ┌─────────────▼────────────────────┐
  │ load_model(config.model)         │   models/__init__.py
  │   → ModelSpec (NamedTuple)       │
  └─────────────┬────────────────────┘
                │
  ┌─────────────▼────────────────────┐
  │ Apply config.constants override  │   model._replace(constants=...)
  └─────────────┬────────────────────┘
                │
  ┌─────────────▼─────────────────────┐
  │ model wants risky steady state?   │
  │   yes → swap steady_state_fn for  │
  │         model.risky_steady_state  │
  │   (e.g. disaster + p_disaster>0)  │
  └─────────────┬─────────────────────┘
                │
  ┌─────────────▼────────────────────┐
  │ create_train_state               │   training/trainer.py
  │   • build network                │   networks/{mlp,lstm,transformer,
  │     (MLP / LSTM / Transformer    │     linear_plus_mlp}.py
  │      / LinearPlusMLP)            │
  │   • build optimizer              │   optimizers/registry.py
  │     (Adam / SGD / NGD / Shampoo  │   → optimizers/{standard,ngd,
  │      / MAO / PCGrad / LBFGS / GN)│       shampoo,mao,mao_kfac,
  │   • init opt_state               │       pcgrad,lbfgs,
  │   • sample initial states        │       gauss_newton}.py
  │     (init_state_fn or near SS)   │
  │   • seed history_state           │
  │     (for sequence policies)      │
  │   → TrainState NamedTuple        │
  └─────────────┬────────────────────┘
                │
  ┌─────────────▼─────────────────────┐
  │ if loss_type == composite:        │   training/composite_loss.py
  │   • linearize_model               │   training/linearize.py
  │     (Blanchard-Kahn QZ → P, Q)    │
  │   • compute ergodic covariance    │
  │   • prepare_composite_data        │
  │     (anchor pts, ss_state,        │
  │      ss_policy)                   │
  └─────────────┬─────────────────────┘
                │
  ┌─────────────▼────────────────────┐
  │ make_train_step                  │   training/trainer.py
  │   dispatch on OptimizerKind:     │
  │   STANDARD | PCGRAD | MAO |      │
  │   LBFGS | GN                     │
  │   composes:                      │
  │   make_rollout_fn (cycle.py)     │   training/cycle.py
  │   + make_grad_step_<name>        │   optimizers/<name>.py
  │   → make_cycle_step (cycle.py)   │
  │   → wrap in jax.jit              │
  └──────────────────────────────────┘
```

## Region 2 — The cycle loop (with JIT boundary)

The **JIT boundary** (dashed box) wraps the entire `cycle_step` —
rollout, minibatch sweep, gradient updates, all of it. The outer Python
loop only does dispatch, logging, and checkpointing.

```text
  ┌──────────────────────────────────────────┐
  │ for ep in 0..total_episodes:             │   training/trainer.py
  │                                          │
  │   ┌────────────────────────────────┐     │
  │   │ shock_scale = curriculum(ep)   │     │
  │   └────────────────┬───────────────┘     │
  │                    │                     │
  │ ╔══════════════════▼═══════════════════╗ │  ╔══════════════════════════╗
  │ ║   JIT BOUNDARY (dashed)              ║ │  ║  This whole box is one    ║
  │ ║   cycle_step(state, lr_scale,        ║─┼──║  jax.jit. XLA fuses       ║
  │ ║              shock_scale)            ║ │  ║  rollout, loss, grad,     ║
  │ ║                                      ║ │  ║  opt-step.                ║
  │ ║   1. rollout_fn (one rollout)        ║ │  ║  training/cycle.py        ║
  │ ║      → trajectory, final_history     ║ │  ╚══════════════════════════╝
  │ ║                                      ║ │
  │ ║   2. build minibatch dataset         ║ │
  │ ║      (IID shuffle OR sorted-within-  ║ │
  │ ║       batch slice)                   ║ │
  │ ║                                      ║ │
  │ ║   3. for epoch in n_epochs:          ║ │
  │ ║        for mb in n_minibatches:      ║ │
  │ ║          state = grad_step(state,    ║ │   optimizers/<name>.py
  │ ║                  batch, lr_scale,    ║ │
  │ ║                  shock_scale)        ║ │
  │ ║                                      ║ │
  │ ║   → returns (TrainState, Metrics)    ║ │
  │ ╚══════════════════╤═══════════════════╝ │
  │                    │                     │
  │   ┌────────────────▼───────────────┐     │
  │   │ log metrics, checkpoint, hook  │     │   training/reporting.py
  │   │   • cycle_hook(model, state,   │     │   training/checkpointing.py
  │   │     ep) — model-specific       │     │
  │   └────────────────────────────────┘     │
  └──────────────────────────────────────────┘
```

Two annotations to highlight on the SVG:

- **`shock_scale` flows everywhere** — into `rollout_fn` (so curriculum
  and `shock_mask` apply to state simulation) AND into `compute_loss`
  (so the expectation matches). Show as a labelled wire.
- **`history_state` threads through the loop** — for sequence policies
  it's persisted in `TrainState` across cycles (`None` for MLP). Show
  as a side-rail wire when distinguishing sequence vs feed-forward
  paths is useful.

## Region 3 — Inside `cycle_step` (the heart)

Show this as a zoomed-in inset (high-level SVG) AND as the central
panel of the detailed SVG. This is the reader's "what does one
gradient step actually do?" view.

```text
            ┌──────────────────────────────────────┐
            │ rollout_fn(state, shock_scale)       │   training/cycle.py
            │   • optional fresh init or ss_reset  │
            │   • run_episode (or                  │   training/episode.py
            │     run_episode_with_history)        │
            │     = lax.scan over episode_length   │
            └────────────────┬─────────────────────┘
                             │ trajectory [T, B, n_states]
                             │
            ┌────────────────▼─────────────────────┐
            │ slice into minibatches               │
            │   IID-shuffled OR sorted within batch│
            └────────────────┬─────────────────────┘
                             │
            ┌────────────────▼─────────────────────┐
            │ FOR EACH MINIBATCH:                  │
            │   grad_step(state, batch, lr_scale,  │   optimizers/<name>.py
            │             shock_scale)             │   (makes its own loss call)
            └────────────────┬─────────────────────┘
                             │
            ┌────────────────▼─────────────────────┐
            │ compute_loss(state, batch, key, ...) │   training/loss.py
            └────────────────┬─────────────────────┘
                             │
       ┌─────────────────────┴────────────────────┐
       │                                          │
  ┌────▼─────────┐                       ┌────────▼──────────┐
  │ sample shocks│                       │ if quadrature:    │   training/shocks.py
  │ (antithetic  │       ──or──          │ tensor-product GH │
  │  MC pairs)   │                       │ nodes + weights   │
  └────┬─────────┘                       └────────┬──────────┘
       │                                          │
       └──────────────────┬───────────────────────┘
                          │ shocks [n_samples, B, n_shocks]
                          │
            ┌─────────────▼──────────────┐
            │ vmap over shocks:          │
            │   compute_residuals(state, │   training/loss.py
            │     policy_fn, batch,      │
            │     shock, target_fn?)     │
            └─────────────┬──────────────┘
                          │
            ┌─────────────▼─────────────────────────┐
            │ INSIDE compute_residuals:             │
            │   policy = policy_fn(state)           │
            │   next_state = step_fn(state, policy, │   models/<name>/dynamics.py
            │                  shock, constants)    │
            │   next_policy = next_fn(next_state)   │
            │     ← stop_gradient if target net     │
            │   residuals = equations_fn(state,     │   models/<name>/equations.py
            │     policy, next_state, next_policy)  │
            │                                       │
            │   if model has mixture branch         │
            │     (e.g. p_disaster > 0):            │
            │     branch d=0 + branch d=1           │
            │     → mixture (1-p)·r₀ + p·r₁         │
            └─────────────┬─────────────────────────┘
                          │ residuals dict per equation
                          │
            ┌─────────────▼──────────────────┐
            │ aggregate:                     │
            │   eq_loss[k] = mean_b(         │
            │     (Σ_s w_s · r_s,b)²         │
            │   )                            │
            │   total_loss = Σ_k weight[k]·  │
            │                  eq_loss[k]    │
            │   weights from reweighting.py  │   training/reweighting.py
            └─────────────┬──────────────────┘
                          │
            ┌─────────────▼─────────────────────┐
            │ if loss_type == composite, add:   │   training/composite_loss.py
            │   + anchor (||π_net - π_lin||²)   │
            │   + jac    (||∂π_net - P||²)      │
            │   + barrier (state/policy bounds) │
            │   + newton (cond, residual)       │
            │ — keys prefixed "aux_" so PCGrad/ │
            │   MAO/reweighting ignore them     │
            └─────────────┬─────────────────────┘
                          │ scalar loss + per-eq dict
                          │
            ┌─────────────▼──────────────────┐
            │ VARIANT-SPECIFIC GRADIENT PATH │
            │ (lives in optimizer file):     │
            │                                │
            │ STANDARD: value_and_grad → opt │   optimizers/standard.py
            │ PCGRAD:   per-eq grads → proj  │   optimizers/pcgrad.py
            │ MAO:      jacrev → per-eq Jac  │   optimizers/mao.py
            │ MAO-KFAC: as MAO + K-FAC prec. │   optimizers/mao_kfac.py
            │ LBFGS:    value+grad+value_fn  │   optimizers/lbfgs.py
            │ GN/LM:    residual Jacobian J  │   optimizers/gauss_newton.py
            │           → -(JᵀJ)⁻¹ Jᵀr      │
            └─────────────┬──────────────────┘
                          │ updates
            ┌─────────────▼──────────────────┐
            │ params = apply_updates(params, │
            │                        updates)│
            │ update opt_state               │
            │ adaptive reweight loss_weights │   training/reweighting.py
            └─────────────┬──────────────────┘
                          │
                  new TrainState
```

## What to put in each diagram

### Conceptual SVG (`deqn_conceptual.svg`)

Pedagogical, no code references. Show only:

- The four nesting levels (CYCLE ⊃ {SIMULATION, TRAINING} ⊃
  STEP/EPOCH ⊃ BATCH).
- Inside STEP: forward pass + total step actions.
- Inside BATCH: forward+backward + update NN actions.
- The `state_episode` data bridge between SIMULATION and TRAINING.
- The cycle-back arrow: STEP's final state `s_{T-1}` seeds next
  cycle's `s_0`. Source the arrow from clearly-empty space (not from
  state_episode pill, which is the *whole tensor* used for training
  data, not the seed).

### High-level (code-level) SVG (`deqn_solver_loop.svg`)

Keep it readable at a glance. Show:

- Setup region with the fork: `TrainConfig → load_model → ModelSpec`,
  then ModelSpec splits to `create_train_state` (→ `TrainState` data)
  and `make_train_step` (→ JIT'd `cycle_step` callable). Both feed the
  loop.
- The Python loop with the JIT boundary on `cycle_step`.
- Inside `cycle_step`: SIMULATION (rollout_fn → state_episode) and
  TRAINING (minibatch sweep with `grad_step ↔ compute_loss`) as two
  labelled sub-regions.
- File-path annotations under every box.
- Config-knob annotations on the boxes whose behaviour they shape
  (e.g. `initialize_each_episode`, `ss_reset_frac`, `mc_samples`,
  `n_epochs_per_rollout`). This is the diagram's main job for an
  experimenter trying to understand what their YAML actually does.

## Visual conventions

- **Solid box** = function call / module.
- **Dashed box** = JIT compilation boundary (highlight the perimeter,
  ideally with a `jax.jit` label on the corner).
- **Diamond** = conditional branch (e.g. `model has mixture branch?`).
- **Cylinder/pytree** = `TrainState` (params, opt_state, episode_state,
  history_state, key, step, episode, loss_weights, reweight_state,
  target_params, aux_params, aux_opt_state) — shown threading through
  the loop.
- **File-path annotation** = monospace label below or beside each box,
  e.g. `training/cycle.py`. Optional `:function` suffix.
- **Color hint**: keep the original drawing's palette; use one accent
  colour for the JIT boundary so the reader's eye locks onto it.

## What's true now that wasn't in the original drawing

These post-original additions need explicit callouts:

1. **`cycle_step` is the JIT entry**, not `train_step`. The single
   JIT boundary now wraps rollout + minibatch sweep + grad updates
   together (extracted into `training/cycle.py` from the old
   monolithic `trainer.py`).
2. **Per-optimizer `grad_step` files** — variant code lives in
   `optimizers/<name>.py`, not in `trainer.py`. `make_train_step`
   composes them.
3. **Composite-loss aux terms** — anchor + jac + barrier + newton
   cluster (`training/composite_loss.py`). Original showed only
   residual MSE.
4. **Mixture branch** — the model-driven fork inside `compute_residuals`
   (e.g. `p_disaster > 0`). Driven by the model spec, not hard-coded.
5. **Target network** — `stop_gradient` on `next_policy` when target
   network mode is on. Show as an optional sub-box.
6. **5 variants of the gradient step** (STANDARD, PCGRAD, MAO+MAO-KFAC,
   LBFGS, GN/LM). Original probably showed only STANDARD (Adam-style).
7. **Constants override + risky-SS swap** in setup region — added
   for parameter-sweep and disaster-style experiments.
8. **`LinearPlusMLP`** as a network choice — original was MLP-only.
   Worth showing in setup as one of {MLP, LSTM, Transformer,
   LinearPlusMLP}.
9. **`history_state` threading** — sequence policies (LSTM,
   Transformer) carry a history window across cycles via
   `TrainState.history_state`. MLP is `None`.
10. **`shock_scale` curriculum wiring** — flows into the rollout
    *and* the loss expectation, not just the loss.
11. **Adaptive reweighting** (`lr_annealing`, `relobralo` in
    `training/reweighting.py`) modifies per-equation loss weights
    each step — show as a side-input to the loss-aggregation box.

## What to keep faithful to the original

- The overall **left-to-right or top-to-bottom flow** of one cycle.
- The **separation of "model code" (equations, step) from "framework
  code" (loss, optimizer)** — DEQN's main pedagogical claim is that
  researchers only have to write the model side.
- Any **personal stylistic touches** (handwriting, sketchy boxes,
  margin notes) — those are part of the project's character. The
  digital version should aim to preserve the feel, not sterilize it.
