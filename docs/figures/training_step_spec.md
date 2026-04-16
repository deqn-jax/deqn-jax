# Training-step diagram — structural spec

Hand this to Claude.ai (Figma / tldraw / pure-JS-canvas) alongside the
original hand-drawn diagram so the digital recreation reflects the
**current** code, not the historical one. The original is the visual
language; this file is the ground truth.

The diagram has two regions: **setup** (one-time, top half, no JIT) and
**training loop** (per-episode, bottom half, with the JIT boundary).

## Region 1 — Setup (one-time, eager)

Linear top-down chain. No branching except where noted.

```
  ┌─────────────────────────┐
  │ TrainConfig (YAML/CLI)  │
  └────────────┬────────────┘
               │
  ┌────────────▼────────────┐
  │ load_model(config.model)│      models/__init__.py
  └────────────┬────────────┘
               │   ModelSpec (NamedTuple)
               │
  ┌────────────▼────────────────────┐
  │ Apply config.constants override │  trainer.py:1197
  └────────────┬────────────────────┘
               │
  ┌────────────▼─────────────────────┐
  │ disaster + p_disaster>0 ?        │
  │   yes → swap steady_state_fn to  │  trainer.py:1213
  │         risky_steady_state       │
  └────────────┬─────────────────────┘
               │
  ┌────────────▼────────────────────┐
  │ create_train_state              │  trainer.py:create_train_state
  │   • build network (MLP/LSTM/    │
  │     Transformer/LinearPlusMLP)  │
  │   • build optimizer (Adam/...)  │
  │   • init opt_state              │
  │   • sample initial states       │
  └────────────┬────────────────────┘
               │
  ┌────────────▼─────────────────────────┐
  │ if loss_type == composite:           │  composite_loss.py
  │   • linearize_model (Blanchard-Kahn  │  linearize.py
  │     QZ → P, Q matrices)              │
  │   • compute ergodic covariance       │
  │   • prepare_composite_data           │
  │     (anchor pts, ss_state, ss_policy)│
  └────────────┬─────────────────────────┘
               │
  ┌────────────▼────────────────────┐
  │ make_train_step                 │  trainer.py:make_train_step
  │   dispatch on OptimizerKind:    │
  │   STANDARD | PCGRAD | MAO |     │
  │   LBFGS | GN                    │
  │   — wrap in jax.jit             │
  └─────────────────────────────────┘
```

## Region 2 — Per-episode loop (with JIT boundary)

The **JIT boundary** (dashed box) wraps everything from "compute loss"
through "apply updates". Anything outside it is Python-side and runs
once per episode.

```
  ┌──────────────────────────────────────┐
  │ for episode in 0..N:                 │
  │                                      │
  │   ┌──────────────────────────────┐   │
  │   │ run_episode (lax.scan)       │   │  episode.py
  │   │   trajectory = simulate      │   │
  │   │   under current policy       │   │
  │   └──────────────┬───────────────┘   │
  │                  │ trajectory        │
  │   ┌──────────────▼───────────────┐   │
  │   │ sample batch from trajectory │   │
  │   └──────────────┬───────────────┘   │
  │                  │                   │
  │   ┌──────────────▼─────────────────┐ │
  │   │ if step % target_update == 0:  │ │
  │   │   target_params = polyak(...)  │ │
  │   └──────────────┬─────────────────┘ │
  │                  │                   │
  │   ┌──────────────▼─────────────────┐ │
  │   │ lr_scale = schedule(step)      │ │
  │   └──────────────┬─────────────────┘ │
  │                  │                   │
  │ ╔════════════════▼═════════════════╗ │  ╔═══════════════════════════╗
  │ ║      JIT BOUNDARY (dashed)       ║ │  ║  This whole box is one     ║
  │ ║   train_step(state, batch, lr)   ║─┼──║  jax.jit — XLA fuses       ║
  │ ║                                  ║ │  ║  loss, grad, opt-step      ║
  │ ║   [variant-specific path]        ║ │  ╚═══════════════════════════╝
  │ ║   → returns new TrainState       ║ │
  │ ╚════════════════╤═════════════════╝ │
  │                  │                   │
  │   ┌──────────────▼─────────────────┐ │
  │   │ log metrics, checkpoint, etc.  │ │
  │   └────────────────────────────────┘ │
  └──────────────────────────────────────┘
```

## Region 3 — Inside the JIT boundary (the heart)

Show this as a zoomed-in inset OR a separate sub-figure. This is the
reader's "what does one gradient step actually do?" view.

```
            ┌──────────────────────────────────────┐
            │ compute_loss(state, batch, key, ...) │   loss.py / composite_loss.py
            └────────────────┬─────────────────────┘
                             │
       ┌─────────────────────┴────────────────────┐
       │                                          │
  ┌────▼─────────┐                       ┌────────▼──────────┐
  │ sample shocks│                       │ if quadrature:    │
  │ (antithetic  │       ──or──          │ tensor-product GH │
  │  MC pairs)   │                       │ nodes + weights   │
  └────┬─────────┘                       └────────┬──────────┘
       │                                          │
       └──────────────────┬───────────────────────┘
                          │ shocks [n_samples, B, n_shocks]
                          │
            ┌─────────────▼──────────────┐
            │ vmap over shocks:          │
            │   compute_residuals(state, │
            │     policy_fn, batch,      │
            │     shock, target_fn?)     │
            └─────────────┬──────────────┘
                          │
            ┌─────────────▼─────────────────────────┐
            │ INSIDE compute_residuals:             │
            │   policy = policy_fn(state)           │
            │   next_state = step_fn(state, policy, │
            │                  shock, constants)    │
            │   next_policy = target_fn(next_state) │
            │     ← stop_gradient if target net     │
            │   residuals = equations_fn(state,     │
            │     policy, next_state, next_policy)  │
            │                                       │
            │   if p_disaster > 0:                  │
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
            └─────────────┬──────────────────┘
                          │
            ┌─────────────▼─────────────────────┐
            │ if loss_type == composite, add:   │
            │   + anchor (||π_net - π_lin||²)   │
            │   + jac    (||∂π_net - P||²)      │
            │   + barrier (state/policy bounds) │
            │   + newton (cond, residual)       │
            │ — keys prefixed "aux_" so PCGrad/ │
            │   MAO/reweighting ignore them     │
            └─────────────┬─────────────────────┘
                          │ scalar loss
                          │
            ┌─────────────▼──────────────────┐
            │ VARIANT-SPECIFIC GRADIENT PATH │
            │ (dispatched at construction):  │
            │                                │
            │ STANDARD: value_and_grad → opt │
            │ PCGRAD:   per-eq grads → proj  │
            │ MAO:      jacrev → per-eq Jac  │
            │ LBFGS:    value+grad+value_fn  │
            │ GN:       residual Jacobian J  │
            │           → -(JᵀJ)⁻¹ Jᵀr      │
            └─────────────┬──────────────────┘
                          │ updates
            ┌─────────────▼──────────────────┐
            │ params = apply_updates(params, │
            │                        updates)│
            │ update opt_state, reweight     │
            └─────────────┬──────────────────┘
                          │
                  new TrainState
```

## Visual conventions to suggest to the artist

- **Solid box** = function call / module
- **Dashed box** = JIT compilation boundary (highlight the perimeter, ideally
  with a "jax.jit" label on the corner)
- **Diamond** = conditional branch (e.g. p_disaster > 0?)
- **Cylinder/pytree** = TrainState (params, opt_state, episode_state, key,
  step, episode, loss_weights, reweight_state, target_params, aux_params,
  aux_opt_state) — shown threading through the loop
- **Color hint**: keep the original drawing's color palette; use one accent
  colour for the JIT boundary so the reader's eye locks onto it

## Things the original drawing probably did NOT have

These are post-original additions that are worth a callout (annotation,
sub-box, footnote):

1. **Composite-loss aux terms** — the "anchor + jac + barrier + newton" cluster.
   In the original, only the residual MSE existed.
2. **Mixture branch** — the `p_disaster > 0` fork inside `compute_residuals`.
3. **Target network** — the `stop_gradient` on `next_policy` when target
   network mode is on. Show as an optional sub-box.
4. **5 variants of the gradient step** — original probably showed only
   STANDARD (Adam-style). PCGRAD/MAO/LBFGS/GN are all new.
5. **Constants override + risky-SS swap** in setup region — added in the
   disaster-experiment branch.
6. **LinearPlusMLP** as a network choice — the original was MLP-only.
   Worth showing in setup as one of {MLP, LSTM, Transformer, LinearPlusMLP}.

## What to keep faithful to the original

- The overall **left-to-right or top-to-bottom flow** of one training step.
- The **separation of "model code" (equations, step) from "framework code"
  (loss, optimizer)** — DEQN's main pedagogical claim is that researchers
  only have to write the model side.
- Any **personal stylistic touches** (handwriting, sketchy boxes,
  margin notes) — those are part of the project's character. The digital
  version should aim to preserve the feel, not sterilize it.
