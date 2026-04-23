# Why DEQN-JAX

DEQN-JAX is a **framework for training Deep Equilibrium Nets on economic models**. You declare a model as data — state/policy variables, equilibrium equations, transition dynamics, calibration — and the framework handles training, expectation integration, diagnostics, and checkpointing. This document is for people who have just landed in the repo and want to know whether it solves their problem.

## What DEQN-JAX is

A thin, opinionated layer over JAX + Equinox + Optax. It provides:

- A declarative `ModelSpec` that registers a model with the framework; everything below consumes it.
- `train_from_config(cfg)` — one entry point that handles MC/quadrature expectations, rollout sampling vs. exogenous-rect sampling, minibatch sweeps, optimizer dispatch, checkpointing, TensorBoard/W&B logging, and warm start from steady state.
- A registry of optimizers — Adam / SGD / AdamW / Lion / Muon alongside specialist choices (NGD, MAO, Shampoo, L-BFGS, Gauss-Newton, Levenberg-Marquardt) — all selected by name in the config.
- A composite-loss path (anchor + Jacobian + barrier + Newton terms) with precomputed linearization, off by default but available to any `ModelSpec`.
- Post-training diagnostic modules (`evaluate`, `irf`) and pure-function plotting primitives (`plots.*`) that work uniformly across models.

The entire train step is a single `jax.jit` boundary. That is the core performance decision — a 3-line bugfix can't break it because there is nothing else to break.

## What DEQN-JAX is not

- **Not a catalogue of economic models.** Exactly the models in `src/deqn_jax/models/` are packaged; new models are a port away, not a plugin away. The existing models are reference implementations meant to be read, forked, and extended.
- **Not a DEQN tutorial.** If you want the "here is what a DEQN is, here is how you would build one from scratch in forty lines of TensorFlow" pedagogical walkthrough, read the Geneva DEQN lecture notebooks first, then come back. DEQN-JAX assumes you already know the object you are training.
- **Not a drop-in replacement for a traditional solver.** For low-dimensional problems with closed-form or value-function-iteration solutions, a traditional solver will be faster, simpler, and more accurate. DEQN shines when state dimension grows past what dense grids handle (typically ≥ 6-8 states, often far more).

## When to use it

- You are training a DEQN on a model with ≥ 3-4 state variables (beyond that the curse of dimensionality starts to bite in traditional methods).
- You want to swap optimizers / reweighting strategies / composite-loss configurations without rewriting the model.
- You want to run dozens of variants (different calibrations, different network architectures) under the same config system for easy diffing.
- You want JAX's speed without managing `jit` boundaries, batching conventions, or PRNG plumbing by hand.

## When not to use it

- **You want to learn DEQN.** Build it yourself once in a notebook first. The reference TensorFlow implementation is explicitly pedagogical; this framework is not.
- **Your model is tiny and has a closed form.** Use the closed form.
- **You need a feature the framework does not support.** The cost of forking depends on how deep the missing piece is. Surface-level things (new optimizer, new model, new loss term) are easy. A change to the train-step dispatch or shock-sampling logic is invasive because those live at the single JIT boundary by design.

## Comparison to alternatives

| | DEQN-JAX | Reference TF implementation | Hand-rolled JAX / PyTorch |
|---|---|---|---|
| Target audience | production / research at scale | pedagogical, one model at a time | someone who already knows what they want |
| Swap optimizer | config change | rewrite loop | rewrite loop |
| Batched model comparison | config-driven, built-in | per-notebook | per-script |
| MC / quadrature expectations | config toggle | hand-coded | hand-coded |
| Composite loss (anchor + Jacobian + barriers) | config toggle | does not exist | reimplement |
| Diagnostic suite (Euler errors, IRFs, ergodic moments) | shared across models | per-notebook | per-script |
| Single JIT boundary | yes | no — Keras graph | depends on author |
| Best for | re-running many variants | first exposure to DEQN | a specific paper's one experiment |

## Where to go next

- [Implementing a model](models/implementing.md) — how to port a new economic model end-to-end, with stochastic Brock-Mirman as the running example.
- [Running experiments](running_experiments.md) — CLI, configs, checkpoint/resume, TensorBoard, W&B, tuning.
- `examples/brock_mirman.ipynb` — the template per-model notebook. Same economics as the reference TF notebook; production-framework code path.
- `examples/bm_deterministic.ipynb` — the minimal case (one state, no shocks, closed-form policy) for sanity-checking the framework itself.
