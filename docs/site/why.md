# Why DEQN-JAX

DEQN-JAX is a **framework for training Deep Equilibrium Nets on economic models**. You declare a model as data — state/policy variables, equilibrium equations, transition dynamics, calibration — and the framework handles training, expectation integration, diagnostics, and checkpointing. This document is for people who have just landed in the repo and want to know whether it solves their problem.

New to the method itself? Read [What is DEQN?](what_is_deqn.md) first for a one-page orientation aimed at economists.

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

### Against other DEQN implementations

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

### Against traditional DSGE solvers

| Method | Approximates | Accurate where | Dimension limit | Best for |
|---|---|---|---|---|
| **Perturbation (Dynare)** | Taylor expansion of the policy around SS | Small neighborhood of SS | effectively unlimited | linear models, impulse-response work, baseline DSGE |
| **Value function iteration** | $V(s)$ on a discrete grid | Wherever the grid is dense | ~4-6 states | low-dim problems with occasionally-binding constraints |
| **Projection methods** | Policy as polynomial / Chebyshev / piecewise-linear basis | Interior of state domain; basis-dependent | ~6-8 states | medium-dim models with good structural properties |
| **Parameterized expectations (PEA)** | Conditional expectation as polynomial in state | Where polynomial fits the conditional | ~6 states | stochastic models where the conditional expectation is the object of interest |
| **DEQN-JAX (this framework)** | Policy as neural network | Wherever training samples reach | no hard limit; tested to ~13 states (disaster model) | higher-dim stochastic models with nonlinearities, rare events, kinks |
| **PINN-HJB / KFE** | Value function / density on continuous state space via PDE residual | Wherever the grid / collocation points reach | ~4-6 continuous states | continuous-time heterogeneous-agent models (Aiyagari-class) |

**When DEQN-JAX is worth it vs. Dynare specifically:**

- **Occasionally-binding constraints.** ZLB, borrowing constraints, irreversibility — Dynare's first/second-order perturbation misses the kink entirely. DEQN handles it natively (Fischer-Burmeister residual, soft bound penalties).
- **Rare disaster / regime-switching.** A 1% disaster probability affects the ergodic distribution and the pricing kernel significantly. Dynare's expectation over shocks is a Taylor truncation that underweights the tail. DEQN integrates the full distribution via MC or quadrature.
- **Higher state dimensions.** Past ~10 states Dynare's perturbation machinery works but starts to lose accuracy far from SS; VFI and projection collapse under the curse. DEQN's NN is smooth interpolation regardless of $d$.
- **Non-local welfare or policy counterfactuals.** "What happens at 3σ from SS?" is where linearization fails and DEQN wins.

**When Dynare is still the right tool:**

- Linear / near-linear models (most New Keynesian work up to ZLB).
- Impulse-response work where first-order dynamics are sufficient.
- Bayesian estimation workflows — Dynare handles likelihood evaluation, posterior sampling, and policy solving in one pipeline. DEQN-JAX is a *solver*, not an estimator.
- Anything published: Dynare's output is the profession's common denominator. First-order perturbation from Dynare should be the baseline against which your DEQN solution is compared.

A typical DEQN-in-a-paper workflow: perturb in Dynare → use it for the linearization (Blanchard-Kahn P matrix) in the composite loss → train DEQN against the nonlinear residuals → cross-validate by comparing near-SS behavior to Dynare's. DEQN-JAX supports all of this (`warm_start_linearize`, `composite_loss.anchor_weight`).

## Scope boundaries

**In scope for DEQN-JAX:**

- Discrete-time recursive general equilibrium models with finite-dimensional state.
- Any number of representative or finite-count agents (OLG with $A$ generations is fine; multi-country RBC is fine).
- Shocks: continuous (Gaussian, transformed to lognormal / AR(1)) or discrete i.i.d., integrated via MC or Gauss-Hermite quadrature.
- Occasionally-binding constraints via Fischer-Burmeister complementarity residuals (see `irbc` and upcoming `bm_labor_constrained`).
- Warm-starting from a linearized solution for disaster-risk and kink-approximation settings.

**Out of scope:**

- **Continuous-time models with Hamilton-Jacobi-Bellman equations.** Aiyagari and Krusell-Smith with full heterogeneous-agent distributional state, stationary distribution characterized by a Kolmogorov forward PDE, continuous-time Bellman optimality. These are the natural fit for **PINN-HJB / finite-difference PDE solvers**, not DEQN.
- **Mean-field games** and any model where the state includes a measure or density that evolves under a continuity equation.
- **Bayesian estimation.** This framework solves a calibrated model; it does not evaluate the likelihood of a dataset given parameters. Use Dynare or the estimation-specific literature for that.

If your problem has a continuous-time HJB + KFE structure, you want a PINN-HJB toolkit, not this one. The two paradigms complement each other: DEQN solves algebraic equilibrium conditions at sampled states, PINN-HJB solves PDEs on discretized continuous state. (At time of writing, a sibling PINN-HJB-KFE project is in development in this research group; this doc will link to it when it stabilizes.)

## Where to go next

- [Implementing a model](models/implementing.md) — how to port a new economic model end-to-end, with stochastic Brock-Mirman as the running example.
- [Running experiments](running_experiments.md) — CLI, configs, checkpoint/resume, TensorBoard, W&B, tuning.
- `examples/brock_mirman.ipynb` — the template per-model notebook. Same economics as the reference TF notebook; production-framework code path.
- `examples/bm_deterministic.ipynb` — the minimal case (one state, no shocks, closed-form policy) for sanity-checking the framework itself.
