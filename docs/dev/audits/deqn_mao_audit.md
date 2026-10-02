# DEQN-JAX vs. DEQN-MAO: Feature Audit

**Purpose.** Before any more work on the Geneva course ports, enumerate
what the upstream TF/Keras reference (`~/Projects/research/DEQN_MAO`) has
and what `deqn-jax` has lost, kept, or added on top. This is the audit
that should have happened before I started porting models.

**Reference commits.** DEQN-MAO as of a local checkout of its master
branch (April 2026). `deqn-jax` as of the current working tree.

**Verdict up front.** `deqn-jax` is a *strict specialisation* of DEQN-MAO
in the training-loop area: rollout-based ergodic training is baked in,
and the `initialize_each_episode` flag that makes DEQN-MAO work on
non-ergodic problems is absent (I've patched it in under the name
`resample_each_cycle`; should rename to match upstream). In the *loss
and optimizer* area, `deqn-jax` has added capability (composite loss,
sequence nets, PCGrad, Gauss-Newton). The bounds / penalty / hooks
infrastructure is weaker in `deqn-jax` and has semantic differences.

---

## 1. Model specification

How a model declares itself to the framework.

| Capability | DEQN-MAO | deqn-jax | Gap |
|---|---|---|---|
| Declarative states list with `bounds.lower/upper` per-variable | `Variables.py` dicts (first-class) | `state_names` tuple in `SPEC`; **no bounds** on states | **gap** |
| Declarative policies list with `bounds` + `activation` hints | `Variables.py` dicts | `POLICY_LOWER`/`POLICY_UPPER` arrays via `variable_spec`; **no `activation` hint** (fixed sigmoid+softplus) | weaker |
| Declarative definitions list with `bounds` | `Variables.py` dicts, auto-generated `_RAW` accessors + penalty coefficients | `definitions_fn` returns dict; **no bounds, no penalty, no `_RAW`** | **gap** |
| Per-variable initialization distribution (e.g. `init: {distribution: uniform, kwargs: ...}`) | Per-state `init` spec consumed by `initialize_states()` | Monolithic `init_state_fn(key, batch, constants)` — no per-variable spec | **different design** (monolithic vs declarative) |
| Constants as config-overridable dict | `constants` dict in Variables.py + `config/constants/*.yaml` override | `constants` dict in `ModelSpec` + YAML `constants:` override | ✓ equivalent |

**Assessment.** `deqn-jax`'s `ModelSpec` is narrower than DEQN-MAO's
`Variables.py` contract. The gap that bites in practice is the missing
penalty-on-definitions mechanism: in DEQN-MAO a model can declare
`c > 1e-8` on consumption and the framework adds a soft penalty to the
loss. In `deqn-jax`, if consumption goes negative during training you
get a NaN. The `state_barrier_fn` in `ModelSpec` is a placeholder that
only the `disaster` model uses, and it's a single function, not a
per-variable declarative thing.

---

## 2. Training loop

How batches are produced and how the cycle is structured.

| Capability | DEQN-MAO | deqn-jax | Gap |
|---|---|---|---|
| Cycle structure: rollout → epochs of minibatches | `run_cycle` → `run_episode` + `run_epoch` loop | `rollout_fn` + minibatch-sweep | ✓ equivalent |
| `N_episode_length` × `N_sim_batch` rollout | ✓ | ✓ (`episode_length × batch_size`) | ✓ |
| `N_epochs_per_episode` (multiple passes over the same rollout) | ✓ | ✓ (`n_epochs_per_rollout`) | ✓ |
| `N_minibatch_size` within epochs | ✓ | ✓ | ✓ |
| **`initialize_each_episode: bool`** — re-draw init states every cycle | **first-class config flag** (line 210 of `Graphs.py`) | **MISSING in vanilla**; I just added as `resample_each_cycle` | **regression** — should match upstream name + be present by default |
| Carry trajectory across cycles (persistent `starting_state`) | Default behaviour | Default behaviour (via `episode_state`) | ✓ |
| `ss_reset_frac` — mix in ±5% SS-neighborhood noise | absent | present | deqn-jax **extra** (but less principled than `initialize_each_episode`) |
| Pre-batch shuffle (`use_new_shuffling`) | ✓ optional | ✓ de-facto (shuffle indices over concatenated rollout) | ≈ |
| Bound-penalty term added to loss | ✓ `penalty_bounds_policy` in `Equilibrium.py` (states + policies + definitions) | `barrier_weight * state_barrier_fn(state)` only; policy/definition bounds not penalized (they're clipped hard) | **gap** |
| Huber vs MSE loss switch | ✓ `loss_choice: mse|huber` | ✓ `loss_choice: mse|huber|log_mse` in `compute_loss` | ✓ |
| Horovod distributed training | ✓ | ✗ | acceptable gap |

**Assessment.** The one gap that matters is `initialize_each_episode`. I
re-invented it as `resample_each_cycle` thinking it was a new concept;
it's a feature that's existed in the reference since the start. Rename
and restore. Every other structural piece is either equivalent or a
deqn-jax addition.

---

## 3. Expectation operator

How the framework handles `E_t[...]` inside equilibrium conditions.

| Capability | DEQN-MAO | deqn-jax | Gap |
|---|---|---|---|
| Equation author writes `E_t(lambda state', policy': integrand)` | ✓ `State.E_t_gen` | ✗ Equation signature is `(state, policy, next_state, next_policy, constants)` — single sample | **different abstraction** |
| Gauss-Hermite product quadrature | ✓ (`expectation_type: product`) | ✓ (`gauss_hermite_nd`) | ✓ |
| Pseudo-random MC | ✓ (`expectation_type: pseudo_random`) | ✓ (`sample_antithetic_shocks`, with antithetic variates) | deqn-jax slightly better (antithetic) |
| Nonlinear transform of E[.] (e.g. `u_c_inv(beta * E[u' * mpk])`) in the equation | **Natural** — just write the code | **Awkward** — framework averages per-sample residuals; if you want $f(E[g])$ you have to linearise or do something clever | **gap for certain residual forms** |
| Monomial quadrature rule (`monomial_rule` for cheaper high-d integration) | ✓ present (unused?) | ✗ | minor |
| Double-precision expectation option (`E_t_gen_double`) | ✓ | ✗ | minor |

**Assessment.** This is the subtlest semantic gap. In DEQN-MAO you write

```python
E_mu_mpk = E_t(integrand)
c_implied = u_c_inv(beta * E_mu_mpk)
euler_error = c_implied / c_t - 1.0
```

In deqn-jax, the framework gives your `equations_fn` *one* next-state at
a time and averages residuals after the fact. Writing `u_c_inv(E[...])`
cleanly would require exposing an expectation operator to `equations_fn`
— which means changing the equation signature. For now: as long as
residuals are *linear* in the next-period objects you're fine (the
Euler residual `u'(c) - β u'(c') mpk'` is linear in `u'(c') mpk'`). All
four Geneva models I've looked at so far *can* be expressed in this
form, so the gap is mostly theoretical for the current scope. But it's
a real design difference and it's why Simon's Equations.py for
brock_mirman looks different from my port.

---

## 4. Step dynamics

How state transitions happen.

| Capability | DEQN-MAO | deqn-jax | Gap |
|---|---|---|---|
| `total_step_random(state, policy)` — stochastic draw for simulation | ✓ | ✓ (implicitly — `step_fn(state, policy, random_shock)`) | ✓ |
| `total_step_spec_shock(state, policy, shock_index)` — deterministic step at a specific quadrature node | ✓ separate function | ✗ — same `step_fn` used for both paths; framework passes quadrature nodes as `shock` | **different design**, but functionally equivalent |
| Split into `AR_step + shock_step + policy_step` sub-components | ✓ idiom | ✗ — single `step_fn` does everything | cosmetic |

**Assessment.** Framework-level: the two are equivalent. Model-file
idiom: DEQN-MAO encourages decomposing into AR + shock + policy
sub-steps for readability; `deqn-jax` leaves that to the model author.

---

## 5. Optimizer and loss reweighting

| Capability | DEQN-MAO | deqn-jax | Gap |
|---|---|---|---|
| Adam | ✓ | ✓ | ✓ |
| NGD (diagonal Fisher) | ✓ `optim/NGD.py` | ✓ `optimizers/ngd.py` | ✓ |
| MAO (multi-adaptive optimizer) | ✓ `optim/mao_optimizer.py` + `run_grads` dispatch | ✓ `optimizers/mao.py` + dispatcher on `OptimizerKind` | ✓ |
| Warm-start-then-switch (`WARM_NGD`) | ✓ first-class | ✓ `switch_episode` / `switch_optimizer` fields | ✓ |
| AdaHessian | ✓ `optim/adahessian.py` (custom vendored copy) | ✗ | minor gap |
| PCGrad (per-equation gradient projection) | ✗ | ✓ | deqn-jax **extra** |
| Gauss-Newton / Levenberg-Marquardt | ✗ | ✓ `optimizers/gauss_newton.py` | deqn-jax **extra** |
| L-BFGS (for warm start) | ✗ | ✓ `optimizers/lbfgs.py` + `warm_start.py` | deqn-jax **extra** |
| Shampoo | ✗ | ✓ | deqn-jax **extra** |
| Loss reweighting | ✓ `softmax`, `inverse`, `focal` schemes | ✓ `lr_annealing`, `relobralo` schemes | different algorithms, both work |
| LR scheduler (ReduceLROnPlateau, etc.) | ✓ | ✓ | ✓ |
| Gradient accumulation | ✓ (`optimizer_kwargs`) | ✗ (directly; `n_minibatches_per_epoch` changes semantics) | minor |
| Gradient clipping | ✓ (`clipvalue`) | ✓ (`grad_clip`) | ✓ |

**Assessment.** `deqn-jax` has strictly more optimizer capability than
DEQN-MAO. The only missing items are AdaHessian and native gradient
accumulation.

---

## 6. Network

| Capability | DEQN-MAO | deqn-jax | Gap |
|---|---|---|---|
| MLP | ✓ | ✓ | ✓ |
| Per-layer activation, per-layer init | ✓ via config | ✓ via config | ✓ |
| Sequence networks (LSTM, Transformer) | ✗ | ✓ `networks/lstm.py`, `networks/transformer.py` | deqn-jax **extra** |
| Custom model-specific `Net.py` override | ✓ idiom | partial — `create_mlp(...)` options; no blanket "model brings its own network" hook | slight gap |
| Multi-head output | ✗ | ✓ `MultiHeadMLP` | deqn-jax **extra** |
| Residual/skip connections | ✗ | ✓ `ResMLP` | deqn-jax **extra** |
| Dropout / batch norm in hidden layers | ✓ via config | ✗ (not exposed) | minor gap |
| `adjust_weight_init` heuristic | ✓ `unit_activation_reinitializer.py` | ✗ | minor gap |

---

## 7. Hooks / diagnostics / lifecycle

| Capability | DEQN-MAO | deqn-jax | Gap |
|---|---|---|---|
| `post_init(state)` — called once before training | ✓ model-level hook | ✗ | **gap** |
| `cycle_hook(state, i)` — called every episode (model can plot, log histograms, etc.) | ✓ model-level hook | ✗ framework has `log_every` but no user-callable per-cycle hook | **gap** |
| `end_hook(state, i)` — called after training | ✓ | ✗ | minor gap |
| TensorBoard scalar + histogram summaries | ✓ | ✓ via `metrics.py` | ✓ |
| W&B integration | ✗ | ✓ | deqn-jax **extra** |
| Plotting utilities per model | ✓ model-level `Hooks.py` | some via `plots/` module (shared, not model-level) | **different location** |

**Assessment.** DEQN-MAO's per-model `Hooks.py` is a nice contract: every
model file directory has five files — Variables, Equations, Dynamics,
Net, Hooks — and the framework calls the Hooks at the right moments.
`deqn-jax` has no equivalent. When writing my bm_deterministic notebook
I ended up hand-rolling diagnostic plots inline; in DEQN-MAO Simon's
`brock_mirman/Hooks.py` does this declaratively and the plots appear
automatically each 100 episodes.

---

## 8. Composite loss / advanced features

Things `deqn-jax` has that DEQN-MAO doesn't — credit where due.

| Capability | DEQN-MAO | deqn-jax | Notes |
|---|---|---|---|
| Composite loss (anchor + Jacobian + barrier + Newton aux) | ✗ | ✓ | Used for the disaster model |
| Model linearization / Blanchard-Kahn solve for anchor | ✗ | ✓ | `training/linearize.py` |
| Risky steady state auto-detection | ✗ | ✓ | `use_risky_steady_state` flag |
| Impulse-response tooling (`irf.py`) | per-model scripts (`impulse_response_EKNR.py`, etc.) | generic `run_irf`, `run_girf` | deqn-jax **cleaner** |
| Benchmark harness | ✗ | ✓ `benchmark.py` | extra |

---

## 9. Config system

| Capability | DEQN-MAO | deqn-jax | Gap |
|---|---|---|---|
| YAML-driven | ✓ Hydra (nested composition) | ✓ Pydantic + YAML (flat) | different idiom |
| Per-aspect config files (run, optimizer, net, constants, variables) | ✓ Hydra `defaults:` | ✗ single YAML file per model | **gap in ergonomics** |
| CLI overrides | ✓ Hydra `--foo.bar=baz` | ✓ `--set foo.bar=baz` | ✓ |
| Auto-resume from checkpoint dir | ✓ `STARTING_POINT: LATEST` | ✓ `checkpoint_dir` + resume machinery | ✓ |
| Validation / type coercion | via Hydra/OmegaConf | ✓ Pydantic v2 (stronger) | deqn-jax **extra** |

---

## Critical gaps (must fix to move forward)

1. **`initialize_each_episode`.** Rename my `resample_each_cycle` patch
   to match upstream. Default to `False` (upstream behaviour). Document.
2. **Per-variable init distributions.** Right now `init_state_fn` is
   monolithic. Adding declarative `init` specs on each state variable
   would bring us closer to the reference and make `ModelSpec` more
   expressive.
3. **Bound penalties on definitions.** If a model says `c > 1e-8`, the
   framework should add a penalty term automatically, not rely on the
   model author to handle boundary crossings. Related: auto-generate
   `_RAW` accessors.
4. **Per-model `Hooks` module** — `post_init`, `cycle_hook`, `end_hook`.
   Currently every model that wants diagnostic plots hand-rolls them in
   a notebook. DEQN-MAO makes this a framework contract.

## Non-critical gaps (nice-to-have)

5. **E_t expectation operator in equations.** Lets you write nonlinear
   transforms of expectations directly. Not needed for the 4 Geneva
   models I've looked at, but it's a real design gap.
6. **Activation hints in policies** (`activation: implied` in
   DEQN-MAO's config). Currently `deqn-jax` hardcodes sigmoid+softplus
   selection based on whether `policy_upper` is finite.
7. **AdaHessian optimizer, gradient accumulation.** Minor.
8. **Hydra-style nested config composition.** Ergonomic, not functional.
9. **Dropout / batch-norm in the default MLP builder.** Trivial to add.
10. **`adjust_weight_init` weight-initialization heuristic.**

## Where `deqn-jax` is ahead

- Sequence networks (LSTM, Transformer)
- PCGrad, Gauss-Newton, L-BFGS, Shampoo optimizers
- L-BFGS warm start to steady state
- Composite loss with anchor / Jacobian / barrier / Newton terms
- Blanchard-Kahn linearization for anchor
- Generic IRF/GIRF tooling (model-agnostic)
- Pydantic v2 config validation
- W&B integration
- Antithetic MC shocks

## Recommended action list

**Short term** (before any more Geneva work):

A. Rename `resample_each_cycle` → `initialize_each_episode` to match
   upstream naming, in `config.py`, `trainer.py`, and my in-flight
   config. Keep behaviour identical. Document as "feature parity with
   DEQN-MAO."
B. Revert the four model trees I added (`bm_*`, `olg_analytic_6`) to
   their state prior to my Geneva-port work. The `_complementarity`
   helper can stay.
C. Add `initialize_each_episode: true` to a fresh `bm_deterministic`
   model file and train it via the framework. That's the real proof
   the framework admits simple models.

**Medium term** (if we want framework parity):

D. Per-variable init distributions in `ModelSpec` (declarative).
E. Auto-generated penalty term from declarative state/policy/definition
   bounds. Wire into `compute_loss`.
F. Per-model `Hooks.py` contract: `post_init`, `cycle_hook`, `end_hook`
   called from the trainer at the right moments.
G. Optional: rework equation signature to accept an `E_t` operator so
   nonlinear-of-expectation residuals can be expressed naturally.

**Long term** (polish):

H. Hydra-style nested config composition.
I. AdaHessian, dropout/batchnorm in MLP, `adjust_weight_init`.

---

Once A–C are done we have a framework that actually admits `bm_deterministic`
without any patch framing, and we've removed my suspect work. D–G are
discretionary — they'd close the remaining gaps to full DEQN-MAO parity
but none of them block re-doing the Geneva port properly.
