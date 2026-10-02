# Reading guide

A walk through the source for contributors who need to find subtle bugs,
propose architectural changes, or add domain knowledge.

For usage, see [Quickstart](getting-started/quickstart.md) and
[Running experiments](running_experiments.md). For the rendered API, see the
[API reference](api/config.md). The same material as diagrams is in
[Architecture](architecture.md).

> Reading order: §1, §2 (one cycle end to end), §3 (constraints the code
> relies on). §4–6 are reference.

---

## 1. Where things live

```
src/deqn_jax/
  api.py            stable public surface (load_model, TrainConfig, train, evaluate, IRFs, checkpoint loader)
  config/           Pydantic v2 configs + YAML/CLI loader (TrainConfig, ...)
  cli/              one module per subcommand (train, models, irf, evaluate, init_config)
  types.py          ModelSpec, TrainState, ReweightState, Metrics (NamedTuples)
  evaluate/         Checkpoint → simulation, diagnostics, IRFs, Dynare comparison
  plots/            Diagnostic plotting helpers (no inbound deps in package)

  models/
    __init__.py     load_model(name) — explicit registry
    variable_spec.py
    _complementarity.py        KKT-style helpers shared by some models
    aiyagari/                  Heterogeneous-agent (incomplete markets); not registered
    bm_deterministic/          Brock-Mirman without shocks
    bm_labor/                  Brock-Mirman + labor margin
    bm_labor_autodiff/         …with autodiff-synthesised Euler residuals
    bm_labor_constrained/      …with an upper labor cap (Fischer-Burmeister)
    brock_mirman/              Minimal RBC reference (1 eq, 2 states)
    brock_mirman_autodiff/     …with autodiff-synthesised Euler residuals
    disaster/                  Full-scale NK-DSGE w/ banking (11/13/11)
    irbc/                      International RBC (two-country)
    olg_analytic_6/            6-period OLG with closed-form SS
    olg_lifecycle/             6-generation life-cycle OLG with borrowing constraints
    olg_lifecycle_56/          56-generation annual version of the same
    rss_trade_ez_ref/          RSS-2019 three-country trade DSGE, Epstein-Zin

  networks/
    factory.py        net_type dispatch used by create_train_state
    common.py         Output-bounding helpers (sigmoid bounds, etc.)
    mlp.py            Equinox MLP factory
    lstm.py           Sequence policy (history-aware)
    transformer.py    Multi-head-attention sequence policy
    linear_plus_mlp.py  Residual on top of Blanchard-Kahn linear policy
    rss_net.py        Policy network for the RSS trade model

  optimizers/
    registry.py     OptimizerKind enum, @register_optimizer, create_optimizer
    _step_common.py pieces shared by the five grad-step variants
    standard.py     STANDARD grad_step (adam, muon, ngd, shampoo)
    pcgrad.py       Per-equation gradient surgery (PCGrad)
    mao.py          Multi-Adaptive Optimizer (per-equation moments)
    ngd.py          Diagonal-Fisher natural gradient
    shampoo.py      Kronecker-factored Shampoo
    lbfgs.py        Thin wrapper around optax.lbfgs (line-search args)
    gauss_newton.py Gauss-Newton / Levenberg-Marquardt

  training/
    trainer.py        train(), train_from_config(), _run_training_loop()
                      — slim orchestrator.
    state_init.py     create_train_state(), make_train_step() — assembles
                      the variant pipeline (re-exported from trainer).
    cycle.py          make_rollout_fn + make_cycle_step.
                      One cycle = one rollout + N minibatch grad steps.
    loop_control.py   Python-side per-episode controllers (optimizer switch,
                      LR / curriculum scale).
    loss.py           compute_residuals, compute_loss (MC + GH quadrature),
                      eq_losses_to_array.
    composite_loss.py Anchor + Jacobian + barrier + Newton aux losses.
    moment_loss.py    Moment-matching aux loss.
    coverage.py       EWM coverage sampling.
    replay.py         Prioritized state-replay buffer.
    episode.py        lax.scan trajectory simulator (run_episode,
                      run_episode_with_history).
    history.py        History-window construction for sequence networks.
    linearize.py      Blanchard-Kahn QZ → P, Q matrices + ergodic cov.
    warm_start.py     L-BFGS fit of policy net to steady state.
    steady_state.py   Generic SS-fitting helpers (per-model SS lives in
                      models/<name>/steady_state.py).
    autodiff.py       jax.grad-based Euler synthesis used by *_autodiff models.
    reweighting.py    lr_annealing / relobralo loss-weight schedulers.
    shocks.py         Antithetic MC + tensor-product Gauss-Hermite sampling.
    checkpointing.py  eqx.tree_serialise_leaves wrappers + resumption.
    metrics.py        TensorBoard / W&B logger backends.
    reporting.py      Console reporting and per-episode logging (out of JIT).
```

Where to look for a bug:

- Behaviour during training: start in `training/cycle.py`, then
  `training/loss.py`. `trainer.py` mostly assembles: it picks the variant and
  wires `cycle_step`.
- Wrong loss values: `loss.py` (mixture branch, expectation aggregation) and
  `composite_loss.py` (aux terms).
- Odd optimizer behaviour: the `grad_step` in `optimizers/<name>.py`, then the
  variant dispatch in `make_train_step` in `training/state_init.py`.
- Model misbehaviour: `models/<name>/equations.py`, including the diagnostic
  dict returned by `definitions()`.
- Config not parsing: the validators in `config/_base.py` (Pydantic v2
  `before` mode handles type coercion).
- Rollout or shock issues: `training/cycle.py` (`rollout_fn`),
  `training/shocks.py`, the model's `step_fn`.

## 2. One cycle, end-to-end

![DEQN solver training loop](figures/deqn_solver_loop.svg)

*The conceptual loop. A cycle repeats `N_cycles` times. Each cycle runs a
rollout (an episode of length `N_episode_length` that alternates shock draw,
policy forward pass and dynamics step, filling `state_episode`), then a
training pass of `N_epochs_per_episode` × `N_minibatches` minibatches through
forward and backward. In the JAX port, `cycle_step` calls a JIT'd rollout and
then a JIT'd `grad_step` per minibatch, each covering loss, forward and
backward.*

The path below goes from `deqn-jax train --config configs/disaster.yaml` to a
single weight update.

### 2.1 Entry point

`cli/__init__.py:main()` parses args and dispatches on the subcommand. For
`train`:

```
cli.train.run
  └─ load_config(...)                         # config/io.py
  └─ train_from_config(config)                # training/trainer.py
```

### 2.2 Setup (one-time, before JIT)

In `training/trainer.py:train_from_config`:

1. Validate combinations: fp64 toggle, composite loss against optimizer,
   `episode_length=1` against `sim_batch` and `shock_mask`.
2. `load_model(config.model)` returns a `ModelSpec` from the explicit
   registry in `models/__init__.py`.
3. Apply `config.constants` with `model._replace(constants={...})`, for
   per-run calibration sweeps.
4. Call the model's optional `setup_fn`. For example, `disaster` with
   `p_disaster > 0` replaces `steady_state_fn` with its
   `risky_steady_state` (a Gourio-style locally flat solver). The swap is
   declared by the model, not hard-coded for `disaster`.
5. `create_train_state` builds the network (Equinox), the optimizer (Optax)
   and the initial states (sampled near SS or from the model's
   `init_state_fn`), seeds `history_state` for sequence policies, and packs
   everything into a `TrainState` NamedTuple.
6. If `loss_type == "composite"`, `linearize_model` runs Blanchard-Kahn QZ
   to get `(P, Q)`, then `prepare_composite_data` precomputes anchor points
   and the ergodic covariance.
7. `make_train_step` picks the `grad_step` variant and builds `cycle_step`
   (`training/cycle.py`) around it.

### 2.3 The episode loop

Pseudocode (real code in `trainer.py:train_from_config`):

```python
for ep in range(start_episode, total_episodes):
    shock_scale = curriculum.scale_at(ep)
    state, metrics = cycle_step(state, lr_scale, shock_scale)
        # JIT'd rollout + grad steps — see §2.4
    log(state, metrics); maybe_checkpoint(state)
    cycle_hook(state, model, ep)   # optional model-specific hook
```

The outer Python loop only dispatches, logs and checkpoints. `shock_scale`
reaches both the rollout (so the curriculum and `shock_mask` apply to state
simulation) and the loss expectation.

### 2.4 Inside the JIT boundary

`cycle_step` (in `training/cycle.py`) is the core of the codebase. It calls
the JIT'd `rollout_fn` (which runs `run_episode` or
`run_episode_with_history`) to fill `trajectory`, then loops over
`epochs × minibatches` calling the optimizer's JIT'd `grad_step`. Five
`grad_step` variants exist. `make_train_step` chooses one at construction
from `OptimizerKind`, and PCGrad when `gradient_surgery="pcgrad"` is set on a
STANDARD optimizer:

| Variant   | File                       | Gradient path                                        |
|-----------|----------------------------|------------------------------------------------------|
| STANDARD  | `optimizers/standard.py`   | `value_and_grad(loss)` → `opt.update(grads, ...)`    |
| PCGRAD    | `optimizers/pcgrad.py`     | per-equation grads → conflict projection → standard  |
| MAO       | `optimizers/mao.py`        | `jacrev(per_eq_loss)` → Jac → `mao.update(jac, ...)` |
| LBFGS     | `optimizers/lbfgs.py`      | `optax.lbfgs` w/ `value`, `grad`, `value_fn`         |
| GN        | `optimizers/gauss_newton.py` | residual Jacobian `J` → step `-(JᵀJ)⁻¹ Jᵀ r`       |

Each `grad_step` calls `compute_loss` (`loss.py`), which:

1. Samples shocks (antithetic MC via `shocks.py`, or tensor-product
   Gauss-Hermite quadrature).
2. `vmap`s `compute_residuals` over shocks.
3. Inside `compute_residuals`:
    - `policy = policy_fn(state)`
    - `next_state = step_fn(state, policy, shock, constants)`
    - `next_policy = next_fn(next_state)` (the target net if active, with
      `stop_gradient` applied)
    - `residuals = equations_fn(state, policy, next_state, next_policy)`
    - If the model uses a mixture branch (e.g. disaster with `p > 0`),
      both branches are computed and mixed as `(1-p)·r₀ + p·r₁`.
4. Aggregates: weighted mean over shocks (E[r]), squared, mean over batch.
5. With the composite loss, adds `aux_anchor`, `aux_jac`, `aux_barrier_*`
   and `aux_newton_*`. The `aux_` prefix makes reweighting and gradient
   surgery ignore them (see §3.2).

The optimizer update then produces new params and a new `TrainState`.

`cycle.py` and `trainer.py` are organised around this path.

## 3. Constraints the code relies on

Violating these breaks things silently, sometimes badly. Read this section
before changing anything fundamental.

### 3.1 Two JIT boundaries per cycle

Each cycle crosses two kinds of `@jax.jit` function: `rollout_fn`, called
once, and the variant's `grad_step` (loss + grad + optimizer step), called
once per minibatch. `cycle_step` is the Python function that drives them.
Keeping each piece inside one JIT is the main performance decision. Splitting
a piece into several JITs:

- loses XLA fusion across the split (substantially slower);
- adds host-device sync points (latency);
- makes JAX traces larger (longer compilation).

A one-off Python operation per step belongs in the outer Python loop, not
inside a JIT'd function.

### 3.2 `aux_` prefix on auxiliary loss keys

Adaptive reweighting (`reweighting.py`: `lr_annealing`, `relobralo`) and
per-equation gradient surgery (PCGrad, MAO) work on per-equation residuals.
They iterate over `eq_losses`, and without a filter they would treat the
anchor, Jacobian, barrier and Newton terms as equilibrium equations.

`eq_losses_to_array` in `loss.py` drops every key in `eq_losses` that starts
with `aux_`.

When you add an auxiliary loss term, prefix its key with `aux_`. Otherwise
reweighting will silently shift training toward it.

### 3.3 `next_policy = next_fn(next_state)` with optional `stop_gradient`

In `compute_residuals`, when a target network is active
(`target_update_every > 0`), `next_policy` is computed from `target_params`
under `jax.lax.stop_gradient`. This cuts the self-referential gradient path
in which the network must satisfy today's equations and match its own future
outputs at once.

Removing the `stop_gradient` while keeping the target-network plumbing makes
the target network useless.

### 3.4 `shock_names` and `step_fn` column order must match

`ModelSpec.shock_names` must list shocks in the order they appear in
`step_fn`'s shock argument:

```python
# example from a multi-shock model
eps_a, eps_b, eps_c = shock[:, 0], shock[:, 1], shock[:, 2]
```

An earlier bug in the disaster model swapped two shocks in `shock_names`
while `step_fn` had them right. IRF analysis ran but labelled the shocks
wrongly. When adding or reordering shocks, update both together.

### 3.5 Steady-state caching keys must include all relevant constants

Some models cache `_solve_steady_state` results so that config sweeps do not
re-solve. The cache key must include every constant the steady state depends
on, not `frozenset(constants.items())` over a hand-picked subset.

An earlier bug used one module-level cache filled at import time, which
returned stale results when the caller passed different constants. The fix
keys on the full constants dict and recomputes when any of them changes.

This applies wherever `risky_steady_state` is used (it depends on the
disaster parameters), to parameter sweeps, and to any future calibration
overrides.

### 3.6 `equations_fn` returns a dict, ordering preserved by Python ≥3.7

`equations_fn` returns a dict of named residuals. Its order matters because:

- `eq_losses_to_array` flattens it to a vector for MAO, PCGrad and GN;
- `EQUATION_NAMES` must list the same order, which diagnostics use.

All supported Python versions preserve insertion order. Do not sort or
rebuild the dict between insertion and iteration.

### 3.7 Model-specific invariants live with the model

Some calibrations have constraints the framework cannot enforce generically:
eigenvalue-count requirements, parameters pinned at the edge of validity, the
log-vs-ratio choice for aggregator residuals under non-Gaussian shocks, and
similar.

These are documented with the model. For the disaster model, see
[Disaster (NK-DSGE)](models/disaster.md).

## 4. Pytrees and side-effect discipline

`jit`, `grad` and `vmap` need pure functions. The codebase keeps them pure as
follows:

- All state lives in `TrainState`, a NamedTuple that JAX treats as a pytree.
  `cycle_step` takes `state` and returns a new `state`. Nothing mutates
  module-level data.
- Equinox modules separate trainable arrays from static config via
  `eqx.filter(model, eqx.is_array)`. The optimizer sees only arrays.
- No `print`, no `float()` conversion and no Python branches on traced values
  inside the JIT'd functions.

Common gotchas:

- `jax.tree.map` treats Python tuples as pytree containers. A mapped function
  that returns a tuple gets it unpacked into the tree structure. Use lists or
  NamedTuples to keep a tuple as a leaf.
- `jax.lax.cond` requires an `operand` argument (pass `None` if the branches
  take no input).
- Python-level `ndim` checks inside `tree_map` callbacks are fine; they
  resolve at trace time.
- Shampoo: create the L and R preconditioners with separate `tree_map` calls,
  never one call that returns a tuple pair.

## 5. Where to add new things

| You want to add a…       | See                                                                  |
|--------------------------|----------------------------------------------------------------------|
| New economic model       | [Implementing a model](models/implementing.md)                       |
| New network              | [Adding a network](networks/adding.md)                               |
| New optimizer            | [Adding an optimizer](optimizers/adding.md)                          |
| New loss term            | `training/composite_loss.py`, prefix the key with `aux_` (§3.2)      |
| New CLI subcommand       | a module under `cli/` with `add_parser(subparsers)`, registered in `cli/__init__.py`                           |
| New config field         | `config/train.py:TrainConfig` + a Pydantic validator on `_ConfigBase` (`config/_base.py`) |
| New checkpoint format    | Don't. Use `eqx.tree_serialise_leaves` / `_deserialise_leaves`.      |

## 6. Things that look odd but are intentional

- The MAO factory takes `n_tasks` lazily (`_MAOFactory` in
  `optimizers/mao.py`). The model's equation count is not known when the
  config is parsed; `create_train_state` resolves it once
  `model.equation_names` is available.
- The cosine LR schedule is applied through `lr_scale`. When a schedule is
  active, the optimizer is created with `lr=1.0` and the actual LR is passed
  to `cycle_step` as a dynamic scalar each cycle, which avoids re-JIT at every
  schedule step.
- The curriculum `shock_scale` reaches both the rollout (so `shock_mask` and
  the curriculum apply to state simulation) and the loss expectation. Before
  2026-04-24 it applied only to the loss.
- `eqx.combine(updated_arrays, model)` is used everywhere instead of mutating
  the model. Equinox modules are immutable and are rebuilt with new arrays.
- `history_state` is part of `TrainState`. For sequence policies it persists
  across cycles; for MLPs it is `None`. `cycle_step` dispatches the same way
  in both cases.

---

## When this guide gets stale

If the guide and the source disagree, trust the code and update the guide.
The guide is meant to prevent surprises, not to be a complete reference; the
complete reference is the source plus mkdocstrings.
