# Reading guide

A code-level narrative for contributors. Read this **first** if you need to
understand the codebase well enough to spot subtle bugs, recommend an
architectural change, or insert domain expertise.

For higher-level "how do I use it?" content, see [Quickstart](getting-started/quickstart.md)
and [Running experiments](running_experiments.md). For the rendered API,
see the [API reference](api/config.md). For diagrams of the same content,
see [Architecture](architecture.md).

> **Reading order**: §1 → §2 (one cycle, end-to-end) → §3 (load-bearing
> constraints). §4–6 are reference material to dip into when needed.

---

## 1. Where things live

```
src/deqn_jax/
  config/           Pydantic v2 configs + YAML/CLI loader (TrainConfig, ...)
  cli.py            argparse → train/list/info/check/evaluate/irf/optimizers
  types.py          ModelSpec, TrainState, ReweightState, Metrics (NamedTuples)
  metrics.py        TensorBoard / W&B / Null logger
  evaluate/         Checkpoint → policy evaluation + residual analysis
  irf.py            Checkpoint → impulse-response simulation
  benchmark.py      Performance harness (per-step timing)
  plots/            Diagnostic plotting helpers (no inbound deps in package)

  models/
    __init__.py     load_model(name) — explicit registry
    variable_spec.py
    _complementarity.py        KKT-style helpers shared by some models
    aiyagari/                  Heterogeneous-agent (incomplete markets)
    bm_deterministic/          Brock-Mirman without shocks
    bm_labor/                  Brock-Mirman + labor margin
    bm_labor_autodiff/         …with autodiff-synthesised Euler residuals
    brock_mirman/              Minimal RBC reference (1 eq, 2 states)
    brock_mirman_autodiff/     …with autodiff-synthesised Euler residuals
    disaster/                  Full-scale NK-DSGE w/ banking (11/13/11)
    irbc/                      International RBC (two-country)
    olg_analytic_6/            6-period OLG with closed-form SS

  networks/
    common.py         Output-bounding helpers (sigmoid bounds, etc.)
    mlp.py            Equinox MLP factory
    lstm.py           Sequence policy (history-aware)
    transformer.py    Multi-head-attention sequence policy
    linear_plus_mlp.py  Residual on top of Blanchard-Kahn linear policy

  optimizers/
    registry.py     OptimizerKind enum, @register_optimizer, create_optimizer
    standard.py     Adam / SGD / AdamW / Lion / Muon (one grad_step path)
    pcgrad.py       Per-equation gradient surgery (PCGrad)
    mao.py          Multi-Adaptive Optimizer (per-equation moments)
    mao_kfac.py     K-FAC variant of MAO
    ngd.py          Diagonal-Fisher natural gradient
    shampoo.py      Kronecker-factored Shampoo
    lbfgs.py        Thin wrapper around optax.lbfgs (line-search args)
    gauss_newton.py Gauss-Newton / Levenberg-Marquardt

  training/
    trainer.py        train(), train_from_config(), _run_training_loop()
                      — slim orchestrator.
    state_init.py     create_train_state(), make_train_step() — assembles
                      the variant pipeline (re-exported from trainer).
    cycle.py          rollout_fn + cycle_step — THE JIT entry point.
                      One cycle = one rollout + N minibatch grad steps.
    loss.py           compute_residuals, compute_loss (MC + GH quadrature),
                      eq_losses_to_array.
    composite_loss.py Anchor + Jacobian + barrier + Newton aux losses.
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
    reporting.py      Console / TB / W&B reporting helpers (out of JIT).
```

When you're hunting for a bug:

- **Behaviour during training** → start in `training/cycle.py` (the JIT
  entry), then walk into `training/loss.py`. `trainer.py` itself is
  mostly assembly: it picks the variant and wires `cycle_step`.
- **Loss values look wrong** → `loss.py` (mixture branch, expectation
  aggregation), `composite_loss.py` (aux terms).
- **Optimizer behaving oddly** → `optimizers/<name>.py`'s `grad_step`,
  then check the variant dispatch in `make_train_step` in
  `training/state_init.py`.
- **Model misbehaviour** → `models/<name>/equations.py`, paying attention
  to the diagnostic dict returned by `definitions()`.
- **Config not parsing right** → `config/_base.py` validators (Pydantic v2
  `before` mode handles type coercion).
- **Rollout / shock issues** → `training/cycle.py` (rollout_fn),
  `training/shocks.py`, model's `step_fn`.

## 2. One cycle, end-to-end

![DEQN solver training loop](figures/deqn_solver_loop.svg)

*Conceptual loop. A cycle repeats `N_cycles` times; each runs a rollout
(episode of length `N_episode_length` alternating shock draw, policy
forward pass, and dynamics step — fills `state_episode`) followed by a
training pass (`N_epochs_per_episode` × `N_minibatches` minibatches
through forward + backward). The current JAX port collapses
loss/forward/backward into a single JIT'd `cycle_step`.*

Follow the path from `deqn-jax train --config configs/disaster.yaml` to a
single weight update.

### 2.1 Entry point

`cli.py:main()` parses args, loads config, dispatches on subcommand. For
`train`:

```
cli.run_train
  └─ TrainConfig.from_yaml + with_overrides   # config.py
  └─ train_from_config(config)                # training/trainer.py
```

### 2.2 Setup (one-time, before JIT)

In `training/trainer.py:train_from_config`:

1. **Validate combinations** — fp64 toggle, composite ↔ optimizer
   compatibility, `episode_length=1` ↔ `sim_batch` ↔ `shock_mask` checks.
2. **`load_model(config.model)`** → `ModelSpec` from the explicit
   registry in `models/__init__.py`.
3. **Apply `config.constants`** → `model._replace(constants={...})`.
   Used for per-run calibration sweeps.
4. **Optional risky-SS swap** — when the active model wants it
   (e.g. `disaster` with `p_disaster > 0`), `steady_state_fn` is
   replaced with the model's `risky_steady_state` (Gourio-style
   locally-flat solver). This is keyed on the model, not hard-coded
   for `disaster`.
5. **`create_train_state`** → builds the network (Equinox), the
   optimizer (Optax), the initial states (sampled near SS or via the
   model's `init_state_fn`), seeds `history_state` for sequence
   policies, and packs everything into a `TrainState` NamedTuple.
6. **If `loss_type == "composite"`** — `linearize_model` runs
   Blanchard-Kahn QZ to get `(P, Q)`, then `prepare_composite_data`
   precomputes anchor points + ergodic covariance.
7. **`make_train_step`** dispatches on `OptimizerKind` and wraps
   `cycle_step` (defined in `training/cycle.py`) in `jax.jit`.

### 2.3 The episode loop

Pseudocode (real code in `trainer.py:train_from_config`):

```python
for ep in range(start_episode, total_episodes):
    shock_scale = curriculum.scale_at(ep)
    state, metrics = cycle_step(state, lr_scale, shock_scale)
        # JIT'd — see §2.4
    log(state, metrics); maybe_checkpoint(state)
    cycle_hook(model, state, ep)   # optional model-specific hook
```

Everything inside `cycle_step` is JIT-compiled; the outer Python loop is
just dispatch, logging, and checkpointing. `shock_scale` flows into
both the rollout (so curriculum and `shock_mask` apply to state
simulation) and the loss expectation.

### 2.4 Inside the JIT boundary

`cycle_step` (in `training/cycle.py`) is the heart of the codebase. It
runs `rollout_fn` (which calls `run_episode` or `run_episode_with_history`)
to fill `trajectory`, then loops over `epochs × minibatches` calling the
optimizer's `grad_step`. Five `grad_step` variants exist, dispatched at
construction by `OptimizerKind`:

| Variant   | File                       | Gradient path                                        |
|-----------|----------------------------|------------------------------------------------------|
| STANDARD  | `optimizers/standard.py`   | `value_and_grad(loss)` → `opt.update(grads, ...)`    |
| PCGRAD    | `optimizers/pcgrad.py`     | per-equation grads → conflict projection → standard  |
| MAO       | `optimizers/mao.py`        | `jacrev(per_eq_loss)` → Jac → `mao.update(jac, ...)` |
| LBFGS     | `optimizers/lbfgs.py`      | `optax.lbfgs` w/ `value`, `grad`, `value_fn`         |
| GN        | `optimizers/gauss_newton.py` | residual Jacobian `J` → step `-(JᵀJ)⁻¹ Jᵀ r`       |

Each `grad_step` calls `compute_loss` (`loss.py`), which:

1. Samples shocks (MC antithetic via `shocks.py` OR tensor-product
   Gauss-Hermite quadrature).
2. `vmap`s `compute_residuals` over shocks.
3. Inside `compute_residuals`:
    - `policy = policy_fn(state)`
    - `next_state = step_fn(state, policy, shock, constants)`
    - `next_policy = next_fn(next_state)` (target net if active,
      `stop_gradient` applied)
    - `residuals = equations_fn(state, policy, next_state, next_policy)`
    - **If the model uses a mixture branch** (e.g. disaster `p > 0`):
      compute both branches and mix `(1-p)·r₀ + p·r₁`.
4. Aggregates: weighted mean over shocks (E[r]), squared, mean over batch.
5. **If composite** — adds `aux_anchor`, `aux_jac`, `aux_barrier_*`,
   `aux_newton_*`. All keyed with `aux_` prefix so reweighting and
   gradient surgery ignore them (see §3.2).

Apply optimizer update → new params → return new `TrainState`.

That's the full path. **Keep this trace in mind** when reading
`cycle.py` and `trainer.py` — the modules are organised around it.

## 3. Load-bearing constraints

These are **invariants the codebase depends on**. Violating them silently
breaks things, sometimes catastrophically. Read this before you change
anything fundamental.

### 3.1 Single JIT boundary

`cycle_step` (loss + grad + opt-step) is **one** `@jax.jit` function.
This is the core performance decision. Splitting it into multiple JITs:

- Loses XLA fusion across the boundary (substantially slower).
- Creates host-device sync points (latency).
- Makes JAX traces larger (longer compilation).

If you must add a one-off Python operation per step, pull it OUT of
`cycle_step` (back into the outer Python loop) — never break the JIT.

### 3.2 `aux_` prefix on auxiliary loss keys

Adaptive reweighting (`reweighting.py`: `lr_annealing`, `relobralo`)
and per-equation gradient surgery (PCGrad, MAO) operate on per-equation
residuals. They iterate over `eq_losses` and would treat anchor / jac /
barrier / newton terms as if they were equilibrium equations, which
they're not.

The convention: any key in the `eq_losses` dict prefixed with `aux_` is
filtered out by `eq_losses_to_array` in `loss.py`.

**When adding a new auxiliary loss term, always prefix its key with
`aux_`.** Otherwise reweighting will silently rebalance the training
toward the auxiliary term.

### 3.3 `next_policy = next_fn(next_state)` with optional `stop_gradient`

In `compute_residuals`, when a target network is active
(`target_update_every > 0`), `next_policy` is computed from
`target_params` and `jax.lax.stop_gradient` is applied. This breaks the
self-referential gradient loop where the network must simultaneously
satisfy today's equations and be consistent with its own future outputs.

**If you remove the `stop_gradient` while keeping the target-network
plumbing**, you've defeated the entire purpose of having a target net.

### 3.4 `shock_names` and `step_fn` column order must match

The `ModelSpec.shock_names` tuple must enumerate shocks in the same
order they appear in `step_fn`'s shock argument:

```python
# example from a multi-shock model
eps_a, eps_b, eps_c = shock[:, 0], shock[:, 1], shock[:, 2]
```

A previous bug in the disaster model had two shocks swapped in
`shock_names` while `step_fn` had them correct — IRF analysis ran fine
but mislabeled which shock was which. **When adding or reordering
shocks, update both at once.**

### 3.5 Steady-state caching keys must include all relevant constants

Some models cache `_solve_steady_state` results to avoid re-solving on
every config sweep. The cache key must include **every constant the SS
depends on** — not just `frozenset(constants.items())` over a hand-picked
subset.

A previous bug used a single module-level cache populated at import
time, which silently returned stale results when the caller passed
different constants. The fix was to key on the full constants dict and
to recompute when any change.

This matters wherever you have `risky_steady_state` (depends on
disaster parameters), parameter-sweep setups, or any future calibration
overrides.

### 3.6 `equations_fn` returns a dict, ordering preserved by Python ≥3.7

`equations_fn` returns a dict of named residuals. The order matters for
two reasons:

- `eq_losses_to_array` flattens it to a vector for MAO / PCGrad / GN.
- `EQUATION_NAMES` must enumerate the same order, used for diagnostics.

Insertion order is preserved by all supported Python versions; just
don't sort or rebuild the dict between insertion and iteration.

### 3.7 Model-specific invariants live with the model

Some calibrations have constraints that the framework can't enforce
generically (eigenvalue-count requirements, parameter-pinning at
validity edges, log-vs-ratio choice on aggregator residuals under
non-Gaussian shocks, etc.).

These belong in the model's docs, not here. For the disaster model in
particular, see [Disaster (NK-DSGE)](models/disaster.md).

## 4. Pytrees and side-effect discipline

JAX requires **pure** functions for `jit` / `grad` / `vmap`. The codebase
maintains this by:

- All state lives in `TrainState` (a NamedTuple — JAX treats it as a
  pytree). `cycle_step` takes `state` in, returns new `state`. No
  module-level mutation.
- Equinox modules separate trainable arrays from static config via
  `eqx.filter(model, eqx.is_array)`. The optimizer sees only arrays.
- No `print`, no `float()` conversion, no Python branches on traced
  values inside `cycle_step`.

Common gotchas:

- `jax.tree.map` treats Python tuples as pytree containers. If a mapped
  function returns a tuple, the tuple gets unpacked into the tree
  structure. Use lists or NamedTuples if you want a tuple as a leaf.
- `jax.lax.cond` requires an `operand` argument (use `None` as a dummy
  if your branches don't need an input).
- Python-level `ndim` checks inside `tree_map` callbacks are fine —
  they resolve at trace time, not run time.
- Shampoo: create L and R preconditioners with separate `tree_map`
  calls, never one call returning a tuple pair.

## 5. Where to add new things

| You want to add a…       | See                                                                  |
|--------------------------|----------------------------------------------------------------------|
| New economic model       | [Implementing a model](models/implementing.md)                       |
| New network              | [Adding a network](networks/adding.md)                               |
| New optimizer            | [Adding an optimizer](optimizers/adding.md)                          |
| New loss term            | `training/composite_loss.py`, prefix the key with `aux_` (§3.2)      |
| New CLI subcommand       | `cli.py:main()`, add an argparse subparser                           |
| New config field         | `config/train.py:TrainConfig` + a Pydantic validator on `_ConfigBase` (`config/_base.py`) |
| New checkpoint format    | Don't. Use `eqx.tree_serialise_leaves` / `_deserialise_leaves`.      |

## 6. Things that look weird but are intentional

- **`OptimizerKind.MAO` factory takes `n_tasks` lazily** — see
  `_MAOFactory` in `optimizers/mao.py`. The model's equation count
  isn't known at config-parse time; it's resolved in
  `create_train_state` when `model.equation_names` is available.
- **Cosine LR schedule baked in via `lr_scale`** — when a schedule is
  active, the optimizer is created with `lr=1.0` and the actual LR is
  passed as a dynamic scalar to `cycle_step` each cycle. This avoids
  re-JIT on every schedule step.
- **Curriculum `shock_scale` flows through everything** — into the
  rollout (so `shock_mask` and the curriculum apply to state
  simulation) AND into the loss expectation. Pre-2026-04-24 it only
  applied to the loss; that bug is now closed.
- **`eqx.combine(updated_arrays, model)`** is used everywhere instead
  of mutating the model. Equinox modules are immutable; you reconstruct
  them with new arrays.
- **`history_state` is part of `TrainState`** — for sequence policies
  it persists across cycles and is `None` for MLPs. The dispatch
  through `cycle_step` is the same either way.

---

## When this guide gets stale

If you find yourself reading the source and the guide says one thing
while the code says another, **trust the code and update the guide**.
The guide is meant to *prevent surprises*, not to be a complete
reference. The complete reference is the source plus mkdocstrings.
