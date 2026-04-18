# DEQN-JAX reading guide

A code-level narrative for contributors. Read this *first* if you need to
understand the codebase well enough to spot subtle bugs, recommend an
architectural change, or insert domain expertise.

For higher-level "how do I use it" content, see the
[user docs](../site/index.md). For the published API, see
[mkdocstrings reference](../site/api/config.md).

> **Reading order**: §1 → §2 (one train step trace) → §3 (load-bearing
> constraints). §4–5 are reference material to dip into when needed.

---

## 1. Where things live

```
src/deqn_jax/
  config.py         Pydantic v2 configs + YAML/CLI loader (TrainConfig, ...)
  cli.py            argparse → train/list/info/check/evaluate/irf/optimizers
  types.py          ModelSpec, TrainState, ReweightState, Metrics (NamedTuples)
  metrics.py        TensorBoard / W&B / Null logger
  evaluate.py       Checkpoint → policy evaluation + residual analysis
  irf.py            Checkpoint → impulse-response simulation
  benchmark.py      Performance harness (per-step timing)

  models/
    __init__.py     load_model(name), explicit registry
    variable_spec.py
    brock_mirman/   Minimal RBC reference (1 eq, 2 states, 1 policy)
    disaster/       Full-scale NK-DSGE with banking (11 eq, 13 states, 11 policies)

  networks/
    common.py       Output bounding helpers
    mlp.py          Equinox MLP factory (create_mlp)
    lstm.py         Sequence policy
    transformer.py  Attention-based sequence policy
    linear_plus_mlp.py  Residual over Blanchard-Kahn linear policy

  optimizers/
    registry.py     OptimizerKind enum, @register_optimizer, create_optimizer
    {adam,sgd,ngd,shampoo,mao,lbfgs,gauss_newton}.py

  training/
    trainer.py      THE main file. train(), train_from_config(),
                    create_train_state(), make_train_step(), 5 step variants
    loss.py         compute_residuals, compute_loss (MC + quadrature)
    composite_loss.py  Anchor + Jacobian + barrier + Newton aux losses
    episode.py      lax.scan trajectory simulator
    history.py      History-window helpers for sequence networks
    linearize.py    Blanchard-Kahn QZ → P, Q matrices
    warm_start.py   L-BFGS fit to SS
    steady_state.py Generic SS-fitting helpers (per-model SS lives in models/)
```

When you're hunting for a bug:

- **Behaviour during training** → start in `training/trainer.py`, then walk
  into `loss.py`.
- **Loss values look wrong** → `loss.py` (mixture branch, expectation
  aggregation), `composite_loss.py` (aux terms).
- **Optimizer behaving oddly** → `optimizers/<name>.py`, then check the
  variant dispatch in `make_train_step` in `trainer.py`.
- **Model misbehaviour** → `models/<name>/equations.py`, paying attention
  to the diagnostic dict returned by `definitions()`.
- **Config not parsing right** → `config.py` validators (Pydantic before-mode).

## 2. One train step, end-to-end

![DEQN solver training loop](../figures/deqn_solver_loop.svg)

*Overall solver loop, adapted from the original DEQN framing. The **cycle** is
an outer repetition; each cycle runs a **rollout** (episode of length
`N_episode_length` alternating random step, policy forward pass, and total
step — fills `state_episode`) followed by a **training** pass
(`N_epochs_per_episode` × `N_batches` minibatches through NN forward+backward).
Our JAX port fuses forward, loss, and backward into a single JIT'd train step
per episode; the conceptual decomposition above is preserved but collapsed.
The raw hand-drawn original is in `docs/figures/deqn_solver_loop_orig_drawing.jpeg`.*


Follow the path from `deqn-jax train --config configs/disaster.yaml` to a
single weight update.

### 2.1 Entry point

`cli.py:main()` parses args, loads config, dispatches on subcommand. For
`train`:

```
cli.run_train
  └─ TrainConfig.from_yaml + with_overrides   # config.py
  └─ train_from_config(config)                # trainer.py
```

### 2.2 Setup (one-time, before JIT)

In `trainer.py:train_from_config`:

1. **`load_model(config.model)`** → `ModelSpec` from the registry.
2. **Apply `config.constants`** → `model._replace(constants={...})`.
   Necessary for per-run calibration sweeps. (`trainer.py:1197`)
3. **Risky-SS swap**: if `model.name == "disaster"` and `p_disaster > 0`,
   replace `steady_state_fn` with `risky_steady_state` (Gourio-style
   locally-flat solver). (`trainer.py:1213`)
4. **`create_train_state`** → builds the network (Equinox), the optimizer
   (Optax), the initial states (sampled near SS), and packs them into a
   `TrainState` NamedTuple.
5. **If `loss_type == "composite"`**: `linearize_model` runs Blanchard-Kahn
   QZ to get `(P, Q)`, then `prepare_composite_data` precomputes anchor
   points + ergodic covariance.
6. **`make_train_step`** dispatches on `OptimizerKind` and wraps the
   chosen variant in `jax.jit`.

### 2.3 The episode loop

Pseudocode (real code in `trainer.py:train_from_config`):

```python
for episode in range(start_episode, total_episodes):
    state, trajectory = run_episode(model, params, episode_state, key)
        # episode.py: lax.scan over episode_length steps
    batch = sample_minibatch(trajectory, batch_size)

    if step % target_update_every == 0:
        target_params = polyak_average(target_params, params, tau)

    lr_scale = lr_schedule(step)

    state = train_step(state, batch, lr_scale)
        # JITTED — see §2.4
    log(state); maybe_checkpoint(state)
```

### 2.4 Inside the JIT boundary

The `train_step` function is the heart of the codebase. Five variants
exist (selected at construction by `OptimizerKind`):

| Variant   | Gradient path                                                  |
|-----------|----------------------------------------------------------------|
| STANDARD  | `value_and_grad(loss)` → `opt.update(grads, opt_state, params)`|
| PCGRAD    | per-equation grads → conflict projection → STANDARD update    |
| MAO       | `jacrev(per_eq_loss_vector)` → Jac → `mao.update(jac, ...)`   |
| LBFGS     | `optax.lbfgs` with `value`, `grad`, `value_fn` for line search|
| GN        | residual Jacobian `J` → step `-(JᵀJ)⁻¹ Jᵀ r`                  |

Each calls `compute_loss` (`loss.py`), which:

1. Samples shocks (MC antithetic OR Gauss-Hermite quadrature).
2. `vmap`s `compute_residuals` over shocks.
3. Inside `compute_residuals`:
   - `policy = policy_fn(state)`
   - `next_state = step_fn(state, policy, shock, constants)`
   - `next_policy = next_fn(next_state)` (target net if active,
     `stop_gradient` applied)
   - `residuals = equations_fn(state, policy, next_state, next_policy)`
   - **If `p_disaster > 0`**: compute both `d=0` and `d=1` branches and
     mix `(1-p)·r₀ + p·r₁`.
4. Aggregates: weighted mean over shocks (E[r]), squared, mean over batch.
5. **If composite**: adds `aux_anchor`, `aux_jac`, `aux_barrier_*`,
   `aux_newton_*`. All keyed with `aux_` prefix so reweighting and
   gradient surgery ignore them.

Apply optimizer update → new params → return new `TrainState`.

That's the full path. **Keep this trace in mind** when reading
`trainer.py` — it's a long file but every section maps to one of the
steps above.

## 3. Load-bearing constraints

These are **invariants the codebase depends on**. Violating them silently
breaks things, sometimes catastrophically. Read this before you change
anything fundamental.

### 3.1 Single JIT boundary

The entire `train_step` (loss + grad + opt-step) is **one** `@jax.jit`
function. This is the core performance decision. Splitting it into
multiple JITs:

- Loses XLA fusion across the boundary (substantially slower).
- Creates host-device sync points (latency).
- Makes JAX traces larger (longer compilation).

If you must add a one-off Python operation per step, pull it OUT of
`train_step` (back into the episode loop) — never break the JIT.

### 3.2 `aux_` prefix on auxiliary loss keys

Adaptive reweighting (`lr_annealing`, `relobralo`) and per-equation
gradient surgery (`pcgrad`, `mao`) operate on per-equation residuals.
They iterate over `eq_losses` and would treat anchor / jac / barrier /
newton terms as if they were equilibrium equations, which they're not.

The convention: any key in the `eq_losses` dict prefixed with `aux_`
is filtered out by `eq_losses_to_array` (`loss.py:eq_losses_to_array`).

**When adding a new auxiliary loss term, always prefix its key with
`aux_`.** Otherwise reweighting will silently rebalance the training
toward the auxiliary term.

### 3.3 `xi_p = 0.6` is pinned by Calvo determinacy

In `models/disaster/variables.py`, the price-stickiness parameter
`xi_p = 0.6` cannot be lowered without recalibrating other constants
simultaneously. Attempting `xi_p = 0.5` produced 14 stable eigenvalues
(determinacy expects 13).

Related: the `pi` policy upper bound is **pinned** at the Calvo
validity edge. The price dispersion formula

```
K_p_inner = (1 - xi_p * (pi_tilda/pi)^-5) / (1 - xi_p)
```

requires `pi < ~1.1*pi_tilda` for `K_p_inner > 0`. Widening the upper
bound triggers gradient explosions through the `soft_floor` at 0.01.

### 3.4 Log-space Calvo aggregator residuals enforce geometric means

`equations.py` historically had Phillips-block residuals in log form:

```python
residuals["eq2b"] = log(eq2_rhs) - log(p.K_p)
```

Under stochastic averaging, this enforces the **geometric** mean of the
Calvo aggregator, not the arithmetic mean (Jensen's inequality). For
small Gaussian shocks, the bias is tiny. For disaster jumps, it's huge.

Current form is **ratio**:

```python
residuals["eq2b"] = eq2_rhs / (p.K_p + eps) - 1.0
```

This enforces the arithmetic mean, which is what the equations actually
say. **Don't switch back to log-form residuals on aggregator equations
without thinking through the Jensen implications.**

### 3.5 `next_policy = next_fn(next_state)` with optional `stop_gradient`

In `compute_residuals`, when a target network is active (target_update_every > 0),
`next_policy` is computed from `target_params` and `jax.lax.stop_gradient`
is applied. This breaks the self-referential gradient loop where the
network must simultaneously satisfy today's equations and be consistent
with its own future outputs.

**If you remove the `stop_gradient` while keeping the target network
plumbing**, you've defeated the entire purpose of having a target net.

### 3.6 `shock_names` and `step_fn` column order must match

The `ModelSpec.shock_names` tuple must enumerate shocks in the same
order they appear in `step_fn`'s shock argument:

```python
eps_shock, mu_ups_shock, mu_z_shock, g_shock, mp_shock = (
    shock[:, 0], shock[:, 1], shock[:, 2], shock[:, 3], shock[:, 4]
)
```

A previous bug had `g` and `mu_z` swapped in `shock_names` while
`step_fn` had them correct — IRF analysis ran fine but mislabeled
which shock was which. **When adding a shock, update both at once.**

### 3.7 Steady-state caching

`models/disaster/steady_state.py` caches `_solve_steady_state` results
keyed on `frozenset(constants.items())`. A previous bug used a single
module-level cache populated at import time, which silently returned
stale results when the caller passed different constants.

**The cache key must include every constant the SS depends on.** This
matters for `risky_steady_state` (depends on `p_disaster`, `theta_disaster`)
and any future calibration overrides.

## 4. Pytrees and side-effect discipline

JAX requires **pure** functions for `jit`/`grad`/`vmap`. The codebase
maintains this by:

- All state lives in `TrainState` (a NamedTuple — JAX treats it as a
  pytree). `train_step` takes `state` in, returns new `state`. No
  module-level mutation.
- Equinox modules separate trainable arrays from static config via
  `eqx.filter(model, eqx.is_array)`. The optimizer sees only arrays.
- No `print`, no `float()` conversion, no Python branches on traced
  values inside `train_step`.

Common gotchas:

- `jax.tree.map` treats Python tuples as pytree containers. If a mapped
  function returns a tuple, the tuple gets unpacked into the tree
  structure. Use lists or NamedTuples if you want a tuple as a leaf.
- `jax.lax.cond` requires an `operand` argument (use `None` as a dummy
  if your branches don't need an input).
- Python-level `ndim` checks inside `tree_map` callbacks are fine —
  they resolve at trace time, not run time.

## 5. Where to add new things

| You want to add a...    | See                                                       |
|-------------------------|------------------------------------------------------------|
| New economic model      | [Adding a model](../site/models/adding.md)                |
| New network             | [Adding a network](../site/networks/adding.md)            |
| New optimizer           | [Adding an optimizer](../site/optimizers/adding.md)       |
| New loss term           | `composite_loss.py`, prefix the key with `aux_`           |
| New CLI subcommand      | `cli.py:main()`, add an argparse subparser                |
| New config field        | `config.py:TrainConfig` + add a `_check_*_type` validator |
| New checkpoint format   | Don't. Use `eqx.tree_serialise_leaves` / `_deserialise_leaves`. |

## 6. Things that look weird but are intentional

- **`OptimizerKind.MAO` factory takes `n_tasks` lazily** — see
  `_MAOFactory` in `optimizers/mao.py`. The model's equation count
  isn't known at config-parse time; it's resolved in `create_train_state`
  when `model.equation_names` is available.
- **Cosine LR schedule baked into Optax via `lr_scale`** — when a
  schedule is active, the optimizer is created with `lr=1.0` and the
  actual LR is passed as a dynamic scalar to `train_step` each episode
  (avoids re-JIT on every schedule step).
- **Soft floor at `0.01` in equations.py aggregators** — the Calvo
  inner term can go negative under aggressive policy moves, which would
  produce NaNs. The floor leaks gradient near zero, but training is
  stable as long as the policy stays in the valid region (which the
  composite anchor + barrier enforces).
- **`eqx.combine(updated_arrays, model)`** is used everywhere instead
  of mutating the model. Equinox modules are immutable; you reconstruct
  them with new arrays.

---

## When this guide gets stale

This file lives at `docs/dev/reading_guide.md`. If you find
yourself reading the source and the guide says one thing while the code
says another, **trust the code and update the guide**. Add a date and
your initials at the top of the changed section if useful.

The guide is meant to *prevent surprises*, not to be a complete
reference. The complete reference is the source plus mkdocstrings.
