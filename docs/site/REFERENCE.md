# DEQN-JAX Reference

> The complete contract for building on top of DEQN-JAX, intended for
> agentic workflows and external tooling. Type-signature-first; every
> public entry point is documented in one place.

This document is the deqn-jax-side equivalent of `docs/REFERENCE.md` in
[BIS-DEQN-LAB](https://github.com/BIS-DEQN-LAB). If you're hand-writing a
model, prefer the prose-first walkthrough in [Implementing a model](models/implementing.md).
If you're building an agent stack on top of deqn-jax (codegen models from
LaTeX, drive training, verify, report), this doc is the contract.

**Stability:** everything in `deqn_jax.api` is the stable surface. Symbols
imported from anywhere else (`deqn_jax.training.trainer`, `deqn_jax.networks.mlp`, etc.)
are *internal* and may be refactored without notice. Use `deqn_jax.api`.

---

## Table of contents

- [Quick start](#quick-start)
- [The public API surface (`deqn_jax.api`)](#the-public-api-surface-deqn_jaxapi)
- [The user contract: `ModelSpec`](#the-user-contract-modelspec)
- [Adding a model](#adding-a-model)
- [Configuration schema](#configuration-schema)
- [Training entry points](#training-entry-points)
- [Networks](#networks)
- [Optimizers](#optimizers)
- [Loss](#loss)
- [Shock expectations](#shock-expectations)
- [Evaluation & verification gates](#evaluation--verification-gates)
- [Impulse responses (IRF / GIRF)](#impulse-responses-irf--girf)
- [Checkpointing & resume](#checkpointing--resume)
- [CLI reference](#cli-reference)
- [Discovery helpers](#discovery-helpers)
- [Repository layout](#repository-layout)
- [Versioning policy](#versioning-policy)
- [Limitations and out of scope](#limitations-and-out-of-scope)

---

## Quick start

### Python (programmatic, agent-friendly)

```python
from deqn_jax.api import (
    TrainConfig, NetworkConfig, OptimizerConfig,
    train_from_config, euler_equation_errors, print_euler_errors,
    load_model,
)

cfg = TrainConfig(
    model="brock_mirman",
    episodes=2000,
    batch_size=128,
    episode_length=1,
    initialize_each_episode=True,
    network=NetworkConfig(hidden_sizes=(50, 50), activation="relu"),
    optimizer=OptimizerConfig(name="adam", learning_rate=3e-4,
                              lr_schedule="cosine", lr_min_factor=0.1),
    verbose=False,
)
state, history = train_from_config(cfg)

diag = euler_equation_errors(state.params, load_model("brock_mirman"))
print_euler_errors(diag)   # log10|residual| distribution; mean < -3 = converged
```

### CLI

```bash
uv run deqn-jax list                                    # available models
uv run deqn-jax optimizers                              # available optimizers
uv run deqn-jax train brock_mirman -n 1000 -q           # smoke train
uv run deqn-jax train --config configs/disaster.yaml -n 50000
uv run deqn-jax evaluate runs/disaster/checkpoint_best.eqx
uv run deqn-jax irf runs/disaster/checkpoint_best.eqx --shock eps_z --horizon 40
```

---

## The public API surface (`deqn_jax.api`)

Everything below is re-exported from `deqn_jax.api`. Import from there.

| Group | Symbols |
| --- | --- |
| **Discovery** | `list_models()`, `list_optimizers()`, `list_networks()`, `load_model(name)` |
| **Registration** | `register_model(spec, *, description=None, overwrite=False)`, `ModelSpec` |
| **Configuration** | `TrainConfig`, `NetworkConfig`, `OptimizerConfig`, `CompositeLossConfig`, `ReplayBufferConfig`, `MomentMatchingConfig`, `load_config` |
| **Core types** | `ModelSpec`, `TrainState`, `ReweightState`, `Metrics`, `make_reweight_state` |
| **Training** | `train_from_config(cfg) -> (params, history)`, `train(...)`, `create_train_state(...)`, `make_train_step(...)` |
| **Evaluation** | `euler_equation_errors`, `print_euler_errors`, `stability_check`, `simulated_moments`, `print_moments`, `market_clearing_errors` |
| **IRF** | `run_irf`, `run_girf`, `load_policy_from_checkpoint`, `save_irf_csv`, `print_irf_summary` |
| **Networks (advanced)** | `MLP`, `ResMLP`, `LSTMPolicy`, `TransformerPolicy`, `create_mlp`, `create_lstm`, `create_transformer` |

If you find yourself importing from `deqn_jax.training.*` or `deqn_jax.optimizers.*`
directly, you've stepped past the stable surface. File an issue requesting that
the symbol be re-exported, or accept that future refactors may move it.

---

## The user contract: `ModelSpec`

A `ModelSpec` (in `deqn_jax.types`, re-exported from `deqn_jax.api`) is a
`NamedTuple` carrying everything the framework needs to train a model. It is
the *only* contract between a model and the framework.

```python
ModelSpec(
    # --- Required ---
    name: str,
    n_states: int,
    n_policies: int,
    n_shocks: int,
    constants: dict[str, float],
    equations_fn: Callable,                 # equilibrium residuals
    step_fn: Callable,                      # state transition

    # --- Strongly recommended (default = empty tuple) ---
    state_names: tuple[str, ...] = (),
    policy_names: tuple[str, ...] = (),
    equation_names: tuple[str, ...] = (),
    shock_names: tuple[str, ...] | None = None,

    # --- Optional but commonly set ---
    steady_state_fn: Callable | None = None,        # warm-start, IRF anchor
    init_state_fn: Callable | None = None,          # initial-state sampler
    definitions_fn: Callable | None = None,         # derived quantities
    policy_lower: jax.Array | None = None,          # per-policy lower bound
    policy_upper: jax.Array | None = None,          # per-policy upper bound

    # --- Optional advanced hooks ---
    clip_state_fn: Callable | None = None,          # eval/IRF only — never training
    state_barrier_fn: Callable | None = None,       # legacy soft barrier
    state_bounds: dict | None = None,               # declarative soft bounds
    definition_bounds: dict | None = None,          # ditto for definitions()
    cycle_hook: Callable | None = None,             # called every log_every
    setup_fn: Callable | None = None,               # pre-training model rewrite
    scalar_diagnostics_fn: Callable | None = None,  # custom logged diagnostics
    composite_aux_fn: Callable | None = None,       # custom composite-loss terms
)
```

### Function signatures (the four required-or-recommended)

#### `equations_fn(state, policy, next_state, next_policy, constants) -> dict[str, Array]`

Returns one residual per equilibrium equation, each of shape `[batch]`. The
framework computes `(E_shock[r])²` per batch element, then mean-aggregates
across batch and equations.

- `state`: `[batch, n_states]`
- `policy`: `[batch, n_policies]`
- `next_state`: `[batch, n_states]`
- `next_policy`: `[batch, n_policies]`
- `constants`: `dict[str, float]` (the same dict you put in `ModelSpec.constants`)

**MC-safe residual form** is the agent's responsibility. Default to raw form
`r = u'(c) − β u'(c')(1+r'−δ)` rather than dimensionless ratios; see the
trap discussion in [implementing.md](models/implementing.md) §2.

#### `step_fn(state, policy, shock, constants) -> next_state`

State transition. Must be smooth (used inside the residual + JIT).

- `state`: `[batch, n_states]`
- `policy`: `[batch, n_policies]`
- `shock`: `[batch, n_shocks]` *or* `[batch, 0]` for deterministic models. Handle
  `shock.ndim` defensively (`shock[:, 0] if shock.ndim > 1 else shock`).
- Return: `[batch, n_states]`. Column order must match `state_names`.

**Do not clip states inside `step_fn`** — that breaks differentiability. Clip
in `clip_state_fn` (used only by `evaluate` / `irf`).

#### `definitions_fn(state, policy, constants) -> dict[str, Array]`

Optional. Returns derived quantities (consumption, output, MPK, …). Each value
must be **scalar** or **`[batch]`-shaped** — never `[batch, 1]`. Available to:

- `equations_fn` (share computation with `t+1`),
- the trainer (histogram logging at every `log_every`),
- the composite-loss path,
- post-training diagnostics (`run_irf` records every definition along the path).

#### `steady_state_fn(constants) -> (ss_state, ss_policy)`

Optional. Returns 1-D arrays of length `n_states` and `n_policies` respectively.
If you don't have a closed form, use
`deqn_jax.training.steady_state.solve_steady_state` (numerical L-BFGS).

Used by:

- `network.type='linear_plus_mlp'` (residual parameterization needs SS),
- `network.type='kf_anchored_mlp'` (anchors K/F outputs to BK linearization),
- input-normalization (`(state - ss) / max(|ss|, 0.01)`),
- warm-start (L-BFGS pre-fit to the SS policy),
- IRF (starting state is SS).

### Shape and dtype invariants (what the framework guarantees)

- All arrays passed to your functions are `jnp.ndarray` of `float32` (or
  `float64` if `TrainConfig.fp64=True`).
- Batch dim is always axis 0.
- `policy_lower` / `policy_upper`, when set, are 1-D arrays of length
  `n_policies`. Use `jnp.inf` for unbounded sides; the framework picks
  sigmoid (finite upper) or softplus (`+inf` upper) per dimension.
- `definitions_fn` is called both inside JIT (during loss/training) and
  outside (during diagnostics). It must therefore be JAX-compatible end to end.

---

## Adding a model

### Path A — In-tree (model ships with deqn-jax)

1. Create `src/deqn_jax/models/<name>/` with the five-file layout
   ([detailed walkthrough](models/implementing.md)):

    ```text
    models/<name>/
      __init__.py        # MODEL: ModelSpec
      variables.py       # SPEC, CONSTANTS, POLICY_LOWER/UPPER, N_SHOCKS
      equations.py       # equations(), definitions(), EQUATION_NAMES
      dynamics.py        # step()
      steady_state.py    # steady_state(), init_state
    ```

2. Add an import + entries to `_MODELS` and `_DESCRIPTIONS` in
   [`src/deqn_jax/models/__init__.py`](api/models.md).
3. Done — `load_model("<name>")` and `deqn-jax train <name>` both work.

### Path B — Programmatic (codegen / plugin)

For agent-codegen'd models, notebook prototyping, or external plugin packages:

```python
from deqn_jax.api import ModelSpec, register_model

MY_MODEL = ModelSpec(
    name="my_model",
    n_states=2,
    n_policies=1,
    n_shocks=1,
    constants={"alpha": 0.36, "beta": 0.99, ...},
    equations_fn=my_equations,
    step_fn=my_step,
    state_names=("k", "z"),
    policy_names=("sav_rate",),
    equation_names=("euler",),
    shock_names=("eps_z",),
    steady_state_fn=my_steady_state,
    init_state_fn=my_init_state,
    definitions_fn=my_definitions,
    policy_lower=jnp.array([1e-6]),
    policy_upper=jnp.array([1 - 1e-6]),
)

register_model(MY_MODEL, description="My agent-built model")

# Now usable through the same load path:
from deqn_jax.api import load_model, train_from_config, TrainConfig
cfg = TrainConfig(model="my_model", episodes=1000)
state, history = train_from_config(cfg)
```

`register_model` semantics:

- Idempotent calls **fail by default**: re-registering an existing name
  raises `ValueError`. Pass `overwrite=True` to replace deliberately.
- Both paths land in the same dict; `list_models()` sees them identically.
- Use `unregister_model(name)` in tests to clean up between cases.

The two paths are orthogonal: a deployed agent stack typically uses Path B
to register codegen'd models at import time, while in-tree shipped models
(brock_mirman, disaster, …) live in Path A so they stay version-controlled
under deqn-jax.

### Validation gates a new model should pass

Before training seriously, verify in this order (corresponds to
implementing.md §8):

1. **Steady-state Euler residual ≈ 0**. Build `(state=ss, policy=ss, shock=0)`,
   call `equations_fn`, assert `max(|residual|) < 1e-6`. If this fails, your
   equations are algebraically inconsistent with your steady state.
2. **Smoke training**: 500 episodes with hidden=(16,), batch=16, mc_samples=2.
   Loss must decrease roughly monotonically. If it diverges or plateaus at the
   initial value, you almost certainly have the residual-form trap.
3. **Ergodic Euler errors**: after a serious run, `euler_equation_errors(...)`
   reports `mean log10(|resid/u'(c)|) < -3`. Above `-2` means undertrained or
   real model bug.
4. **Sanity vs reference**: closed form, linearization, or published solution.

---

## Configuration schema

`TrainConfig` is a Pydantic v2 model. Validation runs at construction; passing
unknown keys (typos) raises `ValueError` with did-you-mean suggestions. Sub-configs
are constructed via `default_factory` — omitting a sub-block is safe.

### `TrainConfig` (top-level)

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `model` | str | `"brock_mirman"` | Registered model name |
| `episodes` | int | 1000 | Outer cycles (rollout + minibatch sweep) |
| `batch_size` | int | 64 | Minibatch size for each gradient step |
| `episode_length` | int | 100 | Trajectory length T per rollout |
| `mc_samples` | int | 5 | MC shock samples per state |
| `seed` | int | 42 | Top-level PRNG seed |
| `network` | NetworkConfig | default | Policy network (see below) |
| `optimizer` | OptimizerConfig | default | Optimizer + LR schedule |
| `loss_type` | str | `"mse"` | `"mse"` or `"composite"` |
| `composite_loss` | CompositeLossConfig | default | Active when `loss_type="composite"` |
| `replay_buffer` | ReplayBufferConfig | default | Active when `enabled=True` |
| `moment_matching` | MomentMatchingConfig | default | Aux loss vs Dynare moments |
| `loss_choice` | str | `"mse"` | `"mse"` or `"huber"` (post-shock-expectation aggregation) |
| `huber_delta` | float | 1.0 | Huber cutoff (ignored for `mse`) |
| `loss_reweight` | str | `"none"` | `"none"`, `"lr_annealing"`, `"relobralo"` |
| `loss_weights` | List[float] \| None | None | Manual per-equation weights |
| `gradient_surgery` | str | `"none"` | `"none"` or `"pcgrad"` |
| `expectation_type` | str | `"mc"` | `"mc"` or `"quadrature"` |
| `n_quadrature_points` | int | 3 | Per-shock-dim node count for GH |
| `initialize_each_episode` | bool | False | True = rect sampling, False = ergodic |
| `ss_reset_frac` | float | 0.0 | Fraction of batch reseeded to SS each rollout |
| `n_epochs_per_rollout` | int | 1 | Sweep epochs per cycle |
| `n_minibatches_per_epoch` | int \| None | None | None = full-trajectory sweep |
| `sim_batch` | int \| None | None | Trajectory count (None = batch_size) |
| `curriculum_episodes` | int | 0 | Linear shock_scale ramp from `curriculum_start` to 1.0 |
| `curriculum_start` | float | 0.1 | Initial shock_scale during curriculum |
| `shock_mask` | List[float] \| None | None | Per-dim mask (length = n_shocks) |
| `warm_start` | bool | False | L-BFGS pre-fit to SS policy |
| `warm_start_linearize` | bool | False | Use BK P-matrix at SS |
| `target_update_every` | int | 0 | Target-network interval (0 = off) |
| `target_tau` | float | 1.0 | Polyak coefficient |
| `tensorboard_dir` | str \| None | None | TB log dir |
| `wandb_project` | str \| None | None | W&B project name |
| `checkpoint_dir` | str \| None | None | Checkpoint dir |
| `checkpoint_every` | int \| None | None | Periodic save interval |
| `max_checkpoints` | int \| None | None | Retention cap |
| `save_best_checkpoint` | bool | True | Persist `checkpoint_best.eqx` on improvements |
| `early_stop_patience` | int \| None | None | Episodes without improvement |
| `early_stop_min_delta` | float | 1e-6 | Counted-as-improvement threshold |
| `resume` | str \| None | None | Path to `.eqx` checkpoint (sibling `config.yaml` is read) |
| `switch_optimizer` | str \| None | None | Mid-training optimizer switch |
| `switch_episode` | int \| None | None | When to switch |
| `switch_lr` | float \| None | None | LR for switched optimizer |
| `constants` | dict[str, float] | {} | Per-run override of `model.constants` |
| `use_risky_steady_state` | bool | True | For disaster: risky vs deterministic SS |
| `verbose` | bool | True | Console output |
| `fp64` | bool | False | JAX x64 mode |
| `log_every` | int | 100 | Logging / `cycle_hook` interval |
| `barrier_weight` | float | 0.0 | Legacy state-barrier penalty (prefer `state_bounds`) |

### `NetworkConfig`

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `type` | str | `"mlp"` | One of `mlp`, `lstm`, `transformer`, `linear_plus_mlp`, `kf_anchored_mlp` |
| `hidden_sizes` | tuple[int, ...] | (64, 64) | |
| `activation` | str | `"tanh"` | `tanh`, `relu`, `gelu`, `silu`, `softplus` |
| `activations` | tuple[str, ...] \| None | None | Per-layer override |
| `init` | str | `"default"` | `default`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal` |
| `multi_head` | bool | False | Per-policy output heads (experimental) |
| `skip_connections` | bool | False | Residual MLP |
| `history_len` | int | 1 | 1 = MLP; >1 = LSTM/Transformer |
| `num_heads` | int | 4 | Transformer attention heads |
| `n_layers` | int | 2 | Transformer block count |
| `init_scale` | float | 0.0 | `linear_plus_mlp` only — MLP delta init scale (0 = start at linear) |
| `use_zlb_feature` | bool | False | `linear_plus_mlp` + disaster only |
| `kf_names` | tuple[str, ...] | `("F_p","K_p","F_w","K_w")` | `kf_anchored_mlp` only |

### `OptimizerConfig`

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `name` | str | `"adam"` | One of: `adam`, `sgd`, `adamw`, `lion`, `muon`, `ngd`, `shampoo`, `lbfgs`, `mao`, `mao_kfac`, `gn`, `ign`, `lm` |
| `learning_rate` | float | 1e-3 | Peak LR |
| `grad_clip` | float \| None | None | Global gradient-norm clipping |
| `weight_decay` | float | 0.0 | adamw / adam / sgd |
| `beta1`, `beta2`, `epsilon` | float | adam defaults | First/second-moment decay + numerical floor |
| `damping` | float | 1e-4 | Preconditioner damping for NGD/GN/IGN/LM |
| `decay` | float | 0.999 | NGD / Shampoo preconditioner EMA |
| `block_size`, `precond_update_freq` | int | 64, 10 | Shampoo |
| `memory_size` | int | 10 | L-BFGS history |
| `ns_steps` | int | 5 | Muon Newton-Schulz iter count |
| `cg_iters`, `cg_tol` | int, float | 20, 1e-6 | Implicit GN conjugate gradient |
| `lr_schedule` | str | `"constant"` | `constant`, `cosine`, `reduce_on_plateau` |
| `lr_warmup` | int | 0 | Linear warmup episodes |
| `lr_min_factor` | float | 0.0 | Cosine / plateau floor as fraction of peak |
| `lr_reduce_factor`, `lr_reduce_patience`, `lr_reduce_cooldown`, `lr_reduce_min_delta` | various | various | `reduce_on_plateau` parameters |

### `CompositeLossConfig`

Active only when `TrainConfig.loss_type == "composite"`. See
[Composite loss](training/composite_loss.md) for the math.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `anchor_weight` | float | 0.1 | Weight on `‖π_net(x) − π_lin(x)‖²` at fixed anchor points near SS |
| `jac_weight` | float | 0.01 | Weight on `‖J_net(SS) − P‖²_F` |
| `jac_anchor_weight` | float | 0.0 | Weight on per-anchor Jacobian match (expensive) |
| `barrier_weight` | float | 0.01 | Net-worth / leverage / consumption barriers |
| `newton_weight` | float | 0.01 | Newton-step diagnostics (disaster-specific) |
| `n_anchor_points` | int | 64 | Sampled near SS at setup time |
| `anchor_sigma` | float | 1.0 | Gaussian spread for anchor sampling |
| `leverage_mult` | float | 5.0 | Leverage barrier fires at `L > leverage_mult * L_ss` |
| `aux_decay_floor` | float | 0.2 | Min retained anchor+jac weight after curriculum (1.0 = no decay) |

### `ReplayBufferConfig` and `MomentMatchingConfig`

See [config_reference.md](config_reference.md) §replay_buffer and §moment_matching.

### YAML loading

Every config is YAML-roundtrippable:

```python
from deqn_jax.api import TrainConfig, load_config

cfg = TrainConfig.from_yaml("configs/disaster.yaml")
cfg = load_config("configs/disaster.yaml", overrides={"optimizer.learning_rate": 1e-4})
cfg.to_yaml("/tmp/copy.yaml")
```

`--set` overrides on the CLI use dot notation: `--set optimizer.learning_rate=0.01`.

---

## Training entry points

### `train_from_config(config) -> (TrainState, dict)`

The high-level entry point. Everything in `TrainConfig` is honored.

```python
from deqn_jax.api import TrainConfig, train_from_config

cfg = TrainConfig(model="brock_mirman", episodes=1000, ...)
state, history = train_from_config(cfg)

# state: TrainState (final). Use state.params for evaluation/IRF/checkpointing.
# history: dict with keys "loss", "grad_norm", "step_time", optionally per-equation.
```

`history["loss"]` is per-cycle (length = `episodes`); `history["grad_norm"]` ditto.
Checkpointing, TensorBoard / W&B logging, early stopping, optimizer switching,
warm start, replay buffer — all driven by `cfg`.

### `train(model_name, episodes, ...)` (legacy wrapper)

Backward-compatible thin wrapper over `train_from_config`. Prefer
`train_from_config(TrainConfig(...))` for new code.

### `create_train_state(...)` and `make_train_step(...)` (low-level)

Use these only when you need to drive the training loop yourself
(custom outer loops, distributed training, hand-coded learning rate
schedules, …).

```python
from deqn_jax.api import (
    create_train_state, make_train_step, load_model,
    NetworkConfig, OptimizerConfig,
)
import jax, jax.numpy as jnp

model = load_model("brock_mirman")
state, opt, kind = create_train_state(
    model, jax.random.PRNGKey(0),
    hidden_sizes=(64, 64), batch_size=64, n_equations=1,
    optimizer_config=OptimizerConfig(name="adam", learning_rate=1e-3),
    network_config=NetworkConfig(hidden_sizes=(64, 64)),
)
train_step = make_train_step(
    model, opt, episode_length=100, mc_samples=5, batch_size=64,
    kind=kind, history_len=1, n_epochs_per_rollout=1,
    n_minibatches_per_epoch=1,
)
for ep in range(1000):
    state, metrics = train_step(state, jnp.array(1.0), jnp.array(1.0))
    # metrics: Metrics(loss, residuals, grad_norm)
```

`train_step` is a single `@jax.jit`-compiled function — the full rollout +
minibatch sweep + gradient updates fuse into one JIT region per cycle.

---

## Networks

| `network.type` | Architecture | Use case | Module |
| --- | --- | --- | --- |
| `mlp` | Plain MLP with sigmoid/softplus output bounds | Most models | `networks.mlp.MLP` |
| `lstm` | LSTM over a history window | History-dependent policies | `networks.lstm.LSTMPolicy` |
| `transformer` | Multi-head attention over a history window | Same | `networks.transformer.TransformerPolicy` |
| `linear_plus_mlp` | `policy = linear(state) + mlp(state)`; init at the BK linearization | Models with a known good local solution | `networks.linear_plus_mlp.LinearPlusMLP` |
| `kf_anchored_mlp` | K/F gauge elimination via BK linearization anchor | CMR-class disaster models | `networks.kf_anchored_mlp` |

Output bounds (per policy dimension) are enforced **at the network output**:

- Finite `policy_upper[i]` → sigmoid scaled to `[lower, upper]`.
- `policy_upper[i] = jnp.inf` → `softplus(x) + lower`.

### Adding a network

See [Adding a network](networks/adding.md). Minimum: write an Equinox
`eqx.Module` with `__call__(state) -> policy`, register a factory in
`networks/__init__.py`, add the type name to
`NetworkConfig.VALID_TYPES`, and dispatch in `trainer.create_train_state`.

---

## Optimizers

13 built-in. List them with `list_optimizers()`. Five families dispatched
at construction time (before JIT):

| Family | Names | Step shape |
| --- | --- | --- |
| **STANDARD** | adam, sgd, adamw, lion, muon, ngd, shampoo | `jax.grad → opt.update(grads, state, params)` |
| **PCGRAD** | (gradient_surgery) | Per-equation gradients with conflict projection |
| **MAO** | mao, mao_kfac | Per-equation Jacobian via `jax.jacrev` → MAO update |
| **LBFGS** | lbfgs | Optax LBFGS with line search |
| **GN** | gn, ign, lm | Gauss-Newton / Levenberg-Marquardt: `Δθ = −(JᵀJ)⁻¹ Jᵀr` |

Composite loss is currently rejected with MAO/GN/IGN/LM/LBFGS and PCGrad
(the optimizer's update path doesn't see the auxiliary terms).
`TrainConfig._validate_ranges` enforces this.

### Adding an optimizer

See [Adding an optimizer](optimizers/adding.md). Minimum: write the
optax-style transform, register with `@register_optimizer(name, kind)`
in your module, and import it in `optimizers/__init__.py` to trigger
registration. STANDARD-family optimizers compose with the existing
`make_grad_step_standard`; other kinds need their own grad-step factory.

---

## Loss

### Base MSE (default)

`loss_type: "mse"`. The framework computes, per batch element:

1. **Per-shock residuals** via `equations_fn`.
2. **Shock-expectation**: weighted mean across MC samples (uniform) or
   GH nodes (Hermite weights).
3. **Square the mean**: `(E_shock[r])²` per equation per batch element.
4. **Aggregate across batch**: mean (or Huber, if `loss_choice="huber"`).
5. **Aggregate across equations**: mean (DEQN-MAO convention).

Aux losses with keys prefixed `aux_*` are excluded from adaptive reweighting.

### Composite loss

`loss_type: "composite"`. Adds anchor + Jacobian + barriers + Newton terms;
see [Composite loss](training/composite_loss.md).

### Custom loss

Pass `compute_loss_fn` to `make_train_step` (advanced; not exposed in
`TrainConfig`). Signature must match `compute_loss`:
`(model, policy_fn, states, key, mc_samples, weights, shock_scale,
quad_nodes, quad_weights, target_policy_fn, loss_choice, huber_delta) -> (Array, dict)`.

---

## Shock expectations

Two paths, set via `expectation_type`:

| Mode | `expectation_type` | Used as |
| --- | --- | --- |
| **Antithetic Monte Carlo** (default) | `"mc"` | `mc_samples` antithetic Gaussian draws per batch element |
| **Gauss-Hermite quadrature** | `"quadrature"`, `"gh"`, `"gauss_hermite"` | Tensor-product GH grid, `n_quadrature_points^n_shocks` total nodes |

MC has constant cost in shock dim; quadrature scales exponentially. Switch to
quadrature when residuals are highly nonlinear in shocks and `n_shocks ≤ 3`.

`shock_scale` multiplies all shocks (curriculum ramping); `shock_mask` zeroes
specific shock dimensions (ablations). Both apply to MC and quadrature
identically and to both the loss path and the rollout path.

---

## Evaluation & verification gates

The standard verification panel for a trained DEQN policy.

### `euler_equation_errors(policy_net, model, n_periods=10_000, seed=123, burn_in=None) -> dict`

Simulates a long stochastic path under the trained policy, computes Euler
residuals at every period, returns the `log10(|residual|)` distribution per
equation. Gold standard for global accuracy (Azinovic et al. 2022).

```python
from deqn_jax.api import euler_equation_errors, print_euler_errors, load_model

diag = euler_equation_errors(state.params, load_model("brock_mirman"))
print_euler_errors(diag)
```

`diag` keys: `"residuals"` (raw `[n_periods, n_eq]`), `"log10_abs"` (per-eq
distribution stats), `"states"`, `"policies"`. CLI exits with code 2 if
configurable thresholds aren't met (see `evaluate.run_evaluate_cli`).

### `stability_check(policy_net, model, ...) -> dict[str, bool]`

Cheap structural sanity panel. Booleans typically include
`"trajectory_finite"`, `"policies_in_bounds"`, `"states_in_reasonable_range"`.
Use this as a fast early-exit gate before the more expensive Euler test.

### `simulated_moments(...)` / `print_moments(...)`

Long-run mean/std/autocorrelation of states and definitions along the
ergodic path. Compare against linearization-implied moments or Dynare
reference moments via `compare_to_dynare_moments`.

### Suggested verification gates (for an outer loop)

| Gate | Threshold | Disposition |
| --- | --- | --- |
| `stability_check` all True | hard | fail → restart with smaller LR |
| `mean log10\|resid/u'(c)\|` per equation | `< -3` | pass; `[-3, -2]` warn; `> -2` fail |
| `90th percentile log10\|resid/u'(c)\|` | `< -2` | pass |
| `simulated_moments.std` vs reference | within 20% | pass; off by >2× → fail |

These are conventions, not framework-enforced. Encode them in your agent's
verifier; the data is in the dicts returned by the calls above.

---

## Impulse responses (IRF / GIRF)

```python
from deqn_jax.api import (
    load_policy_from_checkpoint, run_irf, run_girf,
    save_irf_csv, print_irf_summary, load_model,
)

policy_net = load_policy_from_checkpoint("runs/disaster/checkpoint_best.eqx")
model = load_model("disaster")

# Plain IRF (path - SS):
irf = run_irf(policy_net, model, shock_name="eps_z", shock_size=1.0, horizon=40)

# Generalized IRF (shocked - no-shock counterfactual):
girf = run_girf(policy_net, model, shock_name="eps_z", shock_size=1.0, horizon=40)

print_irf_summary(girf, "eps_z")
save_irf_csv(girf, "/tmp/eps_z.csv")
```

Both return `dict[str, list[float]]` with keys: `"period"`, every state, every
policy, every definition, every equation residual. `run_girf` is the safer
default under risky-steady-state setups (the no-shock baseline drifts on its
own under the disaster mixture, so plain IRF conflates that drift with the
shock response).

---

## Checkpointing & resume

When `TrainConfig.checkpoint_dir` is set:

```text
<checkpoint_dir>/
  config.yaml                  # written once, used by resume to rebuild template
  checkpoint_NNNNNN.eqx        # periodic (every checkpoint_every episodes)
  checkpoint_best.eqx          # best-loss snapshot (when save_best_checkpoint=true)
  checkpoint_best.meta         # episode + loss text record
```

Resume:

```python
cfg = TrainConfig.from_yaml(orig_config_yaml)
cfg = cfg.model_copy(update={"resume": "runs/X/checkpoint_001000.eqx", "episodes": 5000})
state, history = train_from_config(cfg)
```

The resume path reads the sibling `config.yaml` to rebuild the matching pytree
template, then `eqx.tree_deserialise_leaves` restores params, opt state, and
episode counter. Mid-training optimizer switches via `switch_optimizer` /
`switch_episode` / `switch_lr` are also supported and discard the old
optimizer state.

`max_checkpoints` retains only the N most recent periodic snapshots; the best
snapshot is never deleted.

---

## CLI reference

```bash
deqn-jax train MODEL [-n EPISODES] [--config YAML] [--set KEY=VALUE ...] [-q]
deqn-jax list                       # all registered models
deqn-jax optimizers                 # all registered optimizers
deqn-jax evaluate CKPT [opts]       # see evaluate.run_evaluate_cli
deqn-jax irf CKPT --shock NAME [--horizon N] [--mode irf|girf]
```

Common flags:

| Flag | Effect |
| --- | --- |
| `--config <yaml>` | Load TrainConfig from YAML |
| `--set <key=val>` | Dot-notation override (`--set optimizer.learning_rate=0.01`) |
| `-n N` | Override `episodes` |
| `-q` | Quiet (sets `verbose=false`) |
| `--checkpoint-dir <path>` | Sets `checkpoint_dir` |
| `--resume <ckpt>` | Resume from a `.eqx` checkpoint |

Exit codes: 0 = success, non-zero = config / training / verification failure.
Use `deqn-jax evaluate` exit code as the integration gate for an autonomous
outer loop.

---

## Discovery helpers

```python
from deqn_jax.api import list_models, list_optimizers, list_networks

list_models()       # [(name, description), ...]
list_optimizers()   # [name, ...] sorted
list_networks()     # [name, ...] sorted (NetworkConfig.VALID_TYPES)
```

Both in-tree and runtime-registered models appear in `list_models()`.

---

## Repository layout

```text
src/deqn_jax/
  api.py                    # ★ stable agent-facing surface (this doc's contract)
  __init__.py               # legacy re-exports (subset of api.py)
  cli.py                    # entry point: train, list, optimizers, evaluate, irf
  config.py                 # TrainConfig, OptimizerConfig, NetworkConfig (Pydantic v2)
  types.py                  # ModelSpec, TrainState, ReweightState, Metrics
  evaluate.py               # euler_equation_errors, stability_check, moments
  irf.py                    # run_irf, run_girf, load_policy_from_checkpoint
  metrics.py                # TensorBoard / W&B logger backends
  benchmark.py              # train-step performance benchmarks

  models/
    __init__.py             # _MODELS dict + load_model + register_model
    variable_spec.py        # VariableSpec helper for named state/policy access
    <name>/                 # one subpackage per model

  networks/
    common.py               # _normalize_input, _apply_bounds, INIT_FNS
    mlp.py                  # MLP, ResMLP, MultiHeadMLP, create_mlp
    lstm.py                 # LSTMPolicy, create_lstm
    transformer.py          # TransformerPolicy, create_transformer
    linear_plus_mlp.py      # LinearPlusMLP, create_linear_plus_mlp
    kf_anchored_mlp.py      # KFAnchoredMLP

  optimizers/
    registry.py             # OptimizerKind enum, register_optimizer, create_optimizer
    standard.py             # make_grad_step_standard (adam/sgd/adamw/lion/muon/...)
    pcgrad.py               # make_grad_step_pcgrad
    mao.py                  # MAO + factory + make_grad_step_mao
    mao_kfac.py             # MAO with KFAC preconditioner
    lbfgs.py                # make_grad_step_lbfgs (optax wrapper)
    gauss_newton.py         # GN, IGN, LM
    ngd.py                  # Diagonal Fisher NGD
    shampoo.py              # Shampoo

  training/
    trainer.py              # train_from_config, create_train_state, make_train_step
    cycle.py                # rollout_fn + cycle_step (the inner JIT region)
    episode.py              # lax.scan-based trajectory simulation
    loss.py                 # compute_loss, compute_residuals, sample_antithetic_shocks
    composite_loss.py       # anchor / jac / barrier / Newton aux losses
    moment_loss.py          # Dynare-moments aux loss
    history.py              # sliding history window for sequence policies
    linearize.py            # Blanchard-Kahn decomposition (P, Q matrices)
    warm_start.py           # L-BFGS pre-fit
    steady_state.py         # numerical SS solve (optax.lbfgs wrapper)
    reweighting.py          # adaptive reweight strategies
    checkpointing.py        # save / resume / prune
    replay.py               # prioritized state-replay buffer
    shocks.py               # shock-drawing primitives (antithetic, mask, scale)
    reporting.py            # CLI banners + residual tables

  plots/                    # post-training plotting primitives (IRF / policy / ergodic)

configs/                    # YAML training recipes
tests/                      # pytest; use conftest.py for shared fixtures
docs/site/                  # mkdocs source (this directory)
```

---

## Versioning policy

- `deqn_jax.api` is the **stable surface**. Anything imported from it is part
  of the public contract; we only break these on a major version bump
  (currently 0.x → 1.0).
- All other paths (`deqn_jax.training.trainer.create_train_state`,
  `deqn_jax.networks.mlp.MLP`, etc.) are **internal**. They may move
  between modules, gain or lose parameters, or be deleted between minor
  versions. Most won't change in practice — but no promises.
- `ModelSpec` field additions are **non-breaking** when they're optional
  with a sensible default. New required fields are breaking.
- `TrainConfig` field additions are non-breaking when they default to
  current behavior. New validators that reject previously-valid configs
  are breaking.

If you find an internal symbol you need on the stable surface, file an
issue requesting that it be re-exported from `deqn_jax.api`.

---

## Limitations and out of scope

- **No symbolic differentiation.** All residuals are hand-coded or
  built via `jax.grad` from a scalar period payoff (see
  [autodiff.md](autodiff.md)). There is no SymPy-driven KKT codegen
  analogous to BIS-DEQN-LAB's Path B; that lives in the agent stack you
  build on top.
- **No LaTeX → ModelSpec parsing in this library.** Parsers / agents
  that turn a paper into a `ModelSpec` are user-stack territory. The
  contract here is just the `ModelSpec` shape — what an agent emits.
- **No actor-critic, no value-function head.** That work currently lives
  on the [`experimental/actor-critic`](https://github.com/mechanicpanic/deqn-jax/tree/experimental/actor-critic)
  branch and may land later as an isolated module that wraps stable APIs.
- **No distributed training.** Single-device JAX. Multi-device support
  via `pmap` is straightforward in principle but not wired.
- **No GPU-vs-CPU portability layer.** Runs on whatever JAX picks up.
  Set `JAX_PLATFORM_NAME=cpu` for reproducibility on small models.

---

## Further reading

- [Implementing a model](models/implementing.md) — prose-first walkthrough,
  for humans hand-writing a model.
- [Reading guide](reading_guide.md) — code-level narrative for contributors.
- [Architecture](architecture.md) — design decisions and JIT boundary
  discussion.
- [Composite loss](training/composite_loss.md) — the math behind
  anchor + Jacobian + barrier + Newton terms.
- [Adding a network](networks/adding.md) — how to plug a new architecture.
- [Adding an optimizer](optimizers/adding.md) — how to plug a new
  optimizer family.
- Per-module API reference: [config](api/config.md), [types](api/types.md),
  [models](api/models.md), [trainer](api/trainer.md), [loss](api/loss.md),
  [networks](api/networks.md), [optimizers](api/optimizers.md).
