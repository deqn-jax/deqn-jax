# DEQN-JAX Reference

Every public entry point of DEQN-JAX by type signature, for tools and agents
built on the library; the counterpart of `docs/REFERENCE.md` in
[BIS-DEQN-LAB](https://github.com/BIS-DEQN-LAB). To write a model by hand,
read [Implementing a model](models/implementing.md). This page is for agent
stacks that generate models from LaTeX, drive training, verify and report.

Stability: `deqn_jax.api` is the stable surface. Symbols imported from
anywhere else (`deqn_jax.training.trainer`, `deqn_jax.networks.mlp`, etc.)
are internal and may be refactored without notice.

---

## Table of contents

- [Quick start](#quick-start)
- [The public API surface (`deqn_jax.api`)](#the-public-api-surface-deqn_jaxapi)
- [The user contract: `ModelSpec`](#the-user-contract-modelspec)
- [Adding a model](#adding-a-model)
- [Configuration schema](#configuration-schema)
- [Runtime types: `TrainState` and `Metrics`](#runtime-types-trainstate-and-metrics)
- [Training entry points](#training-entry-points)
- [Networks](#networks)
- [Optimizers](#optimizers)
- [Loss](#loss)
- [Shock expectations](#shock-expectations)
- [Evaluation & verification gates](#evaluation-verification-gates)
- [Impulse responses (IRF / GIRF)](#impulse-responses-irf-girf)
- [Checkpointing & resume](#checkpointing-resume)
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
params, history = train_from_config(cfg)

diag = euler_equation_errors(params, load_model("brock_mirman"))
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

All of these are re-exported from `deqn_jax.api`; import them from there.

| Group | Symbols |
| --- | --- |
| **Discovery** | `list_models()`, `list_optimizers()`, `list_networks()`, `load_model(name)` |
| **Registration** | `register_model(spec, *, description=None, overwrite=False)`, `ModelSpec` |
| **Configuration** | `TrainConfig`, `NetworkConfig`, `OptimizerConfig`, `CompositeLossConfig`, `ReplayBufferConfig`, `MomentMatchingConfig`, `load_config` |
| **Core types** | `ModelSpec`, `TrainState`, `ReweightState`, `Metrics`, `make_reweight_state` |
| **Training** | `train_from_config(cfg) -> (params, history)`, `train(...)`, `create_train_state(...)`, `make_train_step(...)` |
| **Evaluation** | `euler_equation_errors`, `print_euler_errors`, `stability_check`, `simulated_moments`, `print_moments`, `market_clearing_errors` |
| **IRF** | `run_irf`, `run_girf`, `load_policy_from_checkpoint`, `save_irf_csv`, `print_irf_summary` |
| **Steady state** | `solve_steady_state`, `verify_steady_state`, `euler_from_period_return` |
| **Networks (advanced)** | `MLP`, `LSTMPolicy`, `TransformerPolicy`, `LinearPlusMLP`, `create_mlp`, `create_lstm`, `create_transformer`, `create_linear_plus_mlp` |

Imports from `deqn_jax.training.*` or `deqn_jax.optimizers.*` are outside the
stable surface: ask for a re-export in an issue, or expect refactors to move
them.

---

## The user contract: `ModelSpec`

A `ModelSpec` (in `deqn_jax.types`, re-exported from `deqn_jax.api`) is a
`NamedTuple` holding everything the framework needs to train a model. It is
the only interface between a model and the framework.

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

The model author must pick a residual form that is safe under Monte Carlo:
use the raw form `r = u'(c) − β u'(c')(1+r'−δ)`, not a dimensionless ratio
(see [implementing.md](models/implementing.md) §2).

#### `step_fn(state, policy, shock, constants) -> next_state`

State transition. Must be smooth (it runs inside the residual, under JIT).

- `state`: `[batch, n_states]`
- `policy`: `[batch, n_policies]`
- `shock`: `[batch, n_shocks]` *or* `[batch, 0]` for deterministic models. Handle
  `shock.ndim` defensively (`shock[:, 0] if shock.ndim > 1 else shock`).
- Return: `[batch, n_states]`. Column order must match `state_names`.

Do not clip states inside `step_fn`; clipping breaks differentiability. Clip
in `clip_state_fn`, which only `evaluate` and `irf` use.

#### `definitions_fn(state, policy, constants) -> dict[str, Array]`

Optional. Returns derived quantities (consumption, output, MPK, …). Each value
must be a scalar or have shape `[batch]`, never `[batch, 1]`. It is used by:

- `equations_fn` (share computation with `t+1`),
- the trainer (histogram logging at every `log_every`),
- the composite-loss path,
- post-training diagnostics (`run_irf` records every definition along the path).

#### `steady_state_fn(constants) -> (ss_state, ss_policy)`

Optional. Returns 1-D arrays of length `n_states` and `n_policies`. Without a
closed form, use the numerical fallback `solve_steady_state` described next.

Used by:

- `network.type='linear_plus_mlp'` (residual parameterization needs SS),
- input-normalization (`(state - ss) / max(|ss|, 0.01)`),
- warm-start (L-BFGS pre-fit to the SS policy),
- IRF (starting state is SS).

#### `solve_steady_state(model, ...) -> (ss_state, ss_policy)`  *(numerical fallback)*

Without an analytical SS, build the rest of the model (`equations_fn`,
`step_fn`, etc.) and call this helper. It runs L-BFGS on the deterministic
residual norm `Σ_eq r(s, π, s, π, c)²` at zero shock.

```python
from deqn_jax.api import solve_steady_state, verify_steady_state, ModelSpec

partial = ModelSpec(name="…", n_states=…, equations_fn=…, step_fn=…,
                    constants={…}, steady_state_fn=None, …)
ss_state, ss_policy = solve_steady_state(partial, max_iter=1000, tol=1e-8)
residuals = verify_steady_state(partial, ss_state, ss_policy, tol=1e-6)
# residuals: dict[str, float] of per-equation residual values
```

Signature:

```python
solve_steady_state(
    model: ModelSpec,
    init_state: Array | None = None,    # default: jnp.ones(n_states)
    init_policy: Array | None = None,   # default: 0.5 * jnp.ones(n_policies)
    max_iter: int = 1000,
    tol: float = 1e-8,                  # ||residual||² < tol → done
    verbose: bool = True,
    force_numerical: bool = False,      # True = ignore an existing analytical SS
) -> Tuple[Array, Array]
```

Behavior notes for code generators:

- With `model.steady_state_fn` set and `force_numerical=False`, the helper
  returns the analytical solution. Generators usually pass
  `force_numerical=False` and let it choose.
- The solve depends on the initial guess. For models far from the
  unit-vector default, pass `init_state` / `init_policy` from a rough
  linearization or a hand-tuned guess.
- Convergence is not guaranteed: check with `verify_steady_state` and reject
  a model whose SS residuals exceed `tol`.

#### `verify_steady_state(model, ss_state, ss_policy, tol=1e-6) -> dict[str, float]`

Returns the per-equation residual at a candidate steady state, analytical or
numerical. Path-A code generation should reject a model whose
`max(|residuals.values()|) > tol`.

### Optional `ModelSpec` hooks (full signatures)

Signatures of the eight optional fields listed above. Each defaults to
`None`. All are called outside JIT unless noted.

#### `init_state_fn(key, batch_size, constants) -> Array`

Initial-state sampler, called at the start of each rollout (or every cycle
if `initialize_each_episode=True`). Returns `[batch_size, n_states]`.
Default: ergodic-like sampling around the steady state.

#### `clip_state_fn(state) -> state`

Used by `evaluate` and `irf` only; in training it would break
differentiability. Keeps simulated states in valid regions (e.g. capital ≥ ε).
Shape is preserved.

#### `state_barrier_fn(state) -> Array`

Legacy soft barrier. Returns a `[batch]` penalty, multiplied by
`TrainConfig.barrier_weight` and added to the loss. Prefer the declarative
`state_bounds` mechanism below.

#### `cycle_hook(state, model, episode) -> None`

Called every `log_every` episodes after scalar and histogram logging, for
side effects only (plots, TensorBoard, etc.). Close over the output directory
and logger when building it. `state` is the current `TrainState`; `model` is
the `ModelSpec` after `setup_fn`.

#### `setup_fn(model, config) -> ModelSpec`

Called once before training; returns the `ModelSpec` the trainer should
use, rewritten from the resolved `TrainConfig` if needed (plain Python
branching works). `disaster` uses it to switch `steady_state_fn` to its
risky-SS variant when `constants["p_disaster"] > 0` and
`config.use_risky_steady_state` allows it.

#### `scalar_diagnostics_fn(model, policy_fn, states, policy_out, defs) -> dict[str, float]`

Called every `log_every` cycles. The returned scalars are logged to
TensorBoard / W&B under the model's namespace prefix: per-equation
decompositions, ratio diagnostics, soft-floor saturation fractions and the
like. If the hook raises, the trainer warns and continues.

- `model`: the `ModelSpec` after `setup_fn`
- `policy_fn`: the trained Equinox module (or sequence-net wrapper)
- `states`: `[batch, n_states]` from the current training minibatch
- `policy_out`: `[batch, n_policies]` policy at `states`
- `defs`: `dict[str, Array]` definitions at `(states, policy_out)`

#### `composite_aux_fn(model, defs, data, weights) -> (dict[str, Array], Array)`

Active only when `loss_type="composite"`. Lets a model add model-specific
losses keyed `aux_*`. Called inside the `make_composite_loss` closure, after
the barrier losses.

- `model`: the `ModelSpec` after `setup_fn`
- `defs`: batch-level `definitions_fn` output
- `data`: `CompositeData` (linearization + steady state precomputed at
  setup time; see [training/composite_loss.md](training/composite_loss.md))
- `weights`: subset of `CompositeLossConfig` weights relevant to this
  model

Returns `(aux_entries, total_contribution)`:

- `aux_entries`: merged into `eq_losses`, so reweighting and logging see
  each unweighted scalar under its `aux_*` key.
- `total_contribution`: scalar added directly to the loss total; the hook
  applies its own weighting. `disaster` uses it for `aux_newton_cond` and
  `aux_newton_resid`.

#### `state_bounds` and `definition_bounds` (declarative soft bounds)

Both are `dict[str, dict[str, float]]` of the form

```python
{"name": {"lower": float, "upper": float,
          "penalty_lower": float, "penalty_upper": float}}
```

When set, the loss gains a soft-penalty term

```text
penalty_lower * mean(max(0, lower - value) ** 2)
```

and the analogous term for `upper`, for each bounded variable. A missing
penalty coefficient defaults to `1 / bound**2` (the upstream DEQN-MAO
convention).

- `state_bounds` keys must match `state_names`.
- `definition_bounds` keys must match keys returned by `definitions_fn`.
- Hard policy bounds are separate: `policy_lower` / `policy_upper`, enforced
  at the network output activation.

### Shape and dtype invariants (what the framework guarantees)

- All arrays passed to your functions are `jnp.ndarray` of `float32` (or
  `float64` if `TrainConfig.fp64=True`).
- The batch dimension is always axis 0.
- `policy_lower` / `policy_upper`, when set, are 1-D arrays of length
  `n_policies`. Use `jnp.inf` for an unbounded side; per dimension the
  framework uses a sigmoid (finite upper) or softplus (`+inf` upper).
- `definitions_fn` is called both inside JIT (loss, training) and outside it
  (diagnostics), so it must be JAX-compatible throughout.

---

## Adding a model

### Path A: in-tree (model ships with deqn-jax)

1. Create `src/deqn_jax/models/<name>/` with the five-file layout
   ([detailed walkthrough](models/implementing.md)):

    ```text
    models/<name>/
      __init__.py        # MODEL: ModelSpec
      variables.py       # SPEC, CONSTANTS, POLICY_LOWER/UPPER, N_SHOCKS, DESCRIPTION
      equations.py       # equations(), definitions(), EQUATION_NAMES
      dynamics.py        # step()
      steady_state.py    # steady_state(), init_state
    ```

2. Add an import and an entry to `_MODELS` in
   [`src/deqn_jax/models/__init__.py`](api/models.md). The `deqn-jax list`
   description comes from `variables.py::DESCRIPTION`; `_DESCRIPTIONS` is
   derived from it, so nothing needs adding there.
3. `load_model("<name>")` and `deqn-jax train <name>` now work.

### Path B: programmatic (codegen / plugin)

For generated models, notebook prototypes, or external plugin packages:

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
params, history = train_from_config(cfg)   # params is the trained Equinox policy net
```

`register_model` semantics:

- Registering a name twice raises `ValueError` by default. Pass
  `overwrite=True` to replace a model on purpose.
- Both paths write to the same dict, and `list_models()` shows them alike.
- In tests, clean up between cases with `unregister_model(name)` (from
  `deqn_jax.models`).

An agent stack usually registers generated models through Path B at import
time. Models shipped in the tree (brock_mirman, disaster, …) use Path A and
are versioned with deqn-jax.

### Validation gates a new model should pass

Before a long run, check these in order (they follow implementing.md §8):

1. Steady-state Euler residual ≈ 0. Build `(state=ss, policy=ss, shock=0)`,
   call `equations_fn`, and assert `max(|residual|) < 1e-6`. A failure means
   the equations are algebraically inconsistent with the steady state.
2. Smoke training: 500 episodes with hidden=(16,), batch=16, mc_samples=2.
   The loss should fall roughly monotonically. If it diverges or stays at its
   initial value, the residual form is the likely cause (implementing.md §2).
3. Ergodic Euler errors: after a full run, `euler_equation_errors(...)`
   should report `mean log10(|resid/u'(c)|) < -3`. Above `-2` means the policy
   is undertrained or the model has a bug.
4. Comparison with a reference: closed form, linearization, or a published
   solution.

---

## Configuration schema

`TrainConfig` is a Pydantic v2 model and validates on construction. Unknown
keys (typos) raise `ValueError` with did-you-mean suggestions. Sub-configs use
`default_factory`, so a sub-block can be omitted.

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
| `type` | str | `"mlp"` | One of `mlp`, `lstm`, `transformer`, `linear_plus_mlp`, `disaster_policy_net`, `rss_market_clearing_net` |
| `hidden_sizes` | tuple[int, ...] | (64, 64) | |
| `activation` | str | `"tanh"` | `tanh`, `relu`, `gelu`, `silu`, `softplus` |
| `activations` | tuple[str, ...] \| None | None | Per-layer override |
| `init` | str | `"default"` | `default`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal` |
| `history_len` | int | 1 | 1 = MLP; >1 = LSTM/Transformer |
| `num_heads` | int | 4 | Transformer attention heads |
| `n_layers` | int | 2 | Transformer block count |
| `init_scale` | float | 0.0 | `linear_plus_mlp` only — MLP delta init scale (0 = start at linear) |
| `use_zlb_feature` | bool | False | `linear_plus_mlp` + disaster only |
| `kf_names` | tuple[str, ...] | `("F_p","K_p","F_w","K_w")` | `disaster_policy_net` only |

### `OptimizerConfig`

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `name` | str | `"adam"` | One of: `adam`, `muon`, `ngd`, `shampoo`, `lbfgs`, `mao`, `gn`, `ign`, `lm` |
| `learning_rate` | float | 1e-3 | Peak LR |
| `grad_clip` | float \| None | None | Global gradient-norm clipping |
| `beta1`, `beta2`, `epsilon` | float | adam defaults | First/second-moment decay + numerical floor |
| `damping` | float | 1e-4 | Preconditioner damping for NGD/GN/IGN/LM |
| `decay` | float | 0.999 | NGD / Shampoo preconditioner EMA |
| `block_size`, `precond_update_freq` | int | 64, 10 | Shampoo |
| `memory_size` | int | 10 | L-BFGS history |
| `ns_steps` | int | 5 | Muon Newton-Schulz iter count |
| `cg_iters`, `cg_tol` | int, float | 20, 1e-6 | Implicit GN conjugate gradient |
| `lr_schedule` | str | `"constant"` | `constant`, `cosine` |
| `lr_warmup` | int | 0 | Linear warmup episodes |
| `lr_min_factor` | float | 0.0 | Cosine / plateau floor as fraction of peak |

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

### `ReplayBufferConfig`

Prioritized state-replay buffer, off by default. When enabled, each cycle
writes its trajectory states to a fixed-shape ring buffer, with priority equal
to the sum of squared equilibrium residuals at write time. A `mix_ratio`
fraction of each gradient minibatch is then drawn from the buffer by
priority.

Sequence networks (`network.history_len > 1`) are not supported in v1;
enabling both raises `NotImplementedError`.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | bool | False | Master switch; False = byte-identical to no-replay |
| `capacity` | int | 65536 | Ring-buffer size. Memory = `capacity * n_states * 4B` |
| `mix_ratio` | float | 0.5 | Fraction of each minibatch drawn from the buffer (0 = none, 1 = all-buffer) |
| `min_fill_frac` | float | 0.25 | Fraction of capacity required before sampling activates |
| `priority_alpha` | float | 0.6 | PER's α: `prob ∝ (priority + eps) ** α`; 0 = uniform, 1 = fully proportional |
| `priority_eps` | float | 1e-6 | Floor added to priorities before exponentiation |
| `eviction` | str | `"fifo"` | Eviction policy. v1 only supports `"fifo"` |

### `MomentMatchingConfig`

Auxiliary loss on the deviation of ergodic moments from a Dynare reference,
added to any base loss (residual MSE, composite, etc.). It estimates moments
from the policy outputs on each minibatch. The gradient flows through
`policy(s)` only; the states come from a separate rollout and are under
`stop_gradient`.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | bool | False | Master switch; False = identical to base loss |
| `weight` | float | 0.1 | Multiplier on the aux loss term added to the total |
| `mean_weight` | float | 1.0 | Within the aux, weight on the squared mean-deviation term |
| `std_weight` | float | 1.0 | Within the aux, weight on the squared std-deviation term |
| `dynare_dir` | str | `"dynare/results"` | Directory containing `dynare_moments.csv` |
| `scale_eps` | float | 1e-3 | Floor on per-variable scale used for relative comparison |

### YAML loading

Every config round-trips through YAML:

```python
from deqn_jax.api import TrainConfig, load_config

cfg = TrainConfig.from_yaml("configs/disaster.yaml")
cfg = load_config("configs/disaster.yaml", overrides={"optimizer.learning_rate": 1e-4})
cfg.to_yaml("/tmp/copy.yaml")
```

CLI `--set` overrides use dot notation: `--set optimizer.learning_rate=0.01`.

---

## Runtime types: `TrainState` and `Metrics`

JAX-pytree-compatible `NamedTuple`s in `deqn_jax.types`, re-exported from
`deqn_jax.api`. The trainer builds both; you read their fields when driving
the low-level `make_train_step` loop.

### `TrainState`

| Field | Type | Notes |
| --- | --- | --- |
| `params` | Equinox module | The trainable policy network |
| `opt_state` | Optax state | Optimizer momentum / preconditioner / etc. |
| `episode_state` | `[batch, n_states]` | Current rollout starting points |
| `key` | PRNG key | Use `jax.random.PRNGKey(int)`, NOT `jax.random.key(int)` (typed keys break Equinox serialization) |
| `step` | int | Total gradient steps taken |
| `episode` | int | Current episode (cycle) counter |
| `loss_weights` | `[n_eq]` | Active per-equation weights (mutated by adaptive reweighting) |
| `reweight_state` | `ReweightState` | EMA / running stats for `lr_annealing`, `relobralo` |
| `target_params` | Equinox module \| None | Frozen policy copy when `target_update_every > 0` |
| `aux_params` | Any \| None | Slot for a second trainable module (actor-critic value net, learned operator, …). Default training loop ignores it. |
| `aux_opt_state` | Any \| None | Optimizer state for `aux_params` if trained with its own optimizer |
| `history_state` | `[batch, H, n_states]` \| None | Sliding history window for sequence policies (`history_len > 1`); `None` for MLP |
| `replay_state` | `ReplayState` \| None | Prioritized state buffer; `None` when off |

### `Metrics`

Returned by every `train_step` call. At runtime the fields are JAX arrays,
not Python scalars; cast explicitly when needed, e.g. `float(metrics.loss)`.

| Field | Type | Notes |
| --- | --- | --- |
| `loss` | scalar Array | Total loss for the step |
| `residuals` | `dict[str, Array]` \| None | Per-equation residual breakdown (when emitted by the loss path) |
| `grad_norm` | scalar Array \| None | Pre-clip global gradient norm |

### `history` dict (returned by `train_from_config`)

The `history` dict has exactly two keys. Each holds a `list[float]` with one
entry per cycle run (`episodes`, or fewer after early stopping):

| Key | What it holds |
| --- | --- |
| `"loss"` | Per-cycle total loss (the same scalar `Metrics.loss` casts to) |
| `"grad_norm"` | Per-cycle pre-clip gradient norm |

Per-equation losses, learning rates, residual histograms and replay metrics
go to TensorBoard / W&B when configured, not to `history`. Older versions of
this page listed other keys; they do not exist.

---

## Training entry points

### `train_from_config(config) -> (params, history)`

The high-level entry point; it honors every `TrainConfig` field.

```python
from deqn_jax.api import TrainConfig, train_from_config

cfg = TrainConfig(model="brock_mirman", episodes=1000, ...)
params, history = train_from_config(cfg)

# params: trained Equinox policy net (the same object you'd pass as
#         policy_fn to evaluate / IRF / checkpoint loading).
# history: dict with EXACTLY the keys {"loss", "grad_norm"}, each a
#          list[float] of length == episodes. Per-equation losses,
#          per-cycle LRs, gradient histograms, etc. are written to
#          TensorBoard / W&B (when configured) — they are *not* in
#          this dict. To read them post-hoc, parse the TB log dir.
```

`cfg` controls checkpointing, TensorBoard / W&B logging, early stopping,
optimizer switching, warm start and the replay buffer. The final
`TrainState` (opt_state, episode_state, PRNG key, replay buffer, …) is not
returned; for it, use the lower-level path below or load a checkpoint with
`load_policy_from_checkpoint`.

### `train(model_name, episodes, ...)` (legacy wrapper)

Thin backward-compatible wrapper over `train_from_config`. New code should
call `train_from_config(TrainConfig(...))`.

### `create_train_state(...)` and `make_train_step(...)` (low-level)

Use these only to drive the training loop yourself (custom outer loops,
distributed training, hand-coded learning-rate schedules, …).

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

Each `train_step` call runs one cycle across two JIT boundaries: a compiled
rollout, then a minibatch sweep of compiled gradient steps.

---

## Networks

| `network.type` | Architecture | Use case | Module |
| --- | --- | --- | --- |
| `mlp` | Plain MLP with sigmoid/softplus output bounds | Most models | `networks.mlp.MLP` |
| `lstm` | LSTM over a history window | History-dependent policies | `networks.lstm.LSTMPolicy` |
| `transformer` | Multi-head attention over a history window | Same | `networks.transformer.TransformerPolicy` |
| `linear_plus_mlp` | `policy = linear(state) + mlp(state)`; init at the BK linearization | Models with a known good local solution | `networks.linear_plus_mlp.LinearPlusMLP` |

Output bounds are enforced per policy dimension at the network output:

- Finite `policy_upper[i]`: sigmoid scaled to `[lower, upper]`.
- `policy_upper[i] = jnp.inf`: `softplus(x) + lower`.

### Adding a network

See [Adding a network](networks/adding.md). At minimum: write an Equinox
`eqx.Module` with `__call__(state) -> policy`, register a factory in
`networks/__init__.py`, add the type name to `NetworkConfig.VALID_TYPES`,
and dispatch on it in `networks/factory.py:build_policy_net`.

---

## Optimizers

Nine are built in; `list_optimizers()` lists them. The train step is chosen
from five families at construction time, before JIT:

| Family | Names | Step shape |
| --- | --- | --- |
| **STANDARD** | adam, muon, ngd, shampoo | `jax.grad → opt.update(grads, state, params)` |
| **PCGRAD** | (gradient_surgery) | Per-equation gradients with conflict projection |
| **MAO** | mao | Per-equation Jacobian via `jax.jacrev` → MAO update |
| **LBFGS** | lbfgs | Optax LBFGS with line search |
| **GN** | gn, ign, lm | Gauss-Newton / Levenberg-Marquardt: `Δθ = −(JᵀJ)⁻¹ Jᵀr` |

Composite loss is rejected with MAO, GN, IGN and LM: their updates
differentiate only the base residuals and miss the auxiliary terms. It works
with the STANDARD family (with or without PCGrad) and with L-BFGS.
`training/state_init.py:_validate_train_config` enforces this.

### Adding an optimizer

See [Adding an optimizer](optimizers/adding.md). At minimum: write the
optax-style transform, register it with `@register_optimizer(name, kind)`
in its module, and import that module in `optimizers/__init__.py` so the
registration runs. STANDARD-family optimizers reuse
`make_grad_step_standard`; other kinds need their own grad-step factory.

---

## Loss

### Base MSE (default)

`loss_type: "mse"`. For each batch element the framework:

1. Computes per-shock residuals with `equations_fn`.
2. Takes the shock expectation: a weighted mean over MC samples (uniform
   weights) or GH nodes (Hermite weights).
3. Squares the mean: `(E_shock[r])²` per equation per batch element.
4. Aggregates over the batch: mean, or Huber if `loss_choice="huber"`.
5. Aggregates over equations: mean (DEQN-MAO convention).

Losses keyed with the `aux_` prefix are excluded from adaptive reweighting.

### Composite loss

`loss_type: "composite"`. Adds anchor, Jacobian, barrier and Newton terms;
see [Composite loss](training/composite_loss.md).

### Custom loss

Pass `compute_loss_fn` to `make_train_step` (advanced; not exposed in
`TrainConfig`). Its signature must match `compute_loss`:
`(model, policy_fn, states, key, mc_samples, weights, shock_scale,
quad_nodes, quad_weights, target_policy_fn, loss_choice, huber_delta) -> (Array, dict)`.

### Path-A autodiff helper: `euler_from_period_return`

Builds `equations_fn` from a scalar period return Π with `jax.grad`, for
"Path A" (planner / autodiff) code generation. From Π the helper derives the
capital Euler residual (envelope theorem) and, optionally, intratemporal FOCs
(∂Π/∂policy[j] = 0).

```python
from deqn_jax.api import euler_from_period_return

def Pi(K, K_next, z, policy, constants):
    """Per-period return. K and K_next are scalars; z is exog vector;
    policy is the full policy vector your network outputs."""
    alpha = constants["alpha"]
    c = z[0] * K**alpha - K_next        # budget closes consumption
    return jnp.log(c)

equations_fn = euler_from_period_return(
    period_return_fn=Pi,
    step_fn=my_step,            # used at zero shock to reconstruct K_{t+2}
    capital_idx=0,              # which state column is the intertemporal capital
    exog_idx=(1,),              # which columns are exogenous (AR(1), shocks, …)
    n_shocks=1,
    equation_name="euler",      # key under which the Euler residual is returned
    intratemporal_policy_idx=(),    # add FOC equations for these policy indices
    intratemporal_equation_names=(),
)
```

It returns an `equations_fn(state, policy, next_state, next_policy, constants)`
with the standard `ModelSpec.equations_fn` signature. Three in-tree models
build their `equations_fn` this way: `brock_mirman_autodiff`,
`bm_labor_autodiff` and `irbc`.

Supported: one intertemporal state dimension, any number of exogenous state
dimensions, any number of intratemporal FOCs. Not yet supported: multi-agent
OLG-style Euler equations, KKT systems with Lagrange multipliers,
Fischer-Burmeister.

The helper and its signature are part of the stable surface.

---

## Shock expectations

`expectation_type` selects one of two methods:

| Mode | `expectation_type` | Used as |
| --- | --- | --- |
| **Antithetic Monte Carlo** (default) | `"mc"` | `mc_samples` antithetic Gaussian draws per batch element |
| **Gauss-Hermite quadrature** | `"quadrature"`, `"gh"`, `"gauss_hermite"` | Tensor-product GH grid, `n_quadrature_points^n_shocks` total nodes |

MC cost does not grow with the number of shocks; quadrature cost grows
exponentially. Use quadrature when residuals are strongly nonlinear in the
shocks and `n_shocks ≤ 3`.

`shock_scale` multiplies all shocks (used for curriculum ramping);
`shock_mask` zeroes chosen shock dimensions (used for ablations). Both act
the same way under MC and quadrature, in the loss and in the rollout.

---

## Evaluation & verification gates

The standard checks for a trained DEQN policy.

### `euler_equation_errors(policy_net, model, n_periods=10_000, seed=123, burn_in=None) -> dict`

Simulates a long stochastic path under the trained policy and computes Euler
residuals at every period; `print_euler_errors` reports their
`log10(|residual|)` distribution per equation. This is the standard measure of
global accuracy (Azinovic et al. 2022). Optional `expectation_type` and
`n_quadrature_points` arguments set the expectation rule (default:
Gauss-Hermite).

```python
from deqn_jax.api import euler_equation_errors, print_euler_errors, load_model

diag = euler_equation_errors(params, load_model("brock_mirman"))
print_euler_errors(diag)
```

`diag` keys: `"residuals"` (raw, `[n_periods - burn_in, n_eq]`),
`"equation_names"`, `"states"`. `deqn-jax evaluate` prints the results and
applies no accuracy threshold.

### `stability_check(policy_net, model, ...) -> dict[str, bool]`

A cheap structural check returning `"nan_free"`, `"bound_hit_pct"` (% of
policy outputs within 1e-4 of a bound), `"max_ss_deviation_pct"` and
`"stable"` (NaN-free, bound hits under 20%, state deviation under 500%). Run
it before the more expensive Euler test.

### `simulated_moments(...)` / `print_moments(...)`

Long-run mean, standard deviation and autocorrelation of states and
definitions along the ergodic path. Compare them with moments implied by a
linearization, or with Dynare reference moments through
`compare_to_dynare_moments` (in `deqn_jax.evaluate.dynare`).

### Suggested verification gates (for an outer loop)

| Gate | Threshold | Disposition |
| --- | --- | --- |
| `stability_check(...)["stable"]` True | hard | fail → restart with smaller LR |
| `mean log10\|resid/u'(c)\|` per equation | `< -3` | pass; `[-3, -2]` warn; `> -2` fail |
| `90th percentile log10\|resid/u'(c)\|` | `< -2` | pass |
| `simulated_moments.std` vs reference | within 20% | pass; off by >2× → fail |

These are conventions; the framework does not enforce them. Encode them in
your own verifier from the dicts the calls above return.

---

## Impulse responses (IRF / GIRF)

```python
from deqn_jax.api import (
    load_policy_from_checkpoint, run_irf, run_girf,
    save_irf_csv, print_irf_summary, load_model,
)

policy_net, _ = load_policy_from_checkpoint("runs/disaster/checkpoint_best.eqx")
model = load_model("disaster")

# Plain IRF (path - SS):
irf = run_irf(policy_net, model, shock_name="eps_z", shock_size=1.0, horizon=40)

# Generalized IRF (shocked - no-shock counterfactual):
girf = run_girf(policy_net, model, shock_name="eps_z", shock_size=1.0, horizon=40)

print_irf_summary(girf, "eps_z")
save_irf_csv(girf, "/tmp/eps_z.csv")
```

Both return `dict[str, list[float]]` with the keys `"period"`, every state,
every policy, every definition and every equation residual. Prefer
`run_girf` from a risky steady state: the no-shock baseline drifts on its own
under the disaster mixture, and a plain IRF mixes that drift into the
response.

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
params, history = train_from_config(cfg)
```

Resume rebuilds the pytree template from the sibling `config.yaml`, then
`eqx.tree_deserialise_leaves` restores params, optimizer state and episode
counter. A mid-training optimizer switch (`switch_optimizer` /
`switch_episode` / `switch_lr`) discards the old optimizer state.

`max_checkpoints` keeps only the N most recent periodic snapshots. The best
snapshot is never deleted.

---

## CLI reference

```bash
deqn-jax train MODEL [-n EPISODES] [--config YAML] [--set KEY=VALUE ...] [-q]
deqn-jax list                       # all registered models
deqn-jax optimizers                 # all registered optimizers
deqn-jax evaluate CKPT [opts]       # see evaluate.run_evaluate_cli
deqn-jax irf CKPT --shock NAME [--horizon N] [--girf]
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

Exit codes: 0 on success, non-zero on an error (invalid config, failed
training). `deqn-jax evaluate` exits 0 whatever the accuracy, so an outer loop
should gate on the evaluation dicts from the Python API.

---

## Discovery helpers

```python
from deqn_jax.api import list_models, list_optimizers, list_networks

list_models()       # [(name, description), ...]
list_optimizers()   # [name, ...] sorted
list_networks()     # [name, ...] sorted (NetworkConfig.VALID_TYPES)
```

`list_models()` includes both in-tree and runtime-registered models.

---

## Repository layout

```text
src/deqn_jax/
  api.py                    # ★ stable agent-facing surface (this doc's contract)
  __init__.py               # legacy re-exports (subset of api.py)
  cli/                      # entry point; one module per subcommand
  config/                   # TrainConfig, OptimizerConfig, NetworkConfig (Pydantic v2)
  types.py                  # ModelSpec, TrainState, ReweightState, Metrics
  evaluate/                 # euler_equation_errors, stability_check, moments
  benchmark.py              # train-step performance benchmarks

  models/
    __init__.py             # _MODELS dict + load_model + register_model
    variable_spec.py        # VariableSpec helper for named state/policy access
    <name>/                 # one subpackage per model

  networks/
    common.py               # _normalize_input, _apply_bounds, INIT_FNS
    mlp.py                  # MLP, create_mlp
    lstm.py                 # LSTMPolicy, create_lstm
    transformer.py          # TransformerPolicy, create_transformer
    linear_plus_mlp.py      # LinearPlusMLP, create_linear_plus_mlp

  optimizers/
    registry.py             # OptimizerKind enum, register_optimizer, create_optimizer
    standard.py             # make_grad_step_standard (adam/muon/...)
    pcgrad.py               # make_grad_step_pcgrad
    mao.py                  # MAO + factory + make_grad_step_mao
    lbfgs.py                # make_grad_step_lbfgs (optax wrapper)
    gauss_newton.py         # GN, IGN, LM
    ngd.py                  # Diagonal Fisher NGD
    shampoo.py              # Shampoo

  training/
    trainer.py              # train, train_from_config, _run_training_loop (slim orchestrator)
    state_init.py           # create_train_state, make_train_step (re-exported from trainer)
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

- `deqn_jax.api` is the stable surface. Everything imported from it is part
  of the public contract and changes incompatibly only at a major version
  bump (currently from 0.x to 1.0).
- All other paths (`deqn_jax.training.trainer.create_train_state`,
  `deqn_jax.networks.mlp.MLP`, etc.) are internal and may move, change
  parameters or disappear between minor versions.
- Adding a `ModelSpec` field is non-breaking when the field is optional with
  a sensible default. A new required field is breaking.
- Adding a `TrainConfig` field is non-breaking when its default keeps the
  current behavior. A new validator that rejects previously valid configs
  is breaking.

To get an internal symbol onto the stable surface, open an issue asking for
it to be re-exported from `deqn_jax.api`.

---

## Limitations and out of scope

- No symbolic differentiation. Residuals are hand-coded or built with
  `jax.grad` from a scalar period payoff (see [autodiff.md](autodiff.md)).
  There is no SymPy-driven KKT code generation like BIS-DEQN-LAB's Path B;
  that belongs in an agent stack built on top.
- No LaTeX-to-ModelSpec parsing. Parsers or agents that turn a paper into a
  `ModelSpec` belong in the user's stack. This library only defines the
  `ModelSpec` shape they must emit.
- No actor-critic and no value-function head. That work is on the
  [`experimental/actor-critic`](https://github.com/deqn-jax/deqn-jax/tree/experimental/actor-critic)
  branch and may later land as a separate module built on the stable APIs.
- No distributed training; JAX runs on a single device. Multi-device
  support through `pmap` is possible but not wired.
- No GPU/CPU portability layer; it runs on whatever device JAX picks.
  Set `JAX_PLATFORM_NAME=cpu` for reproducibility on small models.

---

## Further reading

- [Implementing a model](models/implementing.md): walkthrough for writing a
  model by hand.
- [Reading guide](reading_guide.md): code walkthrough for contributors.
- [Architecture](architecture.md): design decisions and the JIT boundaries.
- [Composite loss](training/composite_loss.md): the math of the anchor,
  Jacobian, barrier and Newton terms.
- [Adding a network](networks/adding.md): how to add an architecture.
- [Adding an optimizer](optimizers/adding.md): how to add an optimizer
  family.
- Per-module API reference: [config](api/config.md), [types](api/types.md),
  [models](api/models.md), [trainer](api/trainer.md), [loss](api/loss.md),
  [networks](api/networks.md), [optimizers](api/optimizers.md).
