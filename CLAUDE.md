# CLAUDE.md

## What This Is

DEQN-JAX is a **Deep Equilibrium Network** trainer for economic models — a JAX/Equinox port of the Azinovic-Maliar-Maliar method. It trains a policy network to satisfy a model's equilibrium conditions across its ergodic distribution, replacing traditional point-by-point solvers (Dynare, time iteration, etc.).

```
State (K, Z) --> Policy Network --> Policy (C, L) --> Equilibrium Equations --> Loss = sum(residuals^2)
```

It's *PINN-adjacent* but not a PINN: collocation points come from on-policy simulation rather than a fixed spatial domain, the operator is an expectation over shocks rather than a differential operator, and equilibrium equations have multiple fixed points (Blanchard-Kahn-style selection is missing). The closer functional cousin is model-based RL with a known dynamics — that framing predicts the right toolkit (replay, off-policy correction, ensembles) for the failure modes we hit on real research models. (A separate project tracks a true PINN for a related class of problems.)

Stack: JAX + Equinox (networks) + Optax (optimizers). No TensorFlow, no PyTorch, no Keras.

## Commands

```bash
uv run pytest tests/ -v                              # 168 tests
uv run deqn-jax train brock_mirman -n 1000            # train a model
uv run deqn-jax train brock_mirman -n 50 -o ngd -q    # quick smoke test
uv run deqn-jax train --config configs/brock_mirman.yaml -n 100
uv run deqn-jax optimizers                             # list all optimizers
uv run deqn-jax list                                   # list all models
```

Always use `uv run`, never activate the venv manually.

## Project Layout

```
src/deqn_jax/
  config.py               # TrainConfig, OptimizerConfig, NetworkConfig, CompositeLossConfig
  cli.py                  # Entry point: train, list, optimizers subcommands
  types.py                # ModelSpec, TrainState, ReweightState, Metrics (all NamedTuples)
  metrics.py              # TensorBoard, W&B, NullLogger
  evaluate.py             # Policy evaluation and residual analysis
  irf.py                  # Impulse response functions from checkpoints
  benchmark.py            # Performance benchmarking

  models/
    __init__.py           # load_model(), list_models(), explicit model registry
    variable_spec.py      # VariableSpec class for named array access
    brock_mirman/         # Simple RBC (1 eq, 2 states, 1 policy)
      variables.py        # SPEC, CONSTANTS, N_SHOCKS, DESCRIPTION
      equations.py        # equations(), definitions(), EQUATION_NAMES
      dynamics.py         # step()
      steady_state.py     # steady_state(), init_state()
    disaster/             # NK-DSGE with financial frictions (11 eq, 13 states, 11 policies)
      variables.py        # SPEC, CONSTANTS, N_SHOCKS, DESCRIPTION, POLICY_LOWER/UPPER
      equations.py        # equations(), definitions(), EQUATION_NAMES
      dynamics.py         # step()
      steady_state.py     # steady_state(), init_state()

  networks/
    common.py             # Shared network utilities
    mlp.py                # Equinox MLP with per-layer activations, custom init, sigmoid output bounds
    lstm.py               # Equinox LSTM for history-dependent policies
    transformer.py        # Transformer with multi-head attention for sequence policies

  optimizers/
    registry.py           # OptimizerKind enum, @register_optimizer, create_optimizer()
    ngd.py                # Diagonal Fisher NGD (optax.GradientTransformation)
    mao.py                # Multi-Adaptive Optimizer (custom class, per-equation moments)
    shampoo.py            # Kronecker-factored Shampoo (optax.GradientTransformation)
    lbfgs.py              # Thin wrapper around optax.lbfgs
    gauss_newton.py       # Gauss-Newton and Levenberg-Marquardt (custom classes)

  training/
    trainer.py            # train(), train_from_config(), create_train_state(), make_train_step()
    loss.py               # MC loss with antithetic variates
    composite_loss.py     # Anchor + Jacobian + barrier + Newton auxiliary losses
    episode.py            # Trajectory simulation via lax.scan
    history.py            # History window construction for sequence models
    linearize.py          # Model linearization for composite loss precomputation
    warm_start.py         # L-BFGS fitting to steady state (uses optax.lbfgs)
    steady_state.py       # Analytical or numerical SS solving (uses optax.lbfgs)

configs/                  # Example YAML configs
tests/                    # test_basic, test_config, test_config_validation, test_optimizers,
                          # test_convergence, test_warm_start
```

## Architecture

### Single JIT Boundary

The entire train step is one `@jax.jit` function. This is the core performance decision -- do not break it into multiple JIT calls.

### Five Train Step Variants

Dispatched at construction time (before JIT), not inside JIT:

- **STANDARD** (adam, sgd, adamw, lion, muon, ngd, shampoo): `jax.grad` -> `opt.update(grads, state, params)`
- **PCGRAD**: per-equation gradients with conflict projection -> `opt.update(projected_grads, state, params)`
- **MAO**: `jax.jacrev(per_eq_loss_vector)` -> per-equation Jacobian -> `mao.update(eq_jac, state, params)`
- **LBFGS**: `optax.lbfgs` (GradientTransformationExtraArgs) -> needs `value`, `grad`, `value_fn` for line search
- **GN**: Gauss-Newton / Levenberg-Marquardt -> residual Jacobian `J`, update = `-(J^T J)^{-1} J^T r`

`OptimizerKind` enum selects which variant; `make_train_step()` dispatches.

### Optimizer Registry

`@register_optimizer(name, kind)` in each module. All modules imported in `optimizers/__init__.py` to trigger registration. `create_optimizer(config)` looks up the registry and chains grad clipping for STANDARD optimizers automatically.

MAO uses `_MAOFactory` for deferred `n_tasks` resolution (resolved in `create_train_state` when the model's equation count is known).

### Config Priority

```
--set overrides  >  CLI args  >  YAML file  >  dataclass defaults
```

`load_config()` in `config.py` handles merging. Dot-notation for nested fields: `--set optimizer.learning_rate=0.01`.

### Types

Everything is a `NamedTuple` for JAX pytree compatibility. `TrainState` bundles all mutable state (params, opt_state, episode_state, key, step, weights) so `train_step` is a pure function.

## Key Patterns

- **Equinox models**: `eqx.filter(model, eqx.is_array)` to get trainable params, `eqx.combine(updated_arrays, model)` to reconstruct
- **Antithetic variates**: shock sampling pairs each epsilon with -epsilon for variance reduction
- **Adaptive reweighting**: `lr_annealing` (inverse EMA) and `relobralo` (softmax of loss ratios) for multi-equation balancing
- **L-BFGS warm start**: fits network to steady state policy in ~10-50 steps using `optax.lbfgs` with flat-parameter loop
- **Composite loss**: optional `loss_type: composite` adds anchor, Jacobian, barrier, and Newton auxiliary terms (see `CompositeLossConfig`)

## Common JAX Gotchas

- `jax.tree.map` treats Python tuples as pytree containers -- don't return tuples from mapped functions if you want them as leaves
- `jax.lax.cond` branches need an operand argument (use `None` as dummy)
- Python-level `ndim` checks inside `tree_map` callbacks are fine (resolved at trace time, not run time)
- No `float()` calls inside JIT-traced functions
- Shampoo: create L and R preconditioners with separate `tree_map` calls, never a single call returning a tuple pair

## Models

| Model | States | Policies | Equations | Shocks | Steady State |
|-------|--------|----------|-----------|--------|--------------|
| `brock_mirman` | 2 (k, z) | 1 (sav_rate) | 1 (euler) | 1 | Analytical |
| `disaster` | 13 (8 endo + 5 exo) | 11 | 11 | 5 | Numerical |

## Testing

```bash
uv run pytest tests/ -v                         # all 168 tests
uv run pytest tests/test_basic.py -v            # 12 core tests
uv run pytest tests/test_config.py -v           # 18 config tests
uv run pytest tests/test_config_validation.py -v # 109 validation + coercion tests
uv run pytest tests/test_optimizers.py -v       # 23 optimizer + training tests
uv run pytest tests/test_convergence.py -v      # 5 convergence tests
uv run pytest tests/test_warm_start.py -v       # 1 warm start test
```

Short training smoke tests use: 3 episodes, hidden=(16,), batch=16, mc_samples=2.

## Git

- Author: Anna Smirnova <anna@example.com>
- Main branch: `main`, current work on `master`
