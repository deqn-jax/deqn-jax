# CLAUDE.md

## What This Is

DEQN-JAX is a **Physics-Informed Neural Network (PINN) for economics**. It trains neural networks to satisfy equilibrium conditions of economic models across the entire state space, replacing traditional point-by-point solvers.

```
State (K, Z) --> Policy Network --> Policy (C, L) --> Equilibrium Equations --> Loss = sum(residuals^2)
```

Stack: JAX + Equinox (networks) + Optax (optimizers). No TensorFlow, no PyTorch, no Keras.

## Commands

```bash
uv run pytest tests/ -v                              # 53 tests
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
  config.py               # TrainConfig, OptimizerConfig, NetworkConfig; YAML + CLI merging
  cli.py                  # Entry point: train, list, optimizers subcommands
  types.py                # ModelSpec, TrainState, ReweightState, Metrics (all NamedTuples)
  metrics.py              # TensorBoard, W&B, NullLogger

  models/
    brock_mirman.py       # Simple RBC (1 eq, 2 states, 1 policy) -- canonical test case
    disaster.py           # NK-DSGE with financial frictions (12 eq, 13 states, 12 policies)
    variables.py          # VariableSpec for named array access

  networks/
    mlp.py                # Equinox MLP with optional sigmoid output bounds
    lstm.py               # Equinox LSTM for history-dependent policies

  optimizers/
    registry.py           # OptimizerKind enum, @register_optimizer, create_optimizer()
    ngd.py                # Diagonal Fisher NGD (optax.GradientTransformation)
    mao.py                # Multi-Adaptive Optimizer (custom class, per-equation moments)
    shampoo.py            # Kronecker-factored Shampoo (optax.GradientTransformation)
    lbfgs.py              # Thin wrapper around optax.lbfgs
    kfac.py               # Falls back to NGD (full kfac-jax integration is future work)
    gauss_newton.py       # Gauss-Newton and Levenberg-Marquardt (custom classes)

  training/
    trainer.py            # train(), train_from_config(), create_train_state(), make_train_step()
    loss.py               # MC loss with antithetic variates
    episode.py            # Trajectory simulation via lax.scan
    warm_start.py         # L-BFGS fitting to steady state (uses optax.lbfgs)
    steady_state.py       # Analytical or numerical SS solving (uses optax.lbfgs)

configs/                  # Example YAML configs
tests/                    # test_basic.py, test_config.py, test_optimizers.py, test_convergence.py
```

## Architecture

### Single JIT Boundary

The entire train step is one `@jax.jit` function. This is the core performance decision -- do not break it into multiple JIT calls.

### Three Train Step Variants

Dispatched at construction time (before JIT), not inside JIT:

- **STANDARD** (adam, sgd, adamw, lion, muon, ngd, shampoo, kfac): `jax.grad` -> `opt.update(grads, state, params)`
- **MAO**: `jax.jacrev(per_eq_loss_vector)` -> per-equation Jacobian -> `mao.update(eq_jac, state, params)`
- **LBFGS**: `optax.lbfgs` (GradientTransformationExtraArgs) -> needs `value`, `grad`, `value_fn` for line search

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
| `disaster` | 13 (8 endo + 5 exo) | 12 | 12 | 5 | Numerical |

## Testing

```bash
uv run pytest tests/ -v                    # all 53 tests
uv run pytest tests/test_basic.py -v       # 12 core tests
uv run pytest tests/test_config.py -v      # 18 config tests
uv run pytest tests/test_optimizers.py -v  # 18 optimizer + training tests
uv run pytest tests/test_convergence.py -v # 5 convergence tests
```

Short training smoke tests use: 3 episodes, hidden=(16,), batch=16, mc_samples=2.

## Git

- Author: Anna Smirnova <anna@example.com>
- Main branch: `main`, current work on `master`
