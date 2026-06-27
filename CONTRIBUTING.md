<!-- Repo + docs URLs below are org-move-gated: a GitHub org move is pending.
     Until it lands, the canonical locations are the mechanicpanic/* ones used here. -->

# Contributing to DEQN-JAX

Thanks for looking under the hood. DEQN-JAX is a global solver for recursive
economic equilibria — you write a model's equilibrium conditions, it returns
globally-solved decision rules and their Euler-equation accuracy. This page is
the contributor on-ramp: how to set up, where things live, and the three ways to
extend it.

> **Honest about maturity.** This is **alpha (`v0.2.0`)**. While the version is
> `0.x.y` the public API is unstable and may change in any minor bump
> (see [`CHANGELOG.md`](CHANGELOG.md)). The plumbing is solid — **571 tests
> pass** — but treat the surface as still-moving. The *validated* stack is
> deliberately small: `adam` + an MLP (or `LinearPlusMLP`) + MSE residual loss +
> antithetic-MC (or Gauss–Hermite) expectations. Everything beyond that is a
> research instrument, not a turnkey recommendation.

## 30-second dev setup

```bash
git clone https://github.com/mechanicpanic/deqn-jax   # org-move-gated; see top of file
cd deqn-jax
uv sync                                   # creates .venv, installs deps + dev tools
uv run pytest tests/                      # 571 tests
```

Always use `uv run` — never activate the venv by hand.

```bash
uv run pytest tests/ -m 'not slow'        # skip the end-to-end training tests (faster)
uv run pytest tests/test_basic.py -v      # the 12 core tests
uv run deqn-jax check                      # environment + JAX device sanity check
uv run deqn-jax list                       # the registered models
uv build                                   # produces wheel + sdist in dist/
```

The CLI exposes nine subcommands — `train`, `list`, `info`, `optimizers`,
`irf`, `evaluate`, `check`, `active-subspace`, `init-config`. Run any with
`--help`.

## The layout in one glance

```
src/deqn_jax/
  api.py             The stable public surface — import from here (see REFERENCE.md)
  cli.py             Entry point for the nine subcommands
  types.py           ModelSpec, TrainState, Metrics (NamedTuples — JAX pytrees)
  config/            Pydantic configs + YAML + CLI override chain
  models/
    <name>/          Per-model: variables / equations / dynamics / steady_state
    __init__.py      The model registry (_MODELS dict + register_model)
  networks/
    factory.py       net_type dispatch (build_policy_net)
    mlp.py  lstm.py  transformer.py  linear_plus_mlp.py
  optimizers/
    registry.py      @register_optimizer, OptimizerKind, create_optimizer()
    __init__.py      imports every module so registration runs
  training/
    trainer.py       Main loop; the single JIT'd train step (5 dispatched variants)
    loss.py  composite_loss.py  episode.py  linearize.py  warm_start.py
  evaluate/          errREE distribution, ergodic moments, stability checks
configs/             Example YAML configs
tests/               571 tests
```

The one document to read before building anything substantial is
**[`docs/site/REFERENCE.md`](docs/site/REFERENCE.md)** — the type-signature-first
contract for every public entry point, and the `ModelSpec` definition that
hand-written and codegen'd models alike must satisfy.

## Three ways to extend it

Each path is a short recipe pointing at real files. The minimal reference for a
model is `src/deqn_jax/models/brock_mirman/`; the full-scale example is
`src/deqn_jax/models/disaster/`.

### 1. Add a model

The `ModelSpec` contract is the whole surface — declare the economics as data.
(Full signatures in [`REFERENCE.md`](docs/site/REFERENCE.md).)

1. Create `src/deqn_jax/models/<name>/` with four files:
   - `variables.py` — a `VariableSpec` (state/policy names), `CONSTANTS`, and
     steady-state reference values.
   - `equations.py` — `equations(state, policy, next_state, next_policy, constants)`
     returns a dict of residuals that must vanish in expectation; plus
     `definitions()` for derived quantities.
   - `dynamics.py` — `step(state, policy, shock, constants)` returns the next state.
   - `steady_state.py` — `steady_state(constants)` returns `(ss_state, ss_policy)`,
     analytical or numerical.
2. Assemble a `ModelSpec` and export it as `MODEL` from the package `__init__.py`.
3. Register it: add an import + an entry to `_MODELS` / `_DESCRIPTIONS` in
   `src/deqn_jax/models/__init__.py`. (Or, for an out-of-tree model, call
   `register_model(spec, description=...)` at runtime — same registry, no fork.)
4. Add a test that trains ~20 episodes and asserts the loss decreases (see the
   patterns in `tests/test_basic.py`).

### 2. Add an optimizer

1. Create `src/deqn_jax/optimizers/<name>.py`. Either return an
   `optax.GradientTransformation` (standard case) or implement a custom class
   with `.init(params)` / `.update(...)`.
2. Decorate the factory: `@register_optimizer("name", kind=OptimizerKind.STANDARD)`.
3. Import the module in `src/deqn_jax/optimizers/__init__.py` so registration runs.

The train step has **five dispatched variants** — `STANDARD`, `PCGRAD`, `MAO`,
`LBFGS`, `GN` — keyed off the four `OptimizerKind` values (`STANDARD`, `MAO`,
`LBFGS`, `GN`) in `optimizers/registry.py`. Most first-order methods are
`STANDARD`; pick the kind that matches what your update needs (per-equation
Jacobian, line-search extras, or a residual Jacobian for Gauss–Newton).
`uv run deqn-jax optimizers` lists what is registered.

### 3. Add a network

1. Subclass `eqx.Module` in `src/deqn_jax/networks/<name>.py` and add a
   `create_<name>(...)` factory.
2. Wire `net_type == "<name>"` into the dispatch in
   `src/deqn_jax/networks/factory.py` (`build_policy_net` — search for
   `create_mlp`).
3. Select it from config with `network.type: "<name>"`.

`MLP` is the validated default; `LinearPlusMLP` (a residual over the
Blanchard–Kahn linearization) is the recommended prior for medium-scale DSGE.

## Code style and the one load-bearing norm

| Convention | What to do |
|---|---|
| Formatting / lint | `ruff` — `uv run ruff format .` and `uv run ruff check .` |
| Environment | always `uv run`; never activate the venv manually |
| State | everything mutable is a `NamedTuple` (pytree) so `jit`/`grad`/`vmap` compose |
| Networks | Equinox modules: `eqx.filter(model, eqx.is_array)` splits trainable from static |
| No-fabrication | advertise only shipped behavior; mark experimental as experimental |

**The single JIT boundary.** The entire train step — loss, gradient, optimizer
step — is one `@jax.jit` function. This is the core performance decision: it
keeps XLA's fusion opportunities alive. Do **not** split it into multiple JIT
calls. New train-step logic is dispatched *before* JIT (at construction time),
not branched inside it.

## Proposing a change

1. Branch off `master`.
2. Make the change; keep the diff focused.
3. Run the suite green: `uv run pytest tests/` (or `-m 'not slow'` while
   iterating). New behavior gets a test.
4. `uv run ruff format .` and `uv run ruff check .`.
5. Open a PR against `master` describing the *what* and the *why*. If you touched
   the public surface, note it — the API is `0.x` and we track breaks in
   [`CHANGELOG.md`](CHANGELOG.md).

Questions, model contributions, or research collaboration:
**Anna Smirnova `<anna.smirnova@unil.ch>`**.

## The ecosystem

[`deqn-agent`](https://github.com/mechanicpanic/deqn-agent) <!-- org-move-gated -->
is a separate repo: an agent stack *on top* of DEQN-JAX that turns a paper or
model description into a trained, residual-checked DEQN policy. It targets the
same `ModelSpec` / `REFERENCE.md` contract you build against here. It is **v0
alpha** — experimental.

## Credit & provenance

DEQN-JAX is a JAX/Equinox reimplementation and extension of the **Deep
Equilibrium Networks** methodology of Simon Scheidegger and collaborators. All
credit for the original method belongs to the upstream authors:

- Azinovic, M., Gaegauf, L., Scheidegger, S. (2022). *Deep Equilibrium Nets.*
  International Economic Review 63(4), 1471–1525.
- Scheidegger, S., Bilionis, I. (2019). *Machine learning for high-dimensional
  dynamic stochastic economies.* Journal of Computational Science 33, 68–82.

## License

By contributing you agree your contributions are licensed under the project's
**MIT** license (see [`LICENSE`](LICENSE)).

