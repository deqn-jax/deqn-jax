# Implementing a Model

This document walks through porting a new economic model to DEQN-JAX, end to end. Stochastic Brock-Mirman (one Euler equation, two states, one shock) is the running example — it is small enough to read in one sitting but has every moving part a larger model has. All code below matches what lives in `src/deqn_jax/models/brock_mirman/` today.

Audience: someone who already knows what DEQN is and has a paper-or-notes description of the model they want to train. If you are new to DEQN itself, read the reference pedagogical notebooks (Geneva Day 2) first.

---

## Before you start

You need, on paper:

- **State variables** and their interpretation — capital, TFP, debt, whatever. Decide whether each is in levels or logs.
- **Policy variables** and their bounds — savings rate in (0, 1), labour supply in (0, ∞), etc. Bounds will be enforced by the network's output activation.
- **Equilibrium equations** written in residual form (LHS − RHS = 0 per equation). One policy variable per equation, typically.
- **Transition dynamics** — next-period state as a function of current state, policy, and exogenous shocks.
- **Calibration** — all constants the equations and dynamics depend on.
- **A sanity check** — closed-form solution, linearization, or a published solution you can diff against. Essential for validating the port.

If your model is new and you are not sure the equations are right, port it anyway: the framework's ergodic residual table (`evaluate.print_euler_errors`) is an effective catch-net for algebra errors that make the FOCs unsolvable.

---

## Directory layout

Every model lives in `src/deqn_jax/models/<name>/` as a five-file subpackage:

```
models/
  <name>/
    __init__.py        # assembles and exports MODEL: ModelSpec
    variables.py       # SPEC, CONSTANTS, POLICY_LOWER/UPPER, N_SHOCKS
    equations.py       # definitions(), equations(), EQUATION_NAMES
    dynamics.py        # step()
    steady_state.py    # steady_state() and init_state_fn
```

The split is pedagogical — nothing enforces it. What matters is that `__init__.py` exports a `MODEL` constant of type `ModelSpec`, and `models/__init__.py` registers it.

Naming tip: pick a short lowercase identifier. It becomes the `model:` field in YAML configs and the CLI argument to `deqn-jax train`. Dashes are fine; avoid spaces.

---

## 1. `variables.py` — what the model is made of

This file declares the static metadata the framework needs to wire shapes and output activations.

```python
import jax.numpy as jnp
from deqn_jax.models.variable_spec import VariableSpec

SPEC = VariableSpec(
    state_names=("k", "z"),
    policy_names=("sav_rate",),
)

CONSTANTS = {
    "alpha": 0.36,
    "beta": 0.99,
    "gamma": 1.0,       # gamma = 1 is log utility
    "delta": 0.1,
    "rho_z": 0.9,
    "sigma_z": 0.04,
}

POLICY_LOWER = jnp.array([1e-6])
POLICY_UPPER = jnp.array([1 - 1e-6])

N_SHOCKS = 1
DESCRIPTION = "Brock-Mirman (1972) optimal growth model"
```

### `SPEC: VariableSpec`

`VariableSpec.unpack_state(state_array)` and `SPEC.unpack_policy(policy_array)` give named attribute access (`s.k`, `p.sav_rate`) — cleaner than `state[:, 0]` everywhere. Both methods work on both batched `[batch, n]` and unbatched `[n]` arrays, so you write your equations once and they trace through `jax.vmap` unchanged.

Name your variables in the order you want them to appear in the state/policy vector. Keep the order consistent across the whole subpackage: if `SPEC.state_names = ("k", "z")`, then `step()` must return `[k_next, z_next]` stacked in that order, and `init_state_fn` must emit columns in that order.

### `CONSTANTS: dict[str, float]`

Every scalar the model uses. The framework passes this dict to every equation/dynamics/steady-state function. Do not hardcode constants inside those functions — putting them in `CONSTANTS` is what lets YAML configs override them for calibration sweeps.

### `POLICY_LOWER`, `POLICY_UPPER`

One-dimensional `jax.numpy.array`s of length `n_policies`. The MLP output layer applies a sigmoid, then rescales to `[POLICY_LOWER, POLICY_UPPER]`. Hard constraints, not soft penalties. For unbounded policies (e.g. consumption in levels, no upper bound), omit the `policy_upper` field in `ModelSpec` and the network will pass sigmoid output through untransformed — or use `(lower, large_number)` if a finite cap is acceptable.

### `N_SHOCKS`

Number of independent exogenous shocks. The framework draws `eps ~ N(0, I_{N_SHOCKS})` and passes them to `step()`. For deterministic models, set `N_SHOCKS = 0` — shock arrays will be `[batch, 0]`, and `step()` should ignore them.

---

## 2. `equations.py` — the equilibrium conditions

Two functions. `definitions()` computes everything derived from state and policy; `equations()` computes the residuals the network is trained to zero.

### `definitions(state, policy, constants) -> dict[str, Array]`

Any quantity used in more than one place, or that you want logged. The returned dict is available:

- inside `equations()` (share computation between period `t` and `t+1`),
- to the trainer (histogram logging at each cycle),
- to the composite-loss path (Jacobians, barriers),
- to post-training diagnostics (`irf.run_irf` records every definition along the impulse path).

```python
def definitions(state, policy, constants):
    s = SPEC.unpack_state(state)
    p = SPEC.unpack_policy(policy)
    alpha = constants["alpha"]
    gamma = constants["gamma"]

    Z = jnp.exp(s.z)
    y = Z * jnp.power(s.k, alpha)
    mpk = alpha * Z * jnp.power(s.k, alpha - 1)
    c = (1 - p.sav_rate) * y
    sav = p.sav_rate * y
    u_c = jnp.power(c, -gamma)

    return {"Z": Z, "y": y, "mpk": mpk, "c": c, "s": sav, "u_c": u_c}
```

Shape contract: every value in the returned dict is either a scalar (for state-independent constants) or a `[batch]`-shaped array. **Never return `[batch, 1]`** — that interacts badly with downstream broadcasting in `plots.*` and `evaluate.*`.

### `equations(state, policy, next_state, next_policy, constants) -> dict[str, Array]`

Returns a residual per equilibrium equation. Each residual is `[batch]`. The framework takes the weighted mean of residuals across shocks (Monte Carlo or quadrature), then squares, then means across the batch. At an equilibrium, `E[residual] = 0` — that is the contract.

```python
EQUATION_NAMES = ("euler",)

def equations(state, policy, next_state, next_policy, constants):
    beta = constants["beta"]
    delta = constants["delta"]
    defs = definitions(state, policy, constants)
    next_defs = definitions(next_state, next_policy, constants)

    u_c = defs["u_c"]
    u_c_next = next_defs["u_c"]
    mpk_next = next_defs["mpk"]

    euler = u_c - beta * u_c_next * (1.0 + mpk_next - delta)
    return {"euler": euler}
```

`EQUATION_NAMES` is the ordered tuple used for logging, reweighting, and the per-equation diagnostic tables. Keep it in sync with the dict keys in the returned residuals.

### Choosing the right residual form — this is a trap

The same FOC can be written many algebraically-equivalent ways. **Under Monte Carlo sampling, they are not equivalent as loss functions.**

The framework computes, per batch element, `E_shock[residual_shock]^2`. "Expectation of the per-shock residual, then square." Three canonical forms for a generic Euler FOC `u'(c) = β E[u'(c') (1 + r' − δ)]`:

1. **Raw**: `resid = u'(c) − β u'(c') (1 + r' − δ)`.
   Linear in the shock-dependent quantity `u'(c')(1 + r' − δ)`. MC-safe: `E[resid] = u'(c) − β E[u'(c')(1+r'−δ)] = 0` at equilibrium. Scale varies across states (goes like `u'(c)`) but that is handled by the mean-squared aggregation.

2. **LHS-normalized dimensionless**: `resid = 1 − β u'(c') (1 + r' − δ) / u'(c)`.
   Still MC-safe (dividing by the shock-independent `u'(c)`), and reads well in tables. **But optimization-hostile**: at bad policies that drive `c → 0`, `u'(c)` explodes, so the dimensionless residual *shrinks* exactly where you need gradient pressure to push the policy back. Training gets stuck in low-consumption local minima.

3. **RHS-normalized dimensionless (Simon's form)**: `resid = 1 − u'(c) / (β E[u'(c') (1 + r' − δ)])`.
   Clean at equilibrium. **Not MC-compatible**: the per-shock form `1 − u'(c) / (β u'(c'_ω)(1 + r'_ω − δ))` requires `E[1/X] = 1/E[X]`, which is Jensen-false for non-degenerate shocks. Safe when the expectation is computed *inside* the residual by Gauss-Hermite; broken under our per-shock averaging.

**Default to form (1).** Report accuracy post-training in form (2) via `evaluate.print_euler_errors` plus a dimensionless table (see `examples/brock_mirman.ipynb` section 8 for the pattern). If you find yourself reaching for form (3), you are building against an MC framework with a GH-framework mental model — either switch to quadrature (`quad_nodes`/`quad_weights` in `compute_loss`) or switch back to form (1).

The `brock_mirman` subpackage docstring in `equations.py` documents this trap inline; keep a similar note in any new model that has an Euler-like FOC.

---

## 3. `dynamics.py` — the state transition

```python
def step(state, policy, shock, constants):
    s = SPEC.unpack_state(state)
    defs = definitions(state, policy, constants)
    delta = constants["delta"]
    rho_z = constants["rho_z"]
    sigma_z = constants["sigma_z"]

    k_next = (1 - delta) * s.k + defs["s"]
    eps = shock[:, 0] if shock.ndim > 1 else shock
    z_next = rho_z * s.z + sigma_z * eps

    return jnp.stack([k_next, z_next], axis=1)
```

Signature: `step(state, policy, shock, constants) -> next_state`. Shapes:

- `state`: `[batch, n_states]`
- `policy`: `[batch, n_policies]`
- `shock`: `[batch, n_shocks]` (scalar batch dim is allowed — the framework's shock sampler always produces 2-D arrays, but traces that pass `shock` through a single-sample path may produce 1-D).
- Return: `[batch, n_states]`, column order matching `SPEC.state_names`.

**Handle both 1-D and 2-D shock.** The idiom above (`shock[:, 0] if shock.ndim > 1 else shock`) is standard across models. Do not assume batched shape.

**Deterministic models.** For `N_SHOCKS = 0`, `shock` is `[batch, 0]`; your `step()` should simply ignore it. No branching needed; the zero-width shock array never broadcasts into the state update.

**Do not clip states inside `step`.** `step` is called every cycle during training and must be smooth. Clipping belongs in `clip_state_fn` (optional; used only by evaluation/IRF paths).

---

## 4. `steady_state.py` — the starting point and the sampler

Two responsibilities: (a) compute a steady state (for warm-start and IRF), (b) sample initial training states.

### Steady state

```python
def steady_state(constants):
    alpha = constants["alpha"]
    beta = constants["beta"]
    delta = constants["delta"]
    k_ss = ((1 / beta - 1 + delta) / alpha) ** (1 / (alpha - 1))
    z_ss = 0.0
    y_ss = k_ss ** alpha
    sav_rate_ss = delta * k_ss / y_ss
    return jnp.array([k_ss, z_ss]), jnp.array([sav_rate_ss])
```

Signature: `steady_state(constants) -> (ss_state, ss_policy)` where both are 1-D arrays of length `n_states` / `n_policies`.

If your model has no closed-form steady state, solve numerically using `deqn_jax.training.steady_state.solve_steady_state` (which is a thin wrapper over `optax.lbfgs`). See `src/deqn_jax/models/disaster/steady_state.py` for an example.

### Initial state sampler

Use `make_init_state_fn` to build the sampler declaratively:

```python
from deqn_jax.models.variable_spec import make_init_state_fn

INIT_SPECS = {
    "k": {"distribution": "uniform", "kwargs": {"minval": 0.9, "maxval": 12.0}},
    "z": {"distribution": "uniform", "kwargs": {"minval": -0.357, "maxval": 0.262}},
}

init_state = make_init_state_fn(SPEC.state_names, INIT_SPECS)
```

Supported distributions: `uniform`, `normal`, `lognormal`, `truncated_normal`, `constant`. States without an entry default to zero. Unknown distributions raise `ValueError` at build time — typos fail fast, before training starts.

Hand-written samplers are still supported for cases where per-variable specs aren't expressive enough (correlated draws, conditional sampling). Just define `def init_state(key, batch_size, constants) -> [batch, n_states]` directly.

### Rect bounds: where do they come from?

The rect you sample from is a **training-time decision**, not an economic one. Choices:

- **Wide rect around the ergodic support** (default): sample uniformly on a rectangle that covers roughly ±3σ of each state's unconditional distribution. Works for most models.
- **Rect matching the reference implementation** for direct comparison against a published solution.
- **Rollout-based** (no rect): set `initialize_each_episode=False` in the config and let trajectories drift into the ergodic set. Riskier — the training distribution concentrates, and the network can overfit to the attractor.

Brock-Mirman uses the reference's exact rect because the whole point of the notebook is side-by-side comparability.

---

## 5. `__init__.py` — assembly

```python
from deqn_jax.types import ModelSpec
from deqn_jax.models.brock_mirman.variables import (
    SPEC, CONSTANTS, N_SHOCKS, POLICY_LOWER, POLICY_UPPER,
)
from deqn_jax.models.brock_mirman.equations import equations, definitions, EQUATION_NAMES
from deqn_jax.models.brock_mirman.dynamics import step
from deqn_jax.models.brock_mirman.steady_state import steady_state, init_state

MODEL = ModelSpec(
    name="brock_mirman",
    n_states=SPEC.n_states,
    n_policies=SPEC.n_policies,
    n_shocks=N_SHOCKS,
    state_names=SPEC.state_names,
    policy_names=SPEC.policy_names,
    equation_names=EQUATION_NAMES,
    shock_names=("eps_z",),
    constants=CONSTANTS,
    equations_fn=equations,
    step_fn=step,
    steady_state_fn=steady_state,
    init_state_fn=init_state,
    definitions_fn=definitions,
    policy_lower=POLICY_LOWER,
    policy_upper=POLICY_UPPER,
)
```

Required fields: `name`, `n_states`, `n_policies`, `n_shocks`, `constants`, `equations_fn`, `step_fn`. Everything else is optional but almost always useful; see `src/deqn_jax/types.py` for the complete list with docstrings.

Always supply `state_names`, `policy_names`, `equation_names`, `shock_names`. Without them, diagnostic output uses index labels (`state_0`, `policy_0`) and the IRF path `run_irf(shock_name="...")` has no way to address a specific shock by name.

### Optional `ModelSpec` fields worth knowing

- **`shock_names: tuple[str, ...]`** — labels for shocks. Used by `irf.run_irf(shock_name=...)` to select which shock to hit. Do not omit for stochastic models.
- **`cycle_hook: Callable[[TrainState, ModelSpec, int], None]`** — called every `log_every` episodes. Side-effect only (saves convergence snapshots to disk, logs to TB, etc.). See `models/bm_deterministic/hooks.py` for the convention.
- **`state_bounds`, `definition_bounds: dict[str, dict[str, float]]`** — soft penalties added to the training loss. Format:
  ```python
  {"c": {"lower": 0.0, "penalty_lower": 1.0}}
  ```
  Penalty is `coef * mean(max(0, lower − value)^2)` for the lower side; analogous for upper. Missing penalty coefficients default to `1/bound^2`. Use `state_bounds` for raw state variables and `definition_bounds` for anything from the `definitions()` dict. Hard constraints on policies use `policy_lower`/`policy_upper`; these are the soft complement for derived quantities.
- **`clip_state_fn: Callable[[state], state]`** — simulation-safety clip used by `evaluate` and `irf` paths only. **Never applied during training** — that would break differentiability of `step`.
- **`state_barrier_fn: Callable[[state], [batch]]`** — legacy soft barrier; prefer `state_bounds` for new models.

---

## 6. Register the model

Add one line in `src/deqn_jax/models/__init__.py`:

```python
from deqn_jax.models.brock_mirman import MODEL as _brock_mirman

_MODELS = {
    ...
    "brock_mirman": _brock_mirman,
}
_DESCRIPTIONS = {
    ...
    "brock_mirman": "Brock-Mirman (1972) optimal growth model",
}
```

After this, `load_model("brock_mirman")` and `deqn-jax train brock_mirman ...` both work.

---

## 7. Write a YAML config

`configs/<name>.yaml` pins a training recipe for your model. The minimum you need:

```yaml
model: brock_mirman
episodes: 20001
batch_size: 128
episode_length: 1
mc_samples: 5

initialize_each_episode: true
n_epochs_per_rollout: 1
n_minibatches_per_epoch: 1

network:
  type: mlp
  hidden_sizes: [50, 50]
  activation: relu
  init: xavier_uniform

optimizer:
  name: adam
  learning_rate: 3.0e-4
  lr_schedule: cosine
  lr_min_factor: 0.1

warm_start: false
log_every: 1000
```

Key decisions:

- **`episode_length: 1` + `initialize_each_episode: true`** gives exogenous-rect sampling (Simon's phase-1 recipe). `episode_length: N` + `initialize_each_episode: false` gives rollout sampling (ergodic / phase-2).
- **`mc_samples`**: number of shock draws for the expectation. 5 is a reasonable starting point; more gives lower-variance loss at linear cost. For deterministic models, set to 1.
- **`warm_start: true`** runs an L-BFGS pre-fit of the network to the steady-state policy before gradient-based training. Speeds up early convergence but can mask bugs in the Euler equation (the network starts near a good answer regardless of whether the loss is correct).

See `docs/running_experiments.md` (when it exists) for the full field-by-field reference.

---

## 8. Validate

The framework won't tell you if your equations are subtly wrong. Validate in layers:

**1. Unit test the equations at steady state.**
```python
def test_euler_zero_at_ss():
    ss_state, ss_policy = steady_state(CONSTANTS)
    # Batch a 1-row state/policy, no shock
    state = ss_state[None, :]
    policy = ss_policy[None, :]
    next_state = step(state, policy, jnp.zeros((1, N_SHOCKS)), CONSTANTS)
    next_policy = policy  # at SS, policy is the same
    resid = equations(state, policy, next_state, next_policy, CONSTANTS)
    assert jnp.abs(resid["euler"]).max() < 1e-6
```

If this test fails, your equations are algebraically inconsistent with your steady state. Fix before anything else.

**2. Smoke train for a handful of episodes.**
```bash
uv run deqn-jax train brock_mirman --config configs/brock_mirman.yaml -n 500
```

Loss should decrease roughly monotonically (with noise). If it diverges or plateaus at the initial value, you almost certainly have the residual-form trap — check against the raw form in `equations.py`.

**3. Check the trained policy against a known solution if you have one.**
For Brock-Mirman under log utility / δ=1 the closed form is `s* = αβ` — trivial. For your model, compute an analytic steady-state policy, a linearized impulse response, or a dynare solve, and diff against the trained output on a grid.

**4. Run the ergodic diagnostic.**
```python
from deqn_jax.evaluate import euler_equation_errors, print_euler_errors
result = euler_equation_errors(policy_net, MODEL, n_periods=10_000)
print_euler_errors(result, label="ergodic path")
```

Target: mean log₁₀|resid/u'(c)| below −3 on a well-converged model. Above −2 means either undertrained or a real problem with the equations.

**5. Ergodic moments sanity.**
For the linearized AR(1)-driven-exogenous states, the unconditional mean should match the deterministic steady state and the unconditional std should match `σ/√(1 − ρ²)`. Large mismatches usually indicate the policy is extrapolating outside the training rect (ergodic distribution has concentrated somewhere the network never saw).

---

## Common pitfalls

- **Wrong residual form** — see section 2. Default to the raw form under MC; reach for dimensionless only in post-training diagnostics.
- **State column order mismatch** — `SPEC.state_names = ("k", "z")` but `step()` returns `stack([z_next, k_next])`. Use `SPEC.unpack_state` / named access to avoid this class of bug entirely.
- **Hardcoded constants** inside `equations` or `dynamics`. Always read from `constants`; calibration sweeps will fail silently otherwise.
- **Clipping state inside `step`** — breaks differentiability. Put any clipping in `clip_state_fn` instead.
- **Missing `shock_names`** — `irf.run_irf(shock_name="eps_z")` will raise with an unhelpful index error. Always supply `shock_names` for stochastic models.
- **Ignoring `gamma = 1`** — `jnp.power(c, -1.0)` is slower than `1.0 / c` but the framework does not special-case. Small cost; not a bug.
- **Training on an attractor-concentrated distribution** — `initialize_each_episode=false` with a strongly attracting system will starve the network of samples outside the attractor, and extrapolation breaks at IRF time. Use the rect sampler unless you have a reason not to.

---

## Checklist

Before the first real training run:

- [ ] `variables.py`: `SPEC`, `CONSTANTS`, `POLICY_LOWER/UPPER`, `N_SHOCKS`, `DESCRIPTION`
- [ ] `equations.py`: `definitions()`, `equations()`, `EQUATION_NAMES`. Residual form deliberately chosen.
- [ ] `dynamics.py`: `step()` handles 1-D and 2-D `shock`.
- [ ] `steady_state.py`: `steady_state()` and `init_state` (declarative or hand-written).
- [ ] `__init__.py`: `MODEL: ModelSpec` with `state_names`, `policy_names`, `equation_names`, `shock_names` all set.
- [ ] `models/__init__.py`: new model registered in `_MODELS` and `_DESCRIPTIONS`.
- [ ] `configs/<name>.yaml`: training recipe pinned.
- [ ] Test: Euler residual = 0 at steady state.
- [ ] Smoke train: 500 episodes, loss decreasing.
- [ ] Ergodic diagnostic: log₁₀|resid/u'(c)| below −2 on a serious run.
- [ ] Sanity check: against closed form, linearization, or published solution.
