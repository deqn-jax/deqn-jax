# Implementing a Model

This page ports a model to DEQN-JAX, using stochastic Brock-Mirman (one Euler equation, two states, one shock) as the example. It is small but has every part a larger model has. The code matches `src/deqn_jax/models/brock_mirman/`.

It assumes you know what DEQN is and have the model written down. If DEQN is new to you, read the reference pedagogical notebooks (Geneva Day 2) first.

---

## Before you start

You need, on paper:

- State variables and their interpretation (capital, TFP, debt, and so on). Decide whether each is in levels or logs.
- Policy variables and their bounds, for example a savings rate in (0, 1) or labour supply in (0, ∞). The network's output activation enforces the bounds.
- Equilibrium equations in residual form (LHS − RHS = 0 per equation), typically one policy variable per equation.
- Transition dynamics: next-period state as a function of current state, policy and exogenous shocks.
- Calibration: every constant the equations and dynamics depend on.
- Something to validate against: a closed-form solution, a linearization or a published solution.

If you are unsure the equations are right, port the model anyway: the ergodic residual table (`evaluate.print_euler_errors`) catches algebra errors that make the FOCs unsolvable.

---

## Directory layout

Each model lives in `src/deqn_jax/models/<name>/` as a five-file subpackage:

```
models/
  <name>/
    __init__.py        # assembles and exports MODEL: ModelSpec
    variables.py       # SPEC, CONSTANTS, POLICY_LOWER/UPPER, N_SHOCKS
    equations.py       # definitions(), equations(), EQUATION_NAMES
    dynamics.py        # step()
    steady_state.py    # steady_state() and init_state_fn
```

The split is a convention; nothing enforces it. What matters is that `__init__.py` exports a `MODEL` constant of type `ModelSpec` and that `models/__init__.py` registers it.

Pick a short lowercase identifier as the name. It becomes the `model:` field in YAML configs and the CLI argument to `deqn-jax train`. Dashes are fine; avoid spaces.

---

## 1. `variables.py`: what the model is made of

Static metadata that sets shapes and output activations.

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

`VariableSpec.unpack_state(state_array)` and `SPEC.unpack_policy(policy_array)` give named attribute access (`s.k`, `p.sav_rate`) instead of `state[:, 0]`. Both accept batched `[batch, n]` and unbatched `[n]` arrays, so the same equations trace through `jax.vmap` unchanged.

List variables in the order you want in the state and policy vectors, and keep that order across the subpackage. If `SPEC.state_names = ("k", "z")`, then `step()` must return `[k_next, z_next]` stacked in that order, and `init_state_fn` must emit columns in that order.

### `CONSTANTS: dict[str, float]`

Every scalar the model uses, passed to every equation, dynamics and steady-state function. Do not hardcode constants in those functions; YAML configs override `CONSTANTS` for calibration sweeps.

### `POLICY_LOWER`, `POLICY_UPPER`

One-dimensional `jax.numpy.array`s of length `n_policies`. The MLP output activation is chosen per dimension from the upper bound:

- Finite `POLICY_UPPER[i]`: sigmoid, rescaled to `[lower, upper]`. A hard constraint.
- `POLICY_UPPER[i] = jnp.inf`: softplus(x) + `lower`. The lower bound is enforced; the upper is unbounded.

For `bm_labor`, with the savings rate in `(0, 1)` and labor supply in `(0, ∞)`:

```python
POLICY_LOWER = jnp.array([1e-6, 1e-6])
POLICY_UPPER = jnp.array([1 - 1e-6, jnp.inf])   # sigmoid on sav_rate, softplus on L
```

These are hard constraints at the network output. For soft penalties on derived quantities (such as positive consumption), use `state_bounds` / `definition_bounds`.

### `N_SHOCKS`

The number of independent exogenous shocks. The framework draws `eps ~ N(0, I_{N_SHOCKS})` and passes them to `step()`. For deterministic models set `N_SHOCKS = 0`; shock arrays are then `[batch, 0]` and `step()` should ignore them.

---

## 2. `equations.py`: the equilibrium conditions

`definitions()` computes quantities derived from state and policy; `equations()` computes the residuals the network is trained to zero.

### `definitions(state, policy, constants) -> dict[str, Array]`

Any quantity used in more than one place or worth logging. The returned dict is used:

- inside `equations()`, to share computation between periods `t` and `t+1`;
- by the trainer, for histogram logging at each cycle;
- by the composite loss (Jacobians, barriers);
- by post-training diagnostics (`irf.run_irf` records every definition along the impulse path).

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

Shape contract: every value in the returned dict is a scalar (for state-independent constants) or a `[batch]` array. Do not return `[batch, 1]`; it breaks downstream broadcasting in `plots.*` and `evaluate.*`.

### `equations(state, policy, next_state, next_policy, constants) -> dict[str, Array]`

Returns one `[batch]` residual per equilibrium equation. The framework takes the weighted mean of each residual across shocks (Monte Carlo or quadrature), squares it, then averages across the batch. The contract is `E[residual] = 0` at an equilibrium.

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

`EQUATION_NAMES` is the ordered tuple used for logging, reweighting and the per-equation diagnostic tables. Keep it in sync with the keys of the returned dict.

### Choosing the residual form

Algebraically equivalent forms of a FOC are not equivalent as loss functions under Monte Carlo sampling. Per batch element the framework computes `E_shock[residual_shock]^2`: the expectation of the per-shock residual, then the square. Three common forms for a generic Euler FOC `u'(c) = β E[u'(c') (1 + r' − δ)]`:

1. Raw: `resid = u'(c) − β u'(c') (1 + r' − δ)`.
   Linear in the shock-dependent quantity `u'(c')(1 + r' − δ)`, so it is MC-safe: `E[resid] = u'(c) − β E[u'(c')(1+r'−δ)] = 0` at equilibrium. Its scale varies across states (it moves with `u'(c)`), which the mean-squared aggregation handles.

2. LHS-normalized, dimensionless: `resid = 1 − β u'(c') (1 + r' − δ) / u'(c)`.
   Still MC-safe, because it divides by the shock-independent `u'(c)`, and it reads well in tables. It is bad for optimization: at policies that drive `c → 0`, `u'(c)` explodes, so the residual shrinks exactly where the policy needs gradient pressure to move back. Training gets stuck in low-consumption local minima.

3. RHS-normalized, dimensionless (Scheidegger's form): `resid = 1 − u'(c) / (β E[u'(c') (1 + r' − δ)])`.
   Clean at equilibrium, but not compatible with MC. The per-shock form `1 − u'(c) / (β u'(c'_ω)(1 + r'_ω − δ))` would need `E[1/X] = 1/E[X]`, which Jensen's inequality rules out for non-degenerate shocks. It is safe when Gauss-Hermite computes the expectation inside the residual, and wrong under per-shock averaging.

Use form (1) by default. Report accuracy after training in form (2) with `evaluate.print_euler_errors` and a dimensionless table (`examples/brock_mirman.ipynb` section 8 shows the pattern). Form (3) assumes the expectation is taken by quadrature inside the residual. If you need it, either switch to quadrature (`quad_nodes`/`quad_weights` in `compute_loss`) or go back to form (1).

The docstring of `brock_mirman/equations.py` records this; keep a similar note in any model with an Euler-like FOC.

---

## 3. `dynamics.py`: the state transition

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
- `shock`: `[batch, n_shocks]`. A 1-D shock is also possible: the framework's shock sampler always produces 2-D arrays, but code paths that pass a single sample may produce 1-D.
- Return: `[batch, n_states]`, columns in the order of `SPEC.state_names`.

Handle both 1-D and 2-D shocks with the idiom above (`shock[:, 0] if shock.ndim > 1 else shock`), as the other models do.

With `N_SHOCKS = 0`, `shock` is `[batch, 0]` and `step()` ignores it; no branching is needed.

Do not clip states inside `step`: it runs every training cycle and must be smooth. Clipping belongs in the optional `clip_state_fn`, used only by evaluation and IRFs.

---

## 4. `steady_state.py`: the starting point and the sampler

This file computes a steady state (for warm start and IRFs) and samples initial training states.

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

Signature: `steady_state(constants) -> (ss_state, ss_policy)`, both 1-D arrays of length `n_states` / `n_policies`.

Without a closed form, solve numerically, for example with `deqn_jax.training.steady_state.solve_steady_state`, a thin wrapper over `optax.lbfgs`. `src/deqn_jax/models/disaster/steady_state.py` is a numerical example; it uses `scipy.optimize.root` (`method="hybr"`).

### Initial state sampler

Build the sampler declaratively with `make_init_state_fn`:

```python
from deqn_jax.models.variable_spec import make_init_state_fn

INIT_SPECS = {
    "k": {"distribution": "uniform", "kwargs": {"minval": 0.9, "maxval": 12.0}},
    "z": {"distribution": "uniform", "kwargs": {"minval": -0.357, "maxval": 0.262}},
}

init_state = make_init_state_fn(SPEC.state_names, INIT_SPECS)
```

Supported distributions: `uniform`, `normal`, `lognormal`, `truncated_normal`, `constant`. States without an entry default to zero. An unknown distribution raises `ValueError` when the sampler is built, so typos fail before training starts.

For what per-variable specs cannot express (correlated draws, conditional sampling), write `def init_state(key, batch_size, constants) -> [batch, n_states]` by hand.

### Where the rectangle bounds come from

The sampling rectangle is a training choice. Options:

- A wide rectangle around the ergodic support (the default), covering roughly ±3σ of each state's unconditional distribution. This works for most models.
- The reference implementation's rectangle, for direct comparison with a published solution.
- Rollouts with no rectangle: set `initialize_each_episode=False` and let trajectories drift into the ergodic set. This is riskier, since the training distribution concentrates and the network can overfit to the attractor.

Brock-Mirman uses the reference's exact rectangle because its notebook is a side-by-side comparison.

---

## 5. `__init__.py`: assembly

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

Required fields: `name`, `n_states`, `n_policies`, `n_shocks`, `constants`, `equations_fn`, `step_fn`. `src/deqn_jax/types.py` documents the optional ones.

Always supply `state_names`, `policy_names`, `equation_names` and `shock_names`. Without them diagnostics use index labels (`state_0`, `policy_0`) and `run_irf(shock_name="...")` cannot select a shock by name.

### Optional `ModelSpec` fields

- `shock_names: tuple[str, ...]`: shock labels, used by `irf.run_irf(shock_name=...)`. Do not omit them for stochastic models.
- `cycle_hook: Callable[[TrainState, ModelSpec, int], None]`: called every `log_every` episodes, for side effects only (saving convergence snapshots, logging to TensorBoard). `models/bm_deterministic/hooks.py` shows the convention.
- `state_bounds`, `definition_bounds: dict[str, dict[str, float]]`: soft penalties added to the training loss. Format:
  ```python
  {"c": {"lower": 0.0, "penalty_lower": 1.0}}
  ```
  The lower-side penalty is `coef * mean(max(0, lower − value)^2)`; the upper side is analogous. Missing penalty coefficients default to `1/bound^2`. Use `state_bounds` for raw states and `definition_bounds` for entries of the `definitions()` dict. Hard constraints on policies go through `policy_lower`/`policy_upper`; these fields are the soft counterpart for derived quantities.
- `clip_state_fn: Callable[[state], state]`: a safety clip used only by `evaluate` and `irf`. Never applied in training, where it would break the differentiability of `step`.
- `state_barrier_fn: Callable[[state], [batch]]`: a legacy soft barrier. Use `state_bounds` in new models.

---

## 6. Register the model

Add one entry in `src/deqn_jax/models/__init__.py`:

```python
from deqn_jax.models.brock_mirman import MODEL as _brock_mirman

_MODELS = {
    ...
    "brock_mirman": _brock_mirman,
}
```

`_MODELS` is the only dict to edit. The one-line description `deqn-jax list`
prints comes from your package's `variables.py::DESCRIPTION`; there is no
second copy in `models/__init__.py`, and a package without a `DESCRIPTION`
fails at import.

After this, `load_model("brock_mirman")` and `deqn-jax train brock_mirman ...` both work.

---

## 7. Write a YAML config

`configs/<name>.yaml` fixes a training recipe. The minimum:

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

The main choices:

- `episode_length: 1` with `initialize_each_episode: true` samples from the exogenous rectangle (Scheidegger's phase-1 recipe). `episode_length: N` with `initialize_each_episode: false` samples from rollouts (ergodic, phase 2).
- `mc_samples` is the number of shock draws per expectation. 5 is a reasonable start; more draws lower the loss variance at linear cost. For deterministic models set it to 1.
- `warm_start: true` pre-fits the network to the steady-state policy with L-BFGS before gradient training. This speeds up early convergence but can hide Euler-equation bugs, because the network starts near a good answer even if the loss is wrong.

[Running experiments](../running_experiments.md) covers the CLI, config, checkpoints and logging in full.

---

## 8. Validate

The framework cannot detect subtly wrong equations. Validate in layers.

### Unit-test the equations at the steady state

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

If this fails, the equations and the steady state disagree. Fix that first.

### Smoke-train for a few hundred episodes

```bash
uv run deqn-jax train brock_mirman --config configs/brock_mirman.yaml -n 500
```

The loss should fall roughly monotonically, with noise. If it diverges or stays flat, suspect the residual form and compare with the raw form in `equations.py`.

### Compare with a known solution

For Brock-Mirman with log utility and δ=1 the closed form is `s* = αβ`. For your model, compute an analytic steady-state policy, a linearized impulse response or a Dynare solution, and compare it with the trained output on a grid.

### Run the ergodic diagnostic

```python
from deqn_jax.evaluate import euler_equation_errors, print_euler_errors
result = euler_equation_errors(policy_net, MODEL, n_periods=10_000)
print_euler_errors(result, label="ergodic path")
```

Target: mean log₁₀|resid/u'(c)| below −3 for a well-converged model. Above −2 means the model is undertrained or the equations are wrong.

### Check the ergodic moments

For exogenous AR(1) states, the unconditional mean should match the deterministic steady state and the standard deviation should match `σ/√(1 − ρ²)`. Large mismatches usually mean the ergodic distribution sits where the network was never trained, so the policy is extrapolating outside the training rectangle.

---

## Common pitfalls

- Wrong residual form: see section 2. Use the raw form under MC; use dimensionless forms only in post-training diagnostics.
- State column order mismatch: `SPEC.state_names = ("k", "z")` but `step()` returns `stack([z_next, k_next])`. Named access through `SPEC.unpack_state` avoids this.
- Constants hardcoded inside `equations` or `dynamics`. Always read them from `constants`; otherwise calibration sweeps fail silently.
- Clipping the state inside `step`, which breaks differentiability. Put clipping in `clip_state_fn`.
- Missing `shock_names`: `irf.run_irf(shock_name="eps_z")` then raises an unhelpful index error. Always supply `shock_names` for stochastic models.
- `gamma = 1`: `jnp.power(c, -1.0)` is slower than `1.0 / c` and the framework does not special-case it. A small cost, not a bug.
- Training only near the attractor: `initialize_each_episode=false` with a strongly attracting system leaves no samples away from it, and extrapolation fails in IRFs. Use the rectangle sampler unless you have a reason not to.

---

## Checklist

Before the first real training run:

- [ ] `variables.py`: `SPEC`, `CONSTANTS`, `POLICY_LOWER/UPPER`, `N_SHOCKS`, `DESCRIPTION`
- [ ] `equations.py`: `definitions()`, `equations()`, `EQUATION_NAMES`. Residual form chosen on purpose.
- [ ] `dynamics.py`: `step()` handles 1-D and 2-D `shock`.
- [ ] `steady_state.py`: `steady_state()` and `init_state` (declarative or hand-written).
- [ ] `__init__.py`: `MODEL: ModelSpec` with `state_names`, `policy_names`, `equation_names`, `shock_names` all set.
- [ ] `models/__init__.py`: new model registered in `_MODELS`.
- [ ] `configs/<name>.yaml`: training recipe fixed.
- [ ] Test: Euler residual = 0 at steady state.
- [ ] Smoke train: 500 episodes, loss decreasing.
- [ ] Ergodic diagnostic: log₁₀|resid/u'(c)| below −2 on a serious run.
- [ ] Sanity check: against closed form, linearization, or published solution.
