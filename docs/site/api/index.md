# API reference

The Python API does what the CLI (`uv run deqn-jax train …`) does, from a
script: configure a run, register a model, train, and read back the
Euler-equation accuracy.

!!! note "The stable surface is `deqn_jax.api`"
    Everything on this page is re-exported from `deqn_jax.api`, the
    version-stable contract. Any change to that module is a breaking change.
    Anything imported from a deeper path (`deqn_jax.training.trainer`,
    `deqn_jax.networks.mlp`, …) is internal and may be refactored without
    notice; import from `deqn_jax.api` only.

    ```python
    from deqn_jax.api import (
        TrainConfig, NetworkConfig, OptimizerConfig,   # configure a run
        ModelSpec, register_model,                     # declare your model
        train_from_config,                             # solve it
        euler_equation_errors, print_euler_errors,     # read the accuracy
    )
    ```

---

## Building a run

Declare the model (`ModelSpec`), configure the run (`TrainConfig`), solve
(`train_from_config`). The loss is rarely set by hand.

- [Config](config.md): solver settings (not the model calibration), validated
  by Pydantic v2. `TrainConfig` nests `NetworkConfig` (the basis),
  `OptimizerConfig` (the inner solve) and `CompositeLossConfig`; the fields
  are the same ones YAML files and `--set` write.
- [Types](types.md): `ModelSpec` is the model contract (states, equilibrium
  residuals, transition, calibration, steady state). `TrainState` holds the
  mutable solve state (params, optimizer state, RNG) so the train step is a
  pure function. `Metrics` is what each step reports.
- [Trainer](trainer.md): `train_from_config(cfg)` runs the solve and returns
  `(policy_net, history)`. `create_train_state` and `make_train_step` expose the
  per-episode step for a custom outer loop.
- [Loss](loss.md): the conditional expectation over next-period shocks
  (antithetic Monte Carlo or Gauss–Hermite) of the Euler, FOC and
  market-clearing errors. The default (`mse`) needs changing only for stiff
  models.

!!! example "Smallest end-to-end solve"
    ```python
    from deqn_jax.api import (
        TrainConfig, NetworkConfig, OptimizerConfig,
        train_from_config, load_model,
        euler_equation_errors, print_euler_errors,
    )

    cfg = TrainConfig(
        model="brock_mirman",
        episodes=1000,
        network=NetworkConfig(type="mlp", hidden_sizes=(64, 64)),
        optimizer=OptimizerConfig(name="adam", learning_rate=1e-3),
    )
    policy_net, history = train_from_config(cfg)       # the global solve

    diag = euler_equation_errors(policy_net, load_model("brock_mirman"))
    print_euler_errors(diag)                           # the errREE you'd quote
    ```
    `adam` + `mlp` + MSE residual + antithetic MC is the validated stack. The
    registries below are for cases where it is not enough; a new model
    usually needs none of them.

---

## Registries

Three registries, queried at runtime and selected by name. When to use each
entry is covered in the [Method Zoo](../method-zoo/index.md).

- [Models](models.md): `load_model(name)`, `list_models()`, and
  `register_model(spec)`, which adds a model at runtime without editing the
  package source. Twelve models are registered: the Brock–Mirman teaching
  family, the occasionally-binding examples (`bm_labor_constrained`, `irbc`,
  `olg_lifecycle`), the experimental `disaster` NK-DSGE, and the
  `rss_trade_ez_ref` reference replica.
- [Networks](networks.md): the decision-rule basis (the role of Chebyshev
  polynomials or splines in projection). Details in the table below.
- [Optimizers](optimizers.md): the inner solve. `create_optimizer(config)`
  resolves a name from the registry of 9; `list_optimizers()` is the source
  of truth.

??? abstract "The 9 registered optimizers (`uv run deqn-jax optimizers`)"
    The live registry is the canonical list. Status and when to use each are
    in the [Method Zoo optimizer section](../method-zoo/index.md#cabinet-optimizer).

    | Name | Family | Status |
    |---|---|---|
    | `adam` | first-order (STANDARD) | validated; the default |
    | `gn`, `ign`, `lm` | Gauss-Newton / Levenberg-Marquardt | experimental; Newton-style refinement (as in GMM/MLE) |
    | `lbfgs` | quasi-Newton | experimental; also the steady-state warm-start engine |
    | `mao` | multi-equation (per-equation moments) | experimental |
    | `muon`, `ngd`, `shampoo` | deep-learning optimizers | experimental; macro models rarely need these |

    `mao` resolves its task count (one moment per equilibrium equation) when
    the train state is built, once the model's equation count is known.

??? abstract "Registered networks (`NetworkConfig.type`)"
    | `type` | Status | Role |
    |---|---|---|
    | `mlp` | validated default | flexible Markov-policy basis |
    | `linear_plus_mlp` | validated | BK linear rule + zero-init MLP correction; the policy equals the BK solution at init |
    | `lstm`, `transformer` | experimental | history-dependent (sequence) policies |
    | `disaster_policy_net` | experimental | LinearPlusMLP + CMR-specific shape priors; not general-purpose |
    | `rss_market_clearing_net` | model-specific | the `rss_trade_ez_ref` policy network, kept for checkpoint parity with the reference solution |

    The classes (`MLP`, `LSTMPolicy`, `TransformerPolicy`, `LinearPlusMLP`) and
    their `create_*` factories are exported from `deqn_jax.api` for the manual
    `create_train_state` / `make_train_step` path. Most runs only set
    `NetworkConfig.type`.

??? abstract "Twelve registered models (`uv run deqn-jax list`)"
    | Name | Tier | What it shows |
    |---|---|---|
    | `brock_mirman` (+ `bm_deterministic`, `bm_labor`, two `*_autodiff` POCs) | canonical / teaching | state `(k, z)`, one policy `sav_rate`, one Euler equation, analytical SS; the 5-minute smoke test |
    | `bm_labor_constrained` | example | smallest occasionally-binding demo (labor cap via Fischer–Burmeister) |
    | `irbc` | example | 2-country irreversibility (Fischer–Burmeister), Gauss–Hermite expectation |
    | `olg_lifecycle` (+ `olg_analytic_6` closed-form check, `olg_lifecycle_56` annual 56-generation variant) | example | 6-generation borrowing constraints, two-stage loss |
    | `disaster` | experimental | NK-DSGE / CMR, 13 states, 11 policies, numerical SS, under validation |
    | `rss_trade_ez_ref` | reference replica | RSS-2019 three-country trade DSGE with Epstein-Zin preferences, reference layout (31 states / 73 policies / 82 residuals; two-stage loss) |

---

## Other tools on `deqn_jax.api`

??? note "Evaluation, IRF, and the steady-state / autodiff helpers"
    A low residual is necessary but not sufficient: the solve can settle on a
    wrong equilibrium branch, and nothing here enforces selection. These tools
    check a solution.

    - Accuracy and verification: `euler_equation_errors` (errREE),
      `market_clearing_errors`, `simulated_moments`, `stability_check`, and
      the printers `print_euler_errors` / `print_moments`.
    - Impulse responses from a checkpoint: `run_irf`, `run_girf`,
      `load_policy_from_checkpoint`, `save_irf_csv`, `print_irf_summary`.
    - Steady state and code generation: `solve_steady_state` /
      `verify_steady_state` (L-BFGS fallback when no analytical SS exists,
      with per-equation residuals to check), and `euler_from_period_return`,
      which builds the Euler residual from a scalar period return via
      `jax.grad` (used by the `*_autodiff` models).

    See [Diagnostics](../method-zoo/index.md#cabinet-diagnostic) for what each
    number tells you, and the [Gallery](../gallery/index.md) for worked models
    with their measured errREE.

!!! tip "Building on deqn-jax? Read REFERENCE first"
    The [ModelSpec reference](../REFERENCE.md) is the type-signature-first
    contract: the `deqn_jax.api` surface, every `ModelSpec` field, the
    `register_model(...)` path and the verification gates. The per-module
    pages ([Config](config.md) · [Types](types.md) · [Trainer](trainer.md) ·
    [Loss](loss.md) · [Models](models.md) · [Networks](networks.md) ·
    [Optimizers](optimizers.md)) are generated by mkdocstrings from
    docstrings and include internal modules.

---

*A JAX/Equinox reimplementation and extension of **Deep Equilibrium Nets**
(Azinovic, Gaegauf & Scheidegger 2022; Scheidegger & Bilionis 2019). All credit
for the original method belongs to the upstream authors.*
