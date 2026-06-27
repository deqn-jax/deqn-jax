# API reference

**The import surface for driving a solve from Python** (alpha, v0.2.0). If the
CLI (`uv run deqn-jax train …`) is how you *run* a model, this is how you
*embed* one — configure a run, register your own model, train, and read its
Euler-equation accuracy back, all from a script or an agent stack.

!!! note "One stable surface — `deqn_jax.api`"
    Everything on this page re-exports from **`deqn_jax.api`**, the curated,
    version-stable contract. A change to anything in that module is a breaking
    change. **Anything imported from a deeper path** (`deqn_jax.training.trainer`,
    `deqn_jax.networks.mlp`, …) **is internal and may be refactored without
    notice** — the autodoc pages below show that internal depth for contributors
    and codegen, but you should import from `deqn_jax.api` only.

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

The imperative path, in the order you touch it: **declare the model
(`ModelSpec`) → configure the run (`TrainConfig`) → solve (`train_from_config`) →
score the residual (the loss you almost never set by hand).** Four imports get a
policy trained and certified.

<div class="grid cards" markdown>

-   :material-file-document-edit-outline:{ .lg .middle } __Config__

    ---

    Your run card — the calibration of the *solver*, not the model. Pydantic v2,
    fully validated: `TrainConfig` plus the nested `NetworkConfig` (which basis),
    `OptimizerConfig` (which inner solve), `CompositeLossConfig`. Same fields the
    YAML and `--set` overrides write into; build it in Python, pass it once.

    [:octicons-arrow-right-24: Config](config.md)

-   :material-shape-outline:{ .lg .middle } __Types__

    ---

    `ModelSpec` is **the whole contract** — states, equilibrium residuals,
    transition, calibration, steady state, as data. `TrainState` bundles the
    mutable solve (params, optimizer state, RNG) so the train step stays a pure
    function; `Metrics` is what each step reports.

    [:octicons-arrow-right-24: Types](types.md)

-   :material-cog-play-outline:{ .lg .middle } __Trainer__

    ---

    `train_from_config(cfg)` runs the whole solve and hands back
    `(state, history)`. For a custom outer loop, `create_train_state` and
    `make_train_step` expose the single-JIT step. The collocation / projection
    *solve*, in ML clothing.

    [:octicons-arrow-right-24: Trainer](trainer.md)

-   :material-function-variant:{ .lg .middle } __Loss__

    ---

    How the equilibrium residual is scored: the conditional expectation over
    next-period shocks (antithetic Monte-Carlo or Gauss–Hermite) of your
    Euler / FOC / market-clearing error. The default (`mse`) is wired for you —
    reach in only when a model is stiff.

    [:octicons-arrow-right-24: Loss](loss.md)

</div>

!!! success "Smallest end-to-end solve"
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
    state, history = train_from_config(cfg)            # the global solve

    diag = euler_equation_errors(state.params, load_model("brock_mirman"))
    print_euler_errors(diag)                           # the errREE you'd quote
    ```
    `adam` + `mlp` + MSE residual + antithetic-MC — the **validated stack**. The
    registries below exist for when this isn't enough; on a new model you touch
    almost none of it.

---

## The registries — what you pick *from*

Three menus, queried live, swapped by name. The **why** and **when** for each
item lives in the [Method Zoo](../method-zoo/index.md); this is the *what's
registered* and *what to import*.

<div class="grid cards" markdown>

-   :material-database-outline:{ .lg .middle } __Models__

    ---

    `load_model(name)`, `list_models()`, and `register_model(spec)` — add a model
    **programmatically**, no edit to the package source. Ten registered today:
    the Brock–Mirman teaching family, the occasionally-binding trio
    (`bm_labor_constrained`, `irbc`, `olg_lifecycle`), and the experimental
    `disaster` NK-DSGE.

    [:octicons-arrow-right-24: Models](models.md)

-   :material-vector-polyline:{ .lg .middle } __Networks__

    ---

    The decision-rule basis — the role Chebyshev/splines play in projection.
    `mlp` (validated default) and `linear_plus_mlp` (a zero-init MLP correction
    on a Blanchard–Kahn linear rule: at init the policy *is* the BK solution).
    `lstm` / `transformer` are experimental sequence policies.

    [:octicons-arrow-right-24: Networks](networks.md)

-   :material-tune:{ .lg .middle } __Optimizers__

    ---

    The inner solve. `create_optimizer(config)` resolves a name from the
    registry of **13**. `adam`/`adamw`/`sgd` are validated; `gn`/`ign`/`lm`/`lbfgs`
    are the Newton-style polish you know from GMM/MLE; `mao`/`mao_kfac` are
    multi-equation. `list_optimizers()` is the source of truth.

    [:octicons-arrow-right-24: Optimizers](optimizers.md)

</div>

??? abstract "The 13 registered optimizers (`uv run deqn-jax optimizers`)"
    The canonical list is always the live registry. Status and *when to reach for
    it* are in the [Method Zoo optimizer cabinet](../method-zoo/index.md#cabinet-optimizer).

    | Name | Family | Status |
    |---|---|---|
    | `adam` | first-order (STANDARD) | **validated — the default** |
    | `adamw` | first-order | validated |
    | `sgd` | first-order | validated |
    | `gn`, `ign`, `lm` | Gauss-Newton / Levenberg-Marquardt | experimental — Newton-style polish (anchor to GMM/MLE) |
    | `lbfgs` | quasi-Newton | experimental — also the steady-state warm-start engine |
    | `mao`, `mao_kfac` | multi-equation (per-equation moments) | experimental |
    | `lion`, `muon`, `ngd`, `shampoo` | deep-learning optimizers | experimental — *a macro model won't need these* |

    `mao_kfac` resolves its task count (one moment per equilibrium equation) at
    train-state construction, when the model's equation count is known.

??? abstract "Networks registered (`NetworkConfig.type`)"
    | `type` | Status | One-line role |
    |---|---|---|
    | `mlp` | **validated default** | flexible Markov-policy basis |
    | `linear_plus_mlp` | validated | BK linear rule + zero-init MLP correction; policy *is* the BK solution at init |
    | `lstm`, `transformer` | experimental | history-dependent (sequence) policies |
    | `disaster_policy_net` | experimental | LinearPlusMLP + CMR-specific shape priors; not general-purpose |
    | `kf_anchored_mlp` | legacy | earlier gauge fix, superseded by `disaster_policy_net` |

    The classes (`MLP`, `LSTMPolicy`, `TransformerPolicy`, `LinearPlusMLP`) and
    their `create_*` factories are exported from `deqn_jax.api` for the rare manual
    `create_train_state` / `make_train_step` path — most runs only ever set
    `NetworkConfig.type`.

??? abstract "Ten registered models (`uv run deqn-jax list`)"
    | Name | Tier | What it shows |
    |---|---|---|
    | `brock_mirman` (+ `bm_deterministic`, `bm_labor`, two `*_autodiff` POCs) | canonical / teaching | state `(k, z)`, one policy `sav_rate`, one Euler eq, analytical SS — the 5-minute smoke test |
    | `bm_labor_constrained` | example | smallest **occasionally-binding** demo (labor cap via Fischer–Burmeister) |
    | `irbc` | example | 2-country irreversibility (Fischer–Burmeister), Gauss–Hermite expectation |
    | `olg_lifecycle` (+ `olg_analytic_6` closed-form check) | example | 6-generation borrowing constraints, two-stage loss |
    | `disaster` | experimental | NK-DSGE / CMR, 13 states, 11 policies, numerical SS, under validation |

---

## Beyond the run — also on `deqn_jax.api`

??? note "Evaluation, IRF, and the steady-state / autodiff helpers"
    The same stable surface carries the verification and inspection tools — a low
    residual is **necessary, not sufficient** (it can pin a wrong equilibrium
    branch, and nothing here enforces selection), so these are first-class:

    - **Accuracy & verification:** `euler_equation_errors` (errREE),
      `market_clearing_errors`, `simulated_moments`, `stability_check`, plus
      `print_euler_errors` / `print_moments` pretty-printers.
    - **Impulse responses from a checkpoint:** `run_irf`, `run_girf`,
      `load_policy_from_checkpoint`, `save_irf_csv`, `print_irf_summary`.
    - **Steady state & codegen backbone:** `solve_steady_state` /
      `verify_steady_state` (L-BFGS fallback when no analytical SS exists, with
      per-equation residuals to gate on), and `euler_from_period_return` —
      synthesizes the Euler residual from a scalar period-return via `jax.grad`
      (the `*_autodiff` models' backbone).

    See [Diagnostics](../method-zoo/index.md#cabinet-diagnostic) for *what each
    number tells you* and the [Gallery](../gallery/index.md) for worked models
    with their measured errREE certificates.

!!! tip "Building on deqn-jax? Read REFERENCE first"
    For the type-signature-first contract — the full stable `deqn_jax.api`
    surface, every `ModelSpec` field, the programmatic `register_model(...)` path,
    and the verification gates — start with the
    [ModelSpec reference](../REFERENCE.md). The per-module autodoc below
    ([Config](config.md) · [Types](types.md) · [Trainer](trainer.md) ·
    [Loss](loss.md) · [Models](models.md) · [Networks](networks.md) ·
    [Optimizers](optimizers.md)) is mkdocstrings-rendered from source docstrings
    and intentionally shows internal depth — treat anything outside
    `deqn_jax.api` as internal.

---

*A JAX/Equinox reimplementation and extension of **Deep Equilibrium Nets**
(Azinovic, Gaegauf & Scheidegger 2022; Scheidegger & Bilionis 2019). All credit
for the original method belongs to the upstream authors.*

