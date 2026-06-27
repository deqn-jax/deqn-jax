# Quickstart

Train the canonical model with the **validated stack**, then read its accuracy
the way you'd report it in a paper — the relative-Euler-error (errREE)
distribution on the ergodic path.

!!! note "Status: alpha (v0.2.0) — the validated stack is small"
    Everything below leads with the combination the test suite and gallery
    actually exercise: `adam` + an `MLP` + an `MSE` residual + antithetic
    Monte-Carlo expectations, on `brock_mirman`. Everything else in the
    registries is a research instrument, not a turnkey recommendation — see the
    [Method Zoo](../method-zoo/index.md) for *when* (and when not) to reach for it.

## 0. Verify the install

```bash
uv sync
uv run deqn-jax check     # JAX backend, devices, registered models & optimizers
uv run deqn-jax list      # the registered models
```

??? abstract "Install detail — source checkout, CUDA, editable mode"
    Alpha is not yet on PyPI; install from a source checkout.

    ```bash
    git clone <repo>
    cd deqn-jax
    uv sync
    uv pip install -e .            # optional: editable mode for hacking
    ```

    GPU build (Linux aarch64 / x86_64):

    ```bash
    uv pip install -U "jax[cuda13]"   # or "jax[cuda12]" for CUDA 12
    ```

    `uv run deqn-jax check` reports the active backend and devices. Always use
    `uv run`; never activate the venv by hand.

## 1. Solve a model in five minutes

`brock_mirman` is the canonical/teaching tier: state $(k, z)$, one decision rule
(the savings rate), one consumption Euler equation, an **analytical** steady
state. It is the smoke test that proves the stack works on your machine.

```bash
uv run deqn-jax train brock_mirman -n 1000 --warm-start \
    --checkpoint-dir checkpoints/brock_mirman
```

You should see the residual loss fall several orders of magnitude in under a
minute on CPU. `--warm-start` fits the network to the steady-state policy first
(an L-BFGS supervised pre-fit), so training begins from a sane economic guess
rather than noise; `--checkpoint-dir` is what lets the next step read the trained
policy back.

!!! tip "What just happened, in your language"
    The network plays the role Chebyshev polynomials or splines play in a
    projection method — a flexible approximation of the decision rule $\pi(s)$.
    "Training" is the collocation/projection solve for its coefficients;
    "minibatches" are collocation states drawn by **simulating the model** (the
    ergodic set), not a fixed tensor grid; the "loss" is the Euler residual,
    integrated over next-period shocks by antithetic Monte Carlo.

## 2. Read its accuracy

A low loss is **necessary but not sufficient** — like any nonlinear global
solver, residual-minimization can land on the wrong answer. So you don't trust
the loss; you check the policy. `evaluate` simulates a long ergodic path and
reports the errREE distribution — the gold-standard accuracy metric (Azinovic et
al. 2022), the number you'd quote.

```bash
uv run deqn-jax evaluate checkpoints/brock_mirman/checkpoint_best.eqx -n 10000
```

It also runs the market-clearing, simulated-moments, and stability checks; the
config is auto-detected from the checkpoint directory. For *measured* errREE
certificates on worked models, see the **[Gallery](../gallery/index.md)** — the
evidence, not a promise.

## 3. The everyday loop

=== "Train"

    ```bash
    uv run deqn-jax train brock_mirman -n 1000 --warm-start \
        --checkpoint-dir checkpoints/brock_mirman
    ```

    Swap in any registered model: `bm_labor_constrained`, `irbc`,
    `olg_lifecycle`. Config-driven runs read a YAML and accept dot-notation
    overrides:

    ```bash
    uv run deqn-jax train --config configs/brock_mirman.yaml \
        --set optimizer.learning_rate=0.001 \
        --checkpoint-dir checkpoints/brock_mirman
    ```

=== "Evaluate"

    ```bash
    uv run deqn-jax evaluate checkpoints/brock_mirman/checkpoint_best.eqx -n 10000
    ```

    The errREE distribution, market-clearing errors, simulated moments, and the
    stability gate. The config is auto-detected from the checkpoint directory.

=== "Shock it"

    ```bash
    uv run deqn-jax irf checkpoints/brock_mirman/checkpoint_best.eqx --shock eps_z
    ```

    Impulse responses from a trained policy. `--girf` gives the generalized
    (state-dependent, no-shock-baseline-subtracted) variant for nonlinear models.
    Run `deqn-jax info brock_mirman` for valid shock names.

## Where to next

<div class="grid cards" markdown>

-   :material-image-multiple:{ .lg .middle } __See the sell, measured__

    ---

    Closed-form pedagogy → the occasionally-binding constraint trilogy
    (`bm_labor_constrained`, `irbc`, `olg_lifecycle`) → an experimental NK-DSGE,
    each with its measured errREE certificate.

    [:octicons-arrow-right-24: Gallery](../gallery/index.md)

-   :material-tune-variant:{ .lg .middle } __Pick your method__

    ---

    The swappable toolkit — networks, optimizers, expectations, diagnostics —
    and *when* to reach for each. The default recipe is on the first screen.

    [:octicons-arrow-right-24: Method Zoo](../method-zoo/index.md)

-   :material-pencil-ruler:{ .lg .middle } __Write your own model__

    ---

    Declare states, equilibrium residuals, transition, calibration — as data.
    The `ModelSpec` contract is the whole surface.

    [:octicons-arrow-right-24: Implementing a model](../models/implementing.md)

</div>

??? abstract "Resume a checkpoint — and the Adam → Newton-style polish"
    Any checkpoint resumes, including with a **different optimizer**. The
    legitimate use is the pipeline the [Method Zoo](../method-zoo/index.md)
    routes you to when a first-order run *plateaus*: rough exploration with
    `adam`, then a **Newton-style polish** — the same quasi-Newton / Gauss-Newton
    machinery you know from GMM / MLE estimation, applied to the equilibrium
    residuals for quadratic convergence *near* a solution.

    ```bash
    # Rough exploration with Adam (the validated first-order method)
    uv run deqn-jax train brock_mirman -n 1000 --warm-start \
        --checkpoint-dir checkpoints/brock_mirman

    # Polish from the checkpoint with L-BFGS (Newton-style; experimental)
    uv run deqn-jax train brock_mirman -n 200 \
        --resume checkpoints/brock_mirman/checkpoint_best.eqx \
        --set optimizer.name=lbfgs \
        --checkpoint-dir checkpoints/brock_mirman
    ```

    The trainer detects the optimizer change, re-initializes optimizer state for
    the new method, and keeps the network weights; the original config is read
    from `<checkpoint_dir>/config.yaml` to reconstruct the pytree template.
    `gn` / `lm` (Gauss-Newton, Levenberg-Marquardt) are the other Newton-style
    members. These are experimental polish steps — `adam` remains the validated
    workhorse, and a stall is more often a *network* fix (`linear_plus_mlp`, the
    Blanchard-Kahn-anchored basis) than an optimizer one.

??? warning "The disaster model is experimental — under validation"
    `disaster` (CMR-style NK-DSGE, 13 states / 11 policies, numerical steady
    state) is the stress test, **not** part of the validated stack. The baseline
    block converges, but the disaster/financial-frictions block is still under
    validation, and the recipe it leans on — `LinearPlusMLP` plus the
    **composite loss** (anchor + Jacobian-match + barrier + Newton auxiliary
    terms) — is itself experimental. Treat it as a research example, not a
    turnkey result.

    ```bash
    uv run deqn-jax train --config configs/disaster.yaml   # experimental
    ```

    See the [gallery landing](../gallery/index.md) and the
    [composite loss](../training/composite_loss.md) note before trusting any
    number it produces.

!!! warning "Two honest limits — stated here, not buried"
    - **A low residual does not pin down the right equilibrium.** Like any
      nonlinear *global* solver, DEQN can settle on the wrong **branch**, and
      nothing here enforces equilibrium *selection*. There is no global analogue
      of the *local* Blanchard-Kahn saddle-path condition — BK is a linear/local
      determinacy criterion, not a global one.
    - **No certified error bounds.** Accuracy is **measured** (the errREE
      distribution), not proven by a theorem. Quote the number; don't assume it.

