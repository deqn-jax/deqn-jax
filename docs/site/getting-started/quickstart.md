# Quickstart

This page trains the canonical model with the validated configuration and then
reads its accuracy as the relative-Euler-error (errREE) distribution on the
ergodic path.

The commands below use the combination that the test suite and the gallery
exercise: `adam`, an `MLP`, an `MSE` residual and antithetic Monte Carlo
expectations, on `brock_mirman`. The other options in the registries are
research tools; the [Method Zoo](../method-zoo/index.md) describes when to use
them.

## 0. Verify the install

```bash
uv sync
uv run deqn-jax check     # JAX backend, devices, registered models & optimizers
uv run deqn-jax list      # the registered models
```

??? abstract "Install details: source checkout, CUDA, editable mode"
    The alpha release is not on PyPI; install from a source checkout.

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
    `uv run`; do not activate the venv by hand.

## 1. Solve a model

`brock_mirman` is the teaching model: state $(k, z)$, one decision rule (the
savings rate), one consumption Euler equation and an analytical steady state.
It serves as the smoke test for an installation.

```bash
uv run deqn-jax train brock_mirman -n 1000 --warm-start \
    --checkpoint-dir checkpoints/brock_mirman
```

The residual loss should fall by several orders of magnitude in under a minute
on CPU. `--warm-start` first fits the network to the steady-state policy with
a supervised L-BFGS pre-fit, so training starts from an economically sensible
policy rather than from random weights. `--checkpoint-dir` saves the trained
policy so the next step can read it.

!!! tip "The same steps in economics terms"
    The network plays the role Chebyshev polynomials or splines play in a
    projection method: a flexible approximation of the decision rule $\pi(s)$.
    "Training" is the collocation/projection solve for its coefficients.
    "Minibatches" are collocation states drawn by simulating the model (the
    ergodic set), not a fixed tensor grid. The "loss" is the Euler residual,
    integrated over next-period shocks by antithetic Monte Carlo.

## 2. Read its accuracy

A low loss is necessary but not sufficient: like any nonlinear global solver,
residual minimization can converge to a wrong solution. Check the policy
itself. `evaluate` simulates a long ergodic path and reports the errREE
distribution, the standard accuracy measure (Azinovic et al. 2022).

```bash
uv run deqn-jax evaluate checkpoints/brock_mirman/checkpoint_best.eqx -n 10000
```

It also runs the market-clearing, simulated-moments and stability checks. The
config is read from the checkpoint directory. The [Gallery](../gallery/index.md)
has measured errREE for the worked models.

## 3. Train, evaluate, shock

=== "Train"

    ```bash
    uv run deqn-jax train brock_mirman -n 1000 --warm-start \
        --checkpoint-dir checkpoints/brock_mirman
    ```

    Any registered model works here, for example `bm_labor_constrained`,
    `irbc` or `olg_lifecycle`. Config-driven runs read a YAML file and accept
    dot-notation overrides:

    ```bash
    uv run deqn-jax train --config configs/brock_mirman.yaml \
        --set optimizer.learning_rate=0.001 \
        --checkpoint-dir checkpoints/brock_mirman
    ```

=== "Evaluate"

    ```bash
    uv run deqn-jax evaluate checkpoints/brock_mirman/checkpoint_best.eqx -n 10000
    ```

    Reports the errREE distribution, market-clearing errors, simulated moments
    and the stability check. The config is read from the checkpoint directory.

=== "Shock it"

    ```bash
    uv run deqn-jax irf checkpoints/brock_mirman/checkpoint_best.eqx --shock eps_z
    ```

    Impulse responses from a trained policy. `--girf` computes the generalized
    IRF for nonlinear models: it subtracts a no-shock baseline path from the
    same initial state, so the response is state-dependent. Run
    `deqn-jax info brock_mirman` for valid shock names.

## Where to next

- [Gallery](../gallery/index.md): closed-form examples, the three
  occasionally-binding-constraint models (`bm_labor_constrained`, `irbc`,
  `olg_lifecycle`) and an experimental NK-DSGE, each with its measured errREE.
- [Method Zoo](../method-zoo/index.md): the interchangeable networks,
  optimizers, expectation operators and diagnostics, and when to use each. The
  default recipe is at the top of the page.
- [Implementing a model](../models/implementing.md): declare states,
  equilibrium residuals, transition and calibration through the `ModelSpec`
  contract.

??? abstract "Resuming a checkpoint, and polishing with a Newton-type method"
    Any checkpoint can be resumed, including with a different optimizer. The
    intended use is the pipeline the [Method Zoo](../method-zoo/index.md)
    recommends when a first-order run plateaus: explore with `adam`, then
    polish with a Newton-type method. These are the quasi-Newton and
    Gauss-Newton methods familiar from GMM and MLE estimation, applied to the
    equilibrium residuals for quadratic convergence near a solution.

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

    The trainer detects the optimizer change, re-initializes the optimizer
    state for the new method and keeps the network weights. The original
    config is read from `<checkpoint_dir>/config.yaml` to rebuild the pytree
    template. `gn` and `lm` (Gauss-Newton, Levenberg-Marquardt) are the other
    Newton-type options. These polish steps are experimental; `adam` is the
    validated optimizer. When a run stalls, changing the network
    (`linear_plus_mlp`, the Blanchard-Kahn-anchored basis) helps more often
    than changing the optimizer.

??? warning "The disaster model is experimental"
    `disaster` (CMR-style NK-DSGE, 13 states / 11 policies, numerical steady
    state) is a stress test and not part of the validated configuration. The
    baseline block converges, but the disaster and financial-frictions block
    is still being validated. Its recipe, `LinearPlusMLP` plus the composite
    loss (anchor + Jacobian-match + barrier + Newton auxiliary terms), is also
    experimental. Treat its output as a research example.

    ```bash
    uv run deqn-jax train --config configs/disaster.yaml   # experimental
    ```

    Read the [gallery landing page](../gallery/index.md) and the
    [composite loss](../training/composite_loss.md) page before relying on any
    number it produces.

!!! warning "Limits"
    - A low residual does not identify the right equilibrium. Like any
      nonlinear global solver, DEQN can converge to the wrong branch, and the
      framework does not enforce equilibrium selection. There is no global
      analogue of the local Blanchard-Kahn saddle-path condition; BK is a
      linear, local determinacy criterion.
    - No certified error bounds. Accuracy is measured (the errREE
      distribution), not proven. Report the number you measured.
