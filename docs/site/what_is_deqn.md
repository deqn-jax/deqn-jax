# What is DEQN?

**The network is the decision-rule basis.** It plays the exact role Chebyshev
polynomials or splines play in a projection method — a flexible approximation of
the policy function $\pi(s)$ — but trained on collocation points drawn by
*simulating the model* (the ergodic set), not laid down on a fixed tensor grid.

That one substitution is the whole method. Everything below anchors it to the
solver you already use.

## The object we approximate

Take a recursive model: states $s_t$, controls $\pi_t$, equilibrium conditions
$r(s_t, \pi_t, s_{t+1}, \pi_{t+1}) = 0$ (Euler, FOCs, market clearing), and a
transition $s_{t+1} = g(s_t, \pi_t, \varepsilon_{t+1})$. The equilibrium is the
decision rule $\pi^\star(s)$ that makes the residual vanish **in expectation over
next-period shocks**:

$$\mathbb{E}_{\varepsilon}\!\left[\, r\bigl(s,\, \pi^\star(s),\, g(s, \pi^\star(s), \varepsilon),\, \pi^\star(g(s, \pi^\star(s), \varepsilon))\bigr)\right] = 0 .$$

DEQN approximates $\pi^\star(s)$ with a network $\mathcal{N}_\theta(s)$ and solves
for the weights $\theta$ that drive those residuals to zero across the states the
economy actually visits. Same target as perturbation, projection, and time
iteration — DEQN is the **global** member that scales in the state dimension and
keeps the kinks.

## Anchored to the method you know

=== "You know projection (Judd)"

    Same collocation idea, two swaps. The **network replaces the Chebyshev /
    spline basis** as the parameterization of $\pi(s)$, and the collocation points
    come from **simulating the ergodic set** instead of a fixed tensor grid. That
    second swap is why the **state dimension doesn't blow up the grid** — there is
    no grid. "Training" is just the inner solve for the basis coefficients.

=== "You know time iteration / PFI"

    Same fixed-point-on-the-policy logic, made **on-policy**: simulate a
    trajectory under the *current* network, then improve the network on the data
    it just generated, and iterate. The end-state of one episode seeds the next,
    so the training distribution converges onto the model's own ergodic support —
    exactly where the equilibrium residual must hold.

=== "You know perturbation (Dynare)"

    A **global, nonlinear** rule rather than a local Taylor expansion at the
    steady state — so **occasionally-binding constraints stay kinked** (ZLB,
    borrowing limits, irreversibility enter as Fischer–Burmeister complementarity
    residuals, not linearized away). And it **composes** with what you have: a
    first-order Blanchard–Kahn linearization — computed in-framework via QZ, or
    imported from Dynare — warm-starts and anchors the solve. DEQN extends
    perturbation; it doesn't ask you to throw it out.

## What you get out

<div class="grid cards" markdown>

-   :material-function-variant:{ .lg .middle } __A decision rule, not coefficients__

    ---

    A trained $\pi(s)$ you can call, simulate, and shock at any state — no
    re-solve per scenario. Consumption, labor, prices fall out of it.

-   :material-ruler-square-compass:{ .lg .middle } __Accuracy you'd quote__

    ---

    Reported as the distribution of **relative Euler errors (errREE)** on the
    ergodic path — the number you already put in a paper, not a black-box loss.

-   :material-cube-outline:{ .lg .middle } __State-dimension scaling__

    ---

    The network approximates a smooth function regardless of dimension, where
    dense-grid methods (VFI, projection) hit the curse of dimensionality past
    ~6–10 states.

-   :material-chart-bell-curve-cumulative:{ .lg .middle } __Kinks stay kinked__

    ---

    Occasionally-binding constraints, disaster / regime-switching expectations,
    rare-event pricing over the full shock distribution — all fit into the
    residual. No special cases.

</div>

!!! warning "Two honest limits — stated here, not in a footnote"

    DEQN-JAX is **alpha (v0.2.0)**, and like any nonlinear **global** solver it
    carries two limits a tenured skeptic should hear up front:

    - **A low residual is necessary but not sufficient.** DEQN can settle on the
      **wrong equilibrium branch**, and nothing in the framework enforces
      equilibrium *selection*. This is a multiplicity / selection gap — there is
      **no global analogue of the *local* Blanchard–Kahn saddle-path condition**.
      (BK is a linear, local determinacy criterion; do not read the global gap as
      "BK selection.") Always sanity-check the policy against a known benchmark
      where one exists.
    - **No certified error bounds.** Accuracy here is **measured** (the errREE
      distribution along the ergodic path), not a theorem. Quote the number;
      don't assume it.

    The **validated stack is deliberately small**: `adam` + an `mlp` (or
    `linear_plus_mlp`) + an `mse` residual + antithetic Monte-Carlo (or
    Gauss–Hermite) expectations. Everything else in the registries is a research
    instrument, not a turnkey recommendation.

## Going deeper

The mechanics below are reference detail — open them only if you're implementing
or debugging.

??? abstract "The training loop, in detail — the four-level nested loop"

    ![DEQN training cycle — conceptual](figures/deqn_conceptual.svg)

    DEQN is a four-level nested loop. Reading from outside in:

    - **CYCLE** — the outer iteration; run until the equilibrium residuals are small.
    - **SIMULATION** (one episode per cycle) — fills a trajectory by stepping the
      model under the *current* network policy.
    - **STEP** (one per timestep) — a forward pass gives the policy at the current
      state; the model dynamics produce the next state.
    - **TRAINING** (on the trajectory just simulated) — sweeps **EPOCH × BATCH**
      updates that adjust $\theta$ to drive the residuals toward zero.

    The end-state of the episode seeds the next cycle's start state — this is what
    makes the procedure **on-policy**: you simulate under the policy you have, then
    improve that policy on the data it just generated, and iterate.

    In code-level form, per cycle:

    1. **Simulate** a trajectory (or draw a rectangular batch of states).
    2. **Forward** the network at each state for $\pi = \mathcal{N}_\theta(s)$.
    3. **Step** under a sampled shock to get $s'$, forward again for $\pi'$.
    4. **Residual**: evaluate $r(s, \pi, s', \pi')$ and take the shock-expectation.
    5. **Loss + backprop**: square, mean over the batch, gradient step on $\theta$.
    6. Optionally sweep several minibatches before the next rollout.

    Repeat for $N$ cycles. Diagnostics during training: per-equation loss
    trajectories, gradient norms, policy plots against known benchmarks, and the
    ergodic Euler-error distribution at the end.

??? abstract "What you optimize — the loss, the expectation, the aggregation"

    The training loss is the mean squared residual, averaged over:

    1. **States** $s$ — either a bounded rectangle (exogenous / pedagogical) or
       simulated trajectories under the current policy (ergodic / on-policy).
    2. **Shocks** $\varepsilon$ — via Monte Carlo (antithetic sampling by default)
       or Gauss–Hermite quadrature when a small-node grid integrates the Gaussian
       shock accurately.

    Per batch element the loss is $\bigl(\mathbb{E}_\varepsilon[r]\bigr)^2$ —
    average over shocks, *then* square. That is the statistically correct target
    for conditions of the form "expected residual equals zero," and it avoids the
    Jensen-inequality bias that $\mathbb{E}[r^2]$ would introduce. (For the
    unbiased small-sample variant, see `loss_choice: aio` in the
    [Method Zoo](method-zoo/index.md).)

    For multi-equation models the per-equation losses aggregate as a mean across
    equations. Adaptive reweighting (`lr_annealing`, `relobralo`) and per-equation
    gradient surgery (`pcgrad`) are available when one loud equation drowns the
    rest — see [Running experiments](running_experiments.md) for the rationale and
    learning-rate implications.

??? abstract "Why it converges — and where it can fail silently"

    If the loss goes to zero, then (modulo sampling noise and network
    expressiveness) the residuals hold in expectation on the training
    distribution. An ergodic equilibrium *is* "residual $= 0$ on the ergodic
    support," so if the training distribution covers that support, the trained
    policy is a valid equilibrium policy. Two failure modes to watch:

    - **No accuracy guarantees at unsampled states.** Rect sampling covers a box
      but extrapolates poorly outside it; ergodic sampling concentrates on the
      attractor but undersamples tails. The ergodic errREE diagnostic is the
      standard post-hoc check.
    - **Training can fail silently.** Loss can fall while the policy is wrong if
      the residual has a degenerate local minimum (e.g. the `bm_labor` "savings
      rate at 0.9 with negative consumption" case). Treat low loss as
      necessary-but-not-sufficient; the [diagnostic cabinet](method-zoo/index.md#cabinet-diagnostic)
      exists precisely for this.

??? quote "ML ↔ economics dictionary"

    | The ML word | What it is, in your language |
    |---|---|
    | neural-network policy | a flexible approximation of $\pi(s)$ — the role Chebyshev / splines play in projection |
    | loss / training residual | the Euler / FOC / market-clearing error |
    | gradient descent / "training" | the inner solve for the approximation's coefficients |
    | on-policy sampling / minibatch | collocation points drawn by **simulating the model** (the ergodic set), not a fixed tensor grid |
    | expectation over shocks | Gauss–Hermite quadrature, or Monte Carlo with antithetic variates |
    | constraint penalty | a Fischer–Burmeister complementarity residual (irreversibility, borrowing limits, ZLB) |
    | "deep equilibrium net" | a global, nonlinear, high-dimensional recursive-equilibrium solver |
    | "converged" / low loss | small relative Euler errors (errREE) on the ergodic path — necessary, **not** sufficient |

## Where to next

<div class="grid cards" markdown>

-   :material-compare-horizontal:{ .lg .middle } __Method-by-method comparison__

    ---

    Perturbation, VFI, projection, PEA, PINN-HJB — and when to reach for DEQN
    over each.

    [:octicons-arrow-right-24: Overview](why.md)

-   :material-tune-variant:{ .lg .middle } __Pick your method__

    ---

    The swappable toolkit — networks, optimizers, expectations, diagnostics — and
    *when* (and when not) to reach for each.

    [:octicons-arrow-right-24: Method Zoo](method-zoo/index.md)

-   :material-image-multiple:{ .lg .middle } __See worked models__

    ---

    The constraint trilogy and a CMR-style NK-DSGE, each with its *measured*
    errREE certificate.

    [:octicons-arrow-right-24: Gallery](gallery/index.md)

-   :material-pencil-ruler:{ .lg .middle } __Write your own model__

    ---

    Declare states, equilibrium equations, transition, calibration — as data. The
    `ModelSpec` contract is the whole surface.

    [:octicons-arrow-right-24: Implementing a model](models/implementing.md)

</div>

??? quote "Lineage & attribution"
    DEQN-JAX is a JAX/Equinox reimplementation and extension of the **Deep
    Equilibrium Nets** method of **Azinovic, Gaegauf & Scheidegger (2022)**,
    building on the all-in-one / deep-learning Euler-error line of Maliar, Maliar &
    Winant. The method, the errREE accuracy metric, and the linear-anchor idea are
    theirs; this repo contributes the trainer, the optimizer/network cabinets, and
    the model library. Full references on the [home page](index.md).

