# What is DEQN?

A one-page introduction to the method, for economists who have not trained a neural network before. If you already know DEQN and want to know whether this framework is right for your problem, skip to [Overview](why.md).

## The object we're approximating

Take a recursive DSGE model: states $s_t$, controls $\pi_t$, equilibrium conditions $r(s_t, \pi_t, s_{t+1}, \pi_{t+1}) = 0$ (Euler, FOCs, market clearing), and a transition law $s_{t+1} = g(s_t, \pi_t, \varepsilon_{t+1})$.

The equilibrium is a **policy function** $\pi^\star(s)$ that satisfies
$$\mathbb{E}_{\varepsilon}\left[\, r\bigl(s,\, \pi^\star(s),\, g(s, \pi^\star(s), \varepsilon),\, \pi^\star(g(s, \pi^\star(s), \varepsilon))\bigr)\right] = 0$$
for every $s$ in the state space.

DEQN approximates $\pi^\star(s)$ with a **neural network** $\mathcal{N}_\theta(s)$ whose weights $\theta$ are trained so that the equilibrium residuals are close to zero across the state space.

That is the whole idea. A neural network is a smooth, flexible function that is cheap to evaluate and cheap to differentiate — the same two things that a good policy approximation needs.

## What we optimize

The training loss is the mean squared residual, averaged over:

1. **States** $s$ drawn from a distribution — either a bounded rectangle (exogenous / pedagogical) or simulated trajectories under the current policy (ergodic / on-policy).
2. **Shocks** $\varepsilon$ — via Monte Carlo (antithetic sampling by default) or Gauss–Hermite quadrature when the shock distribution is Gaussian enough for a small-node grid to integrate exactly.

Per batch element the loss is $\bigl(\mathbb{E}_\varepsilon[r]\bigr)^2$ (average over shocks, then square) — which is the statistically correct target for conditions of the form "expected residual equals zero," and avoids the Jensen-inequality bias that $\mathbb{E}[r^2]$ would introduce.

For multi-equation models the losses are aggregated as a mean across equations (matches DEQN-MAO convention; see `docs/running_experiments.md` for the rationale and LR implications).

## The training loop

![DEQN training cycle — conceptual](figures/deqn_conceptual.svg)

DEQN is a four-level nested loop. Reading from outside in:

- **CYCLE** — the outer iteration; you run this until the equilibrium residuals are small.
- **SIMULATION** (one episode per cycle) — fills a trajectory by stepping the model under the *current* network policy.
- **STEP** (one per timestep in the episode) — a forward pass of the network gives the policy at the current state, then the model dynamics produce the next state.
- **TRAINING** (on the trajectory just simulated) — sweeps **EPOCH × BATCH** updates that adjust the network parameters to drive the equilibrium residuals toward zero.

The end-state of the episode seeds the next cycle's starting state, which is what makes the procedure *on-policy*: you simulate under the policy you currently have, then improve that policy on the data it just generated, and iterate.

In code-level form:

1. **Simulate** a trajectory or draw a rect batch of states.
2. **Forward** the network at each state to get a candidate policy $\pi = \mathcal{N}_\theta(s)$.
3. **Step** the model forward under a sampled shock to get $s'$, then forward again for $\pi'$.
4. **Residual**: evaluate $r(s, \pi, s', \pi')$. Take the shock-expectation.
5. **Loss + backprop**: square, mean over batch, gradient step on $\theta$.
6. Optionally: sweep several minibatches of the trajectory before the next rollout.

Repeat for $N$ cycles. Diagnostics during training: per-equation loss trajectories, gradient norms, policy plots against known benchmarks, ergodic Euler-error distribution at the end.

## Why it works

The convergence argument is straightforward: if the loss goes to zero, then (modulo sampling noise and expressiveness of the network) the residuals hold in expectation on the training distribution. Ergodic equilibrium = residual = 0 on the ergodic support; if the network's training distribution covers the ergodic support, the trained policy is a valid equilibrium policy.

Two caveats worth stating up front:

1. **No accuracy guarantees at unsampled states.** A neural network interpolates, but its accuracy away from the training samples is empirical. Rect sampling covers a box but extrapolation outside is unreliable; ergodic sampling concentrates on the attractor but leaves tails undersampled. Ergodic Euler-error diagnostics (in [evaluate.py](running_experiments.md#cli-quickstart)) are the standard post-hoc check.
2. **Training can fail silently.** Loss curves can decrease while the policy is wrong if the residual has a degenerate local minimum (see the bm_labor "savings rate at 0.9 with negative consumption" story in the `brock_mirman` notebook parity work). Treat low loss as necessary-but-not-sufficient; always sanity-check the policy against a known benchmark where one exists.

## What it gives you that traditional methods don't

- **A policy function**, not a vector of coefficients. Simulate arbitrary paths without re-solving.
- **Global accuracy**, not a linearization around the steady state.
- **Native handling of nonlinearities and kinks**: occasionally-binding constraints (ZLB, borrowing), Fischer-Burmeister complementarity, disaster/regime-switching expectations — all fit into the residual. No special cases.
- **Correct rare-event pricing**: expectations integrate over the full shock distribution, including tails.
- **State-dimension scaling**: the network approximates a smooth function regardless of dimension, where dense-grid methods (VFI, projection) hit curse-of-dimensionality past ~6–10 states.

## What it doesn't give you

- **No analytic error bounds.** You can quantify accuracy via log₁₀|residual| distributions along the ergodic path, but there is no theorem that says "this approximation is within ε of the true policy."
- **Training is not deterministic.** Different seeds / curricula can converge to slightly different policies. Variance across seeds is part of the diagnostic story.
- **Steady-state solves are still needed upstream** for warm-start, composite-loss linearization, and IRFs. DEQN doesn't replace SS-finding; it follows it.

## Cross-method context

See [Overview](why.md) for a method-by-method comparison (perturbation, VFI, projection, PEA, PINN-HJB) and when to reach for DEQN over each.
