# Composite Loss

The composite loss adds auxiliary regularization terms on top of the base MSE (Euler equation residuals). All auxiliary terms are prefixed `aux_` so that MAO/PCGrad/ReLoBRaLo only see the base equilibrium equations.

Toggle via `loss_type: "composite"` in the config (default `"mse"` is unchanged).

## Base loss (unchanged)

Standard DEQN: for each batch of states, evaluate all equilibrium equations (11 for the disaster model) under stochastic shocks (Gauss-Hermite quadrature or Monte Carlo), square the mean residuals, sum them. This is what `compute_loss()` returns.

## Anchor loss (`aux_anchor`)

**What**: `||f_net(x) - f_linear(x)||^2` averaged over 64 fixed points near steady state.

**Why**: The Blanchard-Kahn (BK) linearization gives the exact first-order policy rule near SS: `policy = ss_policy + P * (state - ss_state)`. The anchor loss penalizes the network for deviating from this known-correct local solution.

**How**: At setup time, 64 points are sampled from `N(ss, sigma^2 * Sigma_ergodic)` where `Sigma_ergodic` is the ergodic state covariance (solved via discrete Lyapunov equation `Sigma = Q Sigma Q' + B B'`). These points are fixed for the entire run (no per-step randomness, no gradient noise). Each step evaluates `vmap(policy_fn)(anchor_points)` and compares to the precomputed linear policy at those points.

**Decay**: Weight is multiplied by `max(0, 1 - shock_scale)`. During curriculum (shock_scale ramps 0.1 to 1.0), anchor fades from 90% to 0%. After curriculum, it contributes zero to the loss, and the network is free to learn nonlinear corrections far from SS.

## Jacobian loss (`aux_jac`)

**What**: `||J_net(ss) - P||^2_F` -- Frobenius norm of the difference between the network's Jacobian at SS and the BK policy rule matrix P.

**Why**: Even if the network matches the linear policy at the 64 anchor points, its derivatives might be wrong. This ensures the network has the correct first-order sensitivity to state perturbations at SS.

**How**: `jax.jacfwd(policy_fn)(ss_state)` gives a [9x13] Jacobian, compared element-wise to P [9x13].

**Decay**: Same as anchor -- fades with curriculum.

## Barrier losses (`aux_barrier_n`, `aux_barrier_L`, `aux_barrier_c`)

**What**: Soft penalties that prevent economically infeasible regions.

- `barrier_n`: `mean(max(0, -log(n))^2)` -- penalizes net worth n < 1 (approaching zero means bank insolvency)
- `barrier_L`: `mean((max(0, L - 5*L_ss) / L_ss)^2)` -- penalizes leverage exceeding 5x steady state
- `barrier_c`: `mean(max(0, -log(c))^2)` -- penalizes consumption c < 1 (approaching zero triggers habit formation singularity)

**Why**: The consumption Euler and entrepreneur contract blow up when c approaches 0 or n approaches 0. Rather than relying on soft floors inside the equations (which kill gradients), the barriers provide smooth gradient signal pushing the network away from dangerous regions.

**How**: Evaluated by `vmap(definitions_fn)` over the training batch. Note: this is currently a redundant forward pass since the base loss already evaluates definitions internally (TODO to fix by having `compute_loss` return intermediate defs).

**No decay**: Barriers stay active for the entire run. Feasibility always matters.

## Newton losses (`aux_newton_cond`, `aux_newton_resid`)

**What**: Diagnostics for the omega_bar Newton solver inside `definitions()`.

- `newton_cond`: `mean(max(0, 0.1 - h'(omega))^2)` -- penalizes ill-conditioned solver (h' near zero means Newton diverges)
- `newton_resid`: `mean(newton_residual^2)` -- penalizes high solver residual (Newton didn't converge)

**Why**: omega_bar is solved analytically via 10 fixed Newton iterations in `definitions()`. If the network drives states into regions where the Newton solver doesn't converge (h' near 0), omega_bar is garbage and everything downstream (n, L, c) is wrong. These losses provide gradient signal pushing the network away from such regions.

**No decay**: Always active.

## Wiring

```
train_from_config()
  -> linearize_model()             # BK decomposition: P [9x13], Q [13x13]
  -> prepare_composite_data()      # ergodic covariance, anchor points, SS leverage
  -> make_composite_loss()         # returns closure matching compute_loss() signature
  -> make_train_step(..., compute_loss_fn=custom_loss_fn)
       # All 5 step variants (standard/pcgrad/mao/lbfgs/gn) use the custom loss
```

## Default weights

| Term | Weight | Decays with curriculum? |
|------|--------|------------------------|
| anchor | 0.1 | Yes |
| jac | 0.01 | Yes |
| barrier_n | 0.01 | No |
| barrier_L | 0.01 | No |
| barrier_c | 0.01 | No |
| newton_cond | 0.01 | No |
| newton_resid | 0.01 | No |

## TensorBoard

Base equation residuals log to `eq/<name>`, auxiliary terms log to `aux/<name>`. ReLoBRaLo adaptive reweighting only operates on the `eq/` terms.

## Config example

```yaml
loss_type: composite
composite_loss:
  anchor_weight: 0.1
  jac_weight: 0.01
  barrier_weight: 0.01
  newton_weight: 0.01
  n_anchor_points: 64
  anchor_sigma: 1.0
  leverage_mult: 5.0
```

## Files

- `src/deqn_jax/training/composite_loss.py` -- `CompositeData`, `prepare_composite_data()`, `make_composite_loss()`
- `src/deqn_jax/training/linearize.py` -- `linearize_model()`, `compute_ergodic_covariance()`
- `src/deqn_jax/config/loss.py` -- `CompositeLossConfig` dataclass
- `src/deqn_jax/training/trainer.py` -- wiring in `train_from_config()`
