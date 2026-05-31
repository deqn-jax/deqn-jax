# LinearPlusMLP

A residual policy parameterization: the network learns a *correction* to a
Blanchard-Kahn first-order policy rather than the policy from scratch. This
is the canonical architecture for the disaster model and any DEQN problem
where the bare-MLP residual landscape has degenerate basins.

## Idea

$$
\pi_\theta(s) \;=\; \underbrace{\pi^* + P\,(s - s^*)}_{\text{linear (BK first-order)}} \;+\; \underbrace{\delta_\theta(s)}_{\text{MLP correction}}
$$

The first term is the textbook Blanchard-Kahn linear policy: SS values plus a
linear rule of state. The second term is an MLP whose **final layer is
zero-initialized**, so $\delta_\theta(s) = 0$ for every state at training
step 0. The full policy is *exactly* the BK solution at init.

Training learns $\delta_\theta$ to capture the higher-order curvature of the
true policy that the linear rule misses. Taylor expansion around SS:

$$
\pi^*(s) - \pi_{\text{BK}}(s) \;=\; \tfrac{1}{2}(s - s^*)^\top H (s - s^*) \;+\; \mathcal{O}(\|s - s^*\|^3) \;+\; (\text{boundary kinks})
$$

So the MLP starts with no work to do, and the residual gradient signal slowly
grows the correction in the directions where the true policy departs from
linear.

## Why this beats a bare MLP

A plain MLP trained on equation residuals for the disaster model converges
to a wrong-attractor manifold — the residual loss is set-identifying, not
point-identifying, so multiple self-consistent solutions exist and random
init lands in whichever basin is closest. The linear ansatz puts you in the
BK basin (correct to first order) and gradient descent fine-tunes from
there. The correction has to grow large to leave the basin, which never
happens spontaneously.

## What you need to supply

**At minimum: a model with a working `steady_state_fn`.** That returns
$(s^*, \pi^*)$ for the model's constants. The in-house linearizer
(`training/linearize.py`) consumes that plus `model.equations`,
`model.dynamics_step`, and the policy bounds to compute $P, Q$ via QZ
decomposition.

**No Dynare dependency.** The linearization is pure JAX; same matrices
Dynare's `stoch_simul order=1` would produce, computed from the model
definition you already wrote. You do **not** need:

- Dynare CSVs
- A separate MATLAB / Octave run
- An external linearization pipeline

Optional complements (each independent):

- **`kf_names`** — list of policy names whose MLP correction is masked to
  zero. Those outputs stay exactly $\pi^*_i + P_i(s - s^*)$ for the entire
  run. Use for Calvo-style discounted-sum auxiliaries (`F_p`, `K_p`, `F_w`,
  `K_w` in the disaster model) that carry first-order gauge freedom in the
  residual loss.
- **`init_scale`** — scaling on the final-layer weights at init. `0.0`
  (exact BK at init, the default for disaster) or `0.01` (small random
  perturbation around BK).
- **`use_zlb_feature`** — prepend `(R_lag - R_lb)` as an extra MLP input
  so the correction can learn ELB-regime-dependent shape.
- **Dynare moments** (a *separate* feature: moment-matching loss). If you
  have Dynare-computed unconditional moments and IRFs as CSVs, the
  `moment_matching` loss block supervises the trained policy's ergodic
  moments against them. Composes with LinearPlusMLP but is independent of
  it: LinearPlusMLP does not need them.

## YAML

Basic — exact BK at init, no gauge fix:

```yaml
network:
  type: linear_plus_mlp
  hidden_sizes: [128, 128]
  activation: tanh
  init: xavier_normal
  init_scale: 0.0       # 0.0 = exact BK linear policy at init
```

With K/F gauge fix (recommended for Calvo-NK models):

```yaml
network:
  type: linear_plus_mlp
  hidden_sizes: [128, 128]
  activation: tanh
  init_scale: 0.0
  kf_names: [F_p, K_p, F_w, K_w]    # mask delta on these outputs
```

With ZLB regime feature (disaster-style models with effective lower bound):

```yaml
network:
  type: linear_plus_mlp
  hidden_sizes: [128, 128]
  activation: tanh
  init_scale: 0.0
  use_zlb_feature: true
```

## Composes with

- **Composite loss** (`loss_type: composite`): adds anchor + Jacobian + barrier
  + Newton auxiliary terms. The anchor term supervises $\pi$ to stay near the
  linearization at SS-adjacent points; redundant for $\delta_\theta$ near SS
  by construction (zero-init), but useful as a soft penalty during
  curriculum-driven exploration. See [Composite loss](../training/composite_loss.md).
- **Moment matching** (`moment_matching.enabled: true`): supervised loss
  against Dynare ergodic moments. Anchors the trained policy's long-run
  distribution to the reference solution.

## When not to use

- **Brock-Mirman / simple RBC**: bare MLP is fine; no wrong-attractor
  pathology, no Calvo gauge.
- **Models without a tractable `steady_state_fn`**: you'd need to provide
  one (analytical or numerical) — the linearizer needs SS values.
- **Models that fail the Blanchard-Kahn rank condition**: the linearizer
  raises; there is no linear policy to anchor against. Reformulate the
  model.

## Init behavior in detail

At step 0 with `init_scale: 0.0`:

- final-layer weights $W_n = 0$ (exactly), bias $b_n = 0$
- so $\delta_\theta(s) = W_n h(s) + b_n = 0$ for every state $s$
- so $\pi(s) = \pi^* + P(s - s^*)$ exactly — the BK linear policy
- gradient $\partial \delta / \partial W_n = h(s)$ is **non-zero** even
  though $\delta$ is zero — the hidden layers compute random xavier-init
  features $h(s)$. So the first gradient step is a kernel-regression update
  on those features. Earlier layers only start moving once $W_n \neq 0$
  (step 2+). This warms the network in safely from the BK basin.

## Source

- `src/deqn_jax/networks/linear_plus_mlp.py` — the module.
- `src/deqn_jax/training/linearize.py` — `linearize_model(model)` returning
  $(P, Q)$ from any model with `steady_state_fn`.
- Tests: `tests/test_linear_plus_mlp.py`.
