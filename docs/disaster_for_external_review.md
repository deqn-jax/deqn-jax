# Disaster NK-DSGE — Model + DEQN Solution Method

A self-contained mathematical exposition for external review. No code references; everything below is at the level of equations and methodology.

---

## 0. Ask

A trained deep-equilibrium-network policy on the model below achieves:

- **Equilibrium residual loss ≈ 3.5 × 10⁻⁶** (sum of squared residuals across 11 equations)
- **Median |Δmean|% vs Dynare order-1 reference ≈ 0.44%** (across 11 policies)
- **Median |Δstd|% vs Dynare order-1 reference ≈ 80%** (across 11 policies)

So the network finds policies that *locally* satisfy the FOCs to high precision and reproduce the reference solution's *first* moments to half a percent — but underestimate the *second* moments by nearly an order of magnitude.

I'd like a second opinion on:

1. Is the level/variance dissociation fundamental to residual-loss training on this class of model, or fixable by a tractable methodological change?
2. Is the comparison against Dynare order-1 the right benchmark? If we should be matching Dynare order-2 or order-3, does that close the std gap mechanically (because order-1 imposes certainty equivalence, missing the precautionary effect that DEQN can in principle capture)?
3. Are there architectural or training-procedure changes that the field has converged on for variance accuracy that I might be missing?
4. Is there something fundamentally wrong with the formulation as stated below?

---

## 1. The Model

A medium-scale New Keynesian DSGE with Calvo nominal rigidities, a Bernanke-Gertler-Gilchrist financial accelerator, an effective lower bound on the policy rate, and a Barro-style exogenous capital-destruction "disaster" shock. The non-disaster core is the Christiano-Motto-Rostagno (2014) framework.

### 1.1 State and Policy Spaces

**Endogenous states (8):**
$\pi_{-1}$, $k_{-1}$, $c_{-1}$, $q_{-1}$, $i_{-1}$, $R_{-1}$, $\tilde w_{-1}$, $L_{-1}$

(lagged inflation, capital, consumption, Tobin's q, investment, gross nominal rate, real wage, hours)

**Exogenous states (5):**
$\varepsilon$ (productivity), $\mu_\Upsilon$ (investment-specific tech), $g$ (government spending), $\mu_z$ (trend growth), $m_p$ (monetary shock)

Each follows AR(1) in logs (or in levels with a positivity floor):
$\log x_t = \rho_x \log x_{t-1} + \sigma_x \xi_t$, $\xi_t \sim \mathcal{N}(0,1)$.

**Policy variables (controls — what the network outputs, 11 total):**
$\lambda_z$ (marginal utility of consumption), $i$, $\pi$, $c$, $\tilde w$, $h$, $F_w$, $F_p$, $q$, $K_p$, $K_w$.

The four "auxiliary" policies $F_p, K_p, F_w, K_w$ are recursive Calvo discounted sums — see §1.4.

**Shocks (5):** one for each exogenous state. Plus a Bernoulli disaster with per-period probability $p_{\text{disaster}}$ that destroys fraction $1 - e^{-\theta_{\text{disaster}}}$ of next-period capital. Default calibration: $p_{\text{disaster}} = 0$ (baseline) or up to $0.10$ (stress).

### 1.2 Equilibrium Conditions

Eleven equations in eleven policies. I'll show the structurally important ones; the rest are mechanical.

**Consumption Euler** (with habit formation, consumption tax):
$$
(1 + \tau_c)\,\lambda_{z,t} = \frac{\mu_{z,t}}{c_t \mu_{z,t} - b\, c_{t-1}} - \beta b\, \mathbb{E}_t \!\left[\frac{1}{c_{t+1} \mu_{z,t+1} - b\, c_t}\right]
$$

$\lambda_z$ is a network output (the marginal utility of consumption); the equation pins it to the after-tax marginal utility implied by the habit-adjusted consumption process. The implementation uses an algebraically equivalent multiply-through form to eliminate the divisions and avoid singular gradients near the zero-habit boundary.

**Bond Euler** (intertemporal):
$$
\lambda_{z,t} = R_t \beta\, \mathbb{E}_t \left[ \frac{\lambda_{z,t+1}}{\pi_{t+1} \mu_{z,t+1}} \right]
$$

**Investment Euler** (with adjustment costs). Define $x_t \equiv \mu_{z,t} i_t / i_{t-1}$ (the trend-adjusted investment growth). Then:
$$
1 = \mu_\Upsilon q_t \big(1 - S(x_t) - x_t\, S'(x_t)\big) + \beta\, \mu_\Upsilon\, \mathbb{E}_t\left[ \frac{\lambda_{z,t+1}}{\lambda_{z,t}}\, q_{t+1}\, \mu_{z,t+1}\, (i_{t+1}/i_t)^2\, S'(x_{t+1}) \right]
$$

with $S(x) = \tfrac{1}{2}\kappa(x - \mu_z^{ss})^2$ a quadratic adjustment cost on the trend-adjusted growth rate. The argument of $S, S'$ is $\mu_{z,t}\,i_t/i_{t-1}$ (not bare $i_t/i_{t-1}$) — the trend factor is inside, not outside, the cost function.

**Resource constraint** (closure):
$$
y_{z,t} = g_t + c_t + i_t/\mu_{\Upsilon,t} + \text{[bank monitoring costs]} + \text{[entrepreneur consumption]}
$$

with $y_{z,t} = \varepsilon_t (k_{t-1}/\mu_{z,t})^\alpha h_t^{1-\alpha} - \Phi$ (Cobb-Douglas with fixed costs).

**Taylor rule** (with effective lower bound, applied as a high-sharpness softplus floor):
$$
R_t^{\text{Taylor}} = R^{ss} (R_{t-1}/R^{ss})^{\rho_p}\big[(\pi_t/\pi^{ss})^{\alpha_\pi} (y_t/y^{ss})^{\alpha_y}\big]^{1-\rho_p} \exp(m_{p,t})
$$
$$
R_t = R^{lb} + \frac{1}{\sharp} \log(1 + \exp(\sharp (R_t^{\text{Taylor}} - R^{lb})))
$$

with $R^{lb} = 1$ (zero net rate) and sharpness $\sharp = 500$, giving SS distortion of $\sim 10^{-7}$ but a tight kink at the floor.

### 1.3 Financial Frictions (Bernanke-Gertler-Gilchrist)

Entrepreneurs face an idiosyncratic productivity shock $\omega \sim \log\mathcal{N}(-\sigma_\omega^2/2, \sigma_\omega^2)$ on their capital purchase. They default if $\omega < \bar\omega_t$. Banks recover fraction $1-\mu_{\text{mon}}$ of defaulted assets (monitoring cost).

The default threshold $\bar\omega_t$ is determined by the **bank participation constraint**:
$$
\bar\omega_t (1 - F(\bar\omega_t)) + (1 - \mu_{\text{mon}}) G(\bar\omega_t) = \frac{L_{t-1} - 1}{L_{t-1} \cdot R^k_t / R_{t-1}}
$$

where $F, G$ are the lognormal CDF and partial expectation. This is solved analytically (Newton iteration) inside each forward pass — the network does *not* output $\bar\omega$, it outputs $q$ and $\pi$ from which $R^k$ and hence $\bar\omega$ are computed.

Net worth evolves:
$$
n_t = \frac{\gamma_e}{\pi_t \mu_{z,t}} (1 - \Gamma(\bar\omega_t)) R^k_t q_{t-1} k_{t-1} + w^e
$$

Leverage $L_t = q_t k_t / n_t$ is determined by the balance-sheet identity, not as a separate equation.

The remaining equilibrium equation is the **entrepreneur contract**: the FOC of entrepreneurs choosing $\bar\omega$ optimally, derived from a participation-constrained optimization.

### 1.4 Calvo Auxiliaries — The Phillips Curves and the K/F Variables

Under Calvo pricing with indexation, the optimal-reset-price condition takes a recursive form. Define:

$$
F_{p,t} = \mathbb{E}_t \sum_{j \ge 0} (\beta \xi_p)^j \lambda_{z,t+j} y_{z,t+j} \prod_{k=1}^{j} (\tilde\pi_{t+k}/\pi_{t+k})^{1/(1-\lambda_f)}
$$
$$
K_{p,t} = \mathbb{E}_t \sum_{j \ge 0} (\beta \xi_p)^j \lambda_z y_z s \cdot \lambda_f \prod_{k=1}^{j} (\tilde\pi_{t+k}/\pi_{t+k})^{\lambda_f/(1-\lambda_f)}
$$

These satisfy two relationships:

**Definition** (algebraic, no expectations):
$$
K_{p,t} = F_{p,t} \cdot \left[\frac{1 - \xi_p (\tilde\pi_t / \pi_t)^{1/(1-\lambda_f)}}{1 - \xi_p}\right]^{1-\lambda_f}
$$

**Recursion** (forward-looking, with expectation):
$$
K_{p,t} = \lambda_z y_z s\, \lambda_f + \beta \xi_p\, \mathbb{E}_t [(\tilde\pi_{t+1}/\pi_{t+1})^{\lambda_f/(1-\lambda_f)} K_{p,t+1}]
$$

(and analogously for $F_{p,t}$, $F_{w,t}$, $K_{w,t}$ on the wage side).

**The structural problem with K/F**: each of these auxiliaries appears *only* in its own definition equation and its own recursion equation. They don't feed back into the consumption Euler, investment FOC, or any "level" equation. So the residual landscape has a **gauge freedom**: any uniform rescaling of the K/F policies $(F_p, K_p, F_w, K_w)$ that is compatible with their own definitions and recursions yields a self-consistent fixed point of the residual equations. The bank participation constraint and the resource constraint don't pin them down.

Under residual-loss training, this gauge freedom is what creates the **wrong-attractor** problem — the network slides off the desired equilibrium into a parallel scaling of K/F that's locally just as residual-zero as the right one.

### 1.5 Calibration (default)

| Parameter | Value | Description |
|---|---|---|
| $\beta$ | 0.9985 | discount |
| $b$ | 0.74 | habit |
| $\sigma_L$ | 1.0 | inverse Frisch |
| $\alpha$ | 0.4 | capital share |
| $\delta$ | 0.025 | depreciation |
| $\kappa$ | 2.0 | investment adjustment cost |
| $\Phi$ | 0.606 | fixed cost |
| $\xi_p, \xi_w$ | 0.6 | Calvo stickiness (prices, wages) |
| $\iota$ | 0.9 | price indexation |
| $\rho_p$ | 0.85 | Taylor-rule smoothing |
| $\alpha_\pi$ | 1.5 | Taylor-rule inflation |
| $\alpha_y$ | 0.36 | Taylor-rule output |
| $\gamma_e$ | 0.985 | entrepreneur survival |
| $\sigma_\omega$ | 0.268 | idiosyncratic risk |
| $\mu_{\text{mon}}$ | 0.22 | monitoring cost |
| $p_{\text{disaster}}$ | 0.0–0.1 | per-period disaster probability |
| $\theta_{\text{disaster}}$ | 0.05 | disaster magnitude (capital destruction $1 - e^{-\theta}$) |
| $R^{lb}$ | 1.0 | gross-rate floor (zero net) |

The Calvo parameters are at the conventional New Keynesian calibration. Note that with $\xi_p = 0.6$ and $\lambda_f = 1.2$, the inner Calvo aggregator $1 - \xi_p (\tilde\pi/\pi)^{1/(1-\lambda_f)}$ goes negative if $\pi/\tilde\pi$ exceeds about 1.1 — there is a hard *Calvo validity edge* in policy space. The implementation enforces this with a soft policy cap on $\pi$.

### 1.6 Why This Model Is Numerically Hard

In rough order of severity:

1. **K/F gauge freedom** — see §1.4. Multiple residual-zero fixed points exist; vanilla MLP training lands in the wrong one with measurable consistency.
2. **Effective lower bound** — kink in policy at $R = R^{lb}$. Smooth (softplus) but sharp at sharpness 500. Tanh networks have spectral bias against sharp transitions.
3. **Calvo validity edge** — $\pi$ has a hard upper bound at $\sim 1.1 \tilde\pi$ above which the optimal-reset-price equations have no real solution.
4. **Disaster shock** — when $p_{\text{disaster}} > 0$, the expectation in every Euler-style equation is a mixture: $\mathbb{E}[X_{t+1}] = (1-p) \mathbb{E}[X_{t+1} | \text{no disaster}] + p \mathbb{E}[X_{t+1} | k_{t+1} \text{ destroyed}]$. The conditional disaster integrand can be far in the tail, and quadrature accuracy degrades.
5. **High dimensionality** — 13 states, 11 policies, 5 shocks. The ergodic distribution lives in a thin manifold; off-manifold residuals can be huge but training rarely visits there.

---

## 2. Solution Method (Deep Equilibrium Networks)

Standard DEQN setup as in Azinovic-Maliar-Maliar (2022), Maliar-Maliar-Winant (2021). A neural network policy:
$$
\hat\pi_\theta : \mathbb{R}^{13} \to \mathbb{R}^{11}, \quad s \mapsto \text{policy vector}
$$
is trained to minimize equilibrium residuals over the on-policy ergodic distribution:
$$
\theta^* = \arg\min_\theta\; \mathbb{E}_{s \sim d_\theta}\, \big\|R(s, \hat\pi_\theta(s); \hat\pi_\theta)\big\|^2
$$
where $R(\cdot)$ is the 11-vector of equilibrium-condition residuals and the expectation over $d_\theta$ is the policy-induced ergodic measure. Each evaluation of $R$ at state $s$ requires:

- The current policy $\hat\pi_\theta(s)$
- The next-period state $s' = T(s, \hat\pi_\theta(s), \xi)$ for shock realization $\xi$
- The next-period policy $\hat\pi_\theta(s')$ — used inside the conditional expectation

Expectations over shocks are approximated by Gauss-Hermite quadrature (3 points per shock dimension; tensor product). Some training configurations use Monte Carlo with antithetic sampling instead.

The training distribution is generated by **on-policy simulation**: trajectories of length 20-50 are rolled out using the current $\hat\pi_\theta$ at each gradient step, with periodic resets to states near the deterministic SS (curriculum). Adam with cosine learning-rate schedule, $\text{lr} \in [10^{-4}, 10^{-3}]$.

### 2.1 The Architectural Ansatz

Vanilla MLP policy training (random init, sigmoid output bounds, hidden tanh) on this model converges to the wrong-attractor manifold described in §1.4. Diagnostic measurements: median $|\Delta\text{mean}|$ vs Dynare ≈ 14-30%, with K/F policies systematically scaled away from the reference equilibrium. Equation residuals are small there — the network is at a self-consistent fixed point — but it's the wrong fixed point.

The architectural fix used here is a **residual ansatz with respect to the Blanchard-Kahn linearization**:

$$
\hat\pi_\theta(s) = \underbrace{\pi^{ss} + P\,(s - s^{ss})}_{\pi_{\text{BK}}(s) \text{ — Dynare order-1 policy}} + \underbrace{\delta_\theta(s)}_{\text{MLP correction, zero-init final layer}}
$$

$P$ is the policy-rule matrix from the QZ decomposition of the linearized model around the deterministic SS. The MLP correction $\delta_\theta$ is built so that $\delta_\theta(s) = 0$ for every $s$ at training step 0 (final-layer weights and bias zero-initialized). At init, the policy is *exactly* the BK linear solution — which is correct to first order around SS and, crucially, in the right basin of attraction.

Training learns $\delta_\theta$ to capture the higher-order curvature of the true policy that BK misses. Taylor expansion around SS:
$$
\hat\pi^*(s) - \pi_{\text{BK}}(s) = \tfrac{1}{2}(s - s^{ss})^\top H (s - s^{ss}) + \mathcal{O}(\|s - s^{ss}\|^3) + \text{boundary kinks}
$$

This is a *one-step* PINN-style soft-constraint embedding: the linear part is in the forward pass forever, not just at init. To leave the BK basin during training, $\delta_\theta$ would need to grow large enough to cancel the linear part — which is harder than starting random and sliding into the closest basin.

### 2.2 The K/F Gauge Fix

On top of the residual ansatz, the four K/F output positions of $\delta_\theta$ are *masked to zero* throughout training:
$$
\delta_\theta(s)_j = 0 \quad \forall s, \forall j \in \{F_p, K_p, F_w, K_w\}
$$

So those four policies remain *exactly* equal to their BK linear function for the entire run:
$$
\hat\pi_j(s) = \pi^{ss}_j + P_j(s - s^{ss}) \quad \forall j \in \text{K/F}
$$

This pins the gauge — the K/F variables become deterministic linear functions of state, not network outputs that residual training is allowed to redistribute. The remaining 7 policies (level variables: $\lambda_z, i, \pi, c, \tilde w, h, q$) carry the full $\delta_\theta$ correction.

### 2.3 Training Schedule and Loss Variants

Standard configuration: 5000 episodes, batch size 64, episode length 20, MC samples 2 (or quadrature 3-point per dim). Adam, lr 10⁻³, gradient clipping at 0.5, no LR schedule.

The loss is plain MSE over residuals. An optional "composite loss" mode adds:
- **Anchor term**: $\lambda_a \|\hat\pi_\theta(s) - \pi_{\text{BK}}(s)\|^2$ at sampled SS-adjacent states. Soft pull toward BK.
- **Jacobian term**: $\lambda_j \|D\hat\pi_\theta(s^{ss}) - P\|^2$. Pins the network's tangent at SS.
- **Barrier terms**: smooth penalties for $\pi$ near the Calvo edge, $L$ near zero, etc.
- **Newton auxiliary**: penalizes |Newton iterate residual| from the $\bar\omega$ solver.

Composite loss + the residual ansatz + K/F mask is the strongest configuration measured.

### 2.4 The "Stochastic Steady State" Question

The BK linearization is around the **deterministic** steady state. The on-policy training distribution lives near the **ergodic** distribution, which is shifted relative to DSS by precautionary effects (Coeurdacier-Rey-Winant 2011 risky steady state, Den Haan ergodic moments). For low shock variance and mild nonlinearity these coincide; for the disaster calibration with $p_{\text{disaster}} = 0.10$ they may not.

The Dynare reference being compared against is order-1 — i.e., centered at the same deterministic SS. So "match the Dynare ergodic moments" actually means "match the certainty-equivalent linear-policy ergodic moments". A truly nonlinear DEQN solution should *differ* from this in a structured way (precautionary saving raises mean K above DSS K, etc.), and the comparison at order-1 understates how good the DEQN solution is in absolute terms.

---

## 3. Empirical Findings

### 3.1 Cross-architecture sweep (5000 ep, Adam, MSE loss, 3 seeds, disaster $p_{\text{disaster}} = 0$)

| Architecture | Best loss | $\|\Delta\text{mean}\|\%$ | $\|\Delta\text{std}\|\%$ |
|---|---:|---:|---:|
| Vanilla MLP (64×64, sigmoid bounds) | 4.4e-3 | NaN (sim divergent) | NaN |
| K/F-pinned MLP (4 outputs frozen, 7 free random-init) | 1.3e-2 | 14.0 | 67.9 |
| K/F-pinned + moment-matching aux (w=0.01) | 5.9e-2 | 20.5 | 71.7 |
| Residual ansatz (full $\pi_\text{BK}$ + δ, no K/F mask) | 3.5e-6 | 23.3 | 100.0 |
| **Residual ansatz + K/F mask** | **3.6e-6** | **0.44** | **80.5** |

Three orders of magnitude lower residual loss for the residual-ansatz arms. The K/F mask alone moves $|\Delta\text{mean}|$ from 23% to 0.44% — same residual loss, dramatically different ergodic mean. This is exactly the gauge-freedom story: without the mask, residual training drifts the K/F levels into a self-consistent but globally wrong scaling, contaminating the level variables through the (eq2a / eq4a) definition equations.

### 3.2 What's Striking

The mean accuracy is essentially the noise floor of the comparison (0.44% across 11 policies — likely within Dynare-vs-simulation uncertainty). The std accuracy is uniformly bad (80% off across all arms). The residual loss is uniformly tiny on the winning arm.

This suggests:

- The network *finds* the right deterministic policy locus
- It does not capture the second-order behavior that drives ergodic variance
- Residual loss alone, even at $10^{-6}$, is consistent with arbitrarily wrong second moments

### 3.3 Hypotheses for the std miss (in priority order)

(a) **Wrong reference.** Dynare order-1 imposes certainty equivalence: variances are determined by the linear policy + shock variances, with no precautionary or asymmetric effects. A nonlinear DEQN solution *should* have different variances. Compare against Dynare order-2 or order-3 to test.

(b) **Underexplored ergodic distribution.** On-policy training keeps the distribution narrow once the policy converges; the network never sees deep tails, so its policy in the tails is wrong, so simulated variance shrinks. Could be addressed with a state replay buffer, off-policy correction, or wider curriculum.

(c) **Adversarial accuracy bug.** Even on-policy states have wrong second-derivatives because residual loss is insensitive to local Hessian — gradient descent finds a *value* that minimizes the residual but not necessarily a *function* with the right Hessian. PINN literature suggests Sobolev training (penalize derivative residuals) helps here.

(d) **Architectural ceiling.** $\delta_\theta$ with 128×128 tanh hidden layers and zero-init final layer cannot represent the kind of curvature this model needs. Wider, deeper, or different activations (ReLU/GELU) might help.

(e) **Stochastic-vs-deterministic SS mismatch in the ansatz itself.** $\pi_\text{BK}$ is centered at deterministic SS; the true policy passes through stochastic SS. The MLP correction has to learn a non-zero *level* shift on top of a non-zero *curvature* shift. Reformulating around SSS might be cleaner.

I find (a) and (b) most plausible, (c) intriguing, (d) unlikely (loss is already at $10^{-6}$), (e) possible but high-cost to test.

---

## 4. Specific Questions for External Review

1. **Is the level/variance dissociation a known failure mode of DEQN-style training, or is it specific to this model?** If it's known, what's the field's current best practice for closing it?

2. **Is comparing against Dynare order-1 the right benchmark for a nonlinear solver?** Should I be regenerating reference moments at order-2 or order-3? What's the principled choice?

3. **Does the K/F gauge-fix story generalize?** Concretely: any DSGE with Calvo recursive aggregators (price-, wage-Phillips, sticky-information variants) inherits this gauge freedom. Is the right move always to pin those aux variables to their first-order linear solution, or is there a model class where doing so loses important nonlinear content?

4. **Is the residual-ansatz architecture fragile in ways I haven't measured?** Specifically: does the "MLP correction starts at zero" property bias training toward solutions close to the linear policy in a way that's correct near SS but systematically wrong far from it?

5. **The std miss across all arms (67-100%) is suspicious — could it be a measurement artifact?** The Dynare comparison is built on sampling 2000 simulated periods from the trained policy. Could finite-sample bias on second moments be eating the comparison?

6. **What would you measure next?** I have a 21-cell sweep on disk and a working evaluation pipeline against Dynare. If you wanted to be 80% more confident the architecture is right (or wrong), what's the next experiment?

---

*Notation conventions: $t$ subscripts on policies; $t-1$ on lags; $t+1$ inside expectations. Variables without subscripts are time-$t$ quantities. SS is "deterministic steady state" unless explicitly modified.*
