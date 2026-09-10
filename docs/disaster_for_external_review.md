# Disaster NK-DSGE — Model, Method, and the Certification Record

A self-contained exposition for external review: equations and methodology, no code references. Refreshed 2026-09-10. It supersedes the 2026-05-31 version, whose findings and questions predate the certification program; the May results are summarized in §3.1 for the record and the May "gauge freedom" claim is retracted in §1.4.

---

## 0. Ask

The claim under review is this one, and no wider:

> The disaster model at the *shipped* calibration (disaster probability $p_{\text{disaster}} = 0$, i.e. the Gaussian-shock CMR-BGG economy with an effective lower bound) is **solved by selection by construction**, per the full certification stack of §2.5, at the frozen convention (final checkpoints, fp64, three seeds).

The certificate, per seed:

| certificate | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| policy error at the deterministic SS $s^*$ | 0 (exact) | 1.2e-14 | 9.0e-15 |
| learned-block spectral radius at $s^*$ | 0.976851 | 0.976851 | 0.976851 |
| learned-block spectral radius at the solved rest point $\hat s$ | 0.9768 | 0.976751 | 0.976788 |
| $\|\hat s - s^*\|$ (max relative) | 0.0525% | 0.0525% | 0.0525% |
| max $|\mathbb{E}[r]|$ at $\hat s$ | 5.2e-4 | 4.3e-4 | 6.4e-4 |
| convergence to $\hat s$ at $t = 1000$ (zero shocks) | 7e-14 | 8.4e-14 | 3.8e-14 |
| held-out stress grid, max per-equation $(\mathbb{E}[r])^2$ | 1.50e-3 | 2.11e-3 | 9.8e-4 |

Two of these legs are *donated* by the construction (§2.3) and hold for every weight vector; the others are earned by training. §3 says which is which and what the linear policy alone scores on the same grids.

What we want a second opinion on is not "are the numbers right" but the questions of §4: whether a certificate two of whose legs hold by construction certifies the solution or the construction; what the restriction that holds four policies linear costs; whether the certified closed loop is a property of the model or of a guardrail inside it; how to move the construction to a risky rest point the network must itself compute; and three formulation questions the May exposition hid.

---

## 1. The Model

A medium-scale New Keynesian DSGE with Calvo nominal rigidities, a Bernanke-Gertler-Gilchrist financial accelerator, an effective lower bound on the policy rate, and a Barro-style exogenous capital-destruction "disaster" shock. The non-disaster core is the Christiano-Motto-Rostagno (2014) framework, reduced by the model's author to eleven equilibrium conditions.

### 1.1 State and Policy Spaces

**Endogenous states (8):**
$\pi_{-1}$, $k_{-1}$, $c_{-1}$, $q_{-1}$, $i_{-1}$, $R_{-1}$, $\tilde w_{-1}$, $L_{-1}$

(lagged inflation, capital, consumption, Tobin's $q$, investment, gross nominal rate, real wage, **entrepreneurial leverage** $L = qk/n$). Hours $h$ are a policy, not a state; the May version mislabeled $L_{-1}$ as hours.

**Exogenous states (5):**
$\varepsilon$ (productivity), $\mu_\Upsilon$ (investment-specific technology), $g$ (government spending), $\mu_z$ (trend growth), $m_p$ (monetary shock).

The four positive-level processes follow log-AR(1) around their steady-state values,
$$
\log(x_t/x^{ss}) = \rho_x \log(x_{t-1}/x^{ss}) + \sigma_x \xi_{x,t}, \qquad \xi \sim \mathcal N(0,1),
$$
and the monetary shock enters the Taylor rule as $\exp(m_{p,t})$ with $m_{p,t} = \sigma_{m}\,\xi_{m,t}$, a zero-mean level shock without persistence ($\sigma_m = 0.0049$). The May version's blanket "AR(1) in logs for every exogenous state" was a prose simplification and did not describe $m_p$.

**Policy variables (what the network outputs, 11):**
$\lambda_z$ (marginal utility of consumption), $i$, $\pi$, $c$, $\tilde w$, $h$, $F_w$, $F_p$, $q$, $K_p$, $K_w$.

The four auxiliaries $F_p, K_p, F_w, K_w$ are the recursive Calvo discounted sums of §1.4.

**Shocks (5):** one per exogenous state, plus a Bernoulli disaster with per-period probability $p_{\text{disaster}}$ that destroys the fraction $1 - e^{-\theta}$ of next-period capital. Two calibrations matter: the **shipped** one, $p_{\text{disaster}} = 0$ (the certified claim; $\theta = 0.05$ is a stress knob that never fires there), and the **author's** one, $p_{\text{disaster}} = 1\%$, $\theta = 15\%$ (Barro-consistent), which is the successor program's target and is not covered by any certificate.

### 1.2 Equilibrium Conditions

Eleven equations in eleven policies. The structurally important ones:

**Consumption Euler** (habit formation, consumption tax). With the habit-adjusted consumption $H_t \equiv c_t\mu_{z,t} - b\,c_{t-1}$,
$$
(1 + \tau_c)\,\lambda_{z,t} = \frac{\mu_{z,t}}{H_t} - \beta b\, \mathbb{E}_t\!\left[\frac{1}{H_{t+1}}\right].
$$
The residual actually minimized is this equation multiplied by the *time-$t$-measurable* factor $H_t/\mu_{z,t}$:
$$
r_5 = \frac{(1+\tau_c)\,\lambda_{z,t}\,H_t + \beta b\, H_t\, \mathbb{E}_t[1/H_{t+1}]}{\mu_{z,t}} - 1 .
$$
Because the multiplier is known at $t$, the root is preserved and the residual is affine in $1/H_{t+1}$, so averaging it over next-period shock draws computes $H_t\,\mathbb{E}_t[1/H_{t+1}]$ exactly. Both $H_t$ and $H_{t+1}$ pass through a smooth floor at $10^{-2}$ (a guardrail; at the steady state $H \approx 0.42$ and the floor's wedge is $\sim 10^{-20}$).

**Bond Euler:**
$$
\lambda_{z,t} = R_t \beta\, \mathbb{E}_t \left[ \frac{\lambda_{z,t+1}}{\pi_{t+1} \mu_{z,t+1}} \right].
$$

**Investment Euler** (adjustment costs on trend-adjusted investment growth $x_t \equiv \mu_{z,t} i_t / i_{t-1}$):
$$
1 = \mu_\Upsilon q_t \big(1 - S(x_t) - x_t\, S'(x_t)\big) + \beta\, \mu_\Upsilon\, \mathbb{E}_t\left[ \frac{\lambda_{z,t+1}}{\lambda_{z,t}}\, q_{t+1}\, \mu_{z,t+1}\, (i_{t+1}/i_t)^2\, S'(x_{t+1}) \right],
\qquad S(x) = \tfrac{1}{2}\kappa (x - \mu_z^{ss})^2 .
$$

**Resource constraint:**
$$
y_{z,t} = g_t + c_t + i_t/\mu_{\Upsilon,t} + \text{[bank monitoring costs]} + \text{[entrepreneur consumption]},
\qquad y_{z,t} = \varepsilon_t (k_{t-1}/\mu_{z,t})^\alpha h_t^{1-\alpha} - \Phi .
$$
Output is Cobb-Douglas net of a fixed cost. **There is no price- or wage-dispersion state and no dispersion wedge between inputs and output** anywhere in the eleven equations (see §4.8).

**Taylor rule with a smooth floor:**
$$
R_t^{\text{Taylor}} = R^{ss} (R_{t-1}/R^{ss})^{\rho_p}\big[(\pi_t/\pi^{ss})^{\alpha_\pi} (y_t/y^{ss})^{\alpha_y}\big]^{1-\rho_p} \exp(m_{p,t}),
\qquad
R_t = R^{lb} + \tfrac{1}{\sharp}\log\!\big(1 + e^{\sharp (R_t^{\text{Taylor}} - R^{lb})}\big),
$$
with $R^{lb} = 1$ and $\sharp = 500$. At the deterministic steady state the smoothing is invisible ($\sim 10^{-7}$); *at the bound* it adds $\log 2/\sharp \approx 1.4\times 10^{-3}$ to the gross rate, about 14 basis points per period. The floor was added to the model as a training guardrail (April 2026) and is absent from the author's original formulation; §3.9 and §4.3 say what that implies for the certificate.

### 1.3 Financial Frictions (Bernanke-Gertler-Gilchrist)

Entrepreneurs face an idiosyncratic shock $\omega \sim \log\mathcal N(-\sigma_\omega^2/2, \sigma_\omega^2)$ on their capital purchase and default if $\omega < \bar\omega_t$; banks recover the fraction $1 - \mu_{\text{mon}}$ of defaulted assets. The default threshold is set by the **bank participation constraint**
$$
\bar\omega_t\,(1 - F(\bar\omega_t)) + (1 - \mu_{\text{mon}})\, G(\bar\omega_t) = \frac{L_{t-1} - 1}{L_{t-1}\, R^k_t / R_{t-1}},
$$
with $F, G$ the lognormal CDF and partial expectation, solved by Newton iteration inside every forward pass ($\bar\omega$ is not a network output; $R^k$ follows from $q$ and $\pi$). The left-hand side is not globally monotone (its derivative is $1 - F - \mu_{\text{mon}}\,\bar\omega f$), and neither feasibility nor root selection is checked beyond the Newton residual (§4.8).

Net worth evolves as
$$
n_t = \frac{\gamma_e}{\pi_t \mu_{z,t}} \big(1 - \Gamma(\bar\omega_t)\big) R^k_t\, q_{t-1} k_{t-1} + w^e ,
$$
leverage $L_t = q_t k_t / n_t$ is a balance-sheet identity, and the eleventh equation is the entrepreneur's contract FOC.

### 1.4 Calvo Auxiliaries — and the retraction of "gauge freedom"

Under Calvo pricing with indexation the optimal-reset-price condition is recursive:
$$
F_{p,t} = \mathbb{E}_t \sum_{j \ge 0} (\beta \xi_p)^j\, \lambda_{z,t+j}\, y_{z,t+j} \prod_{k=1}^{j} (\tilde\pi_{t+k}/\pi_{t+k})^{1/(1-\lambda_f)},
\qquad
K_{p,t} = \mathbb{E}_t \sum_{j \ge 0} (\beta \xi_p)^j\, \lambda_z y_z s\, \lambda_f \prod_{k=1}^{j} (\tilde\pi_{t+k}/\pi_{t+k})^{\lambda_f/(1-\lambda_f)} .
$$
Each auxiliary enters two equations. The **definition** is algebraic and links the *ratio* to inflation:
$$
K_{p,t} = F_{p,t} \left[\frac{1 - \xi_p (\tilde\pi_t / \pi_t)^{1/(1-\lambda_f)}}{1 - \xi_p}\right]^{1-\lambda_f} .
$$
The **recursion** is forward-looking and affine, with a current-period source term:
$$
K_{p,t} = \lambda_z y_z s\, \lambda_f + \beta \xi_p\, \mathbb{E}_t \big[(\tilde\pi_{t+1}/\pi_{t+1})^{\lambda_f/(1-\lambda_f)} K_{p,t+1}\big],
$$
and analogously for $F_p$ (source $\lambda_z y_z$) and for $F_w, K_w$ on the wage side.

**Retraction.** The May version claimed a *gauge freedom*: that a common rescaling of $(F_p, K_p, F_w, K_w)$ leaves the equilibrium system invariant, so residual training could slide into a "parallel scaling" that is just as residual-zero as the truth. That is false. The definitions are homogeneous in the ratio and are indeed invariant; the recursions are not: rescaling a solution $K$ by $a$ leaves the residual $(a - 1)\,\lambda_z y_z s \lambda_f$, which vanishes only at $a = 1$. At this calibration $1 - \beta\xi_p = 0.40$, so the source term is not small relative to the level. We verified this numerically (2026-09-10, fp64, at the deterministic steady state, rescaling all four auxiliaries in both the current and the next-period policy):

| common scale $a$ | definition residuals (price, wage) and all seven level equations | recursion residuals (price $F$, price $K$, wage $F$, wage $K$) |
|---|---|---|
| 1.1 | 0 | $-0.035$ to $-0.037$ |
| 2.0 | 0 | $-0.19$ to $-0.20$ |

What *is* true: multiple residual-small basins exist in weight space (§3.2), and the device that in May moved the mean error from 23% to 0.44% at unchanged loss (§3.1) selected among them. It did so by **holding the four auxiliaries exactly at their first-order policy** — a restriction, not a normalization. Its cost is an open question (§4.2), and it matters most exactly where the successor program must go: at the author's calibration the Calvo recursions carry the largest risk shifts of any policy (§3.10).

### 1.5 Calibration

| Parameter | Value | Description |
|---|---|---|
| $\beta$ | 0.9985 | discount |
| $b$ | 0.74 | habit |
| $\tau_c$ | 0.047 | consumption tax |
| $\sigma_L$ | 1.0 | inverse Frisch |
| $\alpha$ | 0.4 | capital share |
| $\delta$ | 0.025 | depreciation |
| $\kappa$ | 2.0 | investment adjustment cost |
| $\Phi$ | 0.606 | fixed cost |
| $\xi_p, \xi_w$ | 0.6 | Calvo stickiness (prices, wages) |
| $\lambda_f$ | 1.2 | price markup |
| $\iota$ | 0.9 | price indexation |
| $\rho_p, \alpha_\pi, \alpha_y$ | 0.85, 1.5, 0.36 | Taylor rule |
| $\gamma_e$ | 0.985 | entrepreneur survival |
| $\sigma_\omega$ | 0.268 | idiosyncratic capital-quality dispersion |
| $\mu_{\text{mon}}$ | 0.22 | monitoring cost |
| $R^{lb}$ | 1.0 | gross-rate floor (zero net) |
| $p_{\text{disaster}}, \theta$ | 0, 0.05 | **shipped** (certified) calibration |
| $p_{\text{disaster}}, \theta$ | 0.01, 0.15 | **author's** calibration (successor target) |

With $\xi_p = 0.6$ and $\lambda_f = 1.2$ the inner Calvo aggregator $1 - \xi_p(\tilde\pi/\pi)^{1/(1-\lambda_f)}$ turns negative once $\pi/\tilde\pi$ exceeds about 1.1: a hard *Calvo validity edge* in policy space, enforced by a soft cap on $\pi$.

### 1.6 Why This Model Is Numerically Hard

1. **Residuals do not select.** The residual objective has several basins of small residual; at final checkpoints every untreated or loss-treated run of the July program landed in a basin whose closed-loop dynamics are mildly *unstable* (spectral radius 1.02–1.06) with the rest point a few percent from the truth, and untreated trajectories escape to a distant attractor created by the state guardrail. Selection had to be built into the network (§2.3).
2. **Effective lower bound.** A kink at $R = R^{lb}$, smoothed at sharpness 500; tanh networks resist sharp transitions, and coverage of the floor region is a measure question (§3.9).
3. **Calvo validity edge.** A hard upper bound on $\pi$ beyond which the reset-price equations have no real solution.
4. **Disaster mixture.** With $p_{\text{disaster}} > 0$ every expectation is a two-branch mixture whose disaster branch lives far in the tail.
5. **Dimensionality.** 13 states, 11 policies, 5 shocks; the ergodic set is a thin manifold and off-manifold residuals are large and rarely visited.

---

## 2. Solution Method (Deep Equilibrium Networks)

Standard DEQN as in Azinovic-Maliar-Maliar (2022) and Maliar-Maliar-Winant (2021): a network policy $\hat\pi_\theta : \mathbb{R}^{13} \to \mathbb{R}^{11}$ trained to make the equilibrium residuals vanish in conditional expectation on states the economy visits,
$$
\min_\theta\; \mathbb{E}_{s \sim d_\theta}\, \tfrac{1}{11}\sum_{e=1}^{11} \big(\mathbb{E}_{\xi}\, r_e(s, \hat\pi_\theta(s), s', \hat\pi_\theta(s'))\big)^2, \qquad s' = T(s, \hat\pi_\theta(s), \xi).
$$
The inner expectation is a tensor Gauss-Hermite rule, 3 nodes per shock ($3^5 = 243$ nodes); the loss squares the *expected* residual, so the Monte-Carlo variance penalty that afflicts $\mathbb{E}[\hat r^2]$ estimators is absent in the certified arm. The gradient flows through today's and tomorrow's policy. The whole run is fp64.

**The training measure, declared.** Trajectories of length 20 are rolled out under the current policy from the previous cycle's end states; 15% of paths are re-seeded near $s^*$ every cycle; the transition is soft-clipped to a box with margins $\ge 2$ from $s^*$; shock scale ramps from 0.1 to 1 over the first 200 of 3000 episodes; batch 64, Adam at $10^{-4}$ with cosine decay, warm-up 100, gradient clip 0.5. The sentence "trained on the ergodic set" is therefore false in three ways, each defensible and none of which used to be stated (§3.9).

### 2.1 The Residual Ansatz

Random-init MLP policies converge to the wrong basin (§1.6). The ansatz is a residual with respect to the Blanchard-Kahn linearization,
$$
\hat\pi_\theta(s) = \underbrace{\pi^{*} + P\,(s - s^{*})}_{\pi_{\text{BK}}(s)} + \delta_\theta(s),
$$
with $P$ from the QZ decomposition at the deterministic steady state and $\delta_\theta$ a $128\times128$ tanh MLP whose final layer starts at zero, so the policy *is* the linear rule at step 0. The additive form does **not** restrict the slope: $D\hat\pi_\theta(s^*) = P + D\delta_\theta(s^*)$, and nothing stops training from moving it. That is what §3.7 measured happening before the first training step, and why §2.3 exists.

### 2.2 The K/F Restriction

The four auxiliary positions of $\delta_\theta$ are masked to zero for the whole run:
$$
\hat\pi_j(s) = \pi^{*}_j + P_j (s - s^{*}) \quad \forall s,\ j \in \{F_p, K_p, F_w, K_w\}.
$$
These four policies are exactly linear forever; the other seven carry the full correction. This was called a "gauge fix" until §1.4; it is a restriction, and a restricted linear auxiliary will not in general satisfy its nonlinear recursion under the corrected level policies, so the optimizer must compromise elsewhere. Consistent with that, the worst equations on the certified arm's stress grid are the wage-side definition and the investment Euler (§3.5).

### 2.3 Selection by Construction: the BK Pin

The certified arm replaces the correction by its own value-and-tangent-free part at $s^*$,
$$
\tilde\delta_\theta(s) = \delta_\theta(s) - \delta_\theta(s^*) - D\delta_\theta(s^*)\,(s - s^*),
$$
so that $\hat\pi_\theta(s^*) = \pi^*$ and $D\hat\pi_\theta(s^*) = P$ **hold for every $\theta$**. Training cannot unlearn Blanchard-Kahn; the residual objective shapes only second-order-and-beyond deviations. Two consequences: the closed-loop linearization at $s^*$ is seed-invariant by algebra (identical to six digits across seeds in §0), and the pin holds the policy at the *certainty-equivalent* rest point, suppressing whatever risk correction the network would otherwise carry (§2.6).

### 2.4 Loss and Optimizer

The certified recipe is *composite*: the residual term plus, every step, (i) an **anchor** term, weight 1.0, penalizing distance from $\pi_{\text{BK}}$ on a fixed 128-point cloud around $s^*$, with a kink-aware **gate** muting the 21 cloud points where the linear Taylor rate lies below the floor (the linearization is the wrong local model there); (ii) a **Jacobian** term, weight 0.1, on $D\hat\pi(s^*) - P$ (redundant under the pin); (iii) barrier terms at the Calvo edge and at zero leverage, and a penalty on the Newton residual of §1.3, weights 0.01. The anchor never decays. Per-equation residual gradients are combined by one-shot **PCGrad** (conflicting components projected out before summing; the auxiliary terms added unprojected), motivated by measured gradient cosines of $-0.9$ between the price and wage Phillips residuals at the unpinned basin.

### 2.5 The Certification Stack — what "solved" means here

A small training loss is not a certificate; best-by-loss checkpoints are systematically the certificate-worst ones (§3.2). Claims close on the following legs, in order of strictness, at the **frozen convention** (final checkpoint of a 3000-episode run, fp64, three fixed seeds):

1. **Held-out and stress residuals**: $(\mathbb{E}[r])^2$ per equation on a pinned base grid and on a stress grid in the floor region ($m_p \in [-0.03, -0.01]$, low $R_{-1}$, below-target $\pi_{-1}$; 512 points).
2. **Learned-block spectral radius** of the closed-loop map $s \mapsto T(s, \hat\pi(s), 0)$ at $s^*$. The full Jacobian splits exactly into an autonomous exogenous block (largest root 0.98699, the investment-technology root times the guardrail's slope) and the learned $8\times8$ endogenous block; the reported number is the learned block's, because the raw radius can never read below 0.98699.
3. **Solved fixed point** $\hat s = T(\hat s, \hat\pi(\hat s), 0)$ by Newton: $\|\hat s - s^*\|$ and the learned radius at $\hat s$.
4. **Per-equation residuals at $\hat s$.**
5. **Long-horizon convergence** of the zero-shock path from $s^*$ to $\hat s$.

Under the pin, legs 2-at-$s^*$ and "policy error at $s^*$" hold by construction (*donated*); legs 1, 3, 4, 5 are *earned*. Every certificate in this document says which.

### 2.6 Deterministic, Risky, and Learned Rest Points

The pin targets the deterministic steady state $s^*$. The true zero-shock rest point of the stochastic economy is the risky steady state (Coeurdacier-Rey-Winant), displaced by precautionary effects the certainty-equivalent rule cannot express. We measured it (§3.4, §3.10) rather than argued about it: at the shipped calibration the Gaussian risky shift is at most 0.10% per state, so anchoring at $s^*$ is nearly free; at the author's calibration the disaster premium is material and *taxes capital* (capital, investment, consumption, wages down; leverage up), with the Calvo recursions the largest movers. The May version's prediction that precautionary saving would raise mean capital is withdrawn: it was not established by the formulation and is contradicted by the measurement. Note also that a constant precautionary level shift does not by itself change variances; the May std question is re-posed properly in §4.7.

---

## 3. Findings (July–September 2026)

### 3.1 Where this started: the May moment comparison, superseded

In May the residual ansatz with the K/F restriction reached a loss of $3.6\times10^{-6}$ (pooled residual RMS $\approx 6\times10^{-4}$), a median mean error of 0.44% against Dynare order-1 ergodic moments, and a median standard-deviation error of 80% — the last uniform across all arms. The comparison could not separate *selection* (which basin) from *accuracy* (how well within it), and the reference itself was the certainty-equivalent linear solution. The program therefore moved from moment matching to the certificates of §2.5. The std discrepancy was never explained and is not claimed to have been; the experiment that would settle it is §4.7. (The evaluator used a single continuous 2000-period path with burn-in, so episode resets and cross-path averaging are not the explanation.)

### 3.2 Residuals do not select (July 7)

Nine arms, three seeds each, full 3000-episode recipe, probed at final checkpoints in fp64:

| treatment | learned radius at $s^*$, per seed | rest-point error, per seed |
|---|---|---|
| baseline (ansatz + K/F restriction + anchor) | 1.057 / 1.021 / 1.023 | 1.1% / 2.8% / 0.7% |
| + kink-aware gate | 1.049 / 1.064 / 1.052 | 2.3% / 2.5% / 2.6% |
| + floor-region coverage | 1.060 (one seed) | 7.4% |
| gate + coverage | 1.064 / 1.148 / 1.293 | 0.9% / 1.3% / 6.2% |
| gate + drift penalty | 1.044 / 1.107 / 1.033 | 1.5% / 2.0% / 1.6% |
| gate + spectral penalty (two weights) | 1.02–1.28 | up to 49.7% |
| gate + PCGrad | **0.975** / 1.027 / 1.022 | **0.29%** / 4.1% / 7.0% |

Fifteen of fifteen runs across the first five treatments have an unstable learned block; the 1.02–1.06 attractor is reached by the baseline itself, and treatments at best match it. Best-by-loss checkpoints are worse than final ones on every certificate (baseline seed 0: radius 1.22 at best vs 1.057 at final): loss-based checkpoint selection selects *against* the certificates. The one crossing (PCGrad, seed 0) has an attribution problem: 1/3 vs 0/24, one-sided Fisher $p \approx 0.11$, optimizer norm effects uncontrolled.

### 3.3 The probe floor, and the first stable-near-truth policy (July 10)

An adversarial external review of the July 7 table produced two confirmed corrections. First, the raw closed-loop radius has a floor at 0.98699 (§2.5, leg 2); the corrected metric is the learned-block eigenvalue, and by it PCGrad seed 0 reads 0.9750 at $s^*$ and 0.9754 at its own rest point — a real, learned crossing that the floor had masked, not manufactured. Second, "contracts to the true rest point" was retracted: the zero-shock path converges (to $7\times10^{-14}$) to a *learned* fixed point displaced 0.83% from $s^*$ (worst: leverage), i.e. a locally stable economy 0.83% from the truth, with policy levels 0.29% off at $s^*$. Baseline and gated fixed points are 3–10% displaced *and* unstable; their trajectories escape to a 529%-displaced attractor created by the state guardrail.

### 3.4 The 0.83% is error, not economics (July 10)

The charitable reading of the displacement — the network found the risky steady state — was tested by computing that point directly: rest point under zero realized shocks with the equations holding in expectation over future Gaussian shocks, first-order future rules, $3^5$ Gauss-Hermite nodes, damped Newton in fp64, with a deterministic re-solve of the same machinery as control. The pure Gaussian risky shift at the shipped calibration is at most 0.10% per state (leverage $+0.100\%$, $q$ $-0.057\%$); the network's displacement is eight times larger *and of the opposite sign* (leverage $-0.827\%$). Conclusions: the displacement is approximation error; the certainty-equivalent anchor forbids a correction of only $\sim0.1\%$ here, so anchoring hard at $s^*$ is validated at this calibration; and $s^*$-referenced certificates are justified here and only here.

### 3.5 Selection by construction: the certified arm (July 14)

The BK pin (§2.3) on top of gate + PCGrad gives the certificate of §0 on 3/3 seeds. On the held-out stress grid its worst per-equation $(\mathbb{E}[r])^2$ is $1.0$–$2.1\times10^{-3}$ (worst equations: the wage-side $K_w$ definition and the investment Euler), against $4.9$–$8.5\times10^{-3}$ for the unpinned PCGrad arm, two of whose seeds are unstable; on the base grid the totals are $0.007$–$0.014$ vs $0.021$–$0.052$. The learned spectrum at $s^*$ is identical across seeds to six digits — that is the pin's algebra, not a training outcome — and the seed lottery that defined every other arm is abolished at the level of the certificate.

Certified claim, verbatim: *the disaster model at the shipped (disasterless) calibration is solved by selection by construction, per the full certification stack.* Not covered: the author's calibration.

### 3.6 What the pin buys over its donor (July 14)

The linearized policy alone, on the same pinned grids: stress max $(\mathbb{E}[r])^2$ $1.64\times10^{-2}$ vs the pinned arm's median $1.50\times10^{-3}$ (11×); base-grid total $3.94\times10^{-2}$ vs $7.67\times10^{-3}$ (5×). The linear rule passes the selection legs by definition and loses the accuracy legs — the division of labor the stack itself describes. Under the pin, network minus linear has zero value and slope at $s^*$ and *is* the learned higher-order content: on the base grid its magnitude scales with distance from $s^*$ at log-log slope 2.26 (2.0 would be pure quadratic), with medians 0.05–0.5% of policy scale and maxima on $q$ (7.3%) and $i$ (3.9%). The four Calvo heads read $\sim10^{-15}$: linear by the restriction of §2.2. The certified rest point's 0.0525% displacement tracks the shared machinery floor (soft clip plus linear future rule, 0.0496% on capital) almost state for state, so it is not network error.

### 3.7 Warm-start contamination (September 2)

Every July arm ran a constant-steady-state least-squares warm start on the anchored network before episode 1. On the shipped baseline recipe that fit alone moved the closed-loop radius at $s^*$ from the exogenous floor to 1.14 and the policies up to 16% off $\pi_{\text{BK}}$ *before the first training step*: the warm start taught the correction to cancel the linear slope. Pinned arms are insulated at first order by construction, so the certificate of §3.5 stands; the July 7 table (§3.2) is now read as "recovered from a contaminated start", and its re-run with the fix is owed. Method lesson adopted: certify what a run starts from, not only what it ends at.

### 3.8 The K/F restriction is not a gauge fix (September 10)

The test of §1.4. The name is corrected in this document; the network documentation and the program's decision ledger still carry the old rationale and are being corrected. The May finding (§3.1) is re-read as basin selection by a restriction.

### 3.9 Standing findings of the whole-library review (September 2)

| finding | content | status |
|---|---|---|
| Three undeclared selection devices | 15% re-seeding at $s^*$, the soft-clipped transition, the fixed 128-point anchor cloud with its own never-decaying loss term — the training measure is *ergodic ∪ SS-transients under the clipped transition*, the objective is *residual + 1.0·anchor + 0.1·Jacobian* | now declared (§2, §2.4); whether any belongs in the claim is §4.5 |
| Donated legs | under the pin the SS-error and radius-at-$s^*$ legs hold for every weight vector | stated in every table here; a pin-alone arm (no gate, no PCGrad) would show what the other devices add and has not been run |
| Anchor never decays | weight 1.0 throughout, in 44 of 44 configurations; not in any claim | open |
| Units | the scalar loss averages squared residuals of eleven equations in mixed units; the single-draw evaluator likewise | open, §4.6 |
| Guardrail inside the certified loop | the soft clip is inside the transition during training and inside the probe; identity to $10^{-6}$ at $s^*$, but the long-horizon and stress legs run under the clipped map, and the raw-radius floor is the exogenous root times the clip's slope | a clip-off probe was never run, §4.3 |

### 3.10 The gap at the author's calibration (July 10 measurement)

At $p_{\text{disaster}} = 1\%$, $\theta = 15\%$ the risky steady state (Bernoulli mixture, three-system decomposition into machinery, Gaussian and disaster parts, Newton to $4\times10^{-15}$) moves materially and coherently: capital, investment, consumption and wages down 0.12–0.14%, leverage $+0.22\%$, $\lambda_z$ $+0.12\%$, inflation and the rate up $\sim0.07\%$, and the Calvo recursions $K_p, K_w, F_p, F_w$ up 0.41–0.66% — linear in $\theta$. The totals the successor program must hit are, on states, leverage $+0.217\%$, consumption $-0.133\%$, capital $-0.115\%$; on policies, $K_p$ $+0.66\%$, $K_w$ $+0.64\%$, $F_p$ $+0.56\%$, $F_w$ $+0.41\%$. The certified $p = 0$ policy, graded against them, carries 14–46% of these shifts: the certainty-equivalent pin holds it at $s^*$ (it carries $\sim0.3$ of even the Gaussian shift at the shipped calibration), which is harmless at $p = 0$ and exactly the mechanism that must change at $p = 1\%$. The risk correction there is of the size of the whole error budget of the best unpinned run.

---

## 4. Questions for External Review

1. **Donated versus earned.** Under the pin, two legs of the certificate are theorems about the construction, not results of training. Is a certificate whose stability leg holds by construction a certificate of the *solution*, and what leg — independent of the pin — would make stability *earned*? Is the learned radius at the solved rest point $\hat s$ (0.9768, earned, but $\hat s$ is 0.05% from the pinned point) enough?

2. **The K/F restriction.** Four of eleven policies are held exactly at the first-order rule; the remaining seven carry the correction and must satisfy recursions whose auxiliaries cannot move. What does that cost, and how should it be measured before training at the author's calibration, where those four are the largest risk movers? Candidates: a restriction-off arm with the auxiliaries normalized by their steady-state scale; or eliminating the definition equations exactly by substituting $K_p = A_p(\pi)\,F_p$ into the recursions. Is there a model class where the restriction is known to lose essential nonlinear content?

3. **The guardrail in the loop.** The certified closed-loop map contains the soft clip, and the floor smoothing adds 14 basis points at the bound. One probe with the clip removed would show whether the certificate is a property of the model. Is there a principled way to certify *the model* when training required the guardrail?

4. **Retargeting the pin.** At $p = 1\%$ the pin must move from $s^*$ to a risky rest point that the network's own policy determines. The value can be pinned to the risky-steady-state solve of §3.10; the tangent has no comparable donor (the first-order risky rule is one choice). What is the right object to pin, and can a pinned network capture a disaster premium no linear anchor expresses?

5. **Selection devices in the claim.** Which of the three devices of §3.9 — re-seeding, the clipped transition, the never-decaying anchor — must appear in the statement of what was solved, and is "trained on ergodic ∪ SS-transients under the clipped map" an acceptable claim?

6. **Units.** The loss averages squared expected residuals of eleven equations in their natural (mixed) units, and the grading evaluator does the same with single draws. Is there a standard dimensionless normalization the field has converged on for residual stacks of this kind, and does the choice change which basin is selected?

7. **The small-shock limit and the order-1 identity.** The May std miss was never explained. Two experiments would settle what it was: (a) confirm that the pin's donor $P$ equals the reference decision rule at identical states and that the linear policy under the linearized transition reproduces the order-1 simulation under identical innovations; (b) solve at shock scales $1, \tfrac12, \tfrac14, \tfrac18$ with disasters off and away from the bound and report signed per-variable ratios $\hat\sigma_j/\sigma_j^{\text{ref}}$ and mean errors in units of $\sigma_j^{\text{ref}}$. Is a persistent ratio far from one in that limit ever attributable to economics, or only to a slope, transition or estimator error?

8. **Formulation.** Three questions the exposition used to hide. (a) There is no price- or wage-dispersion state and no dispersion wedge in the resource constraint; if the author's reduction dropped it deliberately, the model is "nonlinear except for Calvo aggregation", and the risky steady state of §3.10 is computed on that reduced model — does that matter at $p = 1\%$? (b) The bank-participation root is taken by Newton without a feasibility or root-selection check although the left-hand side is not globally monotone. (c) The exogenous processes are now specified per variable (§1.1); is a persistence-free monetary shock the intended reading of the original?
