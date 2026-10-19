# Disaster NK-DSGE — Model Specification

A self-contained mathematical specification of a medium-scale New Keynesian DSGE model with a Bernanke-Gertler-Gilchrist financial accelerator, an effective lower bound on the policy rate, and an exogenous Barro-style disaster shock on capital. Read this cold, ignore any solution method preconceptions, and tell me what you see.

The framework is a small extension of Christiano-Motto-Rostagno (2014) — the same structural blocks (Calvo prices, Calvo wages, entrepreneur financial frictions, monetary policy with smoothing) augmented with (a) a hard effective lower bound on the gross nominal rate, (b) an exogenous capital-destruction disaster mechanism, and (c) the standard 5-shock specification (productivity, investment-specific tech, government spending, trend growth, monetary policy).

The questions at the end are open-ended on purpose. I'm interested in what *you* think is interesting or under-explored, not what I think.

---

## 1. Notation and Timing

All variables are at quarterly frequency. A subscript $t$ denotes period $t$; lagged variables enter as separate state variables denoted with a $\text{lag}$ subscript, e.g., $k_{\text{lag},t} \equiv k_{t-1}$. Expectations $\mathbb{E}_t$ are conditional on the period-$t$ information set.

Variables grouped by economic role:

- **Endogenous predetermined states** (8): $\pi_{\text{lag}}, k_{\text{lag}}, c_{\text{lag}}, q_{\text{lag}}, i_{\text{lag}}, R_{\text{lag}}, \tilde w_{\text{lag}}, L_{\text{lag}}$
- **Exogenous states** (5): $\varepsilon, \mu_\Upsilon, g, \mu_z, m_p$ (each a univariate Markov process — see §11)
- **Controls / period-$t$ choices** (11): $\lambda_z, i, \pi, c, \tilde w, h, F_w, F_p, q, K_p, K_w$
- **Period-$t$ derived quantities** (computed analytically from state + controls): $s, \bar\omega, n, L, R, R^k, y_z, y, k$, plus the financial-friction objects $F(\bar\omega), G(\bar\omega), \Gamma(\bar\omega)$.

The four "auxiliary" controls $F_p, K_p, F_w, K_w$ are recursive Calvo discounted sums — see §4 and §5.

Two stochastic structures coexist:

- **Continuous shocks** $\xi_t = (\xi_\varepsilon, \xi_{\mu_\Upsilon}, \xi_{\mu_z}, \xi_g, \xi_{m_p})$: i.i.d. standard Gaussian, drives the 5 exogenous AR(1) processes.
- **Disaster shock** $d_t \in \{0, 1\}$: i.i.d. Bernoulli with $\mathbb{P}(d_t = 1) = p_{\text{disaster}}$. Independent of $\xi_t$.

---

## 2. Households

A representative household has preferences

$$
\mathbb{E}_0 \sum_{t=0}^\infty \beta^t \left[ \log(c_t \mu_{z,t} - b\, c_{t-1}) - \psi_L \frac{h_t^{1+\sigma_L}}{1+\sigma_L} \right]
$$

with internal habit ($b \in [0,1)$), Frisch elasticity $1/\sigma_L$, trend-growth-adjusted consumption ($\mu_{z,t}$ is the gross trend-growth shock, normalizing units).

Budget constraint (real, after-tax):

$$
(1+\tau_c) c_t + B_t/P_t = (1-\tau_l) \tilde w_t h_t + R_{t-1} B_{t-1}/(P_t \mu_{z,t}) + (\text{transfers})
$$

with $B_t$ nominal one-period bonds, $P_t$ price level, $\tilde w$ real wage, $\tau_c, \tau_l$ consumption and labor taxes.

Define the marginal utility of consumption:
$$
\lambda_{z,t} \equiv \frac{1}{c_t \mu_{z,t} - b\, c_{t-1}} - \beta b\, \mathbb{E}_t \frac{1}{c_{t+1} \mu_{z,t+1} - b\, c_t}
$$

The household FOCs yield:

**Consumption Euler** (recursive form, with habit):
$$
\boxed{\quad (1+\tau_c)\, \lambda_{z,t}\, (c_t \mu_{z,t} - b\, c_{t-1}) + \beta b\, \frac{c_t \mu_{z,t} - b\, c_{t-1}}{c_{t+1} \mu_{z,t+1} - b\, c_t} = \mu_{z,t} \quad}
$$

This is the equation written as a residual in the implementation; the implementation outputs $\lambda_z$ directly as a control rather than computing it from $c, c_{-1}$ via the marginal-utility expression, then enforces consistency via this Euler condition.

**Bond Euler** (gross nominal rate $R_t$):
$$
\boxed{\quad \lambda_{z,t} = R_t\, \beta\, \mathbb{E}_t \frac{\lambda_{z,t+1}}{\pi_{t+1} \mu_{z,t+1}} \quad}
$$

**Labor supply / wage setting**: handled in §5 via Calvo wages, not as a household FOC (in this calibration the household supplies labor through a continuum of unions).

---

## 3. Final-Goods Producers

A competitive aggregator combines a continuum of differentiated intermediate goods $y_t(j)$ via Dixit-Stiglitz with elasticity $\lambda_f / (\lambda_f - 1)$:

$$
y_t = \left[ \int_0^1 y_t(j)^{1/\lambda_f}\, dj \right]^{\lambda_f}, \qquad \lambda_f > 1
$$

Cost minimization gives demand for variety $j$:
$$
y_t(j) = (P_t(j)/P_t)^{-\lambda_f/(\lambda_f-1)} y_t
$$

with the price index $P_t = [\int P_t(j)^{1/(1-\lambda_f)} dj]^{1-\lambda_f}$.

---

## 4. Intermediate-Goods Producers (Calvo Prices)

Each intermediate firm $j$ produces with technology

$$
y_t(j) = \varepsilon_t (k_{t-1}(j)/\mu_{z,t})^\alpha h_t(j)^{1-\alpha} - \Phi
$$

with $\Phi$ a fixed cost (capturing zero economic profits in steady state). Real marginal cost is identical across firms:

$$
s_t = \frac{1}{\varepsilon_t} \left( \frac{\mu_{z,t} h_t}{k_{t-1}} \right)^\alpha \frac{\tilde w_t}{1-\alpha}
$$

The capital rental rate is:
$$
r^k_t = \alpha \varepsilon_t (\mu_{z,t} h_t / k_{t-1})^{1-\alpha} s_t
$$

Each period, a fraction $1 - \xi_p$ of firms can reoptimize their price; the remaining fraction $\xi_p$ index to lagged inflation:
$$
P_t(j) = \tilde\pi_t P_{t-1}(j), \qquad \tilde\pi_t \equiv (\pi^{ss})^\iota \pi_{t-1}^{1-\iota}
$$

Reoptimizing firms choose $\tilde P_t$ to maximize expected discounted profits. The optimality condition can be written recursively. Define:

$$
F_{p,t} \equiv \mathbb{E}_t \sum_{j \ge 0} (\beta \xi_p)^j \lambda_{z,t+j}\, y_{z,t+j} \prod_{k=1}^j \left( \frac{\tilde\pi_{t+k}}{\pi_{t+k}} \right)^{1/(1-\lambda_f)}
$$

$$
K_{p,t} \equiv \mathbb{E}_t \sum_{j \ge 0} (\beta \xi_p)^j \lambda_{z,t+j}\, y_{z,t+j}\, s_{t+j}\, \lambda_f \prod_{k=1}^j \left( \frac{\tilde\pi_{t+k}}{\pi_{t+k}} \right)^{\lambda_f/(1-\lambda_f)}
$$

with $y_{z,t} \equiv \varepsilon_t (k_{t-1}/\mu_{z,t})^\alpha h_t^{1-\alpha} - \Phi$.

These satisfy two relationships, each of which becomes an equilibrium equation:

**Forward recursion of $F_p$** (eq. 1):
$$
\boxed{\quad F_{p,t} = \lambda_{z,t}\, y_{z,t} + \beta \xi_p\, \mathbb{E}_t \left[ (\tilde\pi_{t+1}/\pi_{t+1})^{1/(1-\lambda_f)} F_{p,t+1} \right] \quad}
$$

**Definition of $K_p$ from optimal price** (eq. 2a, *no expectation*):
$$
\boxed{\quad K_{p,t} = F_{p,t} \cdot \left[ \frac{1 - \xi_p (\tilde\pi_t/\pi_t)^{1/(1-\lambda_f)}}{1 - \xi_p} \right]^{1-\lambda_f} \quad}
$$

**Forward recursion of $K_p$** (eq. 2b):
$$
\boxed{\quad K_{p,t} = \lambda_f\, \lambda_{z,t}\, y_{z,t}\, s_t + \beta \xi_p\, \mathbb{E}_t \left[ (\tilde\pi_{t+1}/\pi_{t+1})^{\lambda_f/(1-\lambda_f)} K_{p,t+1} \right] \quad}
$$

Equation 2a is the optimal-reset-price condition (no expectation — it's a static algebraic identity at each $t$). Equation 2b is the forward recursion that makes $K_p$ well-defined. Together they pin down $\pi_t$ given $F_p, K_p$.

**Calvo validity edge.** The bracketed term in the $K_p$ definition is $(1 - \xi_p (\tilde\pi/\pi)^{1/(1-\lambda_f)})/(1-\xi_p)$. For $\lambda_f > 1$, the exponent $1/(1-\lambda_f)$ is negative, so as $\pi/\tilde\pi$ rises, the term $\xi_p (\tilde\pi/\pi)^{1/(1-\lambda_f)}$ rises, and the numerator can go negative. With $\xi_p = 0.6, \lambda_f = 1.2$: validity requires $\pi/\tilde\pi \lesssim 1.1$. Beyond that, $K_p$ has no real solution — the Calvo price aggregator becomes ill-defined. **The natural domain of equilibrium $\pi$ is bounded above** by this hard kink.

---

## 5. Labor Unions (Calvo Wages)

A continuum of unions sets wages with Calvo friction. Each period a fraction $1 - \xi_w$ can reoptimize; the rest index to lagged inflation $\tilde\pi_w \equiv (\pi^{ss})^{\iota_w} \pi_{t-1}^{1-\iota_w}$.

Define wage inflation $\pi_{w,t} \equiv \pi_t \tilde w_t / \tilde w_{t-1}$ and the wage-trend factor $\Lambda_{w,t} \equiv \mu_{z,t}^{\iota_\mu / (1-\lambda_w)} (\mu_z^{ss})^{(1-\iota_\mu)/(1-\lambda_w)}$.

The auxiliary recursive variables:

$$
F_{w,t} \equiv \mathbb{E}_t \sum_{j \ge 0} (\beta \xi_w)^j (1-\tau_l)\, \lambda_{z,t+j}\, h_{t+j} \prod_{k=1}^j \frac{\Lambda_{w,t+k}\, \tilde\pi_{w,t+k}^{1/(1-\lambda_w)}}{\pi_{w,t+k}^{\lambda_w/(1-\lambda_w)} \cdot \pi_{t+k}}
$$

$$
K_{w,t} \equiv \mathbb{E}_t \sum_{j \ge 0} (\beta \xi_w)^j h_{t+j}^{1+\sigma_L} \prod_{k=1}^j \left( \frac{\tilde\pi_{w,t+k} \mu_z^{ss} / \pi_{w,t+k}}{\cdot} \right)^{\lambda_w (1+\sigma_L)/(1-\lambda_w)}
$$

The three resulting equilibrium equations:

**Eq. 3 — Wage Phillips ($F_w$ recursion):**
$$
F_{w,t} = (1-\tau_l)\frac{h_t}{\lambda_w}\lambda_{z,t} + \beta \xi_w\, \mathbb{E}_t \left[ \Lambda_{w,t+1}\, \tilde\pi_{w,t+1}^{1/(1-\lambda_w)}\, \pi_{w,t+1}^{-\lambda_w/(1-\lambda_w)}\, \pi_{t+1}^{-1}\, F_{w,t+1} \right]
$$

**Eq. 4a — Wage definition (no expectation):**
$$
K_{w,t} = \frac{1}{\psi_L} \left[ \frac{1 - \xi_w (\tilde\pi_{w,t} \mu_z^{ss} / \pi_{w,t})^{1/(1-\lambda_w)}}{1 - \xi_w} \right]^{1-\lambda_w(1+\sigma_L)} \tilde w_t F_{w,t}
$$

**Eq. 4b — $K_w$ recursion:**
$$
K_{w,t} = h_t^{1+\sigma_L} + \beta \xi_w\, \mathbb{E}_t \left[ (\tilde\pi_{w,t+1} \mu_z^{ss} / \pi_{w,t+1})^{\lambda_w(1+\sigma_L)/(1-\lambda_w)} K_{w,t+1} \right]
$$

The wage-side Calvo validity edge mirrors the price side.

---

## 6. Capital Producers and Investment

Capital producers transform investment into installed capital subject to a quadratic adjustment cost on the *rate of change* of investment:

$$
S(x) = \tfrac{1}{2} \kappa (x - \mu_z^{ss})^2, \qquad x_t \equiv \mu_{z,t} i_t / i_{t-1}
$$

Capital evolves:

$$
k_t = (1-\delta) k_{t-1}/\mu_{z,t} + (1 - S(x_t))\, i_t
$$

Investment-specific technology shock $\mu_\Upsilon$ multiplies investment in the resource constraint (§10) but not here.

The investment FOC gives Tobin's $q$:

**Eq. 7 — Investment Euler:**
$$
\boxed{\quad 1 = \mu_\Upsilon q_t \big( 1 - S(x_t) - x_t S'(x_t) \big) + \beta \mu_\Upsilon\, \mathbb{E}_t \left[ \frac{\lambda_{z,t+1}}{\lambda_{z,t}}\, q_{t+1} \mu_{z,t+1} (i_{t+1}/i_t)^2 S'(x_{t+1}) \right] \quad}
$$

---

## 7. Entrepreneurs and Financial Frictions (Bernanke-Gertler-Gilchrist)

Entrepreneurs purchase capital using internal net worth $n$ plus loans from a competitive banking sector. Each entrepreneur faces an idiosyncratic productivity shock $\omega \sim \log\mathcal{N}(-\sigma_\omega^2/2,\, \sigma_\omega^2)$ on their capital purchase, realized after the period's aggregate state. Define:

$$
F(\bar\omega) = \Phi\!\left( \frac{\log \bar\omega + \sigma_\omega^2/2}{\sigma_\omega} \right), \quad
G(\bar\omega) = \Phi\!\left( \frac{\log \bar\omega - \sigma_\omega^2/2}{\sigma_\omega} \right)
$$

$$
\Gamma(\bar\omega) = \bar\omega (1 - F(\bar\omega)) + G(\bar\omega), \quad \Gamma'(\bar\omega) = 1 - F(\bar\omega)
$$

with $\Phi$ the standard normal CDF. $F$ is the default probability, $G$ is the partial expectation $\mathbb{E}[\omega \mathbf{1}\{\omega < \bar\omega\}]$, and $\Gamma$ is the bank's expected gross share of returns (before monitoring costs).

Banks pay monitoring cost $\mu_{\text{mon}} G(\bar\omega)$ per unit when default occurs, recovering only $G(\bar\omega) - \mu_{\text{mon}} G(\bar\omega)$ from defaulted assets.

The aggregate gross return on capital (composite of dividend + capital gain net of taxes):

$$
R^k_t = \frac{(1-\tau_k) r^k_t + (1-\delta) q_t}{q_{t-1}}\, \pi_t + \tau_k \delta
$$

**Bank participation constraint** (eq. 8 in implementation; here algebraic — pins down $\bar\omega$ given leverage and prices):
$$
\bar\omega_t (1 - F(\bar\omega_t)) + (1 - \mu_{\text{mon}}) G(\bar\omega_t) = \frac{L_{t-1} - 1}{L_{t-1}\, R^k_t / R_{t-1}}
$$

This is a **smooth, monotonically increasing** function of $\bar\omega$ on $[0, \approx 1.1]$ (at the calibration, $\bar\omega_{ss} \approx 0.488$ and $h'(\bar\omega_{ss}) \approx 0.98$), so the constraint pins down $\bar\omega$ uniquely.

**Net-worth evolution:**
$$
n_t = \frac{\gamma_e}{\pi_t \mu_{z,t}} (1 - \Gamma(\bar\omega_t))\, R^k_t\, q_{t-1}\, k_{t-1} + w^e
$$

with $\gamma_e$ the entrepreneur survival rate and $w^e$ a small wage to newborn entrepreneurs.

**Leverage** (balance-sheet identity):
$$
L_t = \frac{q_t k_t}{n_t}
$$

**Eq. 8 — Entrepreneur contract** (the FOC of optimal default-threshold choice given the participation constraint, written as a residual):
$$
\boxed{\quad \frac{R^k_{t+1}}{R_t} (1 - \Gamma(\bar\omega_{t+1})) - \frac{\Gamma'(\bar\omega_{t+1})}{\Gamma'(\bar\omega_{t+1}) - \mu_{\text{mon}} G'(\bar\omega_{t+1})} \left[ 1 - \frac{R^k_{t+1}}{R_t} (\Gamma(\bar\omega_{t+1}) - \mu_{\text{mon}} G(\bar\omega_{t+1})) \right] = 0 \quad}
$$

This is the only forward-looking financial-friction equation; the rest are algebraic at $t$.

---

## 8. Government and Monetary Authority

**Fiscal**: lump-sum government spending $g_t$ (exogenous AR(1)); distortionary taxes $\tau_c, \tau_l, \tau_k$ on consumption, labor, and capital income, all constant.

**Monetary**: Taylor-type rule with smoothing, augmented with an effective lower bound:
$$
R^{\text{Taylor}}_t = R^{ss}\, (R_{t-1}/R^{ss})^{\rho_p} \left[ (\pi_t/\pi^{ss})^{\alpha_\pi} (y_t/y^{ss})^{\alpha_y} \right]^{1-\rho_p} \exp(m_{p,t})
$$

The realized policy rate is the un-floored Taylor prescription, then floored:
$$
\boxed{\quad R_t = R^{lb} + \frac{1}{\sharp} \log\!\left( 1 + \exp\big(\sharp (R^{\text{Taylor}}_t - R^{lb})\big) \right) \quad}
$$

with $R^{lb} = 1$ (zero net rate) and sharpness $\sharp = 500$. The softplus floor introduces an essentially-hard kink at $R = R^{lb}$ while preserving differentiability.

The bond Euler in §2 uses this $R_t$.

---

## 9. Disaster Mechanism

Each period, with probability $p_{\text{disaster}} \in [0, 0.10]$ in the calibration, a disaster realizes ($d_t = 1$) and capital entering next period is destroyed by fraction $1 - e^{-\theta_{\text{disaster}}}$:

$$
k_t \to k_t \cdot e^{-\theta_{\text{disaster}} d_t}
$$

The disaster shock is i.i.d. and independent of the continuous shocks. It enters every conditional expectation as a mixture:

$$
\mathbb{E}_t[X_{t+1}] = (1 - p_{\text{disaster}})\, \mathbb{E}_t[X_{t+1} | d_{t+1} = 0] + p_{\text{disaster}}\, \mathbb{E}_t[X_{t+1} | d_{t+1} = 1]
$$

The disaster branch evaluates next-period quantities at the destroyed capital level.

This is the only non-Gaussian element. Default calibration: $p_{\text{disaster}} = 0$ (model reduces to standard CMR). Stress runs use $p_{\text{disaster}} = 0.10$, $\theta_{\text{disaster}} = 0.05$.

---

## 10. Resource Constraint and Output

Aggregate resource constraint (eq. 9, written in the same residual form as elsewhere):
$$
y_{z,t} = g_t + c_t + i_t/\mu_{\Upsilon,t} + \Theta \frac{1-\gamma_e}{\gamma_e} (n_t - w^e) + \mu_{\text{mon}} G(\bar\omega_t)\, R^k_t\, \frac{q_{t-1} k_{t-1}}{\mu_{z,t} \pi_t}
$$

The third term is consumption by exiting entrepreneurs ($\Theta$ is a calibration constant). The fourth is bank monitoring costs. $y_{z,t}$ is gross output net of fixed costs (defined in §4); $y_t \equiv y_{z,t}$ in this calibration.

---

## 11. Exogenous Processes

Five univariate AR(1)s in logs (or in levels with reflection at zero where appropriate):

$$
\log \varepsilon_t = \rho_\varepsilon \log \varepsilon_{t-1} + \sigma_\varepsilon \xi_{\varepsilon,t}
$$
$$
\log \mu_{\Upsilon,t} = \rho_{\mu_\Upsilon} \log \mu_{\Upsilon,t-1} + \sigma_{\mu_\Upsilon} \xi_{\mu_\Upsilon,t}
$$
$$
\log \mu_{z,t} = \rho_{\mu_z} \log \mu_{z,t-1} + (1 - \rho_{\mu_z}) \log \mu_z^{ss} + \sigma_{\mu_z} \xi_{\mu_z,t}
$$
$$
\log g_t = \rho_g \log g_{t-1} + (1 - \rho_g) \log g^{ss} + \sigma_g \xi_{g,t}
$$
$$
m_{p,t} = \sigma_{m_p} \xi_{m_p,t} \quad \text{(white noise)}
$$

All $\xi \sim \mathcal{N}(0,1)$ i.i.d. Plus the disaster Bernoulli $d_t \sim \text{Bernoulli}(p_{\text{disaster}})$, independent of the $\xi$s.

---

## 12. Calibration

| Block | Param | Value | Comment |
|---|---|---|---|
| Preferences | $\beta$ | 0.9985 | discount |
| | $b$ | 0.74 | habit |
| | $\sigma_L$ | 1.0 | inverse Frisch |
| | $\psi_L$ | 0.7705 | labor disutility scale |
| Production | $\alpha$ | 0.4 | capital share |
| | $\delta$ | 0.025 | depreciation |
| | $\kappa$ | 2.0 | investment-adj cost curvature |
| | $\Phi$ | 0.606 | fixed cost (zero-profit SS) |
| | $\lambda_f$ | 1.2 | price markup |
| | $\lambda_w$ | 1.2 | wage markup |
| Calvo | $\xi_p$ | 0.6 | price stickiness |
| | $\xi_w$ | 0.6 | wage stickiness |
| | $\iota$ | 0.9 | price indexation |
| | $\iota_w$ | 0.49 | wage indexation |
| | $\iota_\mu$ | 0.94 | trend-growth indexation |
| Mon. policy | $\rho_p$ | 0.85 | Taylor rule smoothing |
| | $\alpha_\pi$ | 1.5 | Taylor rule inflation coef |
| | $\alpha_y$ | 0.36 | Taylor rule output coef |
| | $R^{lb}$ | 1.0 | gross-rate floor |
| Taxes | $\tau_c, \tau_l, \tau_k$ | 0.047, 0.24, 0.32 | distortionary |
| BGG | $\Theta$ | 0.005 | exit-entrepreneur consumption |
| | $\gamma_e$ | 0.985 | entrepreneur survival |
| | $w^e$ | 0.005 | wage to newborn entrepreneurs |
| | $\sigma_\omega$ | 0.268 | idiosyncratic productivity sd |
| | $\mu_{\text{mon}}$ | 0.22 | monitoring cost |
| Targets | $\pi^{ss}$ | 1.006 | quarterly inflation, 2.4% annual |
| | $\mu_z^{ss}$ | 1.0041 | quarterly trend, 1.6% annual |
| | $R^{ss}$ | 1.0117 | quarterly gross rate |
| | $y^{ss}$ | 3.0308 | output target |
| | $g^{ss}$ | 0.616 | gov't spending target |
| Shocks ($\rho, \sigma$) | $\varepsilon$ | (0.809, 0.0046) | productivity |
| | $\mu_\Upsilon$ | (0.987, 0.004) | inv-specific tech |
| | $\mu_z$ | (0.146, 0.00715) | trend growth |
| | $g$ | (0.94, 0.023) | gov't spending |
| | $m_p$ | (—, 0.0049) | monetary (white noise) |
| Disaster | $p_{\text{disaster}}$ | 0.0–0.10 | per-period probability |
| | $\theta_{\text{disaster}}$ | 0.05 | log-fraction destroyed |

The Calvo parameters at $\xi_p = \xi_w = 0.6$ are at the standard New Keynesian calibration; **$\xi_p = 0.6$ together with $\lambda_f = 1.2$ is load-bearing for Blanchard-Kahn determinacy** in this calibration. Lowering $\xi_p$ to 0.5 breaks determinacy (14 stable eigenvalues vs. expected 13) without recalibration of other parameters.

---

## 13. Deterministic Steady State

Setting all shocks to their unconditional means and $d_t = 0$, the steady state is solved numerically (max $|\text{residual}| < 10^{-7}$). Numerical values:

| Variable | SS value |
|---|---|
| $\pi^{ss}$ | 1.012 |
| $k^{ss}$ | 27.42 |
| $c^{ss}$ | 1.594 |
| $q^{ss}$ | 1.000 |
| $i^{ss}$ | 0.795 |
| $R^{ss}$ | 1.018 |
| $\tilde w^{ss}$ | 1.920 |
| $L^{ss}$ (leverage) | 1.966 |
| $\lambda_z^{ss}$ | 0.602 |
| $h^{ss}$ | 0.944 |
| $F_w^{ss}$ | 0.885 |
| $F_p^{ss}$ | 4.736 |
| $K_p^{ss}$ | 4.831 |
| $K_w^{ss}$ | 2.207 |
| $\bar\omega^{ss}$ | 0.488 |

(Note: $R^{ss} = 1.018$ is well above the floor $R^{lb} = 1.0$, so the ELB is slack at SS.)

---

## 14. Structural Properties Worth Naming

**Determinacy.** The linearized model has 13 endogenous state equations; Blanchard-Kahn requires exactly 13 stable generalized eigenvalues. At the default calibration this holds. Sensitivity is sharp around $\xi_p$.

**Calvo validity edge.** As noted in §4, the price aggregator becomes ill-defined for $\pi/\tilde\pi \gtrsim 1.1$. This is a **hard upper bound** on the natural domain of $\pi$ — not a soft penalty, an algebraic obstruction.

**ELB kink.** $R$ floors at $R^{lb} = 1$ via a high-sharpness softplus. Under the default calibration the ELB is slack at SS; under stress shocks it can bind. The softplus is $C^\infty$ but the second derivative is sharply peaked at the kink — locally, the policy function has near-discontinuous slope.

**The K/F variables.** $F_p, K_p, F_w, K_w$ are recursive discounted sums (§4–5). They appear *only* in their own definition equations and their own recursions — they do not enter the consumption Euler, the bond Euler, the investment Euler, or the resource constraint at all. Their *level* matters for $\pi$ and $\tilde w$ (via the algebraic definition equations 2a, 4a), and the recursions 1, 2b, 3, 4b pin them down forward-looking from $\lambda_z, y_z, s, h$ and the inflation indices. So the *analytical* equilibrium pins K/F levels uniquely. However, the equation system has the structural feature that K/F errors propagate into $\pi, \tilde w$ via 2a, 4a *without* feeding back into the household, bond, investment, or financial blocks — making K/F a *propagation channel* rather than a *binding constraint* on the level variables. Empirically this affects how residual-minimizing approximations behave, but whether it has an analytical-equilibrium analogue (a near-degeneracy in the equation Jacobian, or otherwise) is worth examining.

**Disaster mixture.** Conditional expectations split into $(1 - p_{\text{disaster}}) \cdot \mathbb{E}[\cdot | d=0] + p_{\text{disaster}} \cdot \mathbb{E}[\cdot | d=1]$. The disaster branch evaluates next-period quantities at $k \cdot e^{-\theta}$, which for $\theta = 0.05$ shifts the conditional next-period distribution by ~5% of capital. This is a *finite* discrete jump — different in character from continuous shock dispersion.

**Trend stationarity.** The model is written in trend-stationary form: $\mu_{z,t}$ scales out of the dynamics, and SS values are at the balanced growth path. All Euler equations include explicit $\mu_z$ terms accounting for this.

**Effective state dimension.** Of the 13 states, 5 are exogenous AR(1)s with their own dynamics and 8 are endogenous lags. The "effective" state for policy purposes (after accounting for any redundancy from balance-sheet identities and the leverage definition) may be smaller — one can ask whether the 8-dimensional endogenous state is reducible.

**Existence and uniqueness.** Standard NK-DSGE existence/uniqueness arguments cover the model with $p_{\text{disaster}} = 0$. The disaster extension introduces a non-Gaussian forcing whose effect on equilibrium uniqueness has not been formally characterized (to my knowledge).

---

## 15. Open Questions

These are deliberately open. I'm interested in what you find structurally interesting, what you'd want to know more about, what you'd do with the model, and what surprises you.

1. **Numerical character.** What kind of numerical solver would you reach for, given the model as stated? Why? What features of the model push toward certain methods and away from others?

2. **State-space reduction.** Is there structure that suggests the effective endogenous state dimension is less than 8? (e.g., balance-sheet identities reducing $L$, $n$ to functions of more primitive states.)

3. **Calvo validity edge.** How would you handle the hard upper bound on $\pi$ from §4? Is this typically treated as a domain restriction in solver design, or is there a reformulation that avoids it?

4. **The K/F gauge property.** Look at the role of $F_p, K_p, F_w, K_w$ in the system. They are tightly defined by their own recursions and definition equations, but they do not enter the household, bond, investment, or resource equations directly. Does this create a true equilibrium-level multiplicity, or only a numerical-identification issue in particular residual formulations? What's the cleanest way to reason about it?

5. **Stochastic vs deterministic SS.** The SS in §13 is deterministic. With non-trivial $\sigma_\omega$, $\sigma_\varepsilon$, and especially $p_{\text{disaster}} > 0$, the stochastic / risky steady state will differ. By how much, in your estimation, given the shock variances and the curvature of the financial-accelerator block?

6. **Disaster mechanism.** $p_{\text{disaster}} = 0.10$ per quarter is a very high disaster intensity. What does this calibration imply about the asset-pricing implications of the model? What's the natural role for the disaster shock in the welfare/business-cycle structure?

7. **The financial accelerator block.** The bank-participation constraint is smooth and monotonic at $\bar\omega \approx 0.488$, but $\bar\omega$ enters the entrepreneur contract (eq. 8) nonlinearly through $\Gamma, \Gamma', G, G'$. Does this block introduce non-monotonicities or non-uniqueness in the policy correspondence that aren't visible at SS?

8. **Determinacy boundary.** $\xi_p = 0.6$ is at the determinacy edge for this calibration. What's the policy-rate-rule perspective on this — is it a fragile feature of the calibration or a deep structural property?

9. **What's missing from the standard NK-DSGE solver toolkit for handling models like this?** Is there a methodology gap you see?

10. **What surprises you?** Anything in the specification you find unusual, mis-specified, or worth probing harder.

---

## References

- Christiano, Motto, Rostagno (2014), "Risk Shocks", *AER* — the core CMR framework with financial frictions.
- Bernanke, Gertler, Gilchrist (1999), "The Financial Accelerator in a Quantitative Business Cycle Framework" — the BGG entrepreneur block.
- Calvo (1983), "Staggered Prices in a Utility-Maximizing Framework" — the Calvo nominal-rigidity setup.
- Smets, Wouters (2007), "Shocks and Frictions in US Business Cycles" — adjacent calibration target.
- Barro (2006), "Rare Disasters and Asset Markets in the Twentieth Century" — disaster mechanism.
- Coeurdacier, Rey, Winant (2011), "The Risky Steady State" — for §14 stochastic-SS framing.
- Blanchard, Kahn (1980), "The Solution of Linear Difference Models under Rational Expectations" — for §14 determinacy.
