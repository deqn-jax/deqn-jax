# Brock–Mirman (1972): stochastic growth, full depreciation

A classic with a known exact solution, used to show that the method works.

**Households / planner.** Maximize $E_0 \sum_t \beta^t \log c_t$.

**Technology.** $y_t = e^{z_t} k_t^{\alpha}$. Capital fully depreciates:
$k_{t+1} = y_t - c_t$.

**Shock.** $z_{t+1} = \rho z_t + \sigma \varepsilon_{t+1}$, $\varepsilon \sim N(0,1)$.

**Calibration.** $\alpha = 0.36$, $\beta = 0.96$, $\rho = 0.9$, $\sigma = 0.04$.

**Decision.** The savings rate $s_t = k_{t+1} / y_t$.

**Known solution.** $s_t = \alpha\beta$ in every state, i.e. $k_{t+1} = \alpha\beta\, e^{z_t} k_t^{\alpha}$.
