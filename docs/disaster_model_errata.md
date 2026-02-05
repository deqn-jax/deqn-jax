# Errata: CMR Disaster Model Implementation

**Date:** February 2026
**Authors:** Anna Smirnova
**Codebase:** `deqn-jax/src/deqn_jax/models/disaster/`

## Summary

During GPU training of the CMR-style NK-DSGE model with financial frictions
("disaster model"), two equilibrium equations — eq4 (wage Phillips K) and eq11
(resource constraint) — exhibited permanently elevated residuals (~13.1 and
~0.55 respectively) that did not improve with continued training, regardless
of optimizer choice (Adam, NGD, MAO, L-BFGS) or learning rate schedule.

Investigation revealed three compounding bugs in the model implementation.
Two are errors in the economic equations; the third is a consequence of the
first two propagating into the hardcoded steady-state values.

---

## Bug 1: Incorrect Entrepreneur Net Worth Formula

**File:** `equations.py`, `definitions()`, lines 79–82 (old code)
**Severity:** Critical — produces net worth ~2x too large

### The Error

The old code computed net worth as:

```
n = (gamma_e / (pi * mu_z)) * [R_k * q * k  -  R * q * k / L  -  mu * G * R_k * q * k]
    + w_e
    + gamma_e * R / (pi * mu_z) * q * k / L
```

Algebraic simplification reveals that the second and fifth terms cancel
(both equal `gamma_e * R * q * k / (L * pi * mu_z)` with opposite signs),
reducing the expression to:

```
n = (gamma_e / (pi * mu_z)) * (1 - mu * G(omega_bar)) * R_k * q * k  +  w_e
```

### Why This Is Wrong

In the BGG (Bernanke-Gertler-Gilchrist) financial accelerator framework,
the entrepreneur's retained share of gross capital returns is `(1 - Gamma)`,
not `(1 - mu * G)`.

These are the standard BGG distribution functions for the idiosyncratic
productivity shock omega ~ LogNormal:

| Function | Definition | Meaning |
|----------|-----------|---------|
| F(omega_bar) | Pr(omega < omega_bar) | Default probability |
| G(omega_bar) | E[omega \| omega < omega_bar] * F | Expected loss given default |
| Gamma(omega_bar) | omega_bar * (1 - F) + G | Bank's gross share of returns |

The entrepreneur keeps the complement of the bank's gross share:

```
Entrepreneur share = 1 - Gamma(omega_bar)
                   = 1 - [omega_bar * (1 - F) + G]
```

The old code used `(1 - mu * G)` instead, where `mu` is the monitoring cost
fraction and `G` is tiny (the expected value of defaulting entrepreneurs'
output). This expression is structurally incorrect — it mixes a cost
parameter (`mu`) with a conditional expectation (`G`) in a way that has no
economic interpretation in the BGG framework.

### Numerical Impact

At the calibrated `omega_bar = 0.489`, `sigma_omega = 0.268`:

| | Value |
|---|---|
| F(omega_bar) | 0.0056 |
| G(omega_bar) | 0.0025 |
| Gamma(omega_bar) | 0.4883 |
| **Old factor: 1 - mu * G** | **0.9995** |
| **Correct factor: 1 - Gamma** | **0.5117** |
| **Ratio** | **1.95x** |

The old formula gave `n = 27.35`; the correct formula gives `n = 14.01`.
For reference, the deterministic steady-state table in the model
documentation lists `n = 14.005`.

### The Fix

```python
# Old (incorrect):
n = (gamma_e / (pi * mu_z)) * (
    R_k * q * k  -  R * q * k / L
    - mu * G * R_k * q * k
) + w_e + gamma_e * R / (pi * mu_z) * q * k / L

# New (correct):
Gamma_val = Gamma(omega_bar, sigma_omega)
n = (gamma_e / (pi * mu_z)) * (1 - Gamma_val) * R_k * q * k  +  w_e
```

### Likely Origin of the Error

The original code appears to have attempted an alternative decomposition of
entrepreneur income based on "gross returns minus debt repayment minus
monitoring costs," but used `R * q * k / L` (which equals `R * n`, the
risk-free return on own capital) where the debt repayment term should have
been `R * q * k * (1 - 1/L)` (which equals `R * (q*k - n)`, interest on
borrowed funds). After this substitution error, the addition of a separate
`+ gamma_e * R * q * k / L` term at the end cancelled the incorrect
subtraction, leaving only the `(1 - mu*G)` factor — which is not the
correct entrepreneur share in any decomposition.

---

## Bug 2: Tautological Equation 12 (Capital Accumulation)

**File:** `equations.py`, line 183 (old code)
**Severity:** Critical — removes one equation from the system, leaving
leverage L unconstrained

### The Error

The old equation 12 was:

```python
residuals["eq12_capital_accumulation"] = (
    defs["k"] - (1 - delta) * k_lag / mu_z - (1 - S) * i
)
```

But `defs["k"]` is defined in `definitions()` as:

```python
k = (1 - delta) * k_lag / mu_z + (1 - S) * i
```

Therefore eq12 reduces to:

```
eq12 = [(1-d)*k/mu + (1-S)*i]  -  [(1-d)*k/mu]  -  [(1-S)*i]  =  0
```

**This is identically zero for all inputs.** Verified: evaluating with
uniformly random state and policy arrays produces `eq12 = 0.000000e+00`.

### Why This Matters

With eq12 always zero, the training loss was computed over only 11
independent equations for 12 policy variables. The leverage ratio `L`
appeared as a policy output of the neural network but was not constrained
by any equation in the loss function.

During training, the network could freely assign any value to `L` without
penalty. Since `L` feeds into net worth `n` (via the balance sheet identity
`n = q*k/L`), monitoring costs, and entrepreneur consumption — all of which
appear in eq8, eq9, eq11 — the unconstrained `L` corrupted these equations'
residuals.

### Why Capital Accumulation Was the Wrong Equation

In a DEQN (deep equilibrium network) framework, the capital law of motion

```
k_{t+1} = (1 - delta) * k_t / mu_{z,t+1} + (1 - S(...)) * i_t
```

is enforced by the **state transition function** (`dynamics.py:step()`),
which sets `next_state[k] = defs["k"]`. Including the same identity as an
equilibrium equation is redundant — it will always be satisfied by
construction. It is not a condition that constrains the policy network.

The CMR model has 12 policy variables and needs 12 independent equilibrium
conditions. The capital accumulation identity is not one of them; it
is a law of motion. The missing 12th equilibrium condition is the **balance
sheet identity** (leverage definition).

### The Fix

Replace the tautological eq12 with the leverage definition:

```python
# Old (tautology, always = 0):
residuals["eq12_capital_accumulation"] = (
    defs["k"] - (1 - delta) * k_lag / mu_z - (1 - S) * i
)

# New (leverage definition, balance sheet identity):
residuals["eq12_leverage_definition"] = L * n - q * k
```

This closes the financial block: equations 8 (bank zero-profit), 9
(optimal contract), and 12 (leverage definition) jointly determine
`omega_bar`, `L`, and the relationship between leverage and capital
structure.

---

## Bug 3: Inconsistent Hardcoded Steady State

**File:** `variables.py`, `STEADY_STATE` dict; `steady_state.py`
**Severity:** Moderate — degrades warm-start initialization quality

### The Error

The `STEADY_STATE` dictionary contained hand-entered values that were not
a true fixed point of the equilibrium system. Evaluating all 12 equation
residuals at the old steady state (even before the equation fixes above)
produced non-trivial residuals:

| Equation | Residual | Order |
|----------|----------|-------|
| eq3 (wage Phillips F) | -1.253e-02 | O(10^-2) |
| eq4 (wage Phillips K) | -2.476e-02 | O(10^-2) |
| eq11 (resource constraint) | -9.861e-04 | O(10^-3) |
| Other equations | < 3e-04 | O(10^-4) |
| eq12 (capital accumulation) | 0 | tautology |

The wage block (eq3, eq4) had O(10^-2) residuals — two orders of magnitude
worse than the other equations. The Phillips curve auxiliary variables
`F_w` and `F_p` were the most incorrect.

### Why This Matters

When warm-start is enabled, the network is first trained (via L-BFGS) to
reproduce the steady-state policy at the steady-state point. If the steady
state itself has residuals at O(10^-2), the network begins training already
anchored to an inconsistent solution.

Subsequent DEQN training must then "unlearn" the warm start for the
incorrect equations. For eq3/eq4 (wage Phillips curves), the K_w term
involves a steep negative power `(1 - lambda_w * (1 + sigma_L)) = -1.4`,
creating a sharp landscape that makes gradient-based correction difficult.
This explains the observed training behavior: eq4 residuals plateau at
~13.1 and do not improve.

### The Fix

Replaced the hardcoded steady state with a numerical solver
(`scipy.optimize.root` with JAX autodiff Jacobian) that finds the true
fixed point of the equilibrium system to machine precision:

**Key variable changes:**

| Variable | Old | New | Change |
|----------|-----|-----|--------|
| F_p | 4.532 | 4.736 | +4.5% |
| F_w | 0.895 | 0.885 | -1.0% |
| pi | 1.006 | 1.012 | +0.6% |
| c | 1.600 | 1.594 | -0.4% |
| i | 0.799 | 0.795 | -0.5% |
| R | 1.012 | 1.018 | +0.6% |

**All 12 residuals at the corrected steady state are < 1.2e-07** (float32
machine precision).

The solver runs at module import time and caches the result. It uses
the hardcoded values as an initial guess (so they need to be approximately
correct, but not exact).

---

## How the Bugs Interacted

The three bugs formed a reinforcing failure mode:

1. **Bug 1** (wrong net worth) made `n` ~2x too large. But since **Bug 2**
   (tautological eq12) removed the leverage constraint, the network could
   freely adjust `L` to compensate — or not. The financial block was
   internally inconsistent but the loss function could not detect it.

2. **Bug 3** (inconsistent SS) initialized the network at a point where
   the wage Phillips curves had large residuals. The network learned to
   reproduce the wrong `F_w` and `F_p` values during warm start. During
   subsequent training, the gradient signal for eq3/eq4 was fighting
   against both the initialization and the unconstrained financial block.

3. The loss reweighting scheme (`lr_annealing`) is adaptive: it
   up-weights equations with small losses and down-weights equations with
   large losses. With eq3/eq4 permanently elevated, the reweighting
   de-emphasized them further, creating a feedback loop where the hard
   equations received less gradient signal over time.

The net result was that eq4 (wage Phillips K) plateaued at a residual of
~13.1 and eq11 (resource constraint) at ~0.55, with no improvement across
10,000+ training episodes. All three bugs needed to be fixed together
for the training to have any chance of converging.

---

## Verification

After applying all three fixes, the model was verified as follows:

### Steady-State Consistency

All 12 equation residuals at the numerically solved steady state are
below 1.2e-07 (float32 machine precision). The Jacobian of the system
at the steady state has rank 12 with condition number ~12,000 (full rank,
moderately conditioned).

### Warm Start

L-BFGS warm start from the corrected steady state converges in 48
iterations to a loss of 9.69e-07 (vs. the old SS which could not
achieve exact convergence due to the O(10^-2) residuals).

### Training

Short training (30 episodes, batch=64, Adam lr=1e-3) with warm start shows
all 12 equations receiving gradient signal and decreasing:

| Equation | Old (stuck) | New (30 ep) |
|----------|-------------|-------------|
| eq4 (wage Phillips K) | ~13.1 | 1.14e-02 |
| eq11 (resource constraint) | ~0.55 | 1.17e-01 |
| eq12 (leverage/capital) | 0 (tautology) | 5.01e-04 |

### Unit Tests

All 53 existing tests pass (12 basic + 18 config + 18 optimizer + 5
convergence).

---

## Previously Fixed: Divergences from `disaster.tex`

The LaTeX model documentation (`disaster.tex`) was the original reference
used to implement the equilibrium system. Cross-checking against the
published paper (CMR 2014, Appendix B, pp. 50–51) revealed four
substantive errors in the tex that were already corrected during the
initial code implementation. These are documented here for completeness,
since `disaster.tex` remains in the repository and may cause confusion
if consulted without this context.

**The tex document should not be used as a primary reference for the
equations. The published paper is the authoritative source.**

### Tex Problem 1: Eq8 — Default Probability vs Survival Probability

**What the tex says** (line ~183):

```
L_{t-1} * (R^k_t / R_{t-1}) * [omega_bar * F(omega_bar, sigma)
    + (1 - mu) * G(omega_bar, sigma)]  -  L_{t-1}  +  1  =  0
```

The tex multiplies `omega_bar` by `F(omega_bar)`, which is the default
CDF — the probability that an entrepreneur's idiosyncratic return falls
*below* the threshold.

**What the paper says:**

In the BGG framework, the bank's payoff from *surviving* entrepreneurs
(those with `omega > omega_bar`) is `omega_bar * (1 - F(omega_bar))`.
The factor `(1 - F)` is the survival probability. Multiplying by `F`
instead of `(1 - F)` reverses the economic meaning: it would give the
bank's claim on defaulting rather than surviving borrowers.

**What the code implements** (`equations.py`, lines 158–161):

```python
survival_prob = 1.0 - defs["F_val"]
residuals["eq8_bank_participation"] = st.L_lag * (defs["R_k"] / st.R_lag) * (
    p.omega_bar * survival_prob + (1 - c["mu_mon"]) * defs["G_val"]
) - st.L_lag + 1
```

The code correctly uses `(1 - F)`. At calibration, `F = 0.0056` so the
numerical difference between `F` and `(1 - F)` is large (~0.006 vs ~0.994).

### Tex Problem 2: Eq9 — Old vs Corrected Bracket Form

**What the tex says** (lines ~194–199):

```
[Gamma' / (Gamma' - mu * G')] * [R^k/R * (Gamma - mu*G)  -  1]
```

This is an intermediate algebraic form of the optimal contract condition,
written as a "first-order condition minus 1" bracket.

**What the paper says:**

The corrected form (also annotated as "fixed" in the earlier Keras
codebase) rearranges the condition to separate the entrepreneur's return
from the contract constraint:

```
(R^k/R) * (1 - Gamma)  -  [Gamma'/(Gamma' - mu*G')] * [1 - (R^k/R)(Gamma - mu*G)]
```

This formulation is numerically more stable (avoids subtraction of
nearly-equal quantities) and economically more transparent: the first
term is the entrepreneur's share of excess returns; the second term is
the contract constraint weighted by the hazard ratio.

**What the code implements** (`equations.py`, lines 163–171):

```python
Rk_over_R = defs_n["R_k"] / defs["R"]
ratio_term = Gamma_prime_next / (Gamma_prime_next - c["mu_mon"] * G_prime_next + 1e-8)
bracket_term = 1 - Rk_over_R * (Gamma_next - c["mu_mon"] * G_next)
residuals["eq9_entrepreneur_contract"] = (
    Rk_over_R * (1 - Gamma_next) - ratio_term * bracket_term
)
```

The code uses the corrected form.

### Tex Problem 3: Production Function — Leverage vs Labor

**What the tex says** (lines ~240–241):

```
y_{z,t} = epsilon_t * (k_{t-1} / mu_{z,t})^alpha * L_t^{1-alpha}  -  Phi
```

Here `L_t` is used as the labor input to Cobb-Douglas production. But
throughout the rest of the tex and the paper, `L_t` denotes the
**leverage ratio** (a financial variable, `L = q*k/n`). The labor input
is hours worked, denoted `h_t` in both the code and the paper.

This is a notation collision: the tex reuses the symbol `L` for two
unrelated concepts without comment.

**What the paper says:**

Standard Cobb-Douglas: `Y_t = A_t * (K_t / mu_z)^alpha * h_t^{1-alpha} - Phi`,
where `h_t` is aggregate labor supply.

**What the code implements** (`equations.py`, line 83):

```python
y_z = st.eps * (st.k_lag / st.mu_z) ** c["alpha"] * p.h ** (1 - c["alpha"]) - c["Phi"]
```

The code correctly uses `p.h` (hours worked), not `p.L` (leverage).

### Tex Problem 4: Financial Derivatives — Wrong Sign and Missing Denominator

**What the tex says** (lines ~314–318):

```
G'(omega_bar, sigma) = phi((log(omega_bar) + sigma^2/2) / sigma) * (1/sigma)
```

Two errors:

1. **Wrong sign in the argument**: uses `+ sigma^2/2` where it should be
   `- sigma^2/2`. The `G` function is defined with the *minus* shift
   (it conditions on `omega < omega_bar` in the lower tail), while `F`
   uses the plus shift. The tex has them swapped.

2. **Missing `1/omega_bar` in the denominator**: differentiating
   `G(omega_bar) = Phi((log(omega_bar) - sigma^2/2) / sigma)` with respect
   to `omega_bar` requires the chain rule on `log(omega_bar)`, which
   produces a `1/omega_bar` factor. The complete derivative is:

```
G'(omega_bar) = phi((log(omega_bar) - sigma^2/2) / sigma) / (sigma * omega_bar)
```

The tex is missing the `omega_bar` in the denominator, making the
derivative dimensionally incorrect (the units don't cancel).

**What the code implements** (`equations.py`, lines 36–38):

```python
def G_omega_prime(omega_bar, sigma):
    z = (jnp.log(omega_bar) - 0.5 * sigma ** 2) / sigma
    return normal_pdf(z) / (sigma * omega_bar)
```

The code has the correct sign (`- 0.5 * sigma**2`) and the correct
denominator (`sigma * omega_bar`), matching the paper.

### Summary of Tex Divergences

| # | Equation | Tex Error | Code Status |
|---|----------|-----------|-------------|
| 1 | Eq8 (bank participation) | Uses `F` (default CDF) instead of `1-F` (survival) | Correct from initial implementation |
| 2 | Eq9 (entrepreneur contract) | Uses old algebraic bracket form `[... - 1]` | Implements corrected rearrangement |
| 3 | Production function | Writes leverage `L` as labor input | Uses hours `h` correctly |
| 4 | `G'(omega_bar)` derivative | Wrong sign (`+sigma^2/2`), missing `1/omega_bar` | Correct sign and full chain rule |

These were all caught and corrected during the original code translation
from tex to JAX. They are not new bugs — they are documented here because
anyone reading `disaster.tex` alongside the code will notice the
discrepancies and may wonder which version is correct. The code follows
the paper; the tex is wrong on these four points.

---

## Note on Hardcoded Constants

The constants `pi_ss = 1.006`, `R_ss = 1.011678`, and `y_ss = 3.0308`
are calibration targets. The numerically solved steady state has
`pi = 1.012` and `R = 1.018`, which differ from `pi_ss` and `R_ss`
because the Taylor rule with non-zero `alpha_y` and the financial frictions
shift the steady-state inflation rate above the target. This is
economically expected behavior for this class of models and is not a bug.

---

## Files Changed

| File | Change |
|------|--------|
| `equations.py` | Fixed net worth formula (1-Gamma); replaced eq12 tautology with leverage definition; added Gamma_val to definitions dict |
| `variables.py` | Updated STEADY_STATE dict to numerically solved values |
| `steady_state.py` | Replaced hardcoded lookup with numerical solver (scipy.root + JAX autodiff Jacobian) |
