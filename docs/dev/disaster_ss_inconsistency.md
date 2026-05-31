# Disaster Model: π_ss target (1.006) vs solved π (1.012) inconsistency

**Date:** 2026-04-30
**Status:** Analysis-only; no code changes proposed yet.
**Trigger:** GPT-5 Pro asked: *"I would want the steady-state file or residual list explaining why the target π_ss=1.006 becomes a solved π_ss=1.012."* The errata file's explanation ("Taylor rule with α_y > 0 + financial frictions shifts SS inflation above target") is hand-wavy and doesn't survive derivation.

---

## The discrepancy

| | Target (calibration constant) | Solved (numerical SS) | Δ |
|---|---|---|---|
| `π` | 1.006 | 1.012 | +0.6pp / 2.4% → 4.9% annual |
| `R` | 1.011678 | 1.018 | +0.6pp |
| `y` | 3.0308 | 3.031 (≈ target) | small |

The errata file declares this "intentional and economically expected." That's true in the sense that it's not a code bug, but it's not actually *explained*. The mechanism is more interesting than the errata suggests.

---

## What the errata claim, and why it's wrong as stated

> "The Taylor rule with non-zero α_y and the financial frictions shift the steady-state inflation rate above the target."

Check the Taylor rule at the *supposed* SS (π = π_ss, y = y_ss, R_lag = R_ss, no shocks):

$$
R = R^{ss} \cdot 1^{\rho_p} \cdot [1^{\alpha_\pi} \cdot 1^{\alpha_y}]^{1-\rho_p} \cdot \exp(0) = R^{ss}
$$

So the Taylor rule is **automatically self-consistent** at the calibration target. α_y doesn't shift anything when y = y_ss exactly.

Check the bond Euler at the same point: `R = π · µ_z / β = 1.006 · 1.0041 / 0.9985 = 1.011678`. **Exactly matches R_ss.** ✓

So the bond Euler + Taylor rule pair is consistent at π = π_ss. The shift cannot come from there.

---

## Where the shift actually comes from

The Calvo block at any stationary SS (π_t = π_t-1 = π̃, regardless of π's *value*):

- Indexation: π̃ = π_ss^ι · π^(1-ι); at stationary π = π̃, this gives π̃/π = 1
- F_p recursion: `F_p = λ_z·y / (1 − β·ξ_p)` (forced by stationarity)
- K_p recursion: `K_p = λ_f·λ_z·y·s / (1 − β·ξ_p)`
- K_p definition (eq 2a) at ratio=1: `K_p = F_p`
- Combine: `K_p / F_p = λ_f · s = 1`, so **s = 1/λ_f = 0.8333**

Critical observation: **`s = 1/λ_f` is satisfied at ANY stationary SS, regardless of π's value.** Indexation makes π̃ = π automatically when nothing's moving. The Calvo block doesn't pin down π's level — it pins down the marginal-cost ratio.

Verify against solved SS: `s_solved = 0.8330` ≈ 1/1.2. ✓ The Calvo condition holds.

So **π_ss is a free parameter** in the equilibrium equations — *as long as* the Taylor-rule SS-consistency `R_ss = π_ss · µ_z / β` is maintained. The model admits a one-parameter family of stationary SSs indexed by π_ss. Pick π_ss, compute R_ss from bond Euler, and the rest of the system rebalances.

---

## So why does the solver pick 1.012, not 1.006?

Because the *other calibration targets* — `R_ss = 1.011678`, `y_ss = 3.0308` — and the structural parameters (`Φ`, `ψ_L`, `b`, `κ`, `σ_ω`, `µ_mon`, etc.) are **inconsistent with π_ss = 1.006**.

When you set up the full nonlinear system with π = 1.006 and let the solver find self-consistent values for everything else, the residual vector at the supposed SS isn't zero. Something has to give. The Newton solver in `steady_state.py` finds the nearest fixed point in the joint (π, R, y, c, h, k, w̃, q, λ_z, F_p, F_w, K_p, K_w, ω̄) space, and that fixed point happens to have π = 1.012.

**My best guess at the mechanism**: the calibration of `Φ` (fixed cost = 0.606) and `ψ_L` (labor disutility weight = 0.7705) was done in CMR to make the *linearized* SS values match observed first moments (y ≈ 3.0308, h ≈ 0.944, etc.). The linearized model takes the SS as given and approximates around it. The nonlinear model has to *find* the SS, and small inconsistencies in the calibration that linearization smooths over compound into a 0.6pp inflation shift.

Other candidates: capital adjustment cost `S(µ_z·i/i_lag)` is exactly zero at SS (since `µ_z·i/i_lag = µ_z_ss = µ_z`), but `S'` is also exactly zero, which means the investment Euler at SS reduces to `1 = µ_Υ · q · 1 + β · µ_Υ · ... · S'(...) = µ_Υ · q`. With µ_Υ_ss = 1, this forces q = 1. ✓ matches solved q = 1.0. Doesn't constrain π.

Resource constraint `y_z = g + c + i/µ_Υ + entrepreneur_consumption + monitoring_cost`. At SS this is one equation in (y, c, h, k, n, ω̄). The entrepreneur consumption term `Θ·(1-γ_e)/γ_e·(n-w_e)` and monitoring cost depend on the financial-friction SS. If those don't sum to `y_ss - g_ss - i_ss/µ_Υ_ss` at the supposed SS values, the system rebalances. **Most likely culprit.**

---

## Concrete diagnostic to verify

A ~30-line script that evaluates all 11 equation residuals at the *target* SS (π=1.006, R=1.011678, y=3.0308, and "intended" values for everything else) would surface which equations have the largest residuals. That tells you which calibration assumption is most fragile.

Pseudocode:

```python
from deqn_jax.models import load_model
import jax.numpy as jnp

model = load_model("disaster")

# Build "target" SS state and policy:
# - All exogenous states at their unconditional means
# - All endogenous lags at the published CMR target SS values
# - All policies at the published CMR target SS values
state_target = jnp.array([
    1.006,    # pi_lag (target)
    27.421,   # k_lag (use CMR's published value, not the solved one)
    1.594,    # c_lag
    1.0,      # q_lag
    0.795,    # i_lag
    1.011678, # R_lag (target!)
    1.920,    # w_tilda_lag
    1.966,    # L_lag
    1.0, 1.0, 0.616, 1.0041, 0.0,  # exo at means
])
policy_target = jnp.array([
    0.602,    # lambda_z
    0.795,    # i
    1.006,    # pi (target!)
    1.594,    # c
    1.920,    # w_tilda
    0.944,    # h
    0.885,    # F_w
    4.736,    # F_p
    1.0,      # q
    4.831,    # K_p
    2.207,    # K_w
])

# Steady state means state_t = state_t+1, policy_t = policy_t+1, no shocks
residuals = model.equations_fn(
    state_target[None, :],
    policy_target[None, :],
    state_target[None, :],   # next_state = state at SS
    policy_target[None, :],  # next_policy = policy at SS
    model.constants,
)

for name, r in residuals.items():
    print(f"{name:30s}  {float(r[0]):+.4e}")
```

Whichever residual is largest at the target SS tells you the binding inconsistency. Predictions:

- If **eq9_resource_constraint** dominates → the gov't+consumption+investment+entrepreneur+monitoring sum doesn't match output. Shows the calibration's resource block is internally inconsistent.
- If **eq8_entrepreneur_contract** dominates → the BGG block's required ω̄ doesn't match what the bank participation constraint produces given the supposed leverage.
- If **eq2a/eq2b or eq4a/eq4b dominate** → the Calvo block is fragile to small mismatches in s.
- If **eq5/eq6 dominate** → the household block has internal inconsistency in habit/Euler.

Probably resource constraint or entrepreneur contract.

---

## Implications

1. **The model is internally consistent at the solved SS** (all residuals < 1e-7 there per errata). So our trained policy isn't wrong — it's just trained around π = 1.012, not π = 1.006.

2. **All our reported numbers — `|Δmean| = 0.44%`, etc. — are computed against this same solved SS**, so the comparison is internally consistent. No correction needed retrospectively.

3. **But the calibration targets in the codebase are misleading.** When someone reads `pi_ss = 1.006` in `variables.py`, they reasonably assume the model lives at 2.4% annual inflation. It doesn't. It lives at 4.9%.

4. **For the external paper / writeup**: this matters. CMR's calibration was anchored to 2.4% annual US inflation; if our SS is at 4.9%, that's a different economy than the one CMR's parameters were estimated for. The disaster results' interpretation might shift.

5. **For the foundational doc**: the SS table I wrote (1.012, 1.018) is the solved SS, which is what the model actually does. Should add a one-paragraph note acknowledging the target/solved gap and pointing to this analysis.

6. **For the Brock-Mirman SS bug filed earlier**: same flavor of issue. The fixture computes K based on one formula, the simulated dynamics produce a different K. Not the same bug but the same class — calibration-target vs equilibrium-of-equations gap.

---

## Recommended next moves

In priority order:

1. **Run the diagnostic script** (~30 LoC) and identify which equation has the largest residual at the target SS. Maybe 30 minutes of work; gives a definitive answer.

2. **Update the foundational doc** with a one-paragraph note about the target/solved gap (small — adds maybe 5 lines to §13 of `disaster_model_specification.md`).

3. **Decide policy on the calibration**: either (a) accept that CMR's published calibration produces this SS shift and document it clearly, or (b) re-calibrate (Φ, ψ_L, etc.) to make the nonlinear SS match the published targets — but this would mean deviating from CMR's published parameters, which has its own messiness.

4. **Raise with Simon/Eric/Alexandre** if this matters for the paper's claims. The paper assumes a specific economic environment; if the SS implied by their parameters is at 4.9% inflation rather than 2.4%, their disaster results' interpretation might need a footnote.

None of this is urgent — the model still solves, just not at the inflation rate the calibration table implies.
