# Gallery

**Worked equilibrium models you can read end-to-end.** Each notebook introduces
*one* economics step and *one* method capability, trains the model from its
`configs/<model>.yaml` with a single `train_from_config` call, and closes with an
**accuracy certificate** -- not a loss curve. The certificate is the claim:
dimensionless residual quantiles on the model's *own* ergodic states, plus, where
it matters, a closed-loop stability diagnostic and an independent cross-check.
See [What counts as solved](#what-counts-as-solved) before you read the numbers.

The gallery is a learning path in two arcs. Read top to bottom the first time.

!!! warning "Alpha -- and these certificates are claims to re-verify"
    deqn-jax is alpha. The certificate numbers below are quoted from prior
    training runs and are pending a fresh executed render; treat them as the
    target each notebook sets for itself, not a settled benchmark. And keep the
    [two honest limits](../index.md) in view: a low residual is *necessary but
    not sufficient* (a global solver can land on the wrong equilibrium branch,
    and nothing enforces equilibrium *selection* -- there is no global analogue
    of the *local* Blanchard-Kahn saddle-path condition), and there are no
    analytic error bounds. "Certified" here means a spectral-radius +
    residual-quantile + linearization-floor certificate, nothing stronger.

---

## Arc 1 -- Closed-form pedagogy

Start where the answer is known. These four models have an analytic oracle (a
closed-form policy or an analytical benchmark), so a trained DEQN can be checked
*point-for-point* against the truth -- the cleanest possible demonstration that
the machinery works before we point it at a model with no answer key.

| # | notebook | economics step | method capability it shows | certificate |
|---|----------|----------------|----------------------------|-------------|
| 1 | [Deterministic Brock-Mirman](bm_deterministic.ipynb) | one state, one Euler equation, no shocks | DEQN mechanics end-to-end on the smallest possible problem | exact-solution comparison vs the closed form $s^\* = \alpha\beta$ |
| 2 | [Stochastic Brock-Mirman](brock_mirman.ipynb) | stochastic optimal growth | on-policy ergodic sampling; antithetic Monte-Carlo expectations | ergodic Euler error vs analytical benchmarks |
| 3 | [Brock-Mirman with labor](bm_labor.ipynb) | endogenous labor supply | multi-policy, multi-equation training (two FOCs jointly) | joint Euler + labor-FOC accuracy |
| 4 | [6-agent OLG (analytic)](olg_analytic_6.ipynb) | 6-generation overlapping generations | multi-agent policies validated against a closed form (Krueger-Kubler 2004) | exact-solution comparison, $k'^h$ vs $\beta_h\,\mathrm{inc}^h$ |

---

## Arc 2 -- The Fischer-Burmeister trilogy

Real models have kinks: investment that can't go negative, households that can't
borrow, choices that hit a cap. These are KKT complementarity conditions, and
perturbation methods linearize the kink away. The DEQN answer is to make the
complementarity itself a trainable residual via the **Fischer-Burmeister**
function. Three notebooks build the capability from its simplest form to a
multi-country planner problem.

| # | notebook | economics step | method capability it shows | certificate |
|---|----------|----------------|----------------------------|-------------|
| 5 | [Labor under a cap](bm_labor_constrained.ipynb) | an upper labor cap (one occasionally-binding constraint) | **Fischer-Burmeister complementarity** as an analytic wedge; slack/wedge diagnostics | Euler median $10^{-2.9}$, FB median $10^{-3.7}$ |
| 6 | [Life-cycle OLG, borrowing-constrained](olg_lifecycle.ipynb) | 6-generation life-cycle with borrowing limits | **two-stage loss**: an FB residual *wrapping* an expectation, where $\mathbb{E}[\mathrm{fb}] \neq \mathrm{fb}(\mathbb{E})$ | ergodic $\lvert\mathrm{errREE}\rvert \approx 8\times10^{-4}$ |
| 7 | [Two-country IRBC, irreversible investment](irbc.ipynb) | 2-country International RBC, irreversibility | **KKT multipliers as network outputs**; Gauss-Hermite quadrature expectations; **Blanchard-Kahn-anchored stability** | Euler median $10^{-4.3}$, ARC median $10^{-2.9}$, $\rho(\mathrm{SS})=0.98$ |

---

## What counts as solved

Training loss is **not** the claim. This repo has documented cases of the
training loss being misleading in *both* directions
(`docs/dev/disaster_stability_findings.md`). A gallery model is presented as
solved only when all three of these hold:

1. **Its closed-loop dynamics are stable.** Long *unclipped* simulations stay in
   economic territory, and the spectral radius of the closed loop at the steady
   state is below 1. This is equilibrium *selection*, not just equilibrium
   *residuals* -- and it is exactly the check a low loss cannot give you.
2. **Dimensionless residual quantiles are small on the ergodic set**, measured
   with a trustworthy expectation (Gauss-Hermite quadrature, or the unbiased
   AiO estimator -- `docs/dev/aio_loss_estimator.md`) and reported as
   median / p90 / p99, **never a bare mean**.
3. **Something independent agrees** -- a closed form (Arc 1, notebook 4), a
   structural identity (notebook 7's risk-sharing ratio), or the model's own
   linearization used as a floor to beat (the Blanchard-Kahn-anchored models).

## Running a notebook yourself

Every notebook trains its model from scratch -- minutes on a laptop. All
training is config-driven; there is no hand-tuned magic beyond each model's
`configs/` file.

```bash
uv run jupyter nbconvert --to notebook --execute examples/<name>.ipynb \
    --output <name>.ipynb --ExecutePreprocessor.timeout=3600
```

Most notebooks are generated from `_build_<name>_notebook.py` builders: edit the
builder, regenerate, then re-execute.

**In progress** (not yet certified, not yet in the gallery): `aiyagari`
(continuum of agents), a 56-agent OLG benchmark, Krusell-Smith, and the DICE
climate family.

Want to build your own? See [Models & the ModelSpec contract](../models/index.md).
