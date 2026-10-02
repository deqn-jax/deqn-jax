# Gallery

Worked equilibrium models, each readable end to end. A notebook introduces one
economics step and one method capability, trains the model from its
`configs/<model>.yaml` with a single `train_from_config` call, and ends with an
accuracy certificate rather than a loss curve. The certificate reports
dimensionless residual quantiles on the model's own ergodic states. Where it
matters, it adds a closed-loop stability diagnostic and an independent
cross-check. See [What counts as solved](#what-counts-as-solved) before reading
the numbers.

The notebooks form two arcs. Read them in order the first time.

!!! warning "Certificate numbers are quoted from earlier runs"
    The numbers below come from prior training runs and are pending a fresh
    executed render. Treat them as the target each notebook sets for itself,
    not as a settled benchmark. The [two limits](../index.md) on the home page
    apply: a low residual is necessary but not sufficient, since a global solver
    can land on the wrong equilibrium branch and nothing enforces equilibrium
    selection (there is no global analogue of the local Blanchard-Kahn
    saddle-path condition). There are also no analytic error bounds.
    "Certified" here means a spectral-radius, residual-quantile and
    linearization-floor certificate, nothing stronger.

---

## Arc 1 -- Closed-form pedagogy

These four models have an analytic oracle (a closed-form policy or an
analytical benchmark), so the trained DEQN can be checked point for point
against the true solution. This tests the machinery before it is used on a
model with no known answer.

| # | notebook | economics step | method capability it shows | certificate |
|---|----------|----------------|----------------------------|-------------|
| 1 | [Deterministic Brock-Mirman](bm_deterministic.ipynb) | one state, one Euler equation, no shocks | DEQN mechanics end-to-end on the smallest possible problem | exact-solution comparison vs the closed form $s^\* = \alpha\beta$ |
| 2 | [Stochastic Brock-Mirman](brock_mirman.ipynb) | stochastic optimal growth | on-policy ergodic sampling; antithetic Monte-Carlo expectations | ergodic Euler error vs analytical benchmarks |
| 3 | [Brock-Mirman with labor](bm_labor.ipynb) | endogenous labor supply | multi-policy, multi-equation training (two FOCs jointly) | joint Euler + labor-FOC accuracy |
| 4 | [6-agent OLG (analytic)](olg_analytic_6.ipynb) | 6-generation overlapping generations | multi-agent policies validated against a closed form (Krueger-Kubler 2004) | exact-solution comparison, $k'^h$ vs $\beta_h\,\mathrm{inc}^h$ |

---

## Arc 2 -- The Fischer-Burmeister trilogy

Many models have kinks: investment that cannot go negative, households that
cannot borrow, choices that hit a cap. These are KKT complementarity
conditions, and perturbation methods linearize the kink away. DEQN instead
writes the complementarity condition as a trainable residual using the
Fischer-Burmeister function. The three notebooks go from the simplest case to a
multi-country planner problem.

| # | notebook | economics step | method capability it shows | certificate |
|---|----------|----------------|----------------------------|-------------|
| 5 | [Labor under a cap](bm_labor_constrained.ipynb) | an upper labor cap (one occasionally-binding constraint) | Fischer-Burmeister complementarity as an analytic wedge; slack/wedge diagnostics | Euler median $10^{-2.9}$, FB median $10^{-3.7}$ |
| 6 | [Life-cycle OLG, borrowing-constrained](olg_lifecycle.ipynb) | 6-generation life-cycle with borrowing limits | two-stage loss: an FB residual wrapping an expectation, where $\mathbb{E}[\mathrm{fb}] \neq \mathrm{fb}(\mathbb{E})$ | ergodic $\lvert\mathrm{errREE}\rvert \approx 8\times10^{-4}$ |
| 7 | [Two-country IRBC, irreversible investment](irbc.ipynb) | 2-country International RBC, irreversibility | KKT multipliers as network outputs; Gauss-Hermite quadrature expectations; Blanchard-Kahn-anchored stability | Euler median $10^{-4.3}$, ARC median $10^{-2.9}$, $\rho(\mathrm{SS})=0.98$ |

---

## What counts as solved

Training loss is not the claim. This repository has documented cases where the
training loss misled in both directions. A gallery model is presented as solved
only when all three of these hold:

1. Its closed-loop dynamics are stable. Long unclipped simulations stay in
   economically meaningful territory, and the spectral radius of the closed
   loop at the steady state is below 1. This concerns equilibrium selection,
   not only equilibrium residuals, and a low loss cannot establish it.
2. Dimensionless residual quantiles are small on the ergodic set. They are
   measured with a reliable expectation (Gauss-Hermite quadrature, or the
   unbiased AiO estimator in `docs/dev/aio_loss_estimator.md`) and reported as
   median / p90 / p99, never as a bare mean.
3. An independent check agrees: a closed form (Arc 1, notebook 4), a structural
   identity (the risk-sharing ratio in notebook 7), or the model's own
   linearization used as a floor to beat (the Blanchard-Kahn-anchored models).

## Running a notebook yourself

Each notebook trains its model from scratch, which takes minutes on a laptop.
Training is config-driven; the only tuning is in each model's `configs/` file.

```bash
uv run jupyter nbconvert --to notebook --execute examples/<name>.ipynb \
    --output <name>.ipynb --ExecutePreprocessor.timeout=3600
```

Most notebooks are generated by `_build_<name>_notebook.py` builders. Edit the
builder, regenerate, then re-execute.

In progress, not yet certified and not in the gallery: `aiyagari` (continuum of agents), a
56-agent OLG benchmark, Krusell-Smith, and the DICE climate family.

To build your own model, see [Models & the ModelSpec contract](../models/index.md).
