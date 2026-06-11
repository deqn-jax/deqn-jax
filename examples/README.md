# deqn-jax example gallery

Worked equilibrium models, ordered as a learning path: each notebook introduces
one economics step and one method capability, trains end-to-end from its config
(`configs/<model>.yaml`), and reports an **accuracy certificate** — quantiles of
dimensionless residuals on the model's own ergodic states, plus (where relevant)
closed-loop stability diagnostics. Certificates, not loss curves, are the
claim: see *"What counts as solved"* below.

Most notebooks are generated from `_build_<name>_notebook.py` builders
(edit the builder, regenerate, re-execute — see each builder's docstring).

## The path

| # | notebook | economics | method capability | certificate headline |
|---|----------|-----------|-------------------|----------------------|
| 1 | [bm_deterministic](bm_deterministic.ipynb) | deterministic Brock-Mirman | DEQN mechanics on a model with the closed form s\*=αβ | exact-solution comparison |
| 2 | [brock_mirman](brock_mirman.ipynb) | stochastic growth | on-policy sampling, antithetic MC expectations | ergodic Euler error vs analytical benchmarks |
| 3 | [bm_labor](bm_labor.ipynb) | endogenous labor | multi-policy, multi-equation training | joint Euler + labor-FOC accuracy |
| 4 | [bm_labor_constrained](bm_labor_constrained.ipynb) | labor cap | **Fischer-Burmeister complementarity** (analytic wedge); slack/wedge diagnostics | Euler med 10⁻²·⁹, FB med 10⁻³·⁷ |
| 5 | [olg_analytic_6](olg_analytic_6.ipynb) | 6-agent OLG | multi-agent policies, closed-form validation (Krueger-Kubler) | exact-solution comparison |
| 6 | [olg_lifecycle](olg_lifecycle.ipynb) | life-cycle OLG, borrowing constraints | **two-stage loss**: FB wrapping an expectation (E[fb] ≠ fb(E)) | ergodic \|errREE\| ~8·10⁻⁴ |
| 7 | [irbc](irbc.ipynb) | 2-country IRBC, irreversible investment | **KKT multipliers as network outputs**; quadrature expectations; **BK-anchored stability** | Euler med 10⁻⁴·³, ARC med 10⁻²·⁹, ρ(SS)=0.98 |
| 8 | [disaster](disaster.ipynb) | NK-DSGE with financial frictions (13 states, 11 equations) | **certified equilibrium selection**: BK-linear core + tangent anchoring, spectral-radius certificate | ρ(SS) = BK eigenvalue to 6 digits; beats its own linearized anchor |

Appendix: [interp_brock_mirman](interp_brock_mirman.ipynb) — function-approximation
preliminaries on the Brock-Mirman policy.

**In progress:** `aiyagari` (continuum of agents), 56-agent OLG benchmark,
Krusell-Smith, the DICE climate family (incl. Epstein-Zin).

## What counts as solved

Training loss is not the claim — this repo has documented cases of the
training loss being wrong in *both* directions (see
`docs/dev/disaster_stability_findings.md`). A gallery model is presented as
solved when:

1. **its closed-loop dynamics are stable** (long unclipped simulations stay in
   economic territory; spectral radius of the closed loop at SS below 1 —
   equilibrium *selection*, not just equilibrium *residuals*);
2. **dimensionless residual quantiles are small on the ergodic set**, measured
   with a trustworthy expectation (quadrature, or the unbiased AiO estimator —
   `docs/dev/aio_loss_estimator.md`) and reported as median/p90/p99, never a
   bare mean;
3. **something independent agrees** — a closed form (1, 5), a structural
   identity (7's risk-sharing ratio), or the model's own linearization as a
   floor to beat (8).

## Running

```bash
uv run jupyter nbconvert --to notebook --execute examples/<name>.ipynb \
    --output <name>.ipynb --ExecutePreprocessor.timeout=3600
```

Each notebook trains its model from scratch (minutes on a laptop for 1–7;
~20 minutes for the disaster flagship). All training is config-driven; the
notebooks contain no hand-tuned magic beyond their `configs/` files.
