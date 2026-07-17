"""Build ``examples/bm_labor_constrained.ipynb`` from chapter functions.

Each ``chapter_*`` returns a list of nbformat cells; ``main`` assembles them and
writes the .ipynb. Re-run after editing, then execute with::

    uv run jupyter nbconvert --to notebook --execute \\
        examples/bm_labor_constrained.ipynb --output examples/bm_labor_constrained.ipynb

The notebook presents Brock-Mirman with endogenous labor under an upper labor
cap (Geneva 2026 course, Day 2 Exercise 3): the introductory example for
encoding occasionally-binding constraints as Fischer-Burmeister residuals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import nbformat as nbf

NB_PATH = Path(__file__).parent / "bm_labor_constrained.ipynb"


def md(text: str) -> Dict:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> Dict:
    return nbf.v4.new_code_cell(src)


def chapter_intro() -> List[Dict]:
    return [
        md(
            r"""# Labor under a cap — complementarity as a Fischer-Burmeister residual

**Day 2, Exercise 3** of the Geneva 2026 *Deep Learning for Economics & Finance*
course: Brock-Mirman with endogenous labor, where labor supply is subject to an
upper cap. The economics is one step past `bm_labor`; the *method* lesson is how
to make a KKT complementarity trainable.

The labor optimality condition under the cap $L \le L_{\max}$ is

$$\underbrace{L_{\max} - L}_{\text{slack } a}\;\ge\;0,\qquad
\underbrace{\frac{w\,u'(c)}{\psi L^{\theta}} - 1}_{\text{wedge } b}\;\ge\;0,\qquad
a\cdot b = 0 .$$

Interior: the cap is slack and the usual FOC $\psi L^\theta = w\,u'(c)$ holds
($b=0$). Binding: $L = L_{\max}$ and the marginal benefit of labor exceeds its
cost ($b>0$).

**Why not just use the plain FOC residual?** When the cap binds, $\psi L^\theta -
w\,u'(c)$ is strictly negative and *cannot* reach zero — MSE training would fight
an unsatisfiable target. The Fischer-Burmeister function

$$f^{FB}(a,b) \;=\; a + b - \sqrt{a^2 + b^2}$$

is zero **iff** $a\ge 0,\ b\ge 0,\ ab=0$: both the interior FOC and the binding
corner are genuine zeros of one smooth residual, so the standard DEQN loss can
converge in both regimes.

## A faithful-port subtlety: two different labor bounds

The reference notebook uses **two distinct numbers**:

- the network's $L$ output is `1.01 * sigmoid(...)` → $L \in (0,\,1.01)$;
- the FB slack is $b = 1.02 - L$ → the complementarity cap is $L_{\max}=1.02$.

Since the output ceiling (1.01) sits strictly *below* the FB cap (1.02), the
slack is $\ge 0.01$ always: at this calibration **the cap never binds** and the
FB reduces to enforcing the interior FOC. (The unconstrained steady-state labor
is $L_{ss}\approx 0.975$, comfortably interior.) An earlier port set both bounds
to 1.01, which made the cap bind in the high-TFP tail — a deviation from the
reference. deqn-jax matches the 1.01/1.02 split exactly; this notebook verifies
the complementarity diagnostics that follow from it.

**Outline**
- 1 — Inspect the model
- 2 — Train
- 3 — Training loss
- 4 — Ergodic set and diagnostic quantities
- 5 — Policy functions (capital, labor, consumption)
- 6 — Relative Euler errors (signed)
- 7 — LHS vs RHS equilibrium checks
- 8 — Complementarity diagnostics (slack, wedge, FB residual)
- 9 — Ergodic accuracy certificate
- 10 — Summary"""
        ),
        code(
            """import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from deqn_jax.models import load_model

model = load_model("bm_labor_constrained")
print(model.name, "—", "states:", model.state_names, "policies:", model.policy_names)
print("equations:", model.equation_names)
{k: v for k, v in model.constants.items()}"""
        ),
    ]


def chapter_train() -> List[Dict]:
    return [
        md(
            r"""## 2 — Train

64×64 tanh MLP, cosine-decayed Adam, 6000 episodes of on-policy trajectory
sampling (T=100), 5 antithetic MC samples for the expectation. The twin
unconstrained `bm_labor` is trained with the *identical* config (only the model
name differs), so the labor-policy comparison in §5 is apples-to-apples.
The labor output is sigmoid-bounded to
$(0,\,1.01)$ by the policy bounds — the network cannot violate the output
ceiling by construction; the FB residual handles the (slack) complementarity."""
        ),
        code(
            """from deqn_jax.config import load_config
from deqn_jax.training.trainer import train_from_config

config = load_config("../configs/bm_labor_constrained.yaml")
config = config.model_copy(update={"verbose": False})
params, history = train_from_config(config)
print(f"final loss: {history['loss'][-1]:.3e}")"""
        ),
        code(
            """# Unconstrained twin for comparison (Exercise 2 model, same seed/recipe)
config_u = config.model_copy(update={"model": "bm_labor"})
params_u, history_u = train_from_config(config_u)
print(f"final loss (unconstrained bm_labor): {history_u['loss'][-1]:.3e}")"""
        ),
    ]


def chapter_loss() -> List[Dict]:
    return [
        md(
            """## 3 — Training loss

$\\log_{10}$ of the mean-squared equilibrium residual against the training
episode, mirroring the *loss function* panel of the Geneva Day 2 Exercise 3
result set. The unconstrained `bm_labor` twin (trained with the identical
recipe) is overlaid: the Fischer-Burmeister labor residual trains as cleanly as
the plain interior FOC, so the two curves come down together."""
        ),
        code(
            """fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.log10(np.asarray(history["loss"])), lw=0.8, label="bm_labor_constrained (FB)")
ax.plot(np.log10(np.asarray(history_u["loss"])), lw=0.8, alpha=0.7, label="bm_labor (no cap)")
ax.set_xlabel("training episode"); ax.set_ylabel("loss [log10]")
ax.set_title("loss function")
ax.legend(); fig.tight_layout()"""
        ),
    ]


def chapter_diagnostics() -> List[Dict]:
    return [
        md(
            r"""## 4 — Ergodic set and diagnostic quantities

Following the Geneva Day 2 Exercise 3 result figures, all diagnostics are read
off the **simulated ergodic cloud** under the trained policy (scatter, not a
grid line). We compute, at each ergodic point:

- the capital policy $K' = (1-\delta)K + sY$ and labor $L$;
- the **signed relative Euler error** in the Exercise 3 convention,
  $\text{errREE}_{\text{cap}} = 1 - \dfrac{1}{C\,\beta\,\mathbb{E}[\,u'(C')(1-\delta+r')\,]}$,
  with the conditional expectation taken by 64-node Gauss-Hermite quadrature;
- the **labor FB residual** $\text{errREE}_{\text{lab}} = f^{FB}(a,b)$ with
  $a = \dfrac{w}{C\,\psi L^{\theta}} - 1$ and $b = 1.02 - L$ — this is exactly
  the model's `labor_foc` residual.

The first figure is the ergodic set itself (TFP level $Z$ against capital $K$)."""
        ),
        code(
            """from deqn_jax.models.bm_labor.equations import definitions as defs_fn
from deqn_jax.models._complementarity import fischer_burmeister
from deqn_jax.training.loss import gauss_hermite_nd, compute_residuals

beta = model.constants["beta"]; delta = model.constants["delta"]
psi = model.constants["psi"]; theta = model.constants["theta"]; L_max = model.constants["L_max"]
L_CEIL = 1.01  # network output ceiling (Exercise 3: lab_pol = 1.01 * sigmoid)

# On-policy simulation to the ergodic set (scatter cloud, as in the reference).
# burn counts TIME steps: 256 trajectories roll `burn` steps first (discarded),
# then the following steps are collected until ~T states are gathered.
def simulate(net, T=3000, burn=120, seed=0):
    key = jax.random.PRNGKey(seed)
    s = model.init_state_fn(key, 256, model.constants)
    out = []
    n_collect = T // 256 + 1
    for t in range(burn + n_collect):
        key, sub = jax.random.split(key)
        eps = jax.random.normal(sub, (s.shape[0], model.n_shocks))
        s = model.step_fn(s, net(s), eps, model.constants)
        if t >= burn:
            out.append(s)
    return jnp.concatenate(out)[:T]

states = simulate(params)
pol = params(states)
pol_u = params_u(states)
d = defs_fn(states, pol, model.constants)

K = np.asarray(states[:, 0])            # capital
Z = np.asarray(d["Z"])                  # TFP level exp(z)
L = np.asarray(pol[:, 1])               # labor (constrained model)
L_u = np.asarray(pol_u[:, 1])           # labor (unconstrained twin)
C = np.asarray(d["c"])                  # consumption
u_c = d["u_c"]                          # u'(C) = 1/C  (gamma = 1)
Knext = np.asarray((1.0 - delta) * states[:, 0] + d["s"])  # K' = (1-delta)K + sY

# Capital Euler error, signed, Exercise 3 convention.
# euler residual e = u'(C) - beta * E[u'(C')(1 + mpk' - delta)] computed by quadrature,
# so beta * E[...] = u_c - e, and errREE_cap = 1 - u_c / (beta * E[...]).
nodes, weights = gauss_hermite_nd(64, model.n_shocks)
e = 0.0
for i in range(nodes.shape[0]):
    shock = jnp.broadcast_to(jnp.asarray(nodes[i])[None, :], (states.shape[0], model.n_shocks))
    e = e + weights[i] * compute_residuals(model, params, states, shock)["euler"]
RHS_cap = u_c - e                       # = beta * E[u'(C')(1 + mpk' - delta)]
errREE_cap = np.asarray(1.0 - u_c / RHS_cap)
LHS_cap = np.asarray(u_c)               # 1/C
RHS_cap = np.asarray(RHS_cap)

# Labor FB residual (= model labor_foc); a = wedge, b = slack.
slack = L_max - pol[:, 1]
wedge = d["w"] * u_c / (psi * pol[:, 1] ** theta) - 1.0
errREE_lab = np.asarray(fischer_burmeister(wedge, slack))
LHS_lab = np.asarray(d["w"] * u_c)      # w/C
RHS_lab = np.asarray(psi * pol[:, 1] ** theta)  # psi * L^theta
slack = np.asarray(slack); wedge = np.asarray(wedge)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(Z, K, s=8, alpha=0.5)
ax.set_xlabel("TFP level Z"); ax.set_ylabel("capital K")
ax.set_title("simulated ergodic set")
fig.tight_layout()"""
        ),
    ]


def chapter_policies() -> List[Dict]:
    return [
        md(
            r"""## 5 — Policy functions on the ergodic set

The capital, labor, and consumption policies plotted against each state, mirroring
the *cap policy* / *labor policy* / *consumption policy* panels of the reference.

For capital, the $45^\circ$ line $K'=K$ is the reference: where the cloud crosses
it is the (stochastic) steady-state capital. For labor, the output ceiling
$L=1.01$ is drawn; the unconstrained `bm_labor` twin is overlaid as faint dots.
Because the FB cap (1.02) sits above the output ceiling (1.01), the cap is slack
at this calibration and the two labor policies coincide — the faithful-port check
the 1.01/1.02 split implies. The labor axis is fixed to $[0.9, 1.03]$ so the
margin between the ergodic labor mass and the ceiling reads honestly (it is a
genuine gap, not numerical noise)."""
        ),
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(K, K, s=6, color="0.7", label="45°  (K'=K)")
axes[0].scatter(K, Knext, s=8, alpha=0.6, label="K'")
axes[0].set_xlabel("capital K"); axes[0].set_ylabel("next capital K'")
axes[0].legend(fontsize=8)
axes[1].scatter(Z, Knext, s=8, alpha=0.6)
axes[1].set_xlabel("TFP level Z"); axes[1].set_ylabel("next capital K'")
fig.suptitle("capital policy"); fig.tight_layout()"""
        ),
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, x, xl in ((axes[0], K, "capital K"), (axes[1], Z, "TFP level Z")):
    ax.axhline(L_CEIL, color="r", ls=":", lw=1.0, label="output ceiling 1.01")
    ax.scatter(x, L, s=8, alpha=0.6, label="labor L (capped)")
    ax.scatter(x, L_u, s=5, alpha=0.3, label="unconstrained twin")
    ax.set_xlabel(xl); ax.set_ylabel("labor L"); ax.set_ylim(0.9, 1.03)
axes[0].legend(fontsize=8)
fig.suptitle("labor policy  (cap slack: ergodic labor stays below the ceiling)")
fig.tight_layout()"""
        ),
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(K, C, s=8, alpha=0.6); axes[0].set_xlabel("capital K"); axes[0].set_ylabel("consumption C")
axes[1].scatter(Z, C, s=8, alpha=0.6); axes[1].set_xlabel("TFP level Z"); axes[1].set_ylabel("consumption C")
fig.suptitle("consumption policy"); fig.tight_layout()"""
        ),
        md(
            r"""### Constrained vs unconstrained on a fixed grid

The equivalence check the 1.01/1.02 split implies, on a *fixed* grid (not just
the ergodic cloud): because the FB cap is slack at this calibration, **both**
the savings-rate and labor policies of the constrained model must coincide with
the unconstrained twin — including off the ergodic set."""
        ),
        code(
            """k_grid = jnp.linspace(0.9, 12.0, 200)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
max_gap = 0.0
for z_lvl in (0.7, 1.0, 1.3):
    grid_states = jnp.stack([k_grid, jnp.full_like(k_grid, jnp.log(z_lvl))], axis=1)
    pol_cg = np.asarray(params(grid_states))
    pol_ug = np.asarray(params_u(grid_states))
    max_gap = max(max_gap, float(np.abs(pol_cg - pol_ug).max()))
    axes[0].plot(k_grid, pol_cg[:, 0], label=f"Z={z_lvl} (capped)")
    axes[0].plot(k_grid, pol_ug[:, 0], "--", alpha=0.6)
    axes[1].plot(k_grid, pol_cg[:, 1], label=f"Z={z_lvl} (capped)")
    axes[1].plot(k_grid, pol_ug[:, 1], "--", alpha=0.6)
axes[1].axhline(1.01, color="r", lw=0.8, ls=":", label="output ceiling 1.01")
axes[1].axhline(1.02, color="k", lw=0.8, ls=":", label="FB cap 1.02")
axes[0].set_title("savings rate"); axes[1].set_title("labor supply")
for a in axes:
    a.set_xlabel("k"); a.legend(fontsize=7)
fig.suptitle("solid = constrained model, dashed = unconstrained twin")
fig.tight_layout()
print(f"max |constrained - unconstrained| over grid (both policies): {max_gap:.2e}")"""
        ),
    ]


def chapter_releu() -> List[Dict]:
    return [
        md(
            r"""## 6 — Relative Euler errors (signed)

The Exercise 3 *Rel Ee* panels: the **signed, linear** relative Euler error for
the capital condition and the labor FB residual, scattered against each state. A
horizontal zero reference line is drawn — these are *errors centered on zero*, so
the small vertical scale (≈ $10^{-4}$–$10^{-3}$) is the correct result, not an
autoscale artefact. A clean signed band hugging zero is the target; a raw
$\log_{10}|\,\cdot\,|$ would instead spike to $-\infty$ at the sign changes and
look jagged, which is why the signed form is used."""
        ),
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, x, xl in ((axes[0], K, "capital K"), (axes[1], Z, "TFP level Z")):
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.scatter(x, errREE_cap, s=8, alpha=0.6, label="capital Euler errREE")
    ax.scatter(x, errREE_lab, s=8, alpha=0.6, label="labor FB residual")
    ax.set_xlabel(xl); ax.set_ylabel("relative Euler error (signed)")
axes[0].legend(fontsize=8)
fig.suptitle("Rel Ee  (Geneva Day 2 Exercise 3 convention)")
fig.tight_layout()"""
        ),
    ]


def chapter_euler_check() -> List[Dict]:
    return [
        md(
            r"""## 7 — LHS vs RHS equilibrium checks

The reference's pointwise LHS-vs-RHS overlays. For the **capital Euler**, the
large markers are $1/C$ and the small markers are $\beta\,\mathbb{E}[u'(C')(1-\delta+r')]$;
for the **labor FOC**, $w/C$ against $\psi L^{\\theta}$. These are *values*, not
errors, so they are shown at their natural scale — the two series sitting on top
of each other across the whole ergodic set is exactly the equilibrium holding."""
        ),
        code(
            """fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes[0, 0].scatter(K, LHS_cap, s=40, alpha=0.5, label="LHS  1/C")
axes[0, 0].scatter(K, RHS_cap, s=12, alpha=0.7, label=r"RHS  $\\beta\\,E[u'(C')(1-\\delta+r')]$")
axes[0, 0].set_xlabel("capital K"); axes[0, 0].legend(fontsize=8)
axes[0, 1].scatter(Z, LHS_cap, s=40, alpha=0.5); axes[0, 1].scatter(Z, RHS_cap, s=12, alpha=0.7)
axes[0, 1].set_xlabel("TFP level Z")
axes[1, 0].scatter(K, LHS_lab, s=40, alpha=0.5, label="LHS  w/C")
axes[1, 0].scatter(K, RHS_lab, s=12, alpha=0.7, label=r"RHS  $\\psi L^{\\theta}$")
axes[1, 0].set_xlabel("capital K"); axes[1, 0].legend(fontsize=8)
axes[1, 1].scatter(Z, LHS_lab, s=40, alpha=0.5); axes[1, 1].scatter(Z, RHS_lab, s=12, alpha=0.7)
axes[1, 1].set_xlabel("TFP level Z")
axes[0, 0].set_title("capital Euler"); axes[1, 0].set_title("labor FOC")
fig.suptitle("LHS vs RHS — both equilibrium conditions hold pointwise")
fig.tight_layout()"""
        ),
    ]


def chapter_complementarity() -> List[Dict]:
    return [
        md(
            r"""## 8 — Complementarity diagnostics (KKT regime)

A deqn-jax addition beyond the reference figures, making the KKT regime explicit.
On the ergodic set the pair should show: slack $b = 1.02 - L \ge 0.01$ everywhere
(cap never binds), wedge $a \approx 0$ (interior FOC holds), and the FB residual
$\approx 0$ because $a\approx 0$ — *not* because $b = 0$. If a re-calibration ever
pushed the ergodic mass against the cap, the same plots would show $b \to 0$ with
$a > 0$: the FB encoding handles both regimes without any change to the loss."""
        ),
        code(
            """fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
axes[0].hist(slack, bins=60); axes[0].axvline(0.01, color="r", ls=":")
axes[0].set_xlabel("slack  b = L_max − L")
axes[0].set_title(f"slack  (min {float(slack.min()):.4f}, median {float(np.median(slack)):.4f})")
axes[1].hist(wedge, bins=60)
axes[1].set_xlabel("wedge  a = w·u'(C)/(psi·L^theta) − 1")
axes[1].set_title(f"wedge  (median {float(np.median(wedge)):.2e})")
axes[2].hist(np.abs(errREE_lab), bins=60)
axes[2].set_xlabel("|FB residual|")
axes[2].set_title(f"|FB residual|  (median {float(np.median(np.abs(errREE_lab))):.2e})")
fig.tight_layout()
assert float(slack.min()) >= 0.009, 'cap should never bind at this calibration'"""
        ),
    ]


def chapter_accuracy() -> List[Dict]:
    return [
        md(
            r"""## 9 — Ergodic accuracy certificate

Magnitudes of the signed errors above, reported as **quantiles** (median / p90 /
p99), not means — the tails are where complementarity encodings hide their sins.
$|\text{errREE}_{\text{cap}}|$ is the dimensionless relative Euler error; the
labor entry is $|f^{FB}|$, the FB residual magnitude."""
        ),
        code(
            """q_levels = (0.5, 0.9, 0.99)
labels = {0.5: "median", 0.9: "p90", 0.99: "p99"}
for name, v in (("|errREE| capital Euler", np.abs(errREE_cap)),
                ("|FB residual| labor    ", np.abs(errREE_lab))):
    qs = {labels[p]: float(np.quantile(v, p)) for p in q_levels}
    print(name, "  ", {k: f"{val:.2e}" for k, val in qs.items()},
          "  log10:", {k: f"{np.log10(max(val, 1e-300)):.2f}" for k, val in qs.items()})"""
        ),
    ]


def chapter_summary() -> List[Dict]:
    return [
        md(
            r"""## 7 — Summary

- The labor cap enters as a **Fischer-Burmeister residual**: one smooth equation
  whose zeros are exactly the KKT solutions, so the same MSE loss covers
  interior and binding regimes.
- The reference's **1.01 output-ceiling / 1.02 FB-cap split** makes the cap
  slack by construction at this calibration; the constrained policy coincides
  with the unconstrained `bm_labor` twin, which is the faithful-port check.
- The complementarity diagnostics (slack / wedge / FB histograms) are the
  pattern to reuse whenever a constraint *does* bind — see `olg_lifecycle`
  for borrowing constraints that bind on real ergodic mass, where the FB also
  has to wrap an expectation (the two-stage loss).

**Next stops:** `olg_lifecycle.ipynb` (FB under uncertainty, two-stage loss),
`irbc.ipynb` (FB irreversibility in a two-country model)."""
        ),
    ]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = (
        chapter_intro()
        + chapter_train()
        + chapter_loss()
        + chapter_diagnostics()
        + chapter_policies()
        + chapter_releu()
        + chapter_euler_check()
        + chapter_complementarity()
        + chapter_accuracy()
        + chapter_summary()
    )
    nbf.write(nb, NB_PATH)
    print(f"wrote {NB_PATH} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
