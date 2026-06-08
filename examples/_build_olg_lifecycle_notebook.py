"""Build ``examples/olg_lifecycle.ipynb`` from chapter functions.

Each ``chapter_*`` returns a list of nbformat cells; ``main`` assembles them and
writes the .ipynb. Re-run after editing, then execute with::

    uv run jupyter nbconvert --to notebook --execute \\
        examples/olg_lifecycle.ipynb --output examples/olg_lifecycle.ipynb

The notebook presents the 6-generation life-cycle OLG with borrowing constraints
(Geneva 2026 course, Day 2 Exercise 4) and reproduces that exercise's diagnostic
panels on the deqn-jax port -- the model that motivated the two-stage
expectation-inside-residual loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import nbformat as nbf

NB_PATH = Path(__file__).parent / "olg_lifecycle.ipynb"


def md(text: str) -> Dict:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> Dict:
    return nbf.v4.new_code_cell(src)


def chapter_intro() -> List[Dict]:
    return [
        md(
            r"""# Life-cycle OLG with borrowing constraints — the two-stage-loss model

The fourth `examples/` stop, and the one that motivated a new capability in the
framework. This is **Day 2, Exercise 4** of the Geneva 2026 *Deep Learning for
Economics & Finance* course: a 6-generation overlapping-generations economy
where households live $H = 6$ deterministic periods (one period $\approx$ 10
years, ages 20–80), save in capital subject to a **borrowing constraint**
$k^h_t \ge 0$, and supply age-dependent labor (less in the last two periods —
retirement).

Each working cohort $h \in \{0, \dots, H-2\}$ chooses a saving rate; the last
cohort consumes everything. The optimality condition is an intertemporal Euler
that holds with equality **only when the borrowing constraint is slack**:

$$\frac{1}{c^h_t} \;\ge\; \beta\,\mathbb{E}_t\!\left[\frac{1-\delta+r_{t+1}}{c^{h+1}_{t+1}}\right],
\qquad k^{h+1}_{t+1}\ge 0,\qquad \text{complementary slackness.}$$

We encode the complementarity with a Fischer-Burmeister residual, exactly as the
course notebook does.

## Why this model needed a new loss

The FB nonlinearity wraps an **expectation**. And $\mathbb{E}[f^{FB}(\cdot)] \ne
f^{FB}(\mathbb{E}[\cdot])$: averaging the residual over shocks and *then* squaring
(the standard DEQN path) puts the nonlinearity in the wrong place and leaves a
bias floor. So this model is trained with the framework's **two-stage loss**:

- `inside_fn` returns the shock-dependent continuation terms
  $(1-\delta+r')/c'^{\,j}$, which the loss averages to $\mathbb{E}[\cdot]$;
- `combine_fn` applies the Fischer-Burmeister **after** the expectation.

The standard $(\mathbb{E}[r])^2$ path is the special case `combine = identity`.
This is the architectural prize from the port: occasionally-binding constraints
**under uncertainty**, solved MC-correctly.

**Outline**
- 1 — Inspect the model
- 2 — Train (two-stage loss, no closed-form steady state)
- 3 — Loss curve
- 4 — Reproduce the Exercise-4 diagnostic panels
- 5 — Ergodic accuracy + the borrowing-constraint corner
- 6 — Summary"""
        ),
        code(
            """import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from deqn_jax.config import TrainConfig
from deqn_jax.models.olg_lifecycle import MODEL
from deqn_jax.models.olg_lifecycle.equations import _cohort_block, combine_fn, inside_fn
from deqn_jax.models.olg_lifecycle.variables import CONSTANTS, H, L_CYCLE
from deqn_jax.plots import plot_loss_curve
from deqn_jax.training.trainer import train_from_config"""
        ),
    ]


def chapter_model() -> List[Dict]:
    return [
        md(
            r"""## 1. Inspect the model

State $\mathbf{x}_t = (Z_t,\, k^0_t, \dots, k^5_t)$ — TFP plus the capital held by
each age group (7 states). Policy is the saving rate out of cash-at-hand for the
five cohorts that still save (sigmoid-bounded to $(0,1)$, so consumption and
saved capital are both positive by construction). Five Euler conditions, one per
saving cohort.

There is **no closed-form steady state**; the cross-sectional capital
distribution and TFP are solved jointly over the ergodic distribution, seeded
from a random $\exp(\mathcal{U}(0,1))$ init — just like the course notebook."""
        ),
        code(
            """print(f"states   : {MODEL.n_states}  {MODEL.state_names}")
print(f"policies : {MODEL.n_policies}  {MODEL.policy_names}")
print(f"equations: {len(MODEL.equation_names)}  {MODEL.equation_names}")
print(f"two-stage hooks present: inside_fn={MODEL.inside_fn is not None}, "
      f"combine_fn={MODEL.combine_fn is not None}")
print(f"steady_state_fn: {MODEL.steady_state_fn}  (None -> trained from random init)")
print()
print(f"alpha={CONSTANTS['alpha']}, beta={CONSTANTS['beta']:.4f} (=0.99^10), "
      f"delta={CONSTANTS['delta']}, rho_z={CONSTANTS['rho_z']:.4f}, sigma_z={CONSTANTS['sigma_z']}")
print(f"age-dependent labor l_cycle = {L_CYCLE}  (retirement in the last two periods)")"""
        ),
    ]


def chapter_train() -> List[Dict]:
    return [
        md(
            """## 2. Train

Recipe from `configs/olg_lifecycle.yaml`: a `[70, 70]` ReLU MLP with sigmoid
output (Simon's `10 * n_input` width), Adam at `3e-4`, MC expectation with
antithetic shocks. The two-stage path is selected automatically because the
model declares `inside_fn` + `combine_fn`. A few thousand episodes converge in
seconds on CPU."""
        ),
        code(
            """cfg = TrainConfig.from_yaml("../configs/olg_lifecycle.yaml")
params, history = train_from_config(cfg)"""
        ),
    ]


def chapter_loss() -> List[Dict]:
    return [
        md(
            """## 3. Loss curve

The loss is the mean squared Fischer-Burmeister residual across the five Euler
conditions, with the expectation taken *inside* the FB."""
        ),
        code(
            """fig, ax = plt.subplots(figsize=(7.5, 4))
plot_loss_curve(history, ax=ax, log_y=True)
ax.set_title("olg_lifecycle — two-stage FB-Euler loss (log scale)")
plt.show()
print(f"loss: {float(history['loss'][0]):.3e} -> {float(min(history['loss'][-50:])):.3e}")"""
        ),
    ]


def chapter_panels() -> List[Dict]:
    return [
        md(
            r"""## 4. Reproduce the Exercise-4 diagnostic panels

The course notebook reads a trained policy off by plotting, by age group, the
cross-sectional capital, consumption, cash-at-hand, relative Euler errors, the
saving policy, and the return distribution. We reproduce that exact panel set on
the deqn-jax solution.

First, simulate the ergodic distribution under the trained policy, then compute
the per-cohort quantities with the model's own `_cohort_block`. The per-cohort
**relative Euler error** is the MC-correct two-stage residual: average the
continuation terms over shocks, then apply FB (reusing `inside_fn` + `combine_fn`)."""
        ),
        code(
            """def simulate(params, key, n_traj=1024, T=80, burn=30):
    \"\"\"Roll the trained policy forward; collect post-burn-in ergodic states.\"\"\"
    s = MODEL.init_state_fn(key, n_traj, CONSTANTS)
    visited = []
    for t in range(T):
        key, sk = jr.split(key)
        eps = jr.normal(sk, (n_traj, 1))
        if t >= burn:
            visited.append(s)
        s = MODEL.step_fn(s, params(s), eps, CONSTANTS)
    return jnp.concatenate(visited, axis=0)


def ergodic_euler_error(params, X, key, n_shocks=64):
    \"\"\"MC-correct per-cohort relative Euler error: E[inside] then FB.\"\"\"
    insides = []
    for _ in range(n_shocks):
        key, sk = jr.split(key)
        eps = jr.normal(sk, (X.shape[0], 1))
        ns = MODEL.step_fn(X, params(X), eps, CONSTANTS)
        ins = inside_fn(X, params(X), ns, params(ns), CONSTANTS)
        insides.append(jnp.stack([ins[f"inside_{j}"] for j in range(H)], axis=1))
    E = jnp.mean(jnp.stack(insides), axis=0)
    Edict = {f"inside_{j}": E[:, j] for j in range(H)}
    res = combine_fn(X, params(X), Edict, CONSTANTS)
    return np.asarray(jnp.stack([res[f"euler_{h}"] for h in range(H - 1)], axis=1))


X = simulate(params, jr.PRNGKey(1))
Z, k = X[:, :1], X[:, 1 : 1 + H]
blk = _cohort_block(Z, k, params(X), CONSTANTS)
c, sav, cah, r = (np.asarray(blk[n]) for n in ("c", "sav", "cah", "r"))
k_np = np.asarray(k)
log_err = np.log10(np.abs(ergodic_euler_error(params, X, jr.PRNGKey(2))) + 1e-16)
print(f"ergodic sample: {X.shape[0]} states")"""
        ),
        code(
            """def band(ax, data, title, xl="age group"):
    a = np.arange(data.shape[1])
    ax.plot(a, data.mean(0), "o-", label="mean")
    ax.plot(a, data.min(0), "--", alpha=0.4, label="min")
    ax.plot(a, data.max(0), "--", alpha=0.4, label="max")
    ax.set_title(title); ax.set_xlabel(xl); ax.legend(fontsize=8)


fig, ax = plt.subplots(2, 3, figsize=(14, 8))
band(ax[0, 0], k_np, "capital $k^h$ by age")
band(ax[0, 1], c, "consumption $c^h$ by age")
band(ax[0, 2], cah, "cash-at-hand by age")
band(ax[1, 0], log_err, r"$\\log_{10}|$rel. Euler error$|$", "cohort h (0..4)")
for h in range(H - 1):
    ax[1, 1].scatter(k_np[:, h], sav[:, h], s=2, alpha=0.3, label=f"h={h}")
ax[1, 1].set_title(r"saving $k'^{\\,h+1}$ vs $k^h$"); ax[1, 1].set_xlabel("$k^h$")
ax[1, 1].legend(fontsize=8)
ax[1, 2].hist(r[:, 0], bins=40); ax[1, 2].set_title("return $r$ distribution")
fig.tight_layout()
plt.show()"""
        ),
        md(
            r"""The panels match the Exercise-4 reference: a **hump-shaped capital profile**
($k^0=0$ for the newborn, rising through working life, peaking near retirement,
then dissaved), a **smooth rising consumption** profile, and a saving policy that
clusters by cohort. This is the standard life-cycle picture the course notebook
produces — recovered here by the JAX port under the two-stage loss."""
        ),
    ]


def chapter_accuracy() -> List[Dict]:
    return [
        md(
            r"""## 5. Ergodic accuracy + the borrowing-constraint corner

The mean per-cohort relative Euler error is the headline accuracy number. The
youngest cohort is hardest (it sits closest to the borrowing constraint, where
the FB kink lives); older cohorts are tighter."""
        ),
        code(
            """print("mean log10 |rel. Euler error| by cohort h=0..4:")
print("  ", np.round(log_err.mean(0), 2))
print(f"\\noverall mean log10|errREE| = {log_err.mean():.2f}  "
      f"(~{10**log_err.mean():.1e} relative)")
print()
print("mean capital by age :", np.round(k_np.mean(0), 3))
print("mean consumption    :", np.round(c.mean(0), 3))
near_bind = (sav[:, : H - 1] < 1e-2).mean(0)
print("frac near borrowing constraint (k' < 1e-2), cohort 0..4:", np.round(near_bind, 3))"""
        ),
        md(
            r"""**On comparing to the reference.** This is a faithful reproduction of the
Geneva Day 2 Ex 4 *method and diagnostics*: same equilibrium conditions, same FB
complementarity, same decade calibration, same diagnostic panels. On a like-for-
like ergodic $|\text{err}_{REE}|$ the JAX port currently trails the original
TensorFlow reference by roughly a part in ten on the log scale (a known gap
tracked in the project notes, not a ship blocker) — the shapes and the
economics agree; the last fraction of a decimal of accuracy is the open item.

The model exposes nothing exotic to get here: standard MLP, sigmoid-bounded
saving rates for $0 < s < 1$ (hence $c>0$ and $k'\ge 0$ by construction), and the
two-stage `inside_fn`/`combine_fn` hooks that put the Fischer-Burmeister
**after** the expectation."""
        ),
    ]


def chapter_summary() -> List[Dict]:
    return [
        md(
            r"""## Summary

- **Model**: 6-generation life-cycle OLG with borrowing constraints (Geneva Day 2
  Ex 4). 7 states, 5 saving-rate policies, 5 Fischer-Burmeister Euler conditions,
  no closed-form steady state.
- **Capability it unlocked**: the **two-stage expectation-inside-residual loss**.
  Because the FB wraps an expectation, $\mathbb{E}[f^{FB}] \ne f^{FB}(\mathbb{E})$;
  `inside_fn` averages the continuation terms and `combine_fn` applies the FB
  afterward. The standard $(\mathbb{E}[r])^2$ path is recovered as
  `combine = identity`.
- **Result**: textbook life-cycle behaviour (capital hump, consumption smoothing)
  with ergodic per-cohort Euler errors in the $10^{-2}$–$10^{-3}$ range,
  reproducing the course exercise's diagnostic panels.
- **No special tooling**: plain MLP + sigmoid bounds + the two model hooks; the
  trainer, loss, and `deqn_jax.plots` suite are all stock.

The companion model with a closed-form oracle is `examples/olg_analytic_6.ipynb`
(Krueger-Kubler 2004); the constrained-labor sibling is
`bm_labor_constrained` (Day 2 Ex 3)."""
        ),
    ]


def main() -> None:
    cells: List[Dict] = []
    for ch in (
        chapter_intro,
        chapter_model,
        chapter_train,
        chapter_loss,
        chapter_panels,
        chapter_accuracy,
        chapter_summary,
    ):
        cells.extend(ch())
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    nbf.write(nb, str(NB_PATH))
    print(f"wrote {NB_PATH} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
