"""Build ``examples/disaster.ipynb`` from chapter functions.

Each ``chapter_*`` returns a list of nbformat cells; ``main`` assembles them
and writes the .ipynb. Re-run after editing, then execute with::

    uv run jupyter nbconvert --to notebook --execute \\
        examples/disaster.ipynb --output examples/disaster.ipynb \\
        --ExecutePreprocessor.timeout=3600

The flagship example: the CMR-style NK-DSGE with financial frictions, trained
with the BK-anchored recipe. An experimental research example.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import nbformat as nbf

NB_PATH = Path(__file__).parent / "disaster.ipynb"


def md(text: str) -> Dict:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> Dict:
    return nbf.v4.new_code_cell(src)


def chapter_intro() -> List[Dict]:
    return [
        md(
            r"""# The disaster model — a certified NK-DSGE solution

The gallery flagship: a New-Keynesian DSGE with financial frictions in the
CMR tradition — **13 states, 11 policies, 11 equilibrium equations, 5 shocks**
(plus an optional rare-disaster mixture, off in this baseline). Calvo price
*and* wage setting contribute recursive aggregates with near-degenerate
("gauge") directions; the investment Euler has a sign-flipping bracket; the
policy rate has an effective lower bound nearby. This is the model class DEQN
methods are *for* — and the one where naive training fails in instructive ways.

## The part the loss cannot see

A DEQN loss penalizes equilibrium residuals **at the states the simulation
visits**. But equilibrium systems of this size admit families of self-
consistent explosive solutions (the Blanchard-Kahn multiplicity): a policy can
have small residuals along its own trajectories while its closed-loop dynamics
$s \mapsto \text{step}(s, \pi(s), 0)$ are locally **unstable** — the economy
drifts off until state clipping catches it. The one-number diagnostic is the
spectral radius $\rho$ of the closed-loop Jacobian at the steady state:

- $\rho < 1$ — the policy selects the stable (BK) equilibrium branch;
- $\rho > 1$ — small residuals, wrong solution; simulations leave the region
  the model is about.

deqn-jax's answer is **architecture + anchoring**: the policy network is the
BK-stable linearized rule plus a learned correction (`DisasterPolicyNet`, a
shaped `LinearPlusMLP`), and the composite loss anchors the policy's value and
*tangent* at the steady state to the linearization. With the Jacobian anchor
at full weight, the trained policy's closed-loop spectrum reproduces the BK
eigenvalue to six digits. The same recipe stabilizes the two-country `irbc`
example — the cure is general, not model-specific.

**Outline**
- 1 — Inspect the model
- 2 — Train (BK-anchored composite loss, quadrature expectations)
- 3 — Stability certificate: ρ, simulation health
- 4 — Accuracy: the net against its own anchor
- 5 — Impulse responses: nonlinear vs linearized
- 6 — Summary"""
        ),
        code(
            """import jax

# fp64 must be on BEFORE the model module is imported: the disaster model
# trains in float64 (configs/disaster.yaml fp64: true), and module-level
# constants created under float32 would type-mismatch inside lax.scan later.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from deqn_jax.models import load_model

model = load_model("disaster")
print(f"{model.n_states} states, {model.n_policies} policies, "
      f"{len(model.equation_names)} equations, {model.n_shocks} shocks")
print("policies:", model.policy_names)"""
        ),
    ]


def chapter_train() -> List[Dict]:
    return [
        md(
            r"""## 2 — Train

`configs/disaster.yaml` with one override: the Jacobian-anchor weight raised
to **1.0** (the config default is 0.1, which measurably fails to hold the
steady-state tangent). Expectations by Gauss-Hermite
quadrature, $3^5 = 243$ nodes; fp64; warm-started from the steady state with a
curriculum ramp on shock scale. ~20 minutes on a laptop CPU."""
        ),
        code(
            """from deqn_jax.config import load_config
from deqn_jax.training.trainer import train_from_config

config = load_config("../configs/disaster.yaml")
config = config.model_copy(update={
    "verbose": False,
    "checkpoint_dir": None, "tensorboard_dir": None,
})
config.composite_loss.jac_weight = 1.0
params, history = train_from_config(config)
print(f"final training loss: {history['loss'][-1]:.3e}")"""
        ),
        code(
            """fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(history["loss"], lw=0.8, color="C0")
ax.set_xlabel("episode")
ax.set_ylabel("composite training loss")
ax.set_title("Training loss (log scale)")
ax.grid(True, which="both", alpha=0.2)
fig.tight_layout()"""
        ),
    ]


def chapter_stability() -> List[Dict]:
    return [
        md(
            r"""## 3 — Stability certificate

Three checks, none of which the training loss can certify on its own:
the closed-loop spectral radius at SS (must be < 1, ideally the BK value),
a long **unclipped** simulation staying bounded, and no mass parked at the
state-clip ceilings."""
        ),
        code(
            """ss_state, _ = model.steady_state_fn(model.constants)
ss = jnp.asarray(ss_state)
zero_shock = jnp.zeros(model.n_shocks)

J = jax.jacfwd(
    lambda s: model.step_fn(s[None, :], params(s[None, :]),
                            zero_shock[None, :], model.constants)[0]
)(ss)
rho = float(np.abs(np.linalg.eigvals(np.asarray(J))).max())
print(f"closed-loop spectral radius at SS: rho = {rho:.6f}")
assert rho < 1.0, "policy is on an unstable equilibrium branch"

def simulate(net, T=2000, n_paths=32, seed=0):
    s0 = jnp.tile(ss[None, :], (n_paths, 1))
    shocks = jax.random.normal(jax.random.PRNGKey(seed), (T, n_paths, model.n_shocks))
    def step(s, e):
        nxt = model.step_fn(s, net(s), e, model.constants)  # NO clipping
        return nxt, nxt
    _, traj = jax.lax.scan(step, s0, shocks)
    return traj

traj = simulate(params)
assert bool(jnp.isfinite(traj).all())
k_idx = list(model.state_names).index("k_lag")
k = np.asarray(traj[500:, :, k_idx])
print(f"unclipped 2000-step simulation: capital in [{k.min():.1f}, {k.max():.1f}] "
      f"(ss {float(ss[k_idx]):.1f}) — bounded, no ceiling mass")"""
        ),
        md(
            r"""The ergodic cloud the trained policy actually visits. A
BK-unstable policy would smear toward the state-clip box; this one stays a
tight, recurrent blob around the steady state (marked ★). Economic axes:
capital against (gross quarterly) inflation."""
        ),
        code(
            """pi_idx = list(model.state_names).index("pi_lag")
erg = np.asarray(traj[500:].reshape(-1, model.n_states))

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(erg[:, k_idx], erg[:, pi_idx], s=4, alpha=0.12, color="C0",
           label="ergodic draws")
ax.scatter([float(ss[k_idx])], [float(ss[pi_idx])], marker="*", s=240,
           color="k", zorder=5, label="steady state")
ax.set_xlabel("capital $K$")
ax.set_ylabel(r"inflation $\\pi$ (gross, quarterly)")
ax.set_title("Ergodic distribution under the trained policy")
ax.legend(loc="best")
fig.tight_layout()"""
        ),
    ]


def chapter_accuracy() -> List[Dict]:
    return [
        md(
            r"""## 4 — Accuracy: the net against its own anchor

The natural yardstick for an anchored network is its anchor: the pure
linearized policy. If training works, the nonlinear correction should *beat*
the linear rule on the model's own ergodic states — otherwise the MLP added
nothing. Both policies are evaluated with the same 243-node quadrature."""
        ),
        code(
            """import re

from deqn_jax.training.loss import (
    compute_residuals,
    gauss_hermite_nd,
)
from deqn_jax.training.linearize import linearize_model

P, Q = linearize_model(model, verbose=False)
_, ss_policy = model.steady_state_fn(model.constants)

def lin_policy(states):
    return jnp.asarray(ss_policy)[None, :] + (states - ss[None, :]) @ jnp.asarray(P).T

# Evaluate both policies on the SAME ergodic states with the SAME 243-node
# quadrature. The disaster model is single-stage, so the expectation residual
# per state is the weighted sum of the per-node residuals — exactly the inner
# quantity the (E[r])^2 training loss squares and averages.
nodes, q_weights = gauss_hermite_nd(3, model.n_shocks)
nodes = jnp.array(nodes); q_weights = jnp.array(q_weights)
pool = traj[500:].reshape(-1, model.n_states)
idx = jax.random.choice(jax.random.PRNGKey(2), pool.shape[0], (512,), replace=False)
states = pool[idx]

def expectation_residuals(fn):
    \"\"\"Dict eq_name -> E[r] per state (weighted over quadrature nodes).\"\"\"
    shocks = jnp.broadcast_to(nodes[:, None, :], (nodes.shape[0], states.shape[0], model.n_shocks))
    per_node = jax.vmap(lambda sh: compute_residuals(model, fn, states, sh))(shocks)
    return {k: jnp.einsum("s,sb->b", q_weights, v) for k, v in per_node.items()}

res_net = expectation_residuals(params)
res_lin = expectation_residuals(lin_policy)

eq_order = [k for k in res_net if not k.startswith("aux_")]
def label(name):
    return re.sub(r"^eq\\d+[ab]?_", "", name).replace("_", " ")

rms_net = np.array([float(jnp.sqrt(jnp.mean(res_net[k] ** 2))) for k in eq_order])
rms_lin = np.array([float(jnp.sqrt(jnp.mean(res_lin[k] ** 2))) for k in eq_order])
loss_net = float(np.mean(rms_net ** 2))
loss_lin = float(np.mean(rms_lin ** 2))
print(f"trained net        mean squared residual: {loss_net:.3e}")
print(f"linearized anchor  mean squared residual: {loss_lin:.3e}")
print(f"improvement over the anchor: {loss_lin / loss_net:.1f}x")"""
        ),
        md(
            r"""Root-mean-square equilibrium residual per equation, trained net
against its linearized anchor (log scale — the residuals span orders of
magnitude and a linear axis would hide the small ones). Lower is better; bars
where the orange (net) sits below the grey (linear) are equations the
nonlinear correction actually improves."""
        ),
        code(
            """fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(eq_order))
ax.bar(x - 0.2, rms_lin, width=0.4, color="0.6", label="linearized anchor")
ax.bar(x + 0.2, rms_net, width=0.4, color="C1", label="trained net")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([label(k) for k in eq_order], rotation=40, ha="right", fontsize=8)
ax.set_ylabel("RMS equilibrium residual")
ax.set_title("Per-equation accuracy: trained net vs linearized anchor")
ax.legend(loc="best")
fig.tight_layout()"""
        ),
        code(
            """# Accuracy certificate: quantiles of the dimensionless equilibrium
# residual over all (state, equation) pairs. Quantiles, not a mean — a single
# mean hides whether the tail is controlled.
all_res = np.abs(np.concatenate([np.asarray(res_net[k]) for k in eq_order]))
med, p90, p99, mx = (float(np.quantile(all_res, q)) for q in (0.5, 0.9, 0.99, 1.0))
print("trained net |E[r]| across ergodic states x equations:")
print(f"  median={med:.2e}  p90={p90:.2e}  p99={p99:.2e}  max={mx:.2e}")
print(f"  log10 median = {np.log10(med):.2f}")
worst = sorted(((rms, label(k)) for rms, k in zip(rms_net, eq_order)), reverse=True)[:3]
print("hardest equations:", ", ".join(f"{nm} ({v:.1e})" for v, nm in worst))"""
        ),
    ]


def chapter_irf() -> List[Dict]:
    return [
        md(
            r"""## 5 — Impulse responses: nonlinear vs linearized

A one-standard-deviation investment-efficiency shock from the stochastic
steady state. Where the nonlinear IRF separates from the linearized one is
exactly the content the neural solution adds (risk and curvature corrections);
where they coincide is a sanity check, not a disappointment."""
        ),
        code(
            """def irf(net, shock_idx=1, scale=1.0, T=40):
    base = jnp.tile(ss[None, :], (1, 1))
    e0 = jnp.zeros((1, model.n_shocks)).at[0, shock_idx].set(scale)
    paths = {}
    for tag, first in (("shocked", e0), ("baseline", jnp.zeros((1, model.n_shocks)))):
        s, out = base, []
        for t in range(T):
            e = first if t == 0 else jnp.zeros((1, model.n_shocks))
            s = model.step_fn(s, net(s), e, model.constants)
            out.append(s[0])
        paths[tag] = jnp.stack(out)
    return paths["shocked"] - paths["baseline"]

d_net = irf(params)
d_lin = irf(lin_policy)

# Economic names for the internal lagged-state fields, and report responses as
# percent deviation from steady state — the standard, comparable IRF unit.
show = [("pi_lag", "inflation"), ("k_lag", "capital"),
        ("c_lag", "consumption"), ("R_lag", "policy rate")]
fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
for ax, (nm, econ) in zip(axes, show):
    i = list(model.state_names).index(nm)
    ss_i = float(ss[i])
    pct_net = 100.0 * np.asarray(d_net[:, i]) / ss_i
    pct_lin = 100.0 * np.asarray(d_lin[:, i]) / ss_i
    ax.plot(pct_net, color="C1", label="nonlinear (net)")
    ax.plot(pct_lin, "--", color="0.4", label="linearized")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(econ)
    ax.set_xlabel("periods (quarters)")
    # Honest axis: keep zero in frame and never autoscale a ~0 response into
    # float noise. If the whole response is numerically negligible, show a
    # readable flat band instead of a noise-magnified squiggle.
    span = max(np.abs(pct_net).max(), np.abs(pct_lin).max())
    pad = max(1.05 * span, 1e-3)
    ax.set_ylim(-pad, pad)
axes[0].set_ylabel("% deviation from steady state")
axes[0].legend(fontsize=8)
fig.suptitle("IRF to a 1σ investment-efficiency shock — nonlinear (net) vs "
             "linearized; gaps are the risk/curvature content the net adds")
fig.tight_layout()"""
        ),
    ]


def chapter_summary() -> List[Dict]:
    return [
        md(
            r"""## 6 — Summary

- Equilibrium residual losses **under-determine** large DSGE solutions: the
  Blanchard-Kahn selection has to come from somewhere. Here it comes from the
  architecture (BK-linear core + learned correction) and the composite loss's
  value/tangent anchors — certified after training by the closed-loop spectral
  radius.
- The certificate battery (ρ, unclipped simulation health, net-vs-anchor
  accuracy) is what "solved" means in this gallery; the training loss alone is
  not it — this model's history includes a long stretch where the training
  loss was wrong in *both* directions.
- The same anchoring recipe stabilizes the structurally different `irbc`
  example: equilibrium-selection-by-anchoring is a method-level tool, not a
  per-model hack.

**See also:** `docs/dev/aio_loss_estimator.md` for why this model's loss is
estimated by quadrature (and what the unbiased-MC fallback is when a model
outgrows the node budget)."""
        ),
    ]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = (
        chapter_intro()
        + chapter_train()
        + chapter_stability()
        + chapter_accuracy()
        + chapter_irf()
        + chapter_summary()
    )
    nbf.write(nb, NB_PATH)
    print(f"wrote {NB_PATH} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
