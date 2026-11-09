"""Check a model-folder run and write its one-page HTML report with figures."""

from __future__ import annotations

import base64
import csv
import html
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from deqn_jax.evaluate.irf import run_irf  # noqa: E402
from deqn_jax.evaluate.solution import SolutionCheck, check_solution  # noqa: E402
from deqn_jax.models.folder import (  # noqa: E402
    folder_model_name,
    load_exact_policy,
    load_model_file,
)
from deqn_jax.plots import plot_irf_grid, plot_loss_curve  # noqa: E402
from deqn_jax.plots.solution import (  # noqa: E402
    plot_against_exact,
    plot_equation_errors,
    plot_policies_on_path,
    plot_simulated_states,
)
from deqn_jax.training.checkpointing import (  # noqa: E402
    best_checkpoint_path,
    load_policy_from_checkpoint,
)

CSS = """body{font-family:-apple-system,Helvetica,Arial,sans-serif;max-width:1100px;
margin:2em auto;padding:0 1em;color:#222;line-height:1.45}
.v{padding:1em 1.4em;border-radius:8px;font-size:1.25em;margin:1em 0}
.ok{background:#e8f5e9;border:2px solid #2e7d32}.no{background:#ffebee;border:2px solid #c62828}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:.4em .6em;
text-align:left;vertical-align:top}.pass{color:#2e7d32}.fail{color:#c62828}.info{color:#777}
img{max-width:100%}figcaption{color:#555;font-size:.92em;margin-bottom:1.5em}
pre{background:#f6f6f6;padding:1em;white-space:pre-wrap}"""


def check_run(run: Path):
    """Check the run against the frozen model files it was trained with."""
    run = Path(run)
    model, module = load_model_file(run, folder_model_name(run.parent.parent))
    ckpts = run / "checkpoints"
    best = Path(best_checkpoint_path(str(ckpts)))
    ckpt = best if best.exists() else sorted(ckpts.glob("checkpoint_*.eqx"))[-1]
    net, _ = load_policy_from_checkpoint(str(ckpt))
    check = check_solution(
        net, model, load_exact_policy(run), getattr(module, "SHOCK_NODES_PER_DIM", None)
    )
    (run / "check.json").write_text(json.dumps(check.summary(), indent=2))
    return check, net, model


def _loss_figure(run: Path):
    path = run / "history.csv"
    if not path.exists():
        return None
    with path.open() as fh:
        losses = [float(row["loss"]) for row in csv.DictReader(fh)]
    return plot_loss_curve({"loss": losses}).figure if losses else None


def _irf_figure(net, model):
    if model.steady_state_fn is None or not model.n_shocks:
        return None
    shocks = list(model.shock_names) or [f"shock_{i}" for i in range(model.n_shocks)]
    irfs = {s: run_irf(net, model, s, shock_size=1.0, horizon=40) for s in shocks}
    variables = (list(model.state_names) + list(model.policy_names))[:6]
    return plot_irf_grid(
        irfs,
        variables=variables,
        title="Response to a one-standard-deviation shock (dev. from steady state)",
    )


def _embed(fig, path: Path, caption: str) -> str:
    if fig is None:
        return ""
    fig.savefig(path, dpi=130, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{data}"><figcaption>{html.escape(caption)}</figcaption>'


def write_report(run: Path, check: SolutionCheck, net, model) -> Path:
    """``run/report.html`` (self-contained) and ``run/figures/*.png``."""
    run = Path(run)
    figs = run / "figures"
    figs.mkdir(exist_ok=True)
    if check.solved:
        verdict = '<div class="v ok"><b>Solved.</b> Every check passes.</div>'
    else:
        failed = "; ".join(g.name for g in check.gates if g.passed is False)
        verdict = f'<div class="v no"><b>Not solved yet.</b> Failing: {html.escape(failed)}</div>'
    rows = "".join(
        f"<tr><td class={c}>{c}</td><td>{html.escape(g.name)}</td><td>{html.escape(g.value)}"
        f"</td><td>{html.escape(g.meaning)}</td></tr>"
        for g in check.gates
        for c in [{True: "pass", False: "fail", None: "info"}[g.passed]]
    )
    pictures = [
        (
            plot_equation_errors(check),
            "errors",
            "Left of the green line: off by less than "
            "0.1%. Right of the red line: off by more than 1%.",
        ),
        (
            plot_against_exact(check),
            "exact",
            "Only for models whose exact solution is known.",
        ),
        (
            plot_policies_on_path(check),
            "policies",
            "What the network decides where the economy actually goes.",
        ),
        (
            _irf_figure(net, model),
            "irf",
            "Impulse responses computed with the trained network.",
        ),
        (
            plot_simulated_states(check),
            "simulation",
            "Where a line leaves the grey band, the network is extrapolating.",
        ),
        (_loss_figure(run), "training", "A falling loss is necessary, not sufficient."),
    ]
    body = "".join(_embed(f, figs / f"{name}.png", cap) for f, name, cap in pictures)
    model_md = run / "model.md"
    text = model_md.read_text() if model_md.exists() else "(no model.md)"
    title = f"{run.parent.parent.name}, run {run.name}"
    page = (
        f'<!doctype html><meta charset="utf-8"><title>{html.escape(title)}</title>'
        f"<style>{CSS}</style><h1>{html.escape(title)}</h1>{verdict}"
        f"<p>The same fixed checks for every model: the equilibrium conditions on a "
        f"{check.errors.shape[0]:,}-period simulation, with next period's expectation computed "
        f"by quadrature.</p><table><tr><th></th><th>Check</th><th>Value</th>"
        f"<th>What it means</th></tr>{rows}</table>{body}"
        f"<h2>The model as given</h2><pre>{html.escape(text)}</pre>"
    )
    path = run / "report.html"
    path.write_text(page)
    return path
