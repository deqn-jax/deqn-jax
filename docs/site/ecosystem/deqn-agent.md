# deqn-agent -- paper to trained DEQN policy

!!! warning "Status: v0 alpha (`0.1.0a0`), separate package"
    `deqn-agent` is a separate, early-stage project that sits **on top of**
    deqn-jax and does not ship inside it. Its CLI flags, exit codes, and
    skill/prompt contracts may change without notice. This page documents only
    what is shipped today.

## Where it sits in the ecosystem

```
  a research paper / a model.py
            |
            v
   +------------------+
   |   deqn-agent     |   automation: validate -> smoke -> train -> verify -> notebook
   |  (this project)  |   two surfaces: a deterministic CLI + an LLM-driven path
   +------------------+
            |  calls the public API of
            v
   +------------------+
   |     deqn-jax     |   the solver/library: global recursive-equilibrium solver
   |   (the engine)   |   ModelSpec contract, networks, optimizers, training loop
   +------------------+
```

**deqn-jax is the solver.** You hand it a model -- states, equilibrium
conditions, transition law, calibration -- and it returns solved decision rules
and their Euler-equation accuracy. The rest of this site is about that engine.

**deqn-agent is the automation on top.** It takes the workflow an experienced
user would run by hand -- write a contract-conforming `model.py`, smoke-test it,
pick a training config, train, check the residuals, write up the result -- and
turns it into a single command (and, optionally, an agent that can start from
the paper itself).

The two repositories are deliberately separate: deqn-jax has no dependency on
deqn-agent and never will, so the solver stays usable, testable, and citable on
its own. deqn-agent depends on deqn-jax through its public API only.

## The pipeline

Whatever the entry point, the spine is the same five stages:

| Stage | What runs | Backed by |
|---|---|---|
| **Validate** | a panel of contract gates checks a `model.py` against the deqn-jax `ModelSpec` contract -- registry, declared dimensions, JAX-traceability, output shapes, steady-state residual, policy bounds, and (strict mode) an import allowlist | `deqn_agent.validator` |
| **Smoke** | a 2-episode train confirming the loss is finite and not flat (a real gradient signal exists) | `deqn_agent.runner.smoke_train` |
| **Train** | a full training run with a resolved `TrainConfig`, using the deqn-jax trainer | `deqn_agent.runner` |
| **Verify** | a verification panel -- `euler` (per-equation scaled residual), `stability` (max eigenvalue + NaN/bound-hit checks), `moments` -- aggregated to one `pass` / `warn` / `fail` verdict | `deqn_agent.runner` |
| **Notebook** | a post-hoc walkthrough notebook of the run + the verdict | `deqn_agent.notebook` |

!!! note "What a verdict means (and does not)"
    The verification gates are **threshold checks, not proofs**. They inherit
    deqn-jax's core caveat: a low residual is *necessary but not sufficient*,
    and nothing enforces equilibrium selection. A `pass` is a green light to
    look closer, not a correctness certificate. `warn` is acceptable; only
    `fail` triggers the training-escalation loop.

## Two surfaces

### 1. Deterministic CLI -- `solve-paper --from-model`

For a `model.py` that already conforms to the deqn-jax `ModelSpec` contract
(hand-written, or produced earlier by the LLM path). **No LLM is involved** --
this surface is pure Python and reproducible from a seed.

```bash
solve-paper --from-model path/to/model.py --runs-dir ./runs --seed 42
```

It runs validate -> smoke -> train -> verify -> notebook and writes a
self-contained run directory (`config.yaml`, `history.csv`, `checkpoints/`,
`metrics.json` verdict, `notebook.ipynb`). The exit code is the machine-readable
verdict (`0` pass, `1` warn, `2` fail; `3`-`6` are LLM-path outcomes), so it
drops into CI. This is the surface to lean on when **evaluating** the stack: it
is deterministic, touches no external model, and exercises the same deqn-jax
public API a hand-built script would.

### 2. LLM-driven path -- the `solve-paper` orchestrator

!!! warning "Experimental -- requires an agent harness, proven on one fixture"
    This surface runs inside an agent harness (Claude Code via the skill, or a
    generic harness via `AGENTS.md`). It is **not** a deterministic compiler:
    output quality depends on the model and the paper, and it is validated
    mainly on the Brock-Mirman happy-path fixture. Treat it as a research
    preview that does the first draft, not an oracle.

For full paper-to-policy automation, the orchestrator follows a harness-neutral
workflow document and adds a model-preparation phase in front of the
deterministic spine:

- **Phase 1 -- model preparation.** Extract a structured spec, confirm
  ambiguities with the user, emit a `model.py`, and repair contract/smoke
  failures with a bounded retry loop.
- **Phase 2 -- training.** Propose a `TrainConfig`, train, and -- only on a
  `fail` verdict -- escalate along a fixed rung ladder with a bounded budget.
- **Phase 3 -- notebook + verdict.**

Three composable entry points -- a full `paper.tex`/PDF, `--from-spec spec.md`
(skip extraction), or `--from-model` (skip Phase 1, identical to the
deterministic CLI above) -- with `--autonomous` / `--interactive` / `--silent`
modes.

## The skills

The LLM path is built from three Claude Code skills:

- **`solve-paper`** -- the orchestrator. Reads a single workflow document and
  walks the three phases, dispatching subagents and invoking the two retry
  loops. For `--from-model` input it just shells out to the deterministic CLI.
- **`codegen-loop`** (budget 5) -- bounded model repair: validator -> fix ->
  re-validate -> 2-episode smoke, classifying each failure and routing a
  targeted fix, until it passes or the budget is exhausted.
- **`ralph-loop`** (budget 4) -- bounded training escalation, entered only on a
  `fail` verdict. Climbs a fixed rung ladder, recording every patch and verdict
  to a trail.

!!! note "Budgets are a feature"
    On a hard paper, exhausting a budget is an expected, legible outcome -- the
    loop stops, writes its trail, and hands you the recovery options. It does
    not loop forever and it does not silently "succeed."

!!! info "Cross-run learning is consult-only in v0"
    The subagent prompts include a step to consult prior cases, but **v0 ships
    the consult step and not the write infrastructure** -- the lesson files are
    not yet populated, so there is no accumulated experience yet. Learning
    across runs is intended for v1. This is not a self-improving system today.

## Install & honest limits

deqn-agent is **not on PyPI**. It depends on deqn-jax through a local editable
path, so check out both repositories side by side:

```bash
cd deqn-agent
uv sync                       # installs deqn-jax editable from ../deqn-jax
solve-paper --from-model tests/fixtures/brock_mirman_path_a/model.py \
            --runs-dir ./runs --seed 42
```

- **Proven happy-path is Brock-Mirman** (`tests/fixtures/brock_mirman_path_a`).
  Larger research models are exactly where the budgets earn their keep -- and
  where they may legitimately exhaust and hand back to you.
- **The LLM path needs a harness** and is experimental; the deterministic
  `--from-model` CLI needs neither and is the reproducible surface.
- **Verdicts are thresholds, not proofs** -- see deqn-jax's
  [two honest limits](../index.md).

For the contract your `model.py` must satisfy, see deqn-jax's
[Implementing a model](../models/implementing.md) and the
[agent-facing REFERENCE](../REFERENCE.md).
