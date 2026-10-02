# deqn-agent -- paper to trained DEQN policy

!!! warning "Status: v0 alpha (`0.1.0a0`), separate package"
    `deqn-agent` is a separate, early-stage project built on top of deqn-jax;
    it does not ship inside it. Its CLI flags, exit codes, and skill and prompt
    contracts may change without notice. This page documents only what is
    shipped today.

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

deqn-jax, documented on the rest of this site, is the solver: given a model
(states, equilibrium conditions, transition law, calibration) it returns
solved decision rules and their Euler-equation accuracy.

deqn-agent runs the manual workflow in one command: write a
contract-conforming `model.py`, smoke-test it, pick a training config, train,
check the residuals, write up the result. Optionally an agent starts from the
paper itself.

The repositories are separate on purpose. deqn-jax does not and will not
depend on deqn-agent, so the solver stays usable, testable and citable on its
own. deqn-agent uses only the deqn-jax public API.

## The pipeline

Every entry point runs the same five stages:

| Stage | What runs | Backed by |
|---|---|---|
| Validate | a set of contract checks on a `model.py` against the deqn-jax `ModelSpec` contract: registry, declared dimensions, JAX-traceability, output shapes, steady-state residual, policy bounds, and (strict mode) an import allowlist | `deqn_agent.validator` |
| Smoke | a 2-episode training run confirming the loss is finite and not flat, i.e. there is a gradient signal | `deqn_agent.runner.smoke_train` |
| Train | a full training run with a resolved `TrainConfig`, using the deqn-jax trainer | `deqn_agent.runner` |
| Verify | `euler` (per-equation scaled residual), `stability` (max eigenvalue + NaN/bound-hit checks) and `moments`, combined into one `pass` / `warn` / `fail` verdict | `deqn_agent.runner` |
| Notebook | a walkthrough notebook of the run and the verdict, written afterwards | `deqn_agent.notebook` |

!!! note "What a verdict means"
    The verification checks are thresholds, not proofs. They share deqn-jax's
    main caveat: a low residual is necessary but not sufficient, and nothing
    enforces equilibrium selection. A `pass` means the run is worth a closer
    look; it is not a correctness certificate. `warn` is acceptable; only
    `fail` starts the training-escalation loop.

## Two surfaces

### 1. Deterministic CLI -- `solve-paper --from-model`

For a `model.py` that already conforms to the deqn-jax `ModelSpec` contract,
whether hand-written or produced earlier by the LLM path. No LLM is involved:
this surface is plain Python and reproducible from a seed.

```bash
solve-paper --from-model path/to/model.py --runs-dir ./runs --seed 42
```

It runs validate, smoke, train, verify and notebook in order, and writes a
self-contained run directory (`config.yaml`, `history.csv`, `checkpoints/`,
`metrics.json` verdict, `notebook.ipynb`). The exit code is the verdict (`0`
pass, `1` warn, `2` fail; `3`-`6` are LLM-path outcomes), so it can run in CI.
Use this surface to evaluate the stack: it is deterministic, calls no external
model, and uses the deqn-jax public API like a hand-written script.

### 2. LLM-driven path -- the `solve-paper` orchestrator

!!! warning "Experimental: requires an agent harness, tested on one fixture"
    This surface runs inside an agent harness (Claude Code via the skill, or
    another harness via `AGENTS.md`). It is not a deterministic compiler:
    output quality depends on the model and the paper, and it has been
    validated mainly on the Brock-Mirman fixture. Treat its output as a first
    draft.

For full paper-to-policy automation, the orchestrator follows a
harness-neutral workflow document and adds a model-preparation phase before
the deterministic stages:

- Phase 1, model preparation: extract a structured spec, confirm ambiguities
  with the user, emit a `model.py`, and repair contract and smoke failures in
  a bounded retry loop.
- Phase 2, training: propose a `TrainConfig` and train. Only on a `fail`
  verdict, escalate along a fixed ladder of steps with a bounded budget.
- Phase 3: notebook and verdict.

There are three entry points: a full `paper.tex`/PDF, `--from-spec spec.md`
(skips extraction), or `--from-model` (skips Phase 1; the same as the
deterministic CLI above). Each runs in `--autonomous`, `--interactive` or
`--silent` mode.

## The skills

The LLM path is built from three Claude Code skills:

- `solve-paper`: the orchestrator. It reads one workflow document, walks the
  three phases, dispatches subagents and invokes the two retry loops. For
  `--from-model` input it calls the deterministic CLI.
- `codegen-loop` (budget 5): bounded model repair. It runs the validator, a fix,
  re-validation and a 2-episode smoke, classifies each failure and applies a
  targeted fix, until the model passes or the budget runs out.
- `ralph-loop` (budget 4): bounded training escalation, entered only on a
  `fail` verdict. It climbs a fixed ladder of steps and records every patch
  and verdict to a trail.

!!! note "Budgets"
    On a hard paper, running out of budget is an expected outcome. The loop
    stops, writes its trail and lists the recovery options. It does not loop
    forever, and it does not report success when it has none.

!!! info "Cross-run learning is consult-only in v0"
    The subagent prompts include a step to consult prior cases, but v0 ships
    the consult step without the code that writes cases: the lesson files are
    empty, so there is no accumulated experience. Learning across runs is
    planned for v1; the system does not improve itself today.

## Install and limits

deqn-agent is not on PyPI. It depends on deqn-jax through a local editable
path, so check out both repositories side by side:

```bash
cd deqn-agent
uv sync                       # installs deqn-jax editable from ../deqn-jax
solve-paper --from-model tests/fixtures/brock_mirman_path_a/model.py \
            --runs-dir ./runs --seed 42
```

- The tested end-to-end case is Brock-Mirman
  (`tests/fixtures/brock_mirman_path_a`). On larger research models the
  budgets may run out and hand control back to you.
- The LLM path needs a harness and is experimental; the deterministic
  `--from-model` CLI does not.
- Verdicts are thresholds, not proofs; see deqn-jax's
  [limits](../index.md).

For the contract your `model.py` must satisfy, see deqn-jax's
[Implementing a model](../models/implementing.md) and the
[REFERENCE](../REFERENCE.md).
