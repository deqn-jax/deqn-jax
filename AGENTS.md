# deqn-jax
Deep Equilibrium Network trainer for economic models (JAX/Equinox port of
Azinovic–Maliar–Maliar) — for economists solving dynamic stochastic models and
for the research program certifying those solutions.

## What this project is
- **Nature**: research. Relaxations, explicit and per-tool: pyright stays `basic`
  and advisory (103 errors on 2026-09-03; strict mode drowns in JAX/Equinox partial
  stubs; not in CI); no coverage threshold; ruff strict tightening (`B,N,UP,RUF`) measured at 1,997
  errors on 2026-09-03 and deferred to a dedicated branch. Everything else:
  production discipline.
- **Graph**: `@aleph/deqn` (`r71`) — every session starts with `iskron_orient` here.
- **Focus holon**: `#1 «🧮 deqn-jax solver»`.
- **Agent role**: `#9 «🤖 Coding agent»` — adhikarin, stewards `#6 Training contour`
  and `#7 Certification contour` under the focus holon. Your inbox:
  `iskron_orient(realm="@aleph/deqn", focus="#9")` at session start.
- **Owner role**: `#8 «👤 Maintainer-researcher»` (svatantra 主) — questions outside
  your mandate go there as `posed_to` vimarshas, never as silent decisions.
- **Stack**: Python ≥3.10, JAX ≥0.4.20 + Equinox ≥0.11 + Optax ≥0.2, Pydantic v2
  configs, uv. No TensorFlow, no PyTorch, no Keras.
- **Production statement**: public research code (github.com/deqn-jax/deqn-jax,
  docs at deqn-jax.github.io) read by potential collaborators and course
  students. Cost of breakage: wrong research conclusions and lost collaborator
  trust — worse than a crash. Solver claims are gated by certificates (see
  *Certification*), never by green tests alone.

## Persistence rules
State lives in the **repo**, the **private notes repo**, or the **graph** — nowhere
else. The harness's built-in memory (per-project memory directory, conversation
summaries, `/tmp`, machine-local files) is **forbidden entirely, not by category**:
nothing goes there — no project fact, no user preference, no working-style note.
(why: local memory is invisible to every other agent and machine, so it drifts
silently and breaks the reproducibility that makes a second machine or agent possible.)
- **Repo**: code, configs, conventions, code gotchas, dev docs, certification
  records, branch state (the artifact itself).
- **Private notes repo** (`~/Projects/research/deqn-research-notes`, GitHub
  `mechanicpanic/deqn-research-notes`): research narrative, audits, paper basis,
  the maintainer's personal facts — anything not for the public tree.
- **Graph**: decisions, open questions (vimarshas), plans, handoffs, lessons, hints —
  the thinking around the work. Do not retell graph content in the repo; link the
  vimarsha or contour.
- **Fetch state; never reconstruct from memory.** No source for "we decided…"? Stop
  and read the graph, the chronicle (`docs/dev/selection_program_chronicle_2026_07.md`
  §15/§16) or the dev docs before acting.
- **External design/spec files are drafts for intake**, not the record: the graph
  holds decisions; such a file is their view.
- **Parking spots are named, and there are three.** Prose in this file; a
  sinn-phenomenon for something that acts; a lone `context` arrow that silences a
  detector. Finish line of a record: **a node is not recorded until what pulls it is
  named** — which kriya breaks if the node vanishes? None — that is parking, not a
  record.
- **This overrides the harness's own memory instruction.** Route instead, always,
  asking **whose fact is this?** — and before finishing, check that every durable
  fact from the user's context is persisted by this routing; an unpersisted fact is a
  failed task. Repo convention, code fact, this project's procedures, servers,
  deploy pipeline, dated duties → this file / docs / code, or a node in the graph;
  work state, decision, open question → vimarsha in the graph; the maintainer's own
  facts (people, machines, cross-project lessons) and research narrative → the
  private notes repo. Standing preferences ("how to work with me") are instructions,
  not facts: project-scoped ones live in this file, never in the graph or a memory
  directory. A dated duty is a node carrying the date in `attrs`; a README mention
  loses it. Rules of this project never go into another project's graph.
- The local memory directory is **evacuated and frozen**: `MEMORY.md` holds a
  one-line prohibition stub pointing here, and the memory-guard `PreToolUse` hook in
  `.claude/settings.json` blocks any write there (exit 2). What it held is preserved
  in the private notes repo (`dev/agent_memory_evacuation_2026_09_03.md`).

## Session lifecycle
Graph = the work (structure, open questions, what next). Git = how we got here (SHA,
branches, PRs). **Git references never enter the graph** — no SHAs, branch names, PR
numbers or "shipped/merged" in nodes.
- **Session start:** enter through the `iskron:entry` skill — it ends in signs, not
  reading; this file holds only addresses (graph, focus holon, agent and owner roles
  above). Address agreement with the owner by the seq of his role, not the `me`
  sentinel. Then `git status`, and check running DGX jobs (`logs/` on
  `anna@130.223.169.108:~/projects/deqn-jax`) before launching new work.
- **Starting work: graph first, then project, then code.** A substantive task enters
  in three beats: (1) **graph reconnaissance** — what is recorded about the place of
  the change, which vimarshas are open, what was decided and rejected, and what is
  recorded about the external surfaces the work will touch; `iskron:entry` drives
  it; (2) **integration field** — from the focus holon, the steward role and the
  nodes of the change, derive consumers, effects and neighbouring contours through
  `iskron:integrity`; walk relays with `lens="trace"`; design missing kriyas and
  phenomena, weave gaps with `iskron:weaving`; (3) **design** the change
  (`iskron:design`) — only then code. The one exception is **explicit**: the user
  said "work directly" or named another protocol — then go to code and pay the
  reconnaissance debt at the reconcile beat. Silence is not "work directly".
- **A decision is recorded when it is made, not when it is executed.** Wherever it
  arrives — chat, a channel, two agents agreeing — it stands in the graph immediately
  with the modes it really has now (epistemic no higher than `anumita`, ontic
  `anagata`, volitive `chanda`/`adhimoksha`). Record who decided and what counts as
  execution. A changed situation is reflected just as immediately.
- **Every task is described before it is begun, and recorded as what it is.** Before
  the first change outside the graph, the work stands in the graph as its carrier: a
  one-off deed as an `anga` vimarsha on the transformation it moves (a large one as
  its own bianhua) with its before/after; a **kriya only for a repeatable transition**
  (ritual, pipeline step, procedure). While the work runs, the graph moves with it.
  On **merge** (not push — a branch that merged nothing shipped nothing) modes switch
  to what the merge made true; then `iskron:reality-audit`: check the claim against
  the deployed artifact, not the diff.
- **Every merge → update the graph.** A push that only opened or updated a PR shipped
  nothing. The post-merge sequence hangs on the **event** of the merge, never on a
  lull. When merged, each of these is mandatory:
  - **Reconcile with reality.** Record what positions the change in the target
    system: architecture, module APIs, delivery, user experience, integration. Pure
    repo mechanics (lockfile noise, internal refactors without external effect, file
    moves) stay in git. Updating the graph means weaving, not editing prose: zero
    nodes and zero arrows after a substantive wave is an unperformed step — say so
    plainly if there was truly nothing, and why.
  - **Advance the map.** Keep open work attached `anga` to the transformation it
    moves. A thin `genre=hint` seed only for what the graph does not carry — a
    pointer, not a payload.
  - **Close along the axis, not by feeling "done".** `addressed_by` records the
    answer; `visarjana` is a separate volitive act. Release yourself when three things
    coincide: the answer stands in the graph as a node; the repo shows it; reality
    shows it as far as reachable — where unreachable, the user's word stands and you
    asked for it. Otherwise prepare the release and present it to the owner.
  - **Sweep the shipped contour.** A push that realizes designed nodes switches their
    modes (anagata→vartamana, kalpita→pratyakshita) across the *whole* designed
    contour and ends the design vimarshas the shipment resolved.
  - **Work the inbox.** `posed_to` questions the work answered end by the rule above;
    stale ones are parked or grouped.
  - **Reconcile code and graph.** End of every substantive task: the tidy-up beat of
    `iskron:reconcile` — area nodes against the code, code against the graph, the
    three paths (rejected alternatives recorded and referenceable). Remaining debts
    as vimarshas, not narrative.
  - **Feedback reflection** — on merge, and at the close of a session no merge
    crowned: examine the session's experience *of the method itself* — where a skill,
    rule or surface failed, surprised, or worked for the wrong reason; check for what
    is already said and record only a case worth recording, **at its address**: in
    this system's work graph, anchored to the tool's node or contour, addressed to its
    steward (`posed_to="steward"`); method-general ones to the owner role. Driven by
    `iskron:feedback`. **An empty reflection is a valid outcome; invent nothing.**
  - **Vocabulary pass.** Re-read what you are about to land — repo text and graph
    nodes — for borrowed project-management words (ticket, backlog, sprint, epic,
    story, done, blocker, committed). Do **not** substitute: name each to the user and
    ask what it is called in this project.
  - **Certification record.** If the merge touched solver behaviour or a certification
    claim, the cert report (`docs/dev/disaster_cert_report_2026_07_07.md`) or the
    chronicle moved in the same push; superseded claims are kept and marked, never
    deleted.

  `iskron:weaving` / `iskron:design` carry the *how*.
- **Design completeness criterion:** a design is not *ready* until its decisions, risks
  and lifecycle are in the graph — whichever skill elicited it. Saving to the graph is
  memory work, not implementation: design-phase gates on implementation do not apply
  to it. A design/spec file written by another suite is a draft view: intake it **in
  the same session**. Working autonomously: land decisions and risks now; propose a
  transformation with a telos marked for owner confirmation, do not skip it.
- **Execution suites lead execution.** Planning, TDD, debugging, verification, review
  belong to the installed execution suite; the graph carries only the memory/design
  plane. Decisions born in execution still land in the graph **before the session
  ends**.
- **A claim you made is not a claim you accept.** Behavioural claims — "the fix
  works", "the probe passes", "the suite is green" — close on the verdict of the cold
  `verifier` sub-agent, never on your own re-reading. Give it the claim, the carrier
  and the falsifier from *Reality* — and **wait for the verdict**. (why: you see your
  own change as intended, not as it is.)
- **Hook merging.** Where the harness has a hooks file, entries of different suites
  coexist — add beside, never overwrite others'.
- These reminders are automated in Claude Code (`.claude/settings.json`): session-start
  entry, post-`git push` ritual, memory-guard, the anti-freeze `Stop` hook, and the
  spec-write reminder (interop `full`, below). Codex reads this file natively and
  carries no hooks: for Codex the rituals bind through this prose only.
- **Keep this file honest.** It is generated by `iskronify` and stamped at the bottom
  with the contract it came from. Re-run `iskronify` when the installed skill
  announces a newer contract — its number is the first word of the `iskron:iskronify`
  skill description — or when the sources this file is derived from moved after the
  stamp date (`git log -1 --format=%cd -- pyproject.toml .github/workflows configs
  src/deqn_jax` against the date). A mismatch makes the iskronify run the session's
  first move.
- **Keep your toolchain fresh.** Updates are on by default; take them as the channel
  delivers, do not pin. (why: a stale skill drifts from the tool surface it names and
  degrades you silently.)

### Stage self-check
Quality gate green and a coherent stage finished — a PR opened or updated, or you are
about to touch nodes beyond those you started from — re-read your branch diff
against trunk for: bugs, fragile spots, weak error handling, DRY/SOLID violations,
repeated patterns, missing or useless tests, files over 150 lines and god-units
mixing concerns. Fix in the **same branch** and push again — or say plainly that
nothing surfaced. Invent no findings. **Per stage, not only at the end.**

### Cold stage review
**Self-check does not replace cold review.** Re-reading your own work you see what
you meant, not what you wrote. Both, in this order: own first, then cold.

After the self-check of an open/updated PR or a finished large stage, **open a review
by the top-tier sub-agent** (role `reviewer`, `.claude/agents/reviewer.md`). Where
the harness cannot, say plainly that there was no cold review. **Only a push has a
watchman.** A stage closed without a push is held by you alone — an acknowledged gap.

The reviewer's field is four things, all mandatory: the **branch diff against trunk**
(the whole branch); **the repository itself**; **the focus holon and its steward
role** (`#1`, `#8`/`#9`); **references to the graph nodes entering the diff** — the
leading vimarsha and every kriya, phenomenon and rule the branch changes — never a
retelling. The reviewer runs `iskron:integrity` read-only and returns findings plus
an **integration report**. `NEEDS_CONTEXT` names a graph gap: fix the graph
(design, weave, wake the neighbour through `iskron:collaborate`), then repeat. Fix
what came back in the same branch; reject a finding with a recorded "why" (in the PR
or on the node).

### Branch discipline
One branch through to its merge — follow-ups go into it. Small fixes may commit to
`master` directly (current practice); features and anything touching solver
behaviour go through a PR the maintainer squash-merges. After a merge:
1. `git checkout master && git pull`.
2. Delete the merged branch (`git branch -d <name>`); prune others already in `master`.
3. Update the graph: the change on `master`, not in the branch — weave the shipped
   state into the contour, end what the merge resolved (`iskron:weaving`).
4. Confirm the cleanup before the next task.

### Workflow-suite interop (superpowers)
Superpowers itself ratifies this contract: "user instructions always take
precedence", with "User's explicit instructions (CLAUDE.md, GEMINI.md, AGENTS.md,
direct requests)" at highest priority (using-superpowers, Instruction Priority);
"(User preferences for spec location override this default)" (brainstorming).
AGENTS.md is user instructions: everything below lives inside superpowers' own
rules, not as an exception to them.
- **Run brainstorming for creative work** — its Socratic elicitation is welcome. The
  spec it writes (e.g. under `docs/superpowers/specs/`) is a draft view; the design
  record is the graph.
- **Saving decisions to the graph is memory work, not implementation** —
  brainstorming's HARD-GATE ("Do NOT … take any implementation action") does not
  reach it, by its own wording. A design is not ready until its decisions, risks and
  lifecycle are in the graph.
- **The post-brainstorming handoff holds**: first intake the spec into the graph, in
  the same session (user instructions come first by the priority clause), then hand
  off to writing-plans exactly as brainstorming directs.
- **The execution plane is ceded**: planning, TDD, debugging, verification, review
  and their kin — whatever the installed suite ships — lead execution. Decisions born
  mid-implementation still land as graph nodes before the session ends — never
  deferred to a future push.

*(interop: full — verified against superpowers@6.3.0 — re-check on suite upgrade)*

## Working principles
1. **Think before coding.** Name assumptions; ask when unsure — naming *what* is
   unclear, not only "which option". **Questions to a human are asked in text** — in
   the conversation or a channel; the interactive option-menu tool is never used: a
   list of options replaces the question with an answer and hides what is actually
   unclear. Raise competing readings; push back on false premises. Check repo + graph
   before writing; fetch, don't recall. Hit the live system before trusting a type, a
   name, a doc. Questions beyond the boundary or mandate become `posed_to` vimarshas
   to the owner role `#8`.
2. **Simplicity first.** Minimum code for the task. No speculative features, no
   abstractions for single-use code, no handling of impossible errors. Validate at
   boundaries; trust internal invariants. 200 lines that could be 50 → rewrite.
3. **Stay inside the repo boundary.** Never leave this repository's working directory.
   A change belonging to another contour (the private notes repo, the DGX record,
   someone else's repo) is not yours across the boundary: record it as a vimarsha on
   that contour's node, `anga` to the transformation it serves.
4. **A second implementation is an event to report.** About to write what already
   exists — the same helper for a second consumer, the same rule in a second module?
   First derive both places through the integration field (`iskron:integrity`), then
   name them to the user and propose reunion or a named, deliberate fork. Extend the
   existing class instead of forking a near-duplicate (the LinearPlusMLP /
   KfAnchoredMLP fork cost a full sweep to diagnose).
5. **Surgical changes.** Touch only what the task needs. Don't reformat or refactor
   neighbouring code; the linter is authoritative. Delete only dead code your change
   created; flag the rest.
6. **Goal-driven execution.** Tasks → verifiable goals. Bugs: pin with a failing test
   before patching. Multi-step work: `step → verify` pairs. Solver changes: verify
   with the certification stack on real checkpoints on the DGX, never with unit tests
   alone — *Reality* names the carriers. Name the falsifier before looking ("which
   observation would refute this?") and observe the carrier, not the source that
   should have produced it.
7. **Read before answering an open question.** Tasks framed *discuss / think through /
   investigate / design / plan / analyse / "what do you think"* are answered from
   recorded thinking, not training data: query the graph first, several ways (one
   miss ≠ absence); `iskron:entry` drives the protocol.
8. **Think in the graph, speak the project's language.** The graph's structural
   vocabulary (kriya, phenomenon, contour, role, vimarsha, the three mode axes) is for
   reasoning; it never appears in what you say to the user until the user uses it
   first. Translate into the project's own words: arm, certificate, probe, checkpoint,
   pin, anchor, coverage, the frozen convention. Talk *about* work in plain
   description — the question, the change, what is open, what it resolves — never
   ticket, task, sprint, backlog, story, done.
9. **Claims need receipts.** Every quantitative claim in a dev doc carries a commit,
   file, log, or reproduce command. Superseded claims are kept and marked, not
   deleted. Never write "solved" past what the certificate stack showed at the frozen
   convention.

## Integration field — from the graph only
The focus holon `#1` and its steward roles are the only permanent root of the
walk. A list of shared surfaces, consumers and dependencies is **not kept in
AGENTS.md and not asked of the human**: such prose goes stale in the course of the
work itself. For every change, name the graph nodes whose realization enters the
diff and run `iskron:integrity`. Trace a phenomenon both ways with
`iskron_orient(lens="trace")`; for a kriya walk its `next` thread and the relays of
its `ahara`/`utpatti`/`upadhi`; an exit into another holon leads to its steward
role. A dependency the walk did not find is a model defect, not a reason to write a
list: design the missing kriyas and phenomena, weave the unwired edges, pose a
vimarsha for a pending foreign decision and wake its addressee through
`iskron:collaborate`.

## External surfaces — what you use and do not own
JAX/Equinox/Optax APIs, Dynare outputs, the reference implementations (Simon's EWM
notebook, the RSS TensorFlow checkpoint), the DGX container image. The agent
**guesses** these from training memory, and memory is indistinguishable from
knowledge from the inside.
- **Before the work, record the part of the surface the work will touch** — as a graph
  node, with the version you looked at.
- **Sources by seniority — perception before testimony.** Observation with your own
  hands (a call, `--help` of the installed binary, the installed package's types, a
  response you saw) outranks documentation; documentation outranks memory; **memory
  is not a source.** Write the epistemics honestly: `pratyakshita` only for what you
  observed, `anumita` for what you derived from docs.
- **Weave the link.** The surface node is `upadhi` to the kriya that acts through it.
- **Keep in step.** Found a discrepancy or the vendor bumped a version — fix the node
  in the same move.
- **The reference works both ways.** Source that works with an external surface
  carries `(graph @aleph/deqn, node #N)` — and you **read that node before working**.

## Reality — what a claim is checked against
| Claim class | Canonical carrier | How to observe | Who can |
|---|---|---|---|
| "model X is solved / certified" | the **final** checkpoint on the DGX, `runs/<arm>_s<seed>/checkpoint_003000.eqx`, three seeds, fp64 | `JAX_ENABLE_X64=1 uv run python scripts/disaster_ss_probe.py --runs-dir runs/disaster_cert --arms <arm> --seeds 0,1,2` on the DGX host; stress grid via `scripts/ewm_stress_table.py`; report the learned-block ρ, ŝ, residuals at ŝ | agent (ssh) |
| "the code is correct / tests pass" | the suite run on the DGX host, not the laptop | `ssh anna@130.223.169.108 'export PATH=$HOME/.local/bin:$PATH; cd ~/projects/<lane-dir> && uv run pytest tests/ -q -m "not slow"'` | agent |
| "CI is green" | the GitHub Actions run for the PR/commit | `gh pr checks <n> --watch` / `gh run list --branch <b>` | agent |
| "a training recipe behaves" | the run directory on the DGX (checkpoints, `DONE` marker, config, TensorBoard) | `ls runs/<arm>_s<seed>/`, `logs/cert_container*.log` | agent (launch: `run_sweep_in_container.sh` in the NGC container) |
| "the docs are live" | https://deqn-jax.github.io | `curl -sI` the page; deploy is `mkdocs gh-deploy --remote-name pages` | agent observes; maintainer deploys |
| "a config change took" | the resolved `TrainConfig` printed at run start / `config.yaml` in the run dir | `uv run deqn-jax train <model> --config <yaml> -n 1 -q` and read the resolved config | agent |

**Ceiling**: the disaster model's *true* equilibrium (no oracle exists — the whole
research program is about certificates in its absence; "solved" means "passes the
stack at the frozen convention", nothing more); Dynare comparisons (fixtures are not
in the repo — `dynare/` is gitignored, 18 tests skip); parity with external
references (Simon's TF numbers, the RSS checkpoint) is approximate by construction
and closes only by convergence of independent evidence; the user's private
repositories and machines.

**The table grows by use.** When a session teaches you a carrier the table lacks, an
observation that turned out reachable, or one that turned out unreachable (→
*Ceiling*), write the row *then*, in that session.

## Graph ↔ repo: where what lives
| Concern | Repo | Graph |
|---|---|---|
| Code, configs, lockfiles | ✓ | |
| Commands, conventions, gotchas, stack | ✓ (AGENTS.md) | |
| Certification records, chronicle, dev docs | ✓ (`docs/dev/`) | ✓ (vimarshas link to them) |
| Research narrative, audits, paper basis | private notes repo | |
| Branch state, what is in flight | git + PR body | ✓ (`genre=hint` — work without a PR) |
| Methodology, ontology | | ✓ |
| Design decisions, open questions | | ✓ (vimarshas) |
| Plans, session handoffs, lessons | | ✓ (project graph; thin `genre=hint` for the off-map remainder) |
| Commit history, PRs, SHAs | git | (never in the graph) |

**`HANDOVER.md` is not kept — a decision, not an omission.** Branch state already has
homes, and a handwritten file is the only one that diverges silently: branch and
in-flight work — `git branch`/`log` and the open PR; how a claim is checked — the
*Reality* table; why it was decided and what is open — the graph; work under way with
no PR yet — a `genre=hint` seed. Branch state is said **in the PR body**. Forge:
GitHub, CLI `gh` (account `mechanicpanic`); watch with `gh pr checks <n> --watch`.

## Certification (what "solved" means here)
A small training loss is not a certificate — measured repeatedly (best-by-loss
checkpoints are certificate-worst; `save_best_checkpoint` defaults to True and
should not be trusted for claims). The stack, in order of strictness: held-out/stress
residuals → **learned-block** spectral radius (the probe's raw ρ has a floor at the
exogenous root 0.98699 — always report the learned 8×8 block) → solved fixed point
ŝ = T(ŝ): ‖ŝ−s\*‖ and ρ(ŝ) → per-equation residuals AT ŝ → long-horizon convergence,
multi-seed. Frozen convention: **final checkpoints** (`checkpoint_003000.eqx`), fp64.
Under `bk_pin` the SS-error and ρ(s\*) legs are donated by construction; the earned
legs are the stress grid and the residuals at ŝ — say which is which. **Probe episode
0 as well as the end**: certify what a run starts from (2026-09-02 warm-start finding).

## Commands
| task | command |
|---|---|
| test | `uv run pytest tests/ -q` (654 collected @2026-09-03, 10 of them `slow`; 18 skips when Dynare fixtures are absent; full suite runs on the DGX host) |
| lint | `uv run ruff check src/ tests/ scripts/` (zero-error; CI-enforced) |
| format | `uv run ruff format src/ tests/ scripts/` |
| typecheck (advisory) | `uv run pyright` (basic mode; dev group; not in CI) |
| train | `uv run deqn-jax train <model> -n 1000` (`-o ngd -q` for smoke; arm configs via `--config configs/<arm>.yaml`) |
| list models / optimizers | `uv run deqn-jax list` / `uv run deqn-jax optimizers` |
| certificates | `JAX_ENABLE_X64=1 uv run python scripts/disaster_ss_probe.py --runs-dir runs/disaster_cert --arms <a> --seeds 0,1,2` |
| DGX sync | `rsync -az --exclude .venv --exclude .git <worktree>/ anna@130.223.169.108:~/projects/<lane-dir>/` (one directory per lane; never the main checkout) |
| DGX GPU sweep | `LAUNCHER=scripts/cert_sweep_container.py ./scripts/run_sweep_in_container.sh` (DONE-marker resumable) |
| docs deploy | `mkdocs gh-deploy --remote-name pages` |

Always `uv run`; never activate the venv manually. On the DGX, non-interactive shells
need `export PATH=$HOME/.local/bin:$PATH` before `uv`.

## Project structure
```
src/deqn_jax/
  config/        # Pydantic v2: TrainConfig + nested blocks (optimizer, network, composite_loss, coverage, replay_buffer, moment_matching); io.py derives --set dispatch from model_fields
  types.py       # ModelSpec, TrainState, Metrics — NamedTuple pytrees
  cli.py         # train / list / info / optimizers / check / irf / evaluate / active-subspace / init-config
  models/        # 11 registered models (`deqn-jax list`); each: variables, equations, dynamics, steady_state
  networks/      # factory.py + common / mlp / lstm / transformer / linear_plus_mlp / kf_anchored_mlp; models/disaster/network.py (π_BK + δ, bk_pin)
  optimizers/    # registry + standard / pcgrad / mao / lbfgs / gauss_newton (+ ngd, shampoo, mao_kfac)
  training/      # trainer, state_init (dispatch + validators), cycle, loss, composite_loss, coverage, episode, shocks, linearize, warm_start
  evaluate/      # simulate, diagnostics, dynare, cli
configs/         # arm configs (disaster_gated_pcgrad_bkpin.yaml etc.); configs/archive/ is gitignored
scripts/         # gitignored except the whitelist in .gitignore: probes, sweeps, risky-SS, GN polish
tests/           # smoke convention: 3 episodes, hidden=(16,), batch=16, mc_samples=2
docs/dev/        # cert report, chronicle, library review (research state)
.claude/agents/  # reader / worker / verifier / reviewer role agents
```

## Code conventions
- **Meaning lives in the graph, code references it.** A comment carrying a decision's
  rationale, rejected alternatives or integration layout belongs in the graph; leave
  a reference `(graph @aleph/deqn, node #N)` in the code. Mechanics of a step — in the
  comment; meaning and rationale — in the graph. This is public code with readers who
  have no graph access: keep such references to the places where *why* is otherwise
  unrecoverable, and keep the dev docs self-standing.
- Config precedence: `--set` overrides > CLI args > YAML file > defaults
  (`load_config()` in `config/io.py`; dot-notation reaches every nested block, e.g.
  `--set coverage.enabled=true`). When a hyperparameter "didn't take", check for a
  `--set` in the launcher before editing the YAML.
- Two JIT boundaries per cycle (rollout + grad-step sweep); everything
  runtime-variable resolves at construction time, before tracing.
- Five train-step variants (STANDARD/PCGRAD/MAO/LBFGS/GN) dispatched by
  `OptimizerKind` (resolve the kind through `optimizers.registry.get_optimizer_kind`,
  never a name list); when adding a loss feature, extend
  `state_init._validate_train_config` so combos that would silently drop it from the
  gradient are rejected, not ignored.
- Loss-dict keys prefixed `aux_` are excluded from reweighting and gradient surgery
  by contract (`eq_losses_to_array`).
- Equinox patterns: `eqx.filter(model, eqx.is_array)` → update →
  `eqx.combine(arrays, model)`.
- **Test discipline**: unit + regression (bit-identical guards for refactors: same
  seed, same loss history, same parameter hash); certification claims additionally
  need the probe stack on real checkpoints (*Reality*).
- **Standing rules from the maintainer**: be autonomous — push through to the next
  concrete artifact instead of asking after every plan step; never launch the
  multi-agent `/code-review` workflow or agent fan-outs on routine PRs — ask first,
  manual review is the default; Simon's code may be read but never ported verbatim
  into the public tree, and no private paths appear in it; the unpublished paper is
  not cited in tracked files; never write the maintainer's preferred name in tracked
  docs — "the maintainer"; git identity for private/work repos is the official name
  (see the private notes repo).
- **Gotchas**:
  - *Warm start on anchored nets*: `warm_start: true` is a no-op on any network
    carrying the linearization (`P`/`P_kf`) — the constant-SS fit used to run on
    `disaster_policy_net` and flattened the BK slope before episode 1 (ρ(SS) 1.14);
    fixed 2026-09-02, cert record amended.
  - *Checkpoint loading*: the template network must be built from the FULL network
    config through `networks/factory.py` — static fields (`bk_pin`,
    `use_zlb_feature`, reparam flags) change the forward graph and are not repaired
    by leaf deserialization.
  - *JAX*: `jax.tree.map` treats tuples as containers; `lax.cond` needs an operand; no
    `float()` inside JIT; Shampoo L/R preconditioners in separate `tree_map` calls.
  - *Probe ρ floor*: raw closed-loop spectral radius never reads below 0.98699
    (exogenous mu_ups root × soft-clip); report the learned-block eigenvalue.
  - *Compute placement*: training, evaluation and the full pytest run happen on the
    DGX (host CPU for probes and tests, NGC container for GPU training), not on the
    maintainer's laptop. Verify the GPU with `nvidia-smi` before claiming it is down.
  - *Shared DGX state*: `~/projects/deqn-jax` on the DGX is the certification record
    (`runs/`, `logs/`) — append-only; every branch syncs to its own
    `~/projects/deqn-jax-<lane>/` directory and runs its suite there.
  - *aarch64 flake*: `test_convergence.py::TestDisasterTraining::test_loss_decreases`
    is platform-sensitive (chaotic bare-MLP lr=1e-2 path, last-bit bifurcation) —
    `xfail(strict=False)` in place; the module is `slow`.
  - *CI parity gap*: pyright is advisory-only (not in CI) — deliberate, see Nature.
  - *Short disaster smokes*: episodes < `lr_warmup` makes the cosine schedule
    negative — pass `--set optimizer.lr_warmup=5` for runs under ~100 episodes.
  - *Ruff hook*: `.claude/hooks/ruff_on_edit.py` runs after every edit and strips
    unused imports — an import added before its use is removed; re-add it.
  - *Evaluator units*: `deqn-jax evaluate` reports single-draw residuals across mixed
    per-equation units for single-stage models (July audit, unshipped) — do not
    compare its log10 grades across models.

## What to update when
- `AGENTS.md` — by the inverted default: **if it can be learned by reading a graph
  node, it is not here.** Only what is needed BEFORE the agent reaches the graph:
  commands, orientation addresses, code invariants the linter cannot express, forks
  that must stop you before acting — updated when THOSE change (commands, stack,
  conventions, reachability of a reality carrier).
- `docs/dev/disaster_cert_report_2026_07_07.md` — any certification claim.
- `docs/dev/selection_program_chronicle_2026_07.md` — program-level shifts (new
  results, retractions, method lessons).
- Private notes repo — research narrative not for the public tree.
- The project graph `@aleph/deqn` — every merge (see *Session lifecycle*).

## Git workflow
- Conventional commits (`feat:`/`fix:`/`chore:`/`refactor:`/`docs:`/`test:`); bland
  messages; no session-metadata lines. Claude agents append the trailer
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` (maintainer's standing
  choice). Branches `feat/…`, `fix/…`, `chore/…`, `docs/…`; PR titles in the same
  format.
- **Local gate**: ruff runs on every edit via the committed `.claude` hook; run lint +
  the targeted tests locally and the full suite on the DGX before pushing. CI enforces
  lint + tests on every push and PR.
- **Definition of done**: a PR to `master` with CI green (`gh pr checks <n> --watch`),
  squash-merged by the maintainer (or a small fix pushed to `master` directly); for
  solver changes, certificates unregressed at the frozen convention; docs updated per
  *What to update when*; the graph updated on merge.
- Coding agents (Claude, Codex) work in git worktrees, never the main checkout; one PR
  per delivery step; GitHub refuses approvals from the PR's own account, so reviews
  by agents go as comments and approval is the maintainer's click.
- **Never** `--no-verify`, `--force`, `--no-gpg-sign`, or `git reset --hard` without
  explicit user instruction. Pushing `master` is the maintainer's call unless
  explicitly delegated.

*(iskronify: contract `6`, stamp `2026-09-03` — re-run when the installed iskronify
description names a higher contract or when the sources this file is derived from
moved after this date.)*
