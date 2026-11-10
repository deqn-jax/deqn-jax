---
iskron:
  graph: r71
  holon: 1
  agent_role: 9
  contract: 41
  stamp: 2026-11-10
---
# deqn-jax
Deep Equilibrium Network trainer for economic models (JAX/Equinox port of
Azinovic–Maliar–Maliar) — for economists solving dynamic stochastic models and
for the research program certifying those solutions.

## Project
- **Nature**: research. Relaxed, per tool: pyright stays `basic` and advisory, outside the gate (strict mode drowns in JAX/Equinox partial stubs); no coverage threshold; ruff selects `E,F,I,W` — tightening to `B,N,UP,RUF` is deferred to its own lane with two exemptions (economics variable names, Greek in docstrings); no pre-commit hook. Everything else: production discipline.
- **Cost of breakage**: public research code — wrong conclusions and lost collaborator trust, worse than a crash. Solver claims are gated by certificates, never by green tests alone.
- **Consumers**: collaborators and course students reading the public repo and the docs site; the research program consuming the certificates. Breakage surfaces as a wrong research conclusion, caught by the certificate stack or by a reader, not by a crash.
- **Stack**: Python ≥3.10; JAX ≥0.4.20 + Equinox ≥0.11 + Optax ≥0.2; Pydantic v2 configs; uv. No TensorFlow, no PyTorch, no Keras.
- **Gate**: `make check` (`make check-fast` skips the `slow` tests).
- **Layout**: code map — one line per component in *Project structure* (no component READMEs yet; the first to add one moves the line there); gotchas — the graph, as rules on contour `#1`.
- **Feedback reflection**: yes.
- **Workflow-suite interop**: full.

## Persistence rules
State lives in the **repo** or in the **graph** — nowhere else. The harness's built-in memory (memory directory, conversation summaries, `/tmp`, machine-local files) is **forbidden entirely, not by category**; the temporary directory is session scratch only, and the session removes it.
- **Repo**: code, configs, rituals (how to act here), branch state.
- **Graph**: decisions, questions of substance (vimarshas), plans (the transformation map), lessons, gotchas. Do not retell the graph in the repo — link.
- **Case** (a conversation on a graph node): a one-off assignment, question, agreement, acceptance; the course and outcome of work are lines of its summary; the state of what the work changes is in the graph.
- **Private notes repo** (`~/Projects/research/deqn-research-notes`, GitHub `mechanicpanic/deqn-research-notes`): research narrative, audits, paper basis — anything not for the public tree.
- **Fetch state; never recall it.** No source for "we decided…" — read the graph or the repo before acting.
- **A node is not recorded until what pulls it is named** — which action breaks if it vanishes? Hiding places and the form of a record: skill `iskron:writing`.
- **Whose fact is this?** — ask always, over the harness's memory instruction; before finishing, route every durable fact of the conversation: a property of the code, ritual, command, order of this repo → this file; a gotcha → the graph; the meaning of code (decisions, grounds, links) and knowledge that outlives a refactor → this repo's graph, a decision as a node at once; a question of substance or an obligation → a vimarsha; a dated duty → a node carrying the date in `attrs`; a one-off → a word in the case; course of work → modes of nodes; research narrative → the private notes repo; about another project → its graph. The user's own facts are not written here — where they live is decided by the door `iskron:iskron`.
- **Project instructions are committed files of the working tree (AGENTS.md, the hooks file) and the project graph**: never a global instructions file, harness memory or user-level hooks; personal preferences are not copied into the project; installing the delivery (bridge, plugin) is delivery, not configuration.
- The memory directory is **evacuated and frozen**: the memory-guard `PreToolUse` hook holds it (block, exit 2); the `rituals:` field is not read by the hook — when switching `guard` off, remove the hook too. What the directory held is in the private notes repo (`dev/agent_memory_evacuation_2026_09_03.md`).

## Session lifecycle
One line per rule; the full norm is "Cross-cutting norms" of the skill `iskron:iskron`.
- **Graph is the work, git is how we got here.** SHAs, branches, PR numbers, "merged" never enter the graph (bodies, names, attributes); provenance goes to the record's `reasoning`; in a case — freely.
- **Session start** — the "Start" section of the skill `iskron:iskron`, before acting; it ends in readiness (graph and role named, the greeting arrived), not in reading; the role's queue — `iskron_orient(focus=<role>)` on entry and when there is a reason. Standing — on watch or before the first write to the graph or a case, by one `iskron_stand`; a reading session takes no place. `start <graph> <role> <case №N>` enters that case, the first word restates the assignment; a sub-agent is launched by such a line of a child case. A sub-agent has its own satellite bridge (`iskron_stand` with `satellite_of`, `join`, `leave` on it); on the caller's bridge it does not write to the graph and says so in the first line of its result. Addresses come from the cover; the session's person and the route of a question of principle — by "Start", not by a `group` chain. Then `git status`, and check running DGX jobs (`logs/` on `anna@130.223.169.108:~/projects/deqn-jax`) before launching new work.
- **Tidy with every finished piece** ("Tidying" in the skill `iskron:iskron`): scratch and your own development caches in the same move, tree and branch by *Branch discipline*; in cases where no move of yours is awaited — your lines closed with an outcome and `leave` (the leader — `propose_close` or a handover of leading). Sit only where you await an answer or lead something unclosed.
- **Graph craft belongs to the one who decides**: reading for a decision, the decision, weaving, integration and acceptance. Sub-agents on their own satellite bridge take volume under a ready decision: code, gate, review, verification, edits of many touched nodes with references to the decision; search by the `iskron:recall` ladder — `recaller` (the decider opens the find himself), intake of a complex story — `ingester`; the launcher checks the result.
- **Starting work: graph, then project, then code.** (1) graph reconnaissance (`iskron:entry`): what is recorded about the place of the change, open vimarshas, decided and rejected; (2) the integration field (`iskron:integrity`, section below); (3) design (`iskron:design`), then code. Skipped only on an explicit "work directly" or another named protocol, the reconnaissance debt paid at reconcile; the person's silence is not permission.
- **A decision enters the graph when it is made**, wherever it came from (chat, socket, agreement of agents) — in the repo holon, by editing what it touches and by links, the thesis as its own node, modes by `iskron:writing`; what counts as execution — in the node; who decided — in `reasoning` and the case, not in the node. A changed situation — just as immediately.
- **A task is described before it begins — as what it is.** A one-off — a word in a case (assignment, restatement, outcome in one case); arrived without a case — open one (`open_room` on the subject's node or `iskron_case(action="talk", about=<subject>)`) before the first change outside. Into the graph go the transitions it changes and the decisions. A vimarsha is only a question of substance or an obligation; an action (kriya) is a repeatable transition. Telling a task from a lawful one-sided edge action — `iskron:reconcile`, pass 2.
- **Work runs in the graph.** A question of substance or an obligation — a `posed_to` vimarsha with "Answered when"; a decision — a node; a handover — relay and modes; after an event of the work (merge, rollout, decision) the graph catches up in the same move. A case is a conversation; its summary is the last line of each key: write a line when the state of the subject changes, with the observed outcome, not per action; unfinished — `partial`, "on whom, awaiting what". Awaiting an answer — `ask`, an assignment — `say` with `to`, a receipt — with `in_reply_to` (reference `cases.md` of the door `iskron:iskron`). A word to an agent goes into a case; to the person — `ask` or a direct word to his place, he is not invited into a case. Leaving — the transformation's seed and a word in the case.
- **Merge → graph.** A push that opened or updated a PR shipped nothing; the sequence hangs on the event of the merge, not on a lull. All acts are mandatory — except for work by reference from another agent (assignment by references, report in the case): there weaving, closing along the axis and reconciling go back to the assigner; the doer keeps the transformation's seed and the delivery modes (skill `iskron:vahta`):
  - **Weave** (`iskron:weaving`): what shipped enters the target system as nodes and edges, not a paragraph; repo mechanics stay in git. Zero nodes and arrows after a substantive wave — say plainly why.
  - **Advance the map**: open work is `anga` to its transformation; one `genre=hint` seed per transformation — only what reading the graph and cases cannot restore; live cases by a pointer to reading them (`iskron_case(action="at")`, `read`).
  - **Switch modes** by evidence on the carrier (`REALITY.md`, `iskron:reality-audit`), not by the merge: the merge moves nodes of what the code carries; a practice (`anagata→vartamana`) — a rollout or run seen on the carrier (`anumita`), observed — `pratyakshita`.
  - **Close along the axis** (`iskron:inquiry`): `addressed_by` to the carrying node; `visarjana` — when the answer stands as a node, the repo shows it and reality shows it as far as reachable (where not — the user's word); otherwise present it to the owner. In the same move — the `posed_to` inbox (woken by a change at the anchor, not by age), `propose_close` with evidence.
  - **Reconcile — `iskron:reconcile`**: nodes against code, code against graph, the rejected recorded and referenceable; the remainder as vimarshas.
  - **Feedback reflection** — on a merge and at the close of a session without one: experience *of the method* (a skill, rule or surface failed or surprised) — as a case, not an opinion, after checking what is already said, into the work graph anchored on the tool's node, addressed to its steward; about the method in general — by the route of the skill `iskron:feedback`. Zero records is a valid outcome.
  - **Vocabulary pass**: borrowed words (list — principle 8) in the text and nodes being landed — name them to the person and ask what it is called in the project; do not substitute.
  - **Certification record**: a merge that touched solver behaviour or a certification claim moves the cert report (`docs/dev/disaster_cert_report_2026_07_07.md`) or the chronicle in the same push; superseded claims are kept and marked, never deleted.
- **A design is not ready until its decisions, risks and lifecycle are in the graph** — whichever skill elicited it; an external design/spec file is a draft for intake in the same session. Without the owner: decisions and risks now, a transformation with a telos marked for confirmation.
- **Execution suites lead execution** (plan, TDD, debugging, review); the graph carries memory and design: decisions and risks of execution enter the graph before the session ends.
- **A claim you made is not a claim you accept.** A behavioral claim is closed by the cold `verifier`: the brief is the claim, the carrier and the falsifier from `REALITY.md`; wait for the verdict. No role available — observe the carrier yourself, never the source.
- **Hook merging**: entries of different suites coexist in the hooks file — add beside, never overwrite others'.
- Start, push and merge rituals are carried by the delivery, by the cover; repo hooks in `.claude/settings.json`: memory-guard, spec-write, ruff-on-edit. Codex gets no hooks: for it the rituals bind through this prose only.
- **Keep this file honest.** The contract is the number after «КОНТРАКТ AGENTS.md:» at the start of the `iskronify` skill description: compare it with `contract` of the cover, loading nothing. Higher than `contract`, or the sources moved after `stamp` (`git log -1 --format=%cd -- pyproject.toml .github/workflows configs src/deqn_jax Makefile`) — run `iskron:iskronify` as the first move, in a clean tree and a work branch; the run does not wait for the person. **Lower than `contract`** — the file is ahead of the delivery: do not lower it, update the shadowing copy of `iskronify` and re-read the description. A line of this file disagrees with the skill — name it aloud: `contract` lower — the skill is right; higher — the file; equal — a template defect, feedback to the steward of the skills delivery.
- **Keep the toolchain fresh**: updates are on by default, take them as the channel delivers. A plugin update takes effect only after the harness session restarts — check the running version before a run.

### Stage self-check
Gate green and a coherent stage finished (a PR opened or updated, or you are about to touch nodes beyond those you started from) — re-read the branch diff against trunk: bugs, fragile spots, weak error handling, DRY/SOLID violations, missing or useless tests, files over 150 lines and god-units. Fix in the same branch or say plainly that nothing surfaced; invent no findings. Per stage, not only at the end.

### Cold review of a batch
- **Self-check always, cold review after it where stated below**: you see your own work as intended.
- **Review before merge — by this repo's policy**: features and anything touching solver behaviour go through a PR the maintainer squash-merges after reading it; manual review is the default. Never launch the multi-agent `/code-review` workflow or agent fan-outs on routine PRs — ask first. The reviewer is not the author of the diff. CI, tests and a targeted check of a threat to certified claims — on every batch.
- Review by the role `reviewer`, **in a separate worktree** (Claude Code — `isolation: "worktree"`); where the spawning tool gives none, the leader lays a detached tree (`git worktree add --detach .worktrees/<name> <head>`), its path goes into the brief as "work only there", the edit ban is the second line; the report says "isolation manual". Neither — say there was no cold review. Only a push has a watchman: a stage without a push is held by you.
- The standing alternative is Astra through `codex exec` (*Commands*); its findings and dispositions go on the PR as a comment.
- The reviewer's brief: the branch diff against trunk, the focus holon with its steward role and references to the nodes of the diff, never a retelling; it also asks for docstrings, comments and docs in the diff to be checked against the code they describe. `NEEDS_CONTEXT` is a graph defect: design, weave, pose vimarshas, repeat the review. Reject a finding you disagree with by a recorded "why" (in the PR or on the node).

### Role-file tails
Role files are a projection on the machine, not in git; these lines go under `## This repo` of the role's file.
#### reviewer
- Trunk is `master`: `git diff origin/master...origin/<branch>`.
- Check docstrings, comments and doc pages in the diff against the code they describe (graph #59).
#### verifier
- Heavy carriers live on the DGX (`REALITY.md`); never train or run the suite on the laptop (graph #44); a lane directory per branch, never the main DGX checkout (graph #45).
#### worker
- Gate: `make check-fast`, run in a DGX lane directory; on the laptop only ruff and short smokes (graph #44).

### Branch discipline
One branch and one PR per batch: items and sub-agents' work merge into it locally. **The branch lives in its own worktree inside the repo** (`git worktree add .worktrees/<branch> -b <branch> origin/master`; `.worktrees/` is ignored), not in the main checkout: sessions share it, and a `checkout` there wipes someone's work. Small fixes may commit to `master` directly (the maintainer's practice); pushing `master` is the maintainer's call unless explicitly delegated. After a merge: `git worktree remove` your tree, `git branch -d` the merged branch and others already in `master`, your development caches; `git pull` in the main checkout; weave what shipped into the graph; the next branch from fresh `origin/master`; confirm the cleanup before the next task. **An abandoned tree** (branch in `master` or deleted on the forge, copy clean) is removed by any agent of the repo; uncommitted or unmerged work is not touched — name it to its holder or the person.

### Workflow-suite interop (superpowers)
Superpowers itself ratifies this contract: "user instructions always take
precedence", with "User's explicit instructions (CLAUDE.md, GEMINI.md,
AGENTS.md, direct requests)" at highest priority (using-superpowers,
Instruction Priority); "(User preferences for spec location override this
default)" (brainstorming). AGENTS.md is user instructions: everything below
lives inside superpowers' own rules, not as an exception to them.
- **Run brainstorming for creative work** — its Socratic elicitation is
  welcome. The spec it writes (e.g. under `docs/superpowers/specs/`) is a
  draft view; the design record is the graph.
- **Saving decisions to the graph is memory work, not implementation** —
  brainstorming's HARD-GATE ("Do NOT … take any implementation action") does
  not reach it, by its own wording. A design is not ready until its decisions,
  risks and lifecycle are in the graph.
- **The post-brainstorming handoff holds**: first intake the spec into the
  graph, in the same session (user instructions come first by the priority
  clause), then hand off to writing-plans exactly as brainstorming directs.
- **The execution plane is ceded**: planning, TDD, debugging, verification,
  review and their kin — whatever the installed suite ships — lead execution.
  Decisions born mid-implementation still land as graph nodes before the
  session ends — never deferred to a future push.

*(interop: full — verified against superpowers@7.0.0 — re-check on suite upgrade)*

## Working principles
1. **Think before coding.** Name assumptions; when unsure, ask *what exactly* is unclear. **A question to a person is asked in words, not by a choice tool**: in text — only to the one answering now; one that outlives the turn — an `ask` card in a case ("A question to the person" of the door `iskron:iskron`); the harness's option-menu tool — never. Push back on a simpler move or a false premise. Touch the live system before trusting a type, a name, a doc. Outside the mandate — a `posed_to` vimarsha to the addressee by "Addresses" of the door's start reference; a handover to an agent does not replace the act of the entitled person.
2. **Simplicity first.** Minimum for the task; no speculative features, abstractions for single use, handling of the impossible. Validate at boundaries.
3. **Stay inside the repo boundary.** Edits only in the working directory; outside it — reading carriers (the DGX), session scratch and the delivery's home. Another contour (the private notes repo, someone else's repo) — a vimarsha on its node (`anga` to the transformation) and a word to its steward in a case. **Do not take foreign code into context**: what you need comes from the graph or from the steward's word. The reference implementations may be read but never ported verbatim into the public tree.
4. **A second implementation is an event to report.** Derive both places through `iskron:integrity`, name them to the person, propose reunion or a named fork; a new consumer stands as edges in the same move (graph #39).
5. **Surgical changes.** Touch what the task needs; do not reformat or refactor neighbouring code; the linter is authoritative; delete only dead code your change created, flag the rest.
6. **Goal-driven execution.** A bug — a failing test before the patch. Multi-step — `step → check` pairs. Solver changes — the certification stack on real checkpoints on the DGX, never unit tests alone. The falsifier before the look; observe the carrier (`REALITY.md`), not the source.
7. **Read before answering an open question** — from what is recorded, not from training data (skill `iskron:recall`). Search output is a lead, not data: read structure and links by `iskron_look`, `iskron_orient`; to a sub-agent — reconnaissance of code and files (`reader`), graph search (`recaller`); reading the find for a decision is yours.
8. **Think in the graph, speak the project's language.** The graph's vocabulary (kriya, phenomenon, contour, role, vimarsha, modes) is for reasoning; to the person — the project's words (arm, certificate, probe, checkpoint, pin, anchor, coverage, the frozen convention) until he uses the graph's first. Borrowed words about work — ticket, task, sprint, backlog, epic, story, done, blocker, committed — are not substituted: name the word and ask what it is called here; say: the question, the change, what is open, what it resolves.
9. **Be autonomous**: push through to the next concrete artifact instead of asking after every plan step (the maintainer's standing rule).

## Integration field — from the graph only
- The root of the walk is the focus holon and the steward role of the cover. Keep no list of shared surfaces and consumers here and ask the person for none — what the graph does not model is in *External surfaces*.
- For every change name the nodes whose realization enters the diff and run `iskron:integrity`: a phenomenon — `iskron_orient(lens="trace")` both ways; an action — its `next` thread and the `ahara`/`utpatti`/`upadhi` relays; an exit into another holon — to its steward role.
- A dependency the walk did not find is a model defect: design (`iskron:design`), weave (`iskron:weaving`), a wait — a `posed_to` vimarsha and a word in a case. A new consumer is recorded when code and edges appeared in one move.

## External surfaces — what you use and do not own
JAX/Equinox/Optax APIs, Dynare outputs, the reference implementations (the EWM notebook, the RSS TensorFlow checkpoint and source), the DGX container image, the Codex CLI.
- **Before the work, record the touched part of the surface** as a graph node, with the version.
- **Perception before testimony**: an observation by your own hands (a call, `--help`, the installed package's types) outranks docs, docs outrank memory, memory is not a source. `pratyakshita` — only the observed.
- **Weave the link**: the surface node is `upadhi` (or `ahara`/`utpatti`) to the action acting through it.
- **Keep in step**: a discrepancy or a new version — fix the node in the same move, lower the epistemics if you did not observe.
- Source at a surface carries `(graph @aleph/deqn, node #N)` — read that node before working.

## Reality — what a claim is checked against
The carrier table is `REALITY.md` at the root, read when there is a reason: before saying "works" about behaviour, switching a mode or closing a question on a claim of execution — read the row of your claim class and observe its carrier; the row goes whole into the brief of `verifier` and `reviewer`. The first row — "model X is solved / certified": the final checkpoints on the DGX, three seeds, fp64, through the probe stack. Ceiling — classes without a reachable observation — is there too. Learned a carrier the table lacks — write the row then.

## Certification (what "solved" means here)
A small training loss is not a certificate (best-by-loss checkpoints are certificate-worst; `save_best_checkpoint` defaults to True and is not trusted for claims). The stack, in order of strictness: held-out/stress residuals → **learned-block** spectral radius (graph #43) → solved fixed point ŝ = T(ŝ): ‖ŝ−s\*‖ and ρ(ŝ) → per-equation residuals AT ŝ → long-horizon convergence, multi-seed. Frozen convention: **final checkpoints** (`checkpoint_003000.eqx`), fp64. Under `bk_pin` the SS-error and ρ(s\*) legs are donated by construction; the earned legs are the stress grid and the residuals at ŝ — say which is which. **Probe episode 0 as well as the end** (graph #40). "Solved" means "passes the stack at the frozen convention", nothing more.

## Graph ↔ repo: where what lives
|Concern|Repo|Graph|
|---|---|---|
|Code, configs, lockfiles|✓||
|Commands, conventions, stack|✓ (AGENTS.md)||
|Reality carriers|✓ (REALITY.md)||
|Gotchas||✓ (rules on contour `#1`; this file and code reference them)|
|Certification records, chronicle, dev docs|✓ (`docs/dev/`)|✓ (vimarshas link to them)|
|Research narrative, audits, paper basis|private notes repo||
|Branch state, what is in flight|git + PR (case number in the body)|✓ (`genre=hint` seed of the transformation)|
|Methodology, ontology||✓|
|Decisions||✓ (a node, at once)|
|Questions of substance, obligations||✓ (vimarshas)|
|Plans||✓ (the transformation map)|
|A one-off task, conversation, shift handover||✓ (case + the transformation's seed)|
|Commit history, PRs, SHAs|git|(never in the graph)|

**`HANDOVER.md` is not kept**: branch state has homes — `git branch`/`log`, the open PR and the case by the number in its body, modes of nodes, the transformation's seed; a handwritten file is the only one of them that lies silently.

## Commands
|task|command|
|---|---|
|gate|`make check` (ruff check + ruff format check + full pytest; what CI runs); `make check-fast` deselects the `slow` tests|
|lint / format|`make lint` / `make format` (`src/ tests/ scripts/`; `scripts/local/` is excluded)|
|typecheck (advisory)|`uv run pyright` (basic mode; not in the gate)|
|train|`uv run deqn-jax train <model> -n 1000` (`-o ngd -q` for a smoke; arm configs via `--config configs/<arm>.yaml`)|
|list models / optimizers|`uv run deqn-jax list` / `uv run deqn-jax optimizers`|
|IRFs from a checkpoint|`uv run deqn-jax irf <run>/checkpoint_003000.eqx --girf -o irf_out`|
|certificates|`JAX_ENABLE_X64=1 uv run python scripts/cert/disaster_ss_probe.py --runs-dir runs/disaster_cert --arms <a> --seeds 0,1,2`|
|config reference|`uv run python scripts/dev/gen_config_reference.py` (after any config-field change)|
|DGX sync|`rsync -az --exclude .venv --exclude .git --exclude scripts/local <worktree>/ anna@130.223.169.108:~/projects/<lane-dir>/` (one directory per lane; never the main checkout — graph #45)|
|DGX GPU sweep|`LAUNCHER=scripts/dgx/cert_sweep_container.py ./scripts/dgx/run_sweep_in_container.sh` (DONE-marker resumable)|
|docs deploy|`mkdocs gh-deploy --remote-name pages` (the maintainer deploys)|
|cold review (Astra)|`codex exec -m gpt-6-astra --sandbox read-only -c model_reasoning_effort=high "<brief: branch vs origin/master, what to attack, file:line + failing input>"` from the branch's worktree (the Codex companion plugin refuses this model, the CLI does not)|

Always `uv run`; never activate the venv manually. On the DGX, non-interactive shells need `export PATH=$HOME/.local/bin:$PATH` before `uv`. Compute lives on the DGX; the laptop runs ruff and short smokes only (graph #44).

## Project structure
Library under `src/deqn_jax/`, one line per component; `uv run deqn-jax list` names the registered models.
|component|what it holds|
|---|---|
|`api.py`|the stable public surface: load_model, TrainConfig, train, evaluate, IRFs, the checkpoint loader|
|`types.py`|ModelSpec, TrainState, Metrics — NamedTuple pytrees|
|`cli/`|one module per subcommand: train, models (list / info / optimizers / check), irf, evaluate, init_config|
|`config/`|Pydantic v2: TrainConfig and nested blocks; `io.py` derives `--set` dispatch from `model_fields` and tolerates removed fields in saved run configs|
|`models/`|registered models; each: variables, equations, dynamics, steady_state|
|`networks/`|factory + mlp / lstm / transformer / linear_plus_mlp / rss_net; the disaster network (π_BK + δ, `bk_pin`) is `models/disaster/network.py`|
|`optimizers/`|registry + standard / pcgrad / mao / lbfgs / gauss_newton (+ ngd, shampoo); `_step_common` shared by the step variants|
|`training/`|trainer, state_init (dispatch + validators), cycle, loss, composite_loss, coverage, episode, shocks, linearize, warm_start, checkpointing, metrics|
|`evaluate/`|simulate, diagnostics, dynare, irf|
|`plots/`|figure helpers|
|`configs/`|arm configs; `configs/archive/` is ignored|
|`scripts/`|`cert/` (SS probe, stress table, risky SS, GN polish), `dgx/` (container sweeps), `dev/` (plots, config reference, module graph); `scripts/local/` is ignored scratch|
|`tests/`|smoke convention: 3 episodes, hidden=(16,), batch=16; `conftest.py` enables x64 before any import|
|`docs/`|`dev/` — research record (cert report, chronicle, library review, `disaster/`, `audits/`); `site/` — the public docs; `docs/README.md` says which folder serves which reader|

## Code conventions
- **Meaning lives in the graph, code references it**: rationale, rejected alternatives, integration layout — a node; in code — `(graph @aleph/deqn, node #N)`, for the rejected too ("not cached: #N"). Mechanics — in words on the spot. Having cited, check the node says it; diverged — fix the node. Readers of this public code include people without graph access: keep references to places where *why* is otherwise unrecoverable, and keep the dev docs self-standing.
- **Docstrings are claims about the code, and they move with it.** A change that removes, renames or alters behaviour updates, in the same commit, every docstring, comment, Pydantic field description and doc page naming it: grep the old name across `src/`, `docs/`, `README.md` before committing. Never repeat a docstring to the maintainer as fact without checking the code. A docstring enumerating a registry points to the CLI command instead, or is guarded by a test (graph #59).
- Config precedence: `--set` overrides > CLI args > YAML file > defaults (`load_config()` in `config/io.py`; dot-notation reaches every nested block; lists cannot be set from `--set`). A hyperparameter "didn't take" — check the launcher for a `--set` before editing the YAML.
- Two JIT boundaries per cycle (rollout + grad-step sweep); everything runtime-variable resolves at construction time, before tracing (graph #42).
- Train-step variants are dispatched by `OptimizerKind`, which `optimizers.registry.create_optimizer` reads from the registry — never a name list; when adding a loss feature, extend `state_init._validate_train_config` so combinations that would silently drop it from the gradient are rejected, not ignored.
- Loss-dict keys prefixed `aux_` are excluded from reweighting and gradient surgery by contract (`eq_losses_to_array`).
- Equinox pattern: `eqx.filter(model, eqx.is_array)` → update → `eqx.combine(arrays, model)`.
- **Test discipline**: unit + regression; no coverage threshold. Bit-identical guards for refactors (same seed, same loss history, same parameter hash; the step-variant guard compares exactly on its recording platform and skips elsewhere); certification claims additionally need the probe stack on real checkpoints (`REALITY.md`). Precision is set once, in `tests/conftest.py`, before any test module is imported: never toggle x64 in a test module. `test_convergence.py::TestDisasterTraining::test_loss_decreases` is platform-sensitive — `xfail(strict=False)`; the module is `slow`. Tests are local to the change: a component has its own module and tests.
- **Public-tree rules from the maintainer**: no private paths in the tree; the unpublished paper is not cited in tracked files; never the maintainer's preferred name in tracked docs — "the maintainer"; git identity for private/work repos is the official name (see the private notes repo).
- **Harness surface**: `.claude/hooks/ruff_on_edit.py` runs after every edit and strips unused imports — an import added before its use is removed; re-add it.
- **Gotchas do not live here**: they are rules on contour `#1` of the graph (`iskron_look(realm="@aleph/deqn", node_id="#1")`, the INSIDE list: #40–#47); here and in code only the reference.

## What to update when
- `AGENTS.md` — **what can be learned from a graph node is not written here**; it holds what is needed before the agent reaches the graph (commands, entry into orientation, invariants outside the linter, forks that must stop you). Clearing prose is a move into carrying nodes, not a deletion.
- `REALITY.md` — a claim class's carrier appeared, changed or turned out unreachable; dated measurements go to the graph or the cert report.
- `docs/dev/disaster_cert_report_2026_07_07.md` — any certification claim.
- `docs/dev/selection_program_chronicle_2026_07.md` — program-level shifts (new results, retractions, method lessons).
- `docs/site/config_reference.md` — regenerate after any config-field change (*Commands*).
- Private notes repo — research narrative not for the public tree.
- The project graph — every merge (*Session lifecycle*).

## Git workflow
- **Conventional commits** (`feat:`/`fix:`/`chore:`/`refactor:`/`docs:`/`test:`); bland messages; no session-metadata lines; branches `feat/…`, `fix/…`, `chore/…`, `docs/…`; PR titles in the same format.
- **PR body — one or two lines**: what changed, the case number. Course of work stays in the case.
- **Trailer**: Claude agents append `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` (the maintainer's standing choice).
- **Gate — one call**: `make check`; never assemble its steps by hand. CI calls the same target on PRs and on `master`. No pre-commit hook: run the gate before pushing (the ruff hook on every edit is a convenience, not the gate).
- **A batch's branch does not live without a PR**: pushed — open the PR in the same move (draft if the batch is not assembled).
- **Forge**: GitHub, CLI `gh` (account `mechanicpanic`); watch with `gh pr checks <n> --watch`. GitHub refuses approvals from the PR's own account: agents' reviews go as comments, approval is the maintainer's click.
- **Definition of done**: a PR to `master` with CI green, squash-merged by the maintainer (or a small fix pushed to `master` directly); for solver changes, certificates unregressed at the frozen convention; docs updated per *What to update when*; the graph updated on the merge.
- **Never** `--no-verify`, `--force`, `--no-gpg-sign`, `git reset --hard` without explicit instruction.
