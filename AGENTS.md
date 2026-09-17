# deqn-jax
Deep Equilibrium Network trainer for economic models (JAX/Equinox port of
Azinovic–Maliar–Maliar) — for economists solving dynamic stochastic models and
for the research program certifying those solutions.

## Cover
Answers stand as a table, not prose: slot, value, source. Source is `derived`
(from the repo, by script or reading), `agreed: <who>` (the person or agent whose
word it is) or `<не согласовано — #N>` (the consent node). The cover is written by
beat 1 of `iskron:iskronify`, completed by the Start of the `iskron:iskron` door from
the consent node, re-projected by the full arc.

| Slot | Value | Source |
|---|---|---|
| Nature | research. Relaxations, per tool: pyright stays `basic` and advisory (78 errors on 2026-09-15; strict mode drowns in JAX/Equinox partial stubs; not in the gate); no coverage threshold; ruff strict tightening (`B,N,UP,RUF`) measured at 1,949 errors on 2026-09-15 and deferred to a dedicated lane with two documented exemptions (economics variable names, Greek in docstrings). Everything else: production discipline. | agreed: owner (2026-09-03), numbers derived 2026-09-15 |
| Graph | `@aleph/deqn` (`r71`) — every session starts here | agreed: owner |
| Focus holon | `#1 «🧮 deqn-jax solver»` | derived |
| Repository | `github.com/deqn-jax/deqn-jax` — holon attribute `repository`, from origin | derived |
| Agent role | `#9 «🤖 Coding agent»` — adhikarin, stewards `#6 Training contour` and `#7 Certification contour`; inbox `iskron_orient(realm="@aleph/deqn", focus="#9")` | derived |
| Owner role | `#8 «👤 Maintainer-researcher»` — svatantra 主; `posed_to` address for anything outside the mandate, by seq, never the `me` sentinel | derived |
| Stack | Python ≥3.10; JAX ≥0.4.20 + Equinox ≥0.11 + Optax ≥0.2; Pydantic v2 configs; uv. No TensorFlow, no PyTorch, no Keras. | derived |
| Gate | `make check` — ruff check + ruff format check + the full pytest run, one call, the same call CI makes; `make check-fast` skips the `slow` tests locally | `<не согласовано — #48>` |
| Consumers | potential collaborators and course students reading the public repo and the docs site; the research program consuming the certificates. Breakage surfaces as wrong research conclusions, caught by the certificate stack or by a reader, not by a crash. | agreed: owner (2026-09-03, production statement) |
| Cost of breakage | public research code: wrong conclusions and lost collaborator trust — worse than a crash. Solver claims are gated by certificates, never by green tests alone. | agreed: owner (2026-09-03) |
| Reality | table in the section *Reality* | agreed: owner (2026-09-03); rows added by use since |
| Cross-project memory | personal graph: none — the owner declined cross-project memory on 2026-09-03; a fact about the person that no project owns goes to the private notes repo. Global preferences file: no. | agreed: owner (2026-09-03); re-asked on #48 |
| Feedback reflection | yes | agreed: owner (2026-09-03) |
| Workflow-suite interop | full (superpowers 6.3.0) | agreed: owner (2026-09-03) |
| Consent (Согласование) | `#48` — open slots: gate, personal graph | derived |

## Persistence rules
State lives in the **repo** or in the **graph** — nowhere else. The harness's
built-in memory (whatever it calls it: the per-project memory directory,
conversation summaries, `/tmp`, machine-local files) is **forbidden entirely, not
by category**: nothing goes there — no project fact, no user preference, no
working-style note. The one file in the session's temporary directory is the
session ledger (Start of the `iskron:iskron` door): it dies with the session and
moves neither to the graph nor to memory — that is not storage. (why: local memory
is invisible to every other agent and machine, so it drifts silently and breaks the
reproducibility that makes a second machine or a second agent possible.)
- **Repo**: code, configs, rituals (how to act here), branch state — the artifact
  itself.
- **Graph**: methodology, project decisions, open questions (vimarshas), plans,
  handoffs, lessons, gotchas, hints — the thinking around the work. Do not retell
  graph content in the repo; link the vimarsha or the contour.
- **Private notes repo** (`~/Projects/research/deqn-research-notes`, GitHub
  `mechanicpanic/deqn-research-notes`): research narrative, audits, paper basis,
  the maintainer's own facts — anything not for the public tree, and the
  maintainer's user-scoped facts while no personal graph exists.
- **Route "remember / think about / learn":** how to act in this repository
  (ritual, command, order) → this file; a fact about the person → the private
  notes repo (a personal graph, when the owner wants one — consent node #48);
  knowledge about the project, its meanings, ideas and gotchas → the graph, never
  this file.
- **Fetch state; never reconstruct from memory.** No source for "we decided…"?
  Stop and read the graph or the repo before acting.
- **External design/spec files are drafts for intake**, not the record: the graph
  holds decisions; such a file is their view.
- **Parking spots are named, and there are three.** An undisciplined surface is
  free, a disciplined one costs — and under pressure laziness finds the parking:
  prose in this file; a sinn-phenomenon for something that acts; a lone `context`
  arrow that silences a detector. Finish line of a record: **a node is not recorded
  until what pulls it is named** — which action breaks if the node vanishes? None
  — that is parking, not a record. And the fork is read BEFORE writing, not from
  the warning after (it chronicles, it does not stop): a door enters the graph not
  as a thing and not as the meaning "door", but as **the place where a doer acts**
  — `upadhi` to the action that goes through it.
- **This overrides the harness's own memory instruction**, which invites a
  `project` category and will keep inviting — the pull is strongest exactly when
  something seems worth saving and this file is long out of context. Route instead,
  always — and **before finishing, check that every durable fact from the user's
  context is persisted by this routing: an unpersisted fact is a failed task, not a
  nicety** — asking **whose fact is this?**
  Repo convention, code fact, this project's procedures, **its servers and deploy
  pipeline, its dated duties** → this file / docs / code, or a node in this repo's
  graph; work state, decision, open question → vimarsha in the graph.
  A dated duty (renewal, deadline) is a node carrying the date in `attrs`: a
  README mention loses it. Rules of this project never go into another project's
  or the user's graph — not even as a mention on a machine's card.
  Standing preferences ("how to work with me") are instructions, not facts:
  project-scoped ones live in this file; never in the graph or a memory directory.
- The local memory directory is **evacuated and frozen**: `MEMORY.md` holds a
  one-line prohibition stub pointing here, and the memory-guard `PreToolUse` hook
  in `.claude/settings.json` blocks any write there (exit 2) at the moment the
  saving instinct fires. What it held is preserved in the private notes repo
  (`dev/agent_memory_evacuation_2026_09_03.md`).

## Session lifecycle
Graph = the work (structure, open questions, what next). Git = how we got here (SHA,
branches, PRs). **Git references never enter the graph** — no SHAs, branch names, PR
numbers or "shipped/merged" in nodes (they rot on rebase).
- **Session start:** the Start section of the `iskron:iskron` door skill — it ends
  in readiness, not in reading: the graph and the role are named, standing is taken
  by one `iskron_stand` call **only on watch** (the word "вахта", `start`, an
  invitation line, a frame), the greeting arrived; the role's queue and the maps are
  read when there is a reason, not at start. This file holds only addresses: graph,
  focus holon, agent role and owner role from the cover. The address for agreement
  with the owner is the seq of his role, not the `me` sentinel: a person holding
  several roles makes `me` ambiguous. Then `git status`, and check running DGX jobs
  (`logs/` on `anna@130.223.169.108:~/projects/deqn-jax`) before launching new work.
- **Starting work: graph first, then project, then code.** A substantive task
  enters in three beats: (1) **graph reconnaissance** — what is recorded about the
  place of the change, which vimarshas are open, what was decided and rejected,
  **and what is recorded about the external surfaces the work will touch** (see
  *External surfaces*: a recorded observation is older than your memory of someone
  else's API); `iskron:entry` drives it; (2) **integration field** — from the focus
  holon, the steward role and the nodes of the change, derive consumers, effects,
  neighbouring contours and their doers through `iskron:integrity`; walk relays
  with `lens="trace"`, design missing actions and phenomena, weave gaps with
  `iskron:weaving`; (3) **design** the change (`iskron:design`) — only then code.
  Nobody lists "shared surfaces" for you: such prose rots during the work itself;
  the structure is derived from the graph anew each time. Skipping gets dearer
  left to right: code without reconnaissance fixes what was already decided, argues
  with what is recorded and re-walks recorded dead ends. The one exception is
  **explicit**: the user said "work directly" or named another protocol — then go
  to code and pay the reconnaissance debt at the reconcile beat. Silence is not
  "work directly".
- **A decision is recorded when it is made, not when it is executed.** Wherever it
  arrives — the user said it in chat, a person's word came over the socket, two
  agents agreed — it stands in the graph **immediately**, with the modes it really
  has now (epistemic no higher than `anumita`, ontic `anagata`, volitive
  `chanda`/`adhimoksha`). Modes move later, as the thing is built; the record does
  not wait. Record who decided and what counts as execution. A changed situation
  is reflected just as immediately: a graph behind what is known lies to every next
  reader. (why: a decision left in the conversation that carried it dies with that
  conversation; a late record is the same failure with a delay — what you remember
  as decided is not what was decided.)
- **Every task is described before it is begun, and recorded as what it is.**
  Before the first change outside the graph, the work stands in the graph as its
  carrier: a one-off deed as an `anga` vimarsha on the transformation it moves (a
  large one as its own bianhua) with its before/after in the body; **an action
  (kriya) only for a repeatable transition** whose every run eats the same `ahara`
  and produces the same `utpatti` (ritual, pipeline step, procedure). The one-glance
  test: ask the "action" what it will eat and produce on its *next* run — no answer,
  it is a task. A task recorded as an action lies by type, and the lie cascades: its
  "result" degenerates into a state-label phenomenon no action produces — an orphan
  by construction. While the work runs, the graph moves with it. On **merge** (not
  push — a branch that merged nothing shipped nothing) modes switch to what the
  merge made true. Then `iskron:reality-audit`: check the claim against the deployed
  artifact, not the diff, and only then let the node say it. A mode switched before
  the evidence is an unverified claim that reads exactly like a verified one.
- **Every merge → update the graph.** A push that only opened or updated a PR
  shipped nothing: record the answer where it stands and keep bodies and modes
  describing what trunk really carries. The post-merge sequence (check master,
  rebase the next branch from origin/master, remove the merged one, weave the
  merged state into the graph) hangs on the **event** of the merge — never on a
  lull: "when nothing is in flight" never comes for a busy agent, and its mechanics
  are delegable to a sub-agent, so live work need not be interrupted. When merged,
  each move below is mandatory — except for work assigned by reference from another
  agent (a brief by references to a designed area, a report as a ledger frame): there
  weaving, closing along the axis and reconciling go back to the assigner, and the
  doer keeps the transformation's seed, the delivery modes and the rest below
  (`iskron:vahta`):
  - **Reconcile with reality.** Record what positions the change in the target
    system: architecture, module APIs, delivery, user experience, integration with
    neighbouring code. Pure repo mechanics — lockfile noise, internal refactors
    without external effect, file moves — stay in git, not the graph. **Updating the
    graph means weaving, not editing prose:** a paragraph about your work swelling
    inside someone else's description is a smell, and almost always an action or
    phenomenon you did not create and an edge you did not draw. Zero nodes and zero
    arrows after a substantive wave is an unperformed step: say so plainly if there
    was truly nothing, and why.
  - **Advance the map.** Keep open work attached `anga` to the transformation it
    moves. A `genre=hint` seed — one per transformation, not a log: only what
    matters after the session, nothing the graph already shows; the session ledger
    lives in the session file and in frames.
  - **Close along the axis, not by feeling "done".** Record the answer as
    `addressed_by` to the node that carries it — it raises confidence but does not
    end the question. Release (`visarjana`) is a separate volitive act, and what
    precedes it depends on the question: a distinction is answered by its form, a
    behavioral claim needs an observation on its carrier (*Reality*). Release
    yourself when three things coincide: the answer stands in the graph as a node,
    not in your recollection; the repo shows it; reality shows it as far as
    reachable — where unreachable, the user's word stands instead, and you asked
    for it. Otherwise prepare the release and present it to the owner. Release is
    not the only end: park, displace or crystallize what the question taught.
  - **Sweep the shipped contour.** A push that realizes designed nodes switches
    their modes (anagata→vartamana, kalpita→pratyakshita) across the *whole*
    designed contour — not only the touched nodes — and ends the design vimarshas
    the shipment resolved, by the rule above.
  - **Work the inbox.** `posed_to` questions the work answered end by the rule
    above; stale ones are parked or grouped.
  - **Reconcile code and graph.** End of every substantive task: the tidy-up beat
    of `iskron:reconcile` — area nodes against the code (ontics, names, honest
    modes; a task does not pose as an action), code against the graph (a comment
    carrying meaning goes to the graph, the code keeps a reference), the three paths
    — discarded and rejected alternatives recorded and referenceable. Remaining
    debts as vimarshas, not narrative.
  - **Feedback reflection** — on merge, and at the close of a session no merge
    crowned: examine the session's experience *of the method itself* — where a
    skill, rule or surface failed, surprised, or worked for the wrong reason; check
    for what is already said and record only a case worth recording, **at its
    address**: in this system's work graph, anchored to the tool's node or contour
    and addressed to its steward (`posed_to="steward"`); method-general ones to the
    owner role; there is no shared feedback graph; driven by `iskron:feedback`
    (genre, quote of the rule, mechanism, closing criterion). **An empty reflection
    is a valid outcome: zero records beat an opinion**; invent no findings.
  - **Vocabulary pass.** Re-read what you are about to land — repo text and graph
    nodes — for borrowed project-management words (ticket, backlog, sprint, epic,
    story, done, blocker, committed). Do **not** substitute: name each to the user
    and, in the same move, ask what it is called in this project. (why: renaming is
    the owner's act, and a confidently wrong replacement is worse than the displaced
    word: it reads as native and nobody questions it again.)
  - **Certification record.** If the merge touched solver behaviour or a
    certification claim, the cert report
    (`docs/dev/disaster_cert_report_2026_07_07.md`) or the chronicle moved in the
    same push; superseded claims are kept and marked, never deleted.

  `iskron:weaving` / `iskron:design` carry the *how* (ending vimarshas, wiring the
  contour).
- **Design completeness criterion:** a design is not *ready* until its decisions,
  risks and lifecycle are in the graph — whichever skill elicited it. Saving to the
  graph is memory work, not implementation: design-phase gates on implementation
  do not apply to it. A design/spec file written by another suite is a draft view:
  intake it **in the same session** (never defer landing in the graph to a future
  push). Working autonomously (owner absent): land decisions and risks now; propose
  a transformation with a telos marked for owner confirmation, do not skip it.
- **Execution suites lead execution.** Planning, TDD, debugging, verification,
  review and their kin belong to the installed execution suite; the graph carries
  only the memory/design plane. Decisions and risks born in execution still land in
  the graph **before the session ends** — never gated on a future push/commit.
- **A claim you made is not a claim you accept.** Behavioral claims — "the fix
  works", "the probe passes", "the suite is green" — close on the verdict of the
  cold `verifier` sub-agent, never on your own re-reading. Give it the claim, the
  carrier and the falsifier from *Reality* — and **wait for the verdict** before
  ending anything by the rule above. (why: you see your own change as intended, not
  as it is.)
- **Hook merging.** Where the harness has a hooks file, entries of different suites
  coexist — add beside, never overwrite others'.
- These reminders are automated in Claude Code (`.claude/settings.json`): the
  session-start hook, the hook after `git push` / `gh pr create|merge`, the
  memory-guard hook, the anti-freeze `Stop` hook, the writing-moment hook before
  graph writes, and — since the interop stamp below says `full` — the spec-write
  reminder. Codex reads this file natively and carries no hooks: for Codex the
  rituals bind through this prose only.
- **Keep this file honest.** It is generated by `iskron:iskronify` and stamped at the
  bottom with the contract it came from. Propose a rerun when the installed skill
  announces a newer contract — or when the sources this file is derived from moved
  after the stamp: `git log -1 --format=%cd -- pyproject.toml .github/workflows
  configs src/deqn_jax Makefile` against the date decides in one command. **The
  check costs no call:** the installed contract number is the first word of the
  `iskron:iskronify` skill description, and skill descriptions sit in every
  session's context — compare it with the number in the stamp below. A mismatch
  makes proposing the rerun the session's first move (launch on the word of the
  person or the room; silence: propose again at the next wake-up). (why: a stale
  AGENTS.md is read with full confidence every session and misleads more than no
  file.)
- **Keep your toolchain fresh.** Updates are on by default: take them as the
  channel delivers, do not pin. (why: a stale skill drifts from the tool surface it
  names and degrades you silently.)

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
by the top-tier sub-agent** (role `reviewer`, `.claude/agents/reviewer.md`) — **in a
separate worktree** (the Agent tool's `isolation: "worktree"`): the working copy is
one per machine, and the ban in the role's brief is the second line, not the first.
The standing alternative for the skeleton lanes is Astra through `codex exec` (see
*Commands*); its findings and dispositions go on the PR as a comment. Where the
harness cannot, say plainly that there was no cold review. **Only a push has a
watchman.** A stage closed without a push is held by you alone — an acknowledged gap.

The reviewer's field is four things, all mandatory: the **branch diff against trunk**
(the whole branch); **the repository itself**; **the focus holon and its steward
role** (`#1`, `#8`/`#9`); **references to the graph nodes entering the diff** — the
leading vimarsha and every action, phenomenon and rule the branch changes — never a
retelling. The reviewer runs `iskron:integrity` read-only and returns findings plus an
**integration report**. `NEEDS_CONTEXT` names a graph gap: fix the graph (design,
weave, wake the neighbour through `iskron:standing`), then repeat. Fix what came back
in the same branch; reject a finding with a recorded "why" (in the PR or on the node).

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
  mid-implementation still land in the graph before the session ends — never
  deferred to a future push.

*(interop: full — verified against superpowers@6.3.0 — re-check on suite upgrade)*

## Working principles
1. **Think before coding.** Name assumptions; ask when unsure — naming *what* is
   unclear, not only "which option". **Questions to a human are asked in text** — in
   the conversation or a channel; the interactive option-menu tool is never used: a
   list of options replaces the question with an answer, imposes the agent's frame
   and hides what is actually unclear. Raise competing readings; push back when a
   simpler move or a false premise is visible. Check repo + graph before writing;
   fetch, don't recall. Touch the live system before trusting a type, a name, a doc.
   Questions beyond the boundary or mandate become `posed_to` vimarshas to the owner
   role `#8` — not silent decisions and not chat-only questions.
2. **Simplicity first.** Minimum code for the task. No speculative features, no
   abstractions for single-use code, no handling of impossible errors. Validate at
   boundaries; trust internal invariants. 200 lines that could be 50 → rewrite.
3. **Stay inside the repo boundary.** Never leave this repository's working directory.
   A change belonging to another contour (the private notes repo, the DGX record,
   someone else's repo) is not yours across the boundary: record it as a vimarsha on
   that contour's node, anchored where its owner orients, `anga` to the
   transformation it serves.
4. **A second implementation is an event to report.** About to write what already
   exists — the same helper for a second consumer, the same rule in a second module?
   First derive both places through the integration field (`iskron:integrity`), then
   name them to the user and propose reunion or a named, deliberate fork. A new
   consumer stands as edges to actions and phenomena in the same move it appears in
   code. (why: the LinearPlusMLP / KfAnchoredMLP fork cost a full sweep to diagnose;
   the disaster network still copies LinearPlusMLP — graph #39.)
5. **Surgical changes.** Touch only what the task needs. Don't reformat or refactor
   neighbouring code; the linter is authoritative. Delete only dead code your change
   created; flag the rest.
6. **Goal-driven execution.** Tasks → verifiable goals. Bugs: pin with a failing test
   before patching. Multi-step work: `step → verify` pairs. Solver changes: verify
   with the certification stack on real checkpoints on the DGX, never with unit tests
   alone — *Reality* names the carriers and who reaches them. Name the falsifier
   before looking ("which observation would refute this?") and observe the carrier,
   not the source that should have produced it. Ending the questions your change
   touched goes by *Session lifecycle* — along the axis, not by feeling.
7. **Read before answering an open question.** Tasks framed *discuss / think through /
   investigate / design / plan / analyse / "what do you think"* — everything beyond
   "do X concretely" — are answered from recorded thinking, not training data: query
   the graph first, several ways (one miss ≠ absence); `iskron:entry` drives the
   protocol and finds the graph with the answer.
8. **Think in the graph, speak the project's language.** The graph's structural
   vocabulary (kriya, phenomenon, contour, role, vimarsha, the three mode axes) is for
   reasoning; it never appears in what you say to the user until the user uses it
   first. Translate into the project's own words: arm, certificate, probe, checkpoint,
   pin, anchor, coverage, the frozen convention. Talk *about* work in plain
   description — the question, the change, what is open, what it resolves — never
   ticket, task, sprint, backlog, story, done. (why: a borrowed word brings the script
   of its method with it.)

## Integration field — from the graph only
The focus holon `#1` and its steward roles are the only permanent root of the
walk. A list of shared surfaces, consumers and dependencies is **not kept in
AGENTS.md and not asked of the human**: such prose goes stale in the course of the
work itself and cancels the real walk with a false sense of completeness. The ban
is about our own, which the graph already models. What it does not and cannot
model — *External surfaces* below, and that section is kept.

For every change, name the graph nodes whose realization enters the diff and run
`iskron:integrity`. Trace a phenomenon both ways with `iskron_orient(lens="trace")`;
for an action walk its `next` thread and the relays of its `ahara`/`utpatti`/
`upadhi`; an exit into another holon leads to its steward role. That is how
consumers, effects, open vimarshas and the doers a change touches are found.

A dependency the walk did not find is not a reason to write a list: it is a model
defect. Design the missing actions and phenomena, weave the unwired edges through
`iskron:weaving`, pose a vimarsha for a pending foreign decision and wake its
addressee through `iskron:standing`. A new consumer counts as recorded only when
the code and the corresponding edges appeared in one move.

## External surfaces — what you use and do not own
JAX/Equinox/Optax APIs, Dynare outputs, the reference implementations (the EWM
notebook, the RSS TensorFlow checkpoint and source), the DGX container image, the
Codex CLI. The agent **guesses** these from training memory, and memory is
indistinguishable from knowledge from the inside. The price is not ignorance but
confidence: a field that does not exist looks in code exactly like one that does,
and diverges not at build time but on the live call.
- **Before the work, record the part of the surface the work will touch** — not all
  of it: what you touch. As a graph node, with the version you looked at: the version
  is part of the surface's identity, not a footnote.
- **Sources by seniority — perception before testimony.** Observation with your own
  hands (a call, `--help` of the installed binary, the installed package's types, a
  response you saw) outranks documentation; documentation outranks memory; **memory
  is not a source at all** — what is written from it is a guess dressed as fact.
  Write the epistemics honestly: `pratyakshita` only for what you observed,
  `anumita` for what you derived from docs, and never raise it because "that is how
  it usually is".
- **Weave the link.** The surface node is `upadhi` to the action that acts through
  it (or `ahara`/`utpatti` if it takes or gives data). Without the edge it is an
  orphan label: neither the walk nor the next agent will find it.
- **Keep in step.** Found a discrepancy or the vendor bumped a version — fix the node
  in the same move you discovered it, and lower the epistemics if you did not
  observe the new state. A silently diverging node is worse than a missing one:
  people act on it.
- **The reference works both ways, and the second way matters more here.** Source
  that works with an external surface carries `(graph @aleph/deqn, node #N)` — and
  you **read that node before working**. Here the reference is not a footnote for a
  successor: it is your own first move against guessing.

## Reality — what a claim is checked against
| Claim class | Canonical carrier | How to observe | Who can |
|---|---|---|---|
| "model X is solved / certified" | the **final** checkpoint on the DGX, `runs/<arm>_s<seed>/checkpoint_003000.eqx`, three seeds, fp64 | `JAX_ENABLE_X64=1 uv run python scripts/cert/disaster_ss_probe.py --runs-dir runs/disaster_cert --arms <arm> --seeds 0,1,2` on the DGX host; stress grid via `scripts/cert/ewm_stress_table.py`; report the learned-block ρ, ŝ, residuals at ŝ | agent (ssh) |
| "the code is correct / tests pass" | the suite run on the DGX host, not the laptop | `ssh anna@130.223.169.108 'export PATH=$HOME/.local/bin:$PATH; cd ~/projects/<lane-dir> && make check-fast'` | agent |
| "CI is green" | the GitHub Actions run for the PR/commit | `gh pr checks <n> --watch` / `gh run list --branch <b>` | agent |
| "a training recipe behaves" | the run directory on the DGX (checkpoints, `DONE` marker, config, TensorBoard) | `ls runs/<arm>_s<seed>/`, `logs/cert_container*.log` | agent (launch: `scripts/dgx/run_sweep_in_container.sh` in the NGC container) |
| "a refactor changed no numbers" | same-seed short training on master and on the branch, and the bit-identical step guard | `uv run deqn-jax train <model> -n 3 --seed 0` on both trees, compare the final loss; `tests/test_step_common_guard.py` compares exactly on its recording platform (x64 on, per `tests/conftest.py`) | agent |
| "a certified checkpoint still loads" | the certified checkpoints in the DGX record, through the current loader | the probe command above against `../deqn-jax/runs/disaster_cert` from a lane directory; SS error 0, drift@100 0.051%, ρ at the 0.987 floor is the July record | agent (ssh) |
| "the docs are live" | https://deqn-jax.github.io | `curl -sI` the page; deploy is `mkdocs gh-deploy --remote-name pages` | agent observes; maintainer deploys |
| "a config change took" | the resolved `TrainConfig` printed at run start / `config.yaml` in the run dir | `uv run deqn-jax train <model> --config <yaml> -n 1 -q` and read the resolved config | agent |

**Ceiling**: the disaster model's *true* equilibrium (no oracle exists — the whole
research program is about certificates in its absence; "solved" means "passes the
stack at the frozen convention", nothing more); Dynare comparisons (fixtures are not
in the repo — `dynare/` is gitignored, 18 tests skip); parity with external
references (the EWM numbers, the RSS checkpoint) is approximate by construction
and closes only by convergence of independent evidence; the user's private
repositories and machines.

**The table grows by use.** When a session teaches you a carrier the table lacks, an
observation that turned out reachable, or one that turned out unreachable (→
*Ceiling*), write the row *then*, in that session.

## Certification (what "solved" means here)
A small training loss is not a certificate — measured repeatedly (best-by-loss
checkpoints are certificate-worst; `save_best_checkpoint` defaults to True and
should not be trusted for claims). The stack, in order of strictness: held-out/stress
residuals → **learned-block** spectral radius (graph #43: the probe's raw ρ has a
floor at the exogenous root 0.98699 — always report the learned 8×8 block) → solved
fixed point ŝ = T(ŝ): ‖ŝ−s\*‖ and ρ(ŝ) → per-equation residuals AT ŝ → long-horizon
convergence, multi-seed. Frozen convention: **final checkpoints**
(`checkpoint_003000.eqx`), fp64. Under `bk_pin` the SS-error and ρ(s\*) legs are
donated by construction; the earned legs are the stress grid and the residuals at ŝ
— say which is which. **Probe episode 0 as well as the end**: certify what a run
starts from (graph #40, the warm-start finding of 2026-09-02).

## Graph ↔ repo: where what lives
| Concern | Repo | Graph |
|---|---|---|
| Code, configs, lockfiles | ✓ | |
| Commands, conventions, stack | ✓ (AGENTS.md) | |
| Gotchas, lessons, decisions, open questions | (reference only) | ✓ (rules on the contour; vimarshas) |
| Certification records, chronicle, dev docs | ✓ (`docs/dev/`) | ✓ (vimarshas link to them) |
| Research narrative, audits, paper basis | private notes repo | |
| Branch state, what is in flight | git + PR body | ✓ (`genre=hint` — one seed per transformation, only what matters after the session) |
| Methodology, ontology | | ✓ |
| Plans, session handoffs | | ✓ (project graph; the session ledger dies with the session) |
| Commit history, PRs, SHAs | git | (never in the graph) |

**`HANDOVER.md` is not kept — a decision, not an omission.** Branch state already has
homes, and a handwritten file is the only one that diverges silently: branch and
in-flight work — `git branch`/`log` and the open PR; how a claim is checked — the
*Reality* table; why it was decided and what is open — the graph; work under way —
the modes of its nodes and the session ledger; the PR body is assembled from that
ledger. Branch state is said **in the PR body**. Forge: GitHub, CLI `gh` (account
`mechanicpanic`); watch with `gh pr checks <n> --watch`. GitHub refuses approvals
from the PR's own account, so reviews by agents go as comments and approval is the
maintainer's click.

## Commands
| task | command |
|---|---|
| gate | `make check` (lint + format check + full pytest; what CI runs) — `make check-fast` skips the 9 `slow` tests (668 collected @2026-09-17; 21 skips when Dynare fixtures are absent (18) and for the steady-state legs the RSS replica has none for (3)) |
| lint / format | `make lint` / `make format` (`uv run ruff …` on `src/ tests/ scripts/`; `scripts/local/` is excluded) |
| typecheck (advisory) | `uv run pyright` (basic mode; dev group; not in the gate — see Nature) |
| train | `uv run deqn-jax train <model> -n 1000` (`-o ngd -q` for smoke; arm configs via `--config configs/<arm>.yaml`) |
| list models / optimizers | `uv run deqn-jax list` / `uv run deqn-jax optimizers` |
| IRFs from a checkpoint | `uv run deqn-jax irf <run>/checkpoint_003000.eqx --girf -o irf_out` (config auto-detected next to the checkpoint) |
| certificates | `JAX_ENABLE_X64=1 uv run python scripts/cert/disaster_ss_probe.py --runs-dir runs/disaster_cert --arms <a> --seeds 0,1,2` |
| DGX sync | `rsync -az --exclude .venv --exclude .git --exclude scripts/local <worktree>/ anna@130.223.169.108:~/projects/<lane-dir>/` (one directory per lane; never the main checkout — graph #45) |
| DGX GPU sweep | `LAUNCHER=scripts/dgx/cert_sweep_container.py ./scripts/dgx/run_sweep_in_container.sh` (DONE-marker resumable) |
| docs deploy | `mkdocs gh-deploy --remote-name pages` |
| cold review (Astra) | `codex exec -m gpt-6-astra --sandbox read-only -c model_reasoning_effort=high "<brief: branch vs origin/master, what to attack, file:line + failing input>"` from the branch's worktree; the Codex companion plugin refuses this model, the CLI does not. Findings and dispositions go on the PR as a comment |

Always `uv run`; never activate the venv manually. On the DGX, non-interactive shells
need `export PATH=$HOME/.local/bin:$PATH` before `uv`.

## Project structure
```
src/deqn_jax/
  api.py         # the stable public surface: load_model, TrainConfig, train, evaluate, IRFs, the checkpoint loader
  types.py       # ModelSpec, TrainState, Metrics — NamedTuple pytrees
  cli/           # one module per subcommand: train, models (list / info / optimizers / check), irf, evaluate, init_config
  config/        # Pydantic v2: TrainConfig + nested blocks (optimizer, network, composite_loss, coverage, replay_buffer, moment_matching); io.py derives --set dispatch from model_fields and tolerates removed fields in saved run configs
  models/        # 12 registered models (`deqn-jax list`); each: variables, equations, dynamics, steady_state
  networks/      # factory.py + common / mlp / lstm / transformer / linear_plus_mlp / rss_net; models/disaster/network.py (π_BK + δ, bk_pin)
  optimizers/    # registry + standard / pcgrad / mao / lbfgs / gauss_newton (+ ngd, shampoo); _step_common shared by the five step variants
  training/      # trainer, state_init (dispatch + validators), cycle, loss, composite_loss, coverage, episode, shocks, linearize, warm_start, checkpointing (+ the checkpoint loader), metrics
  evaluate/      # simulate, diagnostics, dynare, dynare_io, irf, cli
  plots/         # figure helpers (irf grid)
configs/         # arm configs (disaster_gated_pcgrad_bkpin.yaml etc.); configs/archive/ is gitignored
scripts/         # cert/ (SS probe, stress table, risky SS, GN polish), dgx/ (container sweeps), dev/ (plots, config reference, module graph); scripts/local/ is ignored scratch
tests/           # smoke convention: 3 episodes, hidden=(16,), batch=16; conftest enables x64 before any import
docs/dev/        # cert report, chronicle, library review (research state)
.claude/agents/  # reader / worker / verifier / reviewer role agents
Makefile         # the gate
```

## Code conventions
- **Meaning lives in the graph, code references it.** A comment carrying a decision's
  rationale, rejected alternatives or integration layout belongs in the graph; leave
  a reference `(graph @aleph/deqn, node #N)` in the code. Referenceable also for what
  was rejected: "not cached: #N" beats a paragraph. Mechanics of a step — in the
  comment; meaning, rationale, integration field — in the graph. The link works both
  ways: the node you cite must say what you cite it for — check, and fix the node in
  the same move if it does not. Readers of this public code include people without
  graph access: keep such references to the places where *why* is otherwise
  unrecoverable, and keep the dev docs self-standing.
- Config precedence: `--set` overrides > CLI args > YAML file > defaults
  (`load_config()` in `config/io.py`; dot-notation reaches every nested block, e.g.
  `--set coverage.enabled=true`; lists cannot be set from `--set`). When a
  hyperparameter "didn't take", check for a `--set` in the launcher before editing
  the YAML.
- Two JIT boundaries per cycle (rollout + grad-step sweep); everything
  runtime-variable resolves at construction time, before tracing (graph #42).
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
  seed, same loss history, same parameter hash; the step-variant guard compares
  exactly on its recording platform and skips elsewhere); certification claims
  additionally need the probe stack on real checkpoints (*Reality*). Precision is
  set once, in `tests/conftest.py`, before any test module is imported: never toggle
  x64 in a test module. `test_convergence.py::TestDisasterTraining::test_loss_decreases`
  is platform-sensitive (chaotic bare-MLP path, last-bit bifurcation) —
  `xfail(strict=False)`; the module is `slow`.
- **Standing rules from the maintainer**: be autonomous — push through to the next
  concrete artifact instead of asking after every plan step; never launch the
  multi-agent `/code-review` workflow or agent fan-outs on routine PRs — ask first,
  manual review is the default; the reference implementations may be read but never
  ported verbatim into the public tree, and no private paths appear in it; the
  unpublished paper is not cited in tracked files; never write the maintainer's
  preferred name in tracked docs — "the maintainer"; git identity for private/work
  repos is the official name (see the private notes repo).
- **Harness surface**: `.claude/hooks/ruff_on_edit.py` runs after every edit and
  strips unused imports — an import added before its use is removed; re-add it.
- **Gotchas do not live here**: runtime traps past types and the linter are rules on
  the repository contour of the graph; here and in code only the reference.
  | rule | graph |
  |---|---|
  | the constant-SS warm start never runs on an anchored network | #40 |
  | a checkpoint's template network is rebuilt from the full network config | #41 |
  | JAX tracing rules this codebase learned the hard way | #42 |
  | the probe's raw ρ floors at 0.98699; report the learned block | #43 |
  | compute lives on the DGX; the laptop runs smokes | #44 |
  | the DGX main checkout is the certification record; branches get lane directories | #45 |
  | short disaster smokes need an lr warm-up shorter than the run | #46 |
  | the evaluator's residual grades are single-draw and in mixed units | #47 |

## What to update when
- `AGENTS.md` — by the inverted default: **if it can be learned by reading a graph
  node, it is not here.** Only what is needed BEFORE the agent reaches the graph:
  commands, orientation addresses, code invariants the linter cannot express, forks
  that must stop you before acting — updated when THOSE change (commands, stack,
  conventions, reachability of a reality carrier). "The structure changed" is not a
  reason for a paragraph here. Clearing already-written prose is a reconcile beat
  with a move into carrying nodes, never a deletion.
- `docs/dev/disaster_cert_report_2026_07_07.md` — any certification claim.
- `docs/dev/selection_program_chronicle_2026_07.md` — program-level shifts (new
  results, retractions, method lessons).
- `docs/site/config_reference.md` — regenerate with `uv run python
  scripts/dev/gen_config_reference.py` after any config-field change.
- Private notes repo — research narrative not for the public tree.
- The project graph `@aleph/deqn` — every merge (see *Session lifecycle*).

## Git workflow
- Conventional commits (`feat:`/`fix:`/`chore:`/`refactor:`/`docs:`/`test:`); bland
  messages; no session-metadata lines. Claude agents append the trailer
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` (maintainer's standing
  choice). Branches `feat/…`, `fix/…`, `chore/…`, `docs/…`; PR titles in the same
  format.
- **Local gate — one call, not a list**: `make check` runs the whole chain; call it by
  name, never assemble the steps by hand. CI calls the same target. The ruff hook on
  every edit is a convenience, not the gate; run the gate before pushing.
- **Definition of done**: a PR to `master` with CI green (`gh pr checks <n> --watch`),
  squash-merged by the maintainer (or a small fix pushed to `master` directly); for
  solver changes, certificates unregressed at the frozen convention; docs updated per
  *What to update when*; the graph updated on merge.
- Coding agents (Claude, Codex) work in git worktrees, never the main checkout; one PR
  per delivery step.
- **Never** `--no-verify`, `--force`, `--no-gpg-sign`, or `git reset --hard` without
  explicit user instruction. Pushing `master` is the maintainer's call unless
  explicitly delegated.

*(iskronify: контракт `10`, штамп `2026-09-17` — propose a rerun when the description
of the installed iskronify names a higher contract or when the sources this file is
derived from moved after this date.)*
