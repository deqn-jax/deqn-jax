# deqn-jax
Deep Equilibrium Network trainer for economic models (JAX/Equinox port of
Azinovic–Maliar–Maliar) — for economists solving dynamic stochastic models and
for the research program certifying those solutions.

## What this project is
- **Nature**: research. Relaxations (explicit, per-tool): pyright stays
  `basic` (authored rationale in `pyproject.toml` — strict mode drowns in
  JAX/Equinox partial stubs); no coverage threshold; ruff strict-tightening
  (`B,N,UP,RUF`) measured at 1,652 errors on 2026-07-12 and deliberately
  deferred to a dedicated branch. Everything else: production discipline.
- **NKS realm**: `deqn` — orient at session start
  (`iskron_orient(realm="deqn")` (iskron MCP, formerly nks)); focus holon `#1 «🧮 deqn-jax solver»`. Open
  research questions live there as vimarshas; close (visarjana) the ones your
  push resolves. Repo remains canonical for conventions (this file),
  certification records (cert report), research state
  (`docs/dev/selection_program_chronicle_2026_07.md`, §15/§16), and narrative
  (private notes repo `mechanicpanic/deqn-research-notes`). When no iskron (NKS)
  server is connected (headless runs), the chronicle is the fallback
  orientation.
- **Stack**: Python ≥3.10, JAX + Equinox + Optax, Pydantic v2 configs, uv.
  No TensorFlow, no PyTorch, no Keras.
- **Production statement**: public research code
  (github.com/deqn-jax/deqn-jax; docs at deqn-jax.github.io), read by
  potential collaborators and course students. Cost of breakage: wrong
  research conclusions and lost collaborator trust — worse than a crash.
  Solver claims are gated by certificates (see *Certification*), never by
  green tests alone.

## Persistence rules
State lives in the **repo** or the **private notes repo** — never only in
agent memory, conversation summaries, or `/tmp`.
- **Repo**: code, configs, conventions, gotchas, dev docs, certification
  records.
- **Private notes repo** (`~/Projects/research/deqn-research-notes`, GitHub
  `mechanicpanic/deqn-research-notes`): research narrative, audits, paper
  basis — anything not for the public tree.
- **Claude-side project memory is a recall cache**, not a home: anything
  load-bearing must be mirrored into one of the two repos the same day.
- **Fetch state; never reconstruct from recall.** No source for a "we
  decided…"? Read the chronicle or the dev docs before acting.

## Session lifecycle
- **Start**: read this file; `iskron_orient(realm="deqn")` (iskron MCP, formerly nks) when the server is
  connected (chronicle §15/§16 otherwise). Check `git status` and running
  DGX jobs (`logs/` on `anna@130.223.169.108:~/projects/deqn-jax`) before
  launching new work.
- **Every push**: if the change touches solver behavior or certification
  claims, update the matching dev doc (cert report for certificates,
  chronicle for program-level shifts) in the same push; close the NKS
  vimarshas the push resolved.
- **After a green push — self-review**: re-read your diff for bugs, fragile
  spots, weak error handling, DRY violations, missing or useless tests,
  god-units mixing concerns. Fix in the same branch and push again, or state
  plainly that nothing surfaced. Don't fake findings.
- **Branch discipline**: one branch through to its merge; small fixes commit
  to `master` directly (current practice). After merge: `git checkout master
  && git pull`, delete the merged branch, prune.

## Working principles
1. **Think before coding.** State assumptions; ask when uncertain — name
   *what's* unclear. Push back on false premises. Check the repo and the
   chronicle before writing; fetch, don't recall. Hit the live system before
   trusting a type, a name, or a doc.
2. **Simplicity first.** Minimum code for the task. No speculative features,
   no abstractions for single-use code. Validate at boundaries; trust internal
   invariants.
3. **Surgical changes.** Touch only what the task needs. Don't reformat
   adjacent code; the linter is authoritative. Extend existing classes instead
   of forking near-duplicates (the LinearPlusMLP/KfAnchoredMLP fork cost a
   full sweep to diagnose).
4. **Goal-driven execution.** Bugs: pin with a failing test before patching.
   Multi-step work: `step → verify` pairs. Solver changes: verify with the
   certification stack on real checkpoints, not just unit tests.
5. **Claims need receipts.** Every quantitative claim in a dev doc carries a
   commit, file, log, or reproduce command. Superseded claims are kept and
   marked, not deleted.

## Certification (what "solved" means here)
A small training loss is not a certificate — measured repeatedly (best-by-loss
checkpoints are certificate-worst). The stack, in order of strictness:
held-out/stress residuals → **learned-block** spectral radius (the probe's raw
ρ has a floor at the exogenous root 0.98699 — always report the learned 8×8
block) → solved fixed point ŝ = T(ŝ): ‖ŝ−s\*‖ and ρ(ŝ) → per-equation
residuals AT ŝ → long-horizon convergence, multi-seed. Frozen convention:
**final checkpoints** (`checkpoint_003000.eqx`), fp64. Tool:
`scripts/disaster_ss_probe.py`.

## Commands
| task | command |
|---|---|
| test | `uv run pytest tests/ -q` (638 collected @2026-07-12; 4 Dynare-fixture skips when data absent) |
| lint | `uv run ruff check src/ tests/ scripts/` (zero-error; CI-enforced) |
| format | `uv run ruff format src/ tests/ scripts/` |
| typecheck (advisory) | `uv run pyright` (basic mode; not in CI) |
| train | `uv run deqn-jax train <model> -n 1000` (`-o ngd -q` for smoke) |
| list models / optimizers | `uv run deqn-jax list` / `uv run deqn-jax optimizers` |
| certificates | `JAX_ENABLE_X64=1 uv run python scripts/disaster_ss_probe.py --runs-dir runs/disaster_cert --arms <a> --seeds 0,1,2` |
| DGX sync | `rsync -azR <files> anna@130.223.169.108:~/projects/deqn-jax/` (relative-path form; one rsync per destination) |
| DGX GPU sweep | `LAUNCHER=scripts/cert_sweep_container.py ./scripts/run_sweep_in_container.sh` (DONE-marker resumable) |
| docs deploy | `mkdocs gh-deploy --remote-name pages` |

Always `uv run`; never activate the venv manually.

## Project structure
```
src/deqn_jax/
  config/        # Pydantic v2: TrainConfig, OptimizerConfig, NetworkConfig, CompositeLossConfig, CoverageConfig
  types.py       # ModelSpec, TrainState, Metrics — all NamedTuple pytrees
  cli.py         # train / list / optimizers / check
  models/        # 11 registered models (see `deqn-jax list`); each: variables, equations, dynamics, steady_state
  networks/      # factory.py + mlp / lstm / transformer / linear_plus_mlp; models/disaster/network.py (π_BK + δ, bk_pin)
  optimizers/    # registry + ngd / mao / shampoo / lbfgs / gauss_newton / pcgrad
  training/      # trainer (orchestrator), state_init (dispatch + validators), cycle, loss, composite_loss, coverage, episode, linearize, warm_start
  evaluate/      # simulate, diagnostics, dynare, cli
configs/         # arm configs (disaster_gated_pcgrad_bkpin.yaml etc.)
scripts/         # gitignored except whitelist: probes, sweeps, risky-SS, GN polish
tests/           # 638 tests; smoke convention: 3 episodes, hidden=(16,), batch=16, mc_samples=2
docs/dev/        # certification record + chronicle (research state)
```

## Code conventions
- Config precedence: `--set` overrides > CLI args > YAML file > defaults
  (`load_config()` in `config/io.py` merges; dot-notation for nested fields,
  e.g. `--set optimizer.learning_rate=0.01`). When a hyperparameter "didn't
  take", check for a `--set` in the launcher before editing the YAML.
- Two JIT boundaries per cycle (rollout + grad-step sweep); everything
  runtime-variable resolves at construction time, before tracing.
- Five train-step variants (STANDARD/PCGRAD/MAO/LBFGS/GN) dispatched by
  `OptimizerKind`; they differentiate *different objects* — when adding a loss
  feature, extend `state_init._validate_train_config` so combos that would
  silently drop it from the gradient are rejected, not ignored.
- Loss-dict keys prefixed `aux_` are excluded from reweighting and gradient
  surgery by contract (`eq_losses_to_array`).
- Equinox patterns: `eqx.filter(model, eqx.is_array)` → update →
  `eqx.combine(arrays, model)`.
- **Test discipline**: unit + regression (bit-identical guards for refactors);
  certification claims additionally need the probe stack on real checkpoints.
- **Gotchas**:
  - *Checkpoint loading*: the template network must be built from the FULL
    network config — static fields (`bk_pin`, `use_zlb_feature`, reparam
    flags) change the forward graph and are not repaired by leaf
    deserialization (`irf.py`, fixed 2026-07-11 after an impossible probe).
  - *JAX*: `jax.tree.map` treats tuples as containers; `lax.cond` needs an
    operand; no `float()` inside JIT; Shampoo L/R preconditioners in separate
    `tree_map` calls.
  - *Probe ρ floor*: raw closed-loop spectral radius never reads below
    0.98699 (exogenous mu_ups root × soft-clip); report the learned-block
    eigenvalue.
  - *Compute placement*: training, evaluation, and full pytest run on the DGX
    (host CPU for probes, NGC container for GPU training), not on the
    maintainer's laptop.
  - *aarch64 flake*: `test_convergence.py::TestDisasterTraining::
    test_loss_decreases` is platform-sensitive on the DGX (chaotic bare-MLP
    lr=1e-2 path, documented last-bit bifurcation) — not a regression signal.
  - *CI parity gap*: pyright is advisory-only (not in CI) — deliberate, see
    Nature relaxations.
  - *Short disaster smokes*: episodes < `lr_warmup` makes the cosine schedule
    negative — pass `--set optimizer.lr_warmup=5` for runs under ~100
    episodes.

## What to update when
- `AGENTS.md` — commands, structure, conventions, or stack change.
- `docs/dev/disaster_cert_report_2026_07_07.md` — any certification claim.
- `docs/dev/selection_program_chronicle_2026_07.md` — program-level shifts
  (new results, retractions, method lessons).
- Private notes repo — research narrative not for the public tree.

## Git workflow
- Conventional commits (`feat:`/`fix:`/`docs:`/`chore:`/`test:`); bland
  messages; no session-metadata lines. Claude agents append their standard
  co-author trailer.
- Local gate: ruff runs on every edit via the committed `.claude` hook; run
  lint + tests before pushing regardless. CI enforces both.
- Definition of done: CI green (tests + lint); for solver changes,
  certificates unregressed at the frozen convention; docs updated per *What
  to update when*.
- **Never** `--no-verify`, `--force`, `--no-gpg-sign`, or `git reset --hard`
  without explicit user instruction. Pushing `master` is the maintainer's
  call unless explicitly delegated.
