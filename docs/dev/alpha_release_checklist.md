# Alpha-release readiness checklist (weekend)

From the 2026-06-27 multi-agent readiness audit. Verdict: **a weekend alpha is
realistic** — the framework is solid (567 tests pass, ruff clean, `uv build`
produces wheel+sdist, all 10 models contract-compliant). The work is
truth-in-labeling + packaging hygiene, not engineering. Blockers ≈ 3.6h.

## Blockers (must-fix before tagging)

- [x] `__init__.py` `__version__` 0.1.0 → 0.2.0 (runtime was lying) — *done c0f67c0*
- [x] CLI `--version` / `-V` flag — *done c0f67c0*
- [x] REFERENCE.md API table: drop `ResMLP` (not exported), add the actually-exported `LinearPlusMLP`/`KfAnchoredMLP`/`solve_steady_state`/… — *done c0f67c0*
- [x] `examples/cli.md`: 7 → 9 subcommands (`active-subspace`, `init-config`) + `--version` — *done c0f67c0*
- [x] README line 23: version/test-count(241→567)/subcommand list — *done c0f67c0 (folded into front-door reframe)*
- [ ] **Gallery re-execution** (7 notebooks, plot fixes already applied; pure re-run) — *heat-gated → offload/batch; disaster excluded*
- [ ] **Org-move URL sweep**: 9 `mechanicpanic` URLs across mkdocs.yml + REFERENCE.md + installation.md + api/models.md + models/{brock_mirman,disaster}.md; update `site_url`; then deploy — *gated on the org move*
- [ ] Pin `deqn-agent`'s `deqn-jax` dep off `path=../deqn-jax` → git/version — *gated on org move*

## The version-to-tag decision (yours)

`pyproject` is 0.2.0 but the tree is **173 commits ahead of the existing `v0.2.0`
tag** — a `v0.2.0` tag would collide/mislead. Pick `v0.2.1` or `v0.3.0a0` before
tagging.

## Should-fix (as time allows)

- [ ] README docs-link + badges (post-org-move URL)
- [ ] `[project.urls]` in pyproject (repo/docs/issues, final org URLs)
- [ ] CONTRIBUTING.md, CODE_OF_CONDUCT.md (Contributor Covenant)
- [ ] CLI error parity: `train invalid_model` dumps a traceback; `info nonexistent` is clean
- [ ] bare `deqn-jax` (no args) exit 0 not 1
- [ ] `(default: brock_mirman)` in train model-arg help
- [ ] link deqn-agent from index.md "where to go next"
- [ ] archive/flag 4 experimental configs using deprecated options
- [ ] CLI unit tests (currently smoke-only in CI)
- [ ] `ModelSpec` status field (stable/experimental/internal) surfaced in `list_models()`

## Docs-site findings (the agent that flubbed structured output, recovered manually)

- [x] IRF helper 1e12 blowup (near-zero baseline) — *done a13445b*
- [ ] broken internal anchors in REFERENCE.md (3) and api/loss.md (1)
- [ ] griffe: undocumented `config` params (registry.py:154, trainer.py:271)
- [ ] **docs deps are an optional extra** (`uv sync --extra docs` required) — note in CONTRIBUTING / a docs-build CI job
- [ ] gallery not yet on the site (needs re-exec + mkdocs-jupyter wiring)

## Weekend sequence (dependency-aware)

1. SAT — start gallery re-exec as a batched/remote job (excl. disaster); in parallel, the zero-heat doc/version fixes *(mostly done)*.
2. SAT — full test suite (cheap) to confirm; commit.
3. SAT — execute the GitHub **org move** (gate for all URL/deploy work).
4. SAT — URL sweep + `site_url` + pin deqn-agent dep + `[project.urls]` + README docs-link; `mkdocs gh-deploy` *only now*.
5. SUN — collect re-executed notebooks, wire mkdocs-jupyter gallery, commit.
6. SUN — should-fix polish; decide tag version; `uv build` + verify clean install + `--version` + tag.
