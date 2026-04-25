# Ralph Loop: trainer.py + composite_loss.py refactor

Paste this body into the `/ralph-loop` slash command. Suggested invocation:

```
/ralph-loop "$(cat .claude/ralph_refactor_prompt.md)" --completion-promise "REFACTOR COMPLETE" --max-iterations 20
```

(If shell substitution doesn't pass through `/ralph-loop` cleanly, paste the body below directly.)

---

You are running a Ralph Loop iteration to extract model-specific code from the framework core in this repo (`deqn-jax`). Each iteration picks ONE smell, makes the smallest change that fixes it, runs tests, commits, and exits.

# Discover state

1. `git log --oneline -8` — see what previous iterations did.
2. `cat /Users/aleph/.claude/projects/-Users-aleph-Projects-research-deqn-jax/memory/MEMORY.md` — project context.
3. `wc -l src/deqn_jax/training/trainer.py src/deqn_jax/training/composite_loss.py` — track shrinkage.

# Pattern precedent

Commit `9926b14` extracted ~190 lines of disaster-specific code from `trainer.py` via two new generic `ModelSpec` hooks (`setup_fn` and `scalar_diagnostics_fn`). Use this pattern for analogous leaks: add a generic hook to `ModelSpec`, implement the model-specific part in the model's own package.

# Smell list, ordered. Pick the first not-yet-addressed.

1. **composite_loss.py disaster knowledge.**
   `composite_loss.py` reads `defs["newton_h_prime"]`, `defs["newton_residual"]`, `defs["n"]`, `defs["L"]`, `defs["c"]` directly. Other models can't use composite loss without claiming those names.
   **Done when** `rg -c 'newton_h_prime|newton_residual' src/deqn_jax/training/composite_loss.py` returns 0.

2. **Five `_make_grad_step_*` factories in trainer.py.**
   `_make_grad_step_standard`, `_make_grad_step_pcgrad`, `_make_grad_step_mao`, `_make_grad_step_lbfgs`, `_make_grad_step_gn`. ~600 lines of near-duplicated structure. Move each variant body into the corresponding `optimizers/<name>.py` and extend `@register_optimizer` to also register a `make_grad_step` factory.
   **Done when** `rg -c '^def _make_grad_step_' src/deqn_jax/training/trainer.py` returns 0.

3. **`train_from_config` is ~700 lines.**
   Decompose into named setup helpers (`_apply_fp64_if_needed`, `_validate_*`, `_resolve_model`, `_build_initial_state`, `_assemble_train_step`).
   **Done when** `train_from_config`'s function body is < 250 lines.

4. **Inline checkpoint I/O.**
   Lift save/prune/resume out of the train loop into `training/checkpointing.py` with `save_checkpoint`, `prune_checkpoints`, `resume_from`.
   **Done when** `rg -c 'tree_serialise|tree_deserialise|checkpoint_best\.eqx' src/deqn_jax/training/trainer.py` returns 0.

# Iteration protocol

1. Identify the first un-addressed smell (use the "Done when" rg checks).
2. State your plan in 1–2 sentences.
3. Make the smallest change that fixes it. Extraction over rewrite.
4. Run `uv run pytest tests/ -q`. Pass count must be ≥ 308. (1 skipped is normal.)
5. If tests fail: `git restore .`, append a one-line summary to `docs/refactor_log.md` (create if absent) explaining what failed, then exit so a human can review.
6. If tests pass: `git add -A && git commit -m "<focused message>"` and exit.

# Hooks

A `PostToolUse` ruff check fires on every `Edit` / `Write` / `NotebookEdit`. Fix lint before continuing.

# Stop condition

When all four smells' "Done when" rg checks return 0, output exactly:

<promise>REFACTOR COMPLETE</promise>

# Out of scope

- New features.
- Public CLI surface or `train_from_config` argument changes.
- Renaming public symbols without keeping an alias.
- Editing tests except to update import paths.
- Editing docs except cross-refs touched by the refactor.

# Now: do one iteration.
