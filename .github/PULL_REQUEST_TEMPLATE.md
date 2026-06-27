<!--
Thanks for contributing to DEQN-JAX. It's alpha — PRs are welcome and the bar is
honesty, not polish. Tell us plainly what you validated and what you didn't; an
experimental contribution clearly labelled as experimental is more useful than an
overclaimed one. Work happens on `master`.
-->

## What this does

<!-- One or two sentences. What changes, and why? Link any related issue: Fixes #123 -->

## Type of change

- [ ] Bug fix
- [ ] New model
- [ ] New optimizer / network / loss term
- [ ] Docs / examples
- [ ] Refactor / internal (no user-facing change)
- [ ] Other:

## Checklist

- [ ] **Tests pass** — `uv run pytest tests/` (or `-m 'not slow'` while iterating)
- [ ] **Added tests** for new behavior <!-- a new model needs the 20-episode loss-decreases test -->
- [ ] **Ruff clean** — `uv run ruff check .` and `uv run ruff format --check .`
- [ ] **Docs updated** if this is user-facing (README, `docs/site/`, and `docs/site/REFERENCE.md` if the contract changed)
- [ ] **Noted in `CHANGELOG.md`** if this changes the public API — we're `0.x`, so breaks get tracked
- [ ] Used `uv run` throughout — did not activate the venv manually

## Validated vs experimental

<!-- Be honest here — this is the most important part of the PR at alpha.
     What did you actually run, on what model, and what did you observe? -->

- **What I validated:** <!-- e.g. "trained brock_mirman 1000 episodes, errREE on the
     ergodic set matches the gallery; the new optimizer's tests pass." -->
- **What I did NOT validate / what's experimental:** <!-- e.g. "only tested on CPU",
     "converges on brock_mirman but untested on a binding-constraint model",
     "second-order path, less exercised than the Adam default." -->
- **Stack used:** <!-- validated core (Adam + MLP/LinearPlusMLP + MSE + antithetic-MC /
     Gauss-Hermite), or a research-instrument path? -->

## Notes for the reviewer

<!-- Anything tricky, anything you're unsure about, anything you'd like a second
     opinion on. "I don't know if this is the right seam" is a fine thing to say. -->

