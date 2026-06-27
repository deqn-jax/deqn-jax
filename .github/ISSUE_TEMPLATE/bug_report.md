---
name: Bug report
about: Something broke, or a result looks wrong
title: "[bug] "
labels: bug
assignees: ''
---

<!--
Thanks for trying DEQN-JAX at alpha — bug reports are genuinely useful right now.
The more of the repro you can fill in, the faster this gets fixed.
A minimal, copy-pasteable command beats a long description every time.
-->

## What happened

<!-- One or two sentences. What did you run, and what went wrong? -->

## Repro

**Model + config**
<!-- Which model (`uv run deqn-jax list`)? A built-in config, a `--config` file,
     or a set of `--set` overrides? Paste the config or the relevant fields. -->

```yaml
# config here, if any
```

**Command**

```bash
# the exact command, e.g.
uv run deqn-jax train brock_mirman -n 1000 --warm-start
```

**Expected vs actual**

- Expected:
- Actual:

<!-- For "result looks wrong" (rather than a crash): what did you read off, and
     against what? errREE on the ergodic set is the number we trust — a residual
     can be low and still sit on the wrong equilibrium branch, so tell us how you
     decided the answer was wrong. -->

**Traceback / output** (if it crashed)

```
# paste full traceback here
```

## Environment

- DEQN-JAX version / commit: <!-- `v0.2.0`, or `git rev-parse --short HEAD` -->
- Output of `uv run deqn-jax check`: <!-- paste it — it reports JAX + accelerator -->
- JAX version:
- Python / `uv` version:
- Accelerator: <!-- CPU / CUDA 12 / CUDA 13 / Apple Metal -->
- OS:

## Anything else

<!-- Does it reproduce on `brock_mirman` (the 5-minute smoke test)? Is it
     deterministic, or does it depend on the seed? Was this the validated stack
     (Adam + MLP/LinearPlusMLP + MSE + antithetic-MC / Gauss-Hermite) or a
     research-instrument path (second-order optimizer, sequence policy, composite
     loss)? Knowing which narrows it down fast. -->

