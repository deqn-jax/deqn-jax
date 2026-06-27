---
name: Feature request
about: Propose a capability, optimizer, network, loss term, or diagnostic
title: "[feature] "
labels: enhancement
assignees: ''
---

<!--
DEQN-JAX is alpha and the API still moves. That's the good time to shape it.
Concrete proposals tied to a real solve land best — vague "make it better" ones
are hard to act on.
-->

## What's missing

<!-- The capability you want. If it's a model or paper you'd like supported,
     please use the "Model / paper support request" template instead — it asks
     the right questions. -->

## Why — the use case

<!-- What are you trying to solve that you can't today? A short description of
     the model class, state dimension, or constraint structure helps us judge
     whether this is a one-off or a pattern. -->

## Where it fits

<!-- Optional, but it speeds things up. Roughly which seam? -->

- [ ] New optimizer (`@register_optimizer`, one of the five train-step variants)
- [ ] New network architecture (`eqx.Module` + factory in `networks/factory.py`)
- [ ] New loss term (`aux_`-prefixed, `training/composite_loss.py`)
- [ ] New diagnostic / CLI subcommand
- [ ] Docs / examples
- [ ] Other

## Prior art

<!-- A paper, a reference implementation, an equation. Anything that pins down
     "done" concretely. -->

## Scope check

<!-- DEQN-JAX is a focused global solver, not a kitchen sink. We say no to keep
     the validated core small and honest. A quick note on how central this is to
     your work helps us prioritize. -->

