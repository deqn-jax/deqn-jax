---
name: Model / paper support request
about: Ask us to support a specific model or paper, or scope porting it yourself
title: "[model] "
labels: model-request
assignees: ''
---

<!--
DEQN-JAX is model-agnostic: a model is four small files (variables, equations,
dynamics, steady_state) plus a registry entry — see "Adding a new model" in the
README. The work is in writing down the equilibrium conditions cleanly, so this
template asks for exactly that. Fill in what you can; even a partial spec is a
useful start, and it's the same spec a human or an agent would target.
-->

## The model

**Name / short description**

**Paper reference** <!-- author, year, title, journal/working-paper; DOI or link if handy -->

<!-- A reference implementation (Dynare .mod, MATLAB, Julia, the authors' code)
     is gold if one exists — link it. -->

## Equilibrium conditions

<!-- The heart of it. What residuals should the network drive to zero? -->

- **States** <!-- endogenous + exogenous, with dimension, e.g. (k, z) -->
- **Policies / controls** <!-- what the network outputs, e.g. (c, l, savings, prices) -->
- **Equations** <!-- Euler equations, FOCs, market clearing — list them, or paste them -->
- **Shocks** <!-- how many, distribution, law of motion -->
- **Occasionally-binding constraints?** <!-- ZLB, borrowing limit, irreversible
     investment? These are the cases DEQN is *for* — a Fischer–Burmeister
     complementarity residual keeps the kink intact. Call them out. -->

## Calibration

<!-- Parameter values, or a pointer to the table/section they live in.
     Without calibration we can't reach a steady state to anchor the solve. -->

## Steady state

- [ ] Analytical (you can write it down)
- [ ] Numerical (solve required)
- [ ] Not sure

<!-- If analytical, the expressions help. If numerical, a known target vector to
     check against is invaluable. -->

## Why this model

<!-- What does solving it globally buy you that perturbation/Dynare doesn't? A
     big state space, a binding constraint, a genuinely nonlinear rule? This both
     helps us prioritize and tells us whether DEQN is the right tool. -->

## Are you offering to port it?

<!-- Totally fine either way. If yes, the README's "Adding a new model" section is
     the path, and we're happy to review a draft PR. If you'd like us to take a
     swing, a complete spec above is what makes that possible. -->

- [ ] I'd like to port it (with review help)
- [ ] I'm requesting it

