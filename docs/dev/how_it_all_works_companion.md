# The Companion — the same story, in plain words

**What this is:** the conversational twin of `docs/how_it_all_works.md`, distilled from the
2026-07-06 session. The main doc is the formal, receipted version; this one is the version you
reread when you've forgotten what any of it means. Same facts, same honesty discipline, cat
included. Where the main doc has a receipt, this one just points at it.

---

## 1. Why DEQN was ever a good idea

Solving an economic model means finding a **rule** — in every situation, what's the optimal
thing to do — and the equations that define the rule contain the rule itself (today's optimum
depends on expectations of tomorrow's). Two classical tools, two walls: **perturbation**
(Dynare) linearizes at the resting point — fast, but blind far away, blind to kinks
(constraint that sometimes bind), blind to risk at first order. **Grids** are right everywhere
but their cost explodes exponentially in the number of state variables — dead by ~8 states,
and the questions people care about (heterogeneity, disasters, the ZLB) live past that wall.

DEQN's bet: neural nets are the one function-fitter whose cost doesn't explode with dimension,
and you don't need to tile the state cube anyway — **simulate** the economy under the current
guess and enforce the equations only on states that occur. The economy lives on a thin sheet
inside the huge cube; grids pay for the cube, simulation pays for the sheet. The founding paper
solved a 56-generation OLG model no grid can touch. That's the promise.

The catch discovered in this repo, measured and named: **"loss went down" and "model solved"
are different claims.** The method ships without brakes. The brakes are what this project
actually built.

## 2. The cat (the ergodic set, the steady state, and the policy)

A cat lives in a big house (the state space — everything that *could* happen). There's a
heated cushion (the **steady state** — where the story would end if the world went quiet). The
world never goes quiet: shocks arrive every period. The cat has **habits** (the policy — the
actual solution of the model: a rule, not a place). Watch the cat for 45,000 days and dot
every place you find it: a thin worn patch of carpet around the cushion. That smudge is the
**ergodic set** — a *consequence* of habits + noise, not a separate object. The house is
enormous; the smudge is tiny; you will never find the cat on the fridge.

Since the economy lives on the sheet, you only need the policy to be right *there* — that's
the whole scaling miracle. And the trap: **the sheet is defined by the policy you haven't
found yet.** Train where the wrong policy walks and it certifies itself on its own footprints.
There is no oracle for the set; there are only two islands of exact knowledge in an unsolvable
model: the **physics** (Γ — how the world moves given a decision; always known) and the
**cushion's neighborhood** (the steady state and its correct local behavior — computable in
any model). Every fix we validated is built from those two islands.

Interactive versions: `~/Projects/vibes/mathgraph/ergodic.html` (Brock–Mirman, true habits,
exact sheet) and `ergodic2.html` (irbc: two trained networks, same shocks — one paces its
sheet, one quietly eats half its capital and moves to the basement; plus the amber EWM
training states in the corner the walk never visits).

## 3. What ρ(SS) is (and the two rhos)

Two different rhos exist in this project. ρ in the shock process (z' = ρz + σε) is boring
persistence. **ρ(SS)** is the stability certificate: put the economy exactly at its resting
point, nudge it once, and ask whether the nudge fades or grows. Each period the deviation gets
multiplied by roughly a constant factor — that factor is ρ(SS). Below 1: marble in a bowl,
nudges melt (0.98 → half-life ~36 quarters). Above 1: marble on an upturned bowl — 1.23 means
deviations grow 23% per period and double every ~3. The equations are satisfied by both bowls;
economics wants the right-side-up one; the training loss can't tell them apart; ρ(SS) can, and
it's cheap (differentiate the trained one-step map at SS).

## 4. Why multi-seed is not Seed Engineering

A classical solver's answer is a function of the problem — seeds don't exist. DEQN's answer is
a function of problem + initialization + draws, and because the loss has **multiple basins**,
different seeds give *different answers*, not the same answer with noise (plain irbc:
ρ = 1.09…1.29 across five seeds — five different wrong economies). One run is an anecdote about
one lottery ticket. Five a-priori-fixed seeds, all reported including failures, is the cheapest
honest test that a claim is about the *recipe* and not about seed 42. Seed engineering is
selecting on the outcome; this is refusing the seed as a degree of freedom. And the win
condition is that multi-seed becomes boring: the composition arm landed all ten anchored runs
on ρ = 0.9808 with zero variance — a working selection device doesn't improve your lottery
odds, it closes the lottery. Seed-variance is itself the diagnostic.

## 5. IRBC, EWM, and how it helped

**IRBC**: two countries, each with capital and wobbling productivity; a planner decides how
much to eat vs invest and where; the wrinkle is **irreversibility** — you can't un-build
factories, so investment ≥ 0 occasionally binds. Exactly the kink linearization can't see;
small enough (4 states) to verify thoroughly. No closed form exists.

**EWM** (Scheidegger & Schaab, arXiv:2606.23463): the diagnosis is self-confirming solutions —
train only where you walk and you only certify where you walk. The prescription: stop letting
the student pick the exam questions — train the same residuals on a mixture (the walk + stress
states rolled through the true dynamics + local nudges), weights fixed a priori. Their
evidence: pathwise verifies 0/10 seeds; coverage 8–9/10. Fine print: their "verified" is a
training-loop stationarity + on-coverage-accuracy notion, not closed-loop stability.

**Our verdict** (5 seeds, refereed, committed): the claim replicates on the half it makes —
stress-region error down ~1.1 decades at zero on-measure cost. It does not fix stability
(0/5). Certification and selection are different diseases; coverage cures the one it
diagnosed; the **BK anchor** (bolt the known-correct local solution into the network) cures
the other. **Composed** (2026-07-06): both certificates, five for five, ρ = 0.9808 zero
variance, coverage adding a further ~2.3× in the binding corner. The anchor holds the center
of the map; coverage patrols the edges. That table is the thing to show.

## 6. What "solved" means, and where irbc stands

Certificates, in ascending strictness: residuals on the ergodic set → residuals on a held-out
stress region → closed-loop stability ρ(SS) < 1 → reproduces the steady state in level →
agreement with an independent solution. Small training loss is *not* a certificate — §4 of the
main doc is a tour of networks that pass it and fail everything else.

**irbc passes the whole stack as of 2026-07-06** (5/5 stable, SS-exact to <1e-3 on all ten
anchored checkpoints, on/off-path certified). It's the first model in the repo solved without
a known answer to lean on — Brock–Mirman is where trust is calibrated; irbc is where the
method stands alone.

## 7. The disaster model, from the very start

A CMR-style New Keynesian DSGE with financial frictions — Alex's LaTeX, three years of
rewrites. Cast: households (habit), Calvo-sticky prices and wages, BGG entrepreneurs (luck
draws, defaults, monitoring costs — the financial accelerator), a Taylor-rule central bank
with a soft interest-rate floor, five shocks, and nominally a capital-destroying disaster.
13 states, 11 network outputs, 11 equations: consumption Euler, bond Euler/Fisher, the price
and wage Phillips blocks (3 + 3, split so gradients don't cancel), investment Euler/Tobin's q,
the loan contract, and the resource constraint.

**The 2026-07-06 audit verdict: nothing is wrong with the equations.** All eleven re-derived
by hand and independently by a cross-family referee (Codex); steady state satisfies the system
to 2e-16; the tex's SS tables pass consistency checks they were never designed to advertise
(the Calvo relation reproduces K_p to four decimals; the resource constraint closes to four
figures). Three years of rewrites did not corrupt the math. What's wrong lives in three
*joints*:

1. **The constants lie about which economy it is.** y_ss = 3.0308 in the calibration vs 3.0044
   actually produced; the Taylor rule reads the phantom gap every quarter and the solved
   economy runs 5% inflation under a 2.4% label. Established by counterfactual re-solves (the
   tex's old "financial-friction compounding" story was wrong and is now rewritten — errata
   item 12). The honest calibration exists and is runnable (`configs/disaster_recal.yaml`,
   π* = 1.005918 verified) but doubles ELB exposure — opt-in, decision coupled to the ELB fix.
2. **The model lives on its own kink.** The Taylor rate wants to cross its floor one quarter
   in eight; every linear tool is blind exactly there.
3. **The disaster is a costume.** p = 0 ships (mechanism inert), θ = 5% vs Alex's Barro-sized
   15%, and Alex's σ_d channel — idiosyncratic dispersion *doubling* in a crisis, the thing
   that would make the financial accelerator fire — was never implemented at all.

## 8. Why the trained policy doesn't sit on its steady state

The puzzle: the recipe starts SS-perfect (`init_scale: 0` = exact linearized policy at step
zero) and the anchor pulls toward SS throughout — yet training drags it off (June 2026
finding). The measured prime suspect: **the anchor and the residuals disagree about 12% of the
world.** The anchor teaches the no-floor linearization; the economy actually visits the floor
one quarter in eight; the network's compromise sells exactness at one point (the cushion) to
buy fit over a fat slice of the distribution (the wall). irbc's kink is off-path so its anchor
never fights — same recipe, different geometry, opposite outcome. Caveats: the June finding
was entangled with a since-fixed fp64-load bug, and no old checkpoint survives the refactors —
a fresh 1,500-episode run at HEAD was training on the DGX as this was written; its probe
(levels at SS, zero-shock drift, ρ) settles the question with current data. Falsifiable
prediction: SS errors concentrate in the monetary block, and a kink-aware anchor restores
irbc-grade exactness.

## 9. Alex's weird calibration (the original table)

Three finds in `docs/disaster.tex`: (a) **σ_p = 0.49** — the monetary shock std listed 100×
too large (percent written as decimal); code correctly ships 0.0049; if you ever ran the table
verbatim and watched it detonate, that row is why. (b) **The disaster was real**: p = 1%/quarter,
θ = 15% ("Own calibration", Barro-consistent) vs the shipped p = 0, θ = 5%. (c) **The ghost
parameter σ_d = 0.54 = 2×σ_ω**, "risk entrepreneurs credit crisis" — disasters were meant to
double idiosyncratic dispersion (a financial crisis, not just a capital loss). Never
implemented. Restoring (b)+(c) is spec-let 4 in the handoff.

## 10. The ELB origin story (the patch that became the protagonist)

Before 2026-04-17 the Taylor rule had **no floor at all** (nothing in Alex's tex). Commit
fdfbf2b added the softplus ELB because a *diverged training run* was dissected and showed
R bottoming at 0.95 — a numerical guardrail, economics second. That commit message is also the
birth certificate of the wrong "~1e-7 SS distortion" claim corrected on 2026-07-06 (true
equilibrium effect: −8e-5 log π). Later, the v0.3.0 probes attacked the kink from the network
side (`use_zlb_feature` regime inputs; ReLU nets) — verdicts not in the tracked record. Three
months after its birth as a band-aid, the floor is the model's most economically alive feature:
the one nonlinearity the economy actually lives with, the blind spot of the anchor, the prime
suspect of §8, and the target of the next move. Kink attacked three ways so far: in the model
(softplus), in the network (features/ReLU), and next in the anchor (two-regime P/Q — the only
one with a validated pattern behind it, namely the composition result).

## 11. Status at end of day (2026-07-06)

**Done and receipted:** irbc solved (full certificate stack, 5/5); EWM claim replicated +
split verdict; composition = both certificates; in-repo evaluation path; the main doc written
and cross-family refereed (its own "single JIT boundary" folklore caught by Codex and fixed
everywhere); disaster equations verified clean; disaster tex slop fixed (errata 12);
recalibrated config shipped opt-in; evaluator overhaul specced.

**The disasterless model is NOT solved.** Fitted (≈3e-5 on-measure), math verified, but: fails
SS-level consistency (pending the HEAD re-measure), ρ(SS) never measured multi-seed, no stress
certification, evaluator numbers provisional. Distance to done = four specced items: kink-aware
anchor (spec-let 1), disaster stress-seed map (spec-let 2), the irbc certification protocol run
on disaster, evaluator overhaul (gated on sign-off).

**Decisions on the maintainer's desk:** push to origin; commit/publish the doc + handoff + house audit;
adopt `disaster_recal` (couples to ELB fix); evaluator sign-off; spec-let 4 (Alex's restored
disaster).

**Where everything lives:** `docs/how_it_all_works.md` (the formal doc);
`docs/dev/handoff_2026_07_06.md` (spec-lets + state);
`docs/superpowers/specs/2026-06-29-ewm-coverage-sampling-design.md` (EWM + composition
results); `docs/dev/disaster_house_audit.md` (untracked, refereed);
`docs/disaster_corrected.tex` (the model, errata 12 current); `scripts/ewm_stress_table.py`
(the certification table); mathgraph `ergodic.html` / `ergodic2.html` (the pictures).
