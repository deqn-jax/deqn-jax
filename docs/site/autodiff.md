# Autodiff-synthesized equations (design note)

> Vision: a researcher writes down a Lagrangian, the state variables, and the law of motion. The framework autodiffs out the equilibrium residuals, hands them to the DEQN trainer, and the model is solved. No hand-derivation of FOCs.

This document sketches how that path fits into DEQN-JAX today and where it's heading. A working proof of concept lives at `src/deqn_jax/models/brock_mirman_autodiff/` — same economics as `brock_mirman`, but the Euler residual is synthesized from a single scalar function rather than hand-derived.

## What the researcher writes

For a representative-agent problem with capital `K` as the intertemporal state, the minimal input is one function:

```python
def period_return(K, K_next, z, constants):
    """Pi(K_t, K_{t+1}, z_t) = u(C_t)."""
    alpha, delta, gamma = constants["alpha"], constants["delta"], constants["gamma"]
    Z = jnp.exp(z)
    y = Z * K ** alpha
    c = y - (K_next - (1 - delta) * K)          # budget constraint baked in
    return jnp.log(c) if gamma == 1.0 else (c ** (1 - gamma) - 1) / (1 - gamma)
```

Everything else — production function, budget identity, utility form — is already in this single expression. The researcher never writes `u'(c) - β E[u'(c')(1 + r' - δ)]` anywhere.

## What the framework synthesizes

Differentiating `Π(K, K', z) = u(C(K, K', z))` via `jax.grad`:

- `∂Π/∂K_{t+1}` evaluated at `(K_t, K_{t+1}, z_t)` — the cost today of investing one more unit.
- `∂Π/∂K_t` evaluated at `(K_{t+1}, K_{t+2}, z_{t+1})` — the marginal benefit tomorrow of having that unit.

The Euler condition is their sum (in expectation over `z_{t+1}`): `0 = ∂Π/∂K_{t+1} + β·E[∂Π/∂K_t]`. That's the residual the trainer gets. `K_{t+2}` is reconstructed from `next_state + next_policy` using the model's own capital-accumulation law — the one place the dynamics reappear inside the residual.

Check: with log utility + Cobb-Douglas production, this simplifies algebraically to `-u'(C_t) + β · u'(C_{t+1})·(1 + r_{t+1} − δ)`, which is the hand-derived form up to sign. The autodiff variant passes a parity test against `brock_mirman`'s hand-derived residuals to float32 noise on a random batch of policy-consistent transitions.

## The eventual API shape

The POC wires the autodiff directly into the model's `equations.py`. The next step is a framework-level helper — something like:

```python
from deqn_jax.training.autodiff import euler_from_period_return

MODEL = ModelSpec(
    ...,
    equations_fn=euler_from_period_return(
        period_return_fn=period_return,
        capital_state="K",              # which state dim is the intertemporal link
        investment_law="lom",           # optional; defaults to inferring from step_fn
    ),
)
```

At that point, the `ModelSpec` declaration for a Brock-Mirman–class model becomes:
- `variables.py` — SPEC, constants
- `period_return.py` — the scalar Π function (the Lagrangian / objective)
- `dynamics.py` — step function (the law of motion)
- `__init__.py` — assembly; no explicit `equations_fn` needed

Three things have to be true before that helper lands:

1. **Multi-policy models.** With labor or other intratemporal choices, there's a second FOC class (`∂Π/∂L = 0`) that needs its own autodiff path. Generalizes cleanly but the helper needs to know which policy dimensions are intratemporal vs state-determining.
2. **Multi-shock / multi-state Euler.** OLG-style models have one Euler per savings-choosing agent. The helper needs to vmap over agents.
3. **Non-separable constraints.** Borrowing constraints with Lagrange multipliers (KKT) don't come out of pure autodiff on Π — they need the full Lagrangian including the multiplier. Simon's OLG benchmark uses Fischer-Burmeister here. The generalized helper should support supplying additional constraint residuals alongside the autodiff-Euler.

The POC covers case (0): single representative agent, single intertemporal state, single policy, utility-only objective. That's the easiest and most common. The rest is a progression of generality.

## Where Claude fits in

Simon's framing: researcher writes down the Lagrangian in something close to paper notation, Claude (or any LLM) transcribes it into the framework's `period_return` + state schema + dynamics. The mechanical part — autodiff FOCs, neural architecture search, loss reweighting, curriculum — is then framework work with no per-paper plumbing.

Concretely: a Claude-authored `bring_your_own_paper` tool would need

- a parse of the problem statement (state variables, controls, objective, constraints),
- translation into `period_return_fn` + `step_fn` + variable/shock schema,
- a round-trip check: autodiff residuals zero at a declared / solved steady state.

That last item is the "did I transcribe it right" gate. It's a cheap, automatic sanity check that catches most transcription errors without the user ever running training.

## Current status

- POC: `src/deqn_jax/models/brock_mirman_autodiff/` (model registered as `brock_mirman_autodiff`).
- Parity tests: `tests/test_autodiff_equations.py` (residual match + SS zero + registration, 3 tests).
- Still to build: the framework-level `euler_from_period_return` helper; extension to multi-policy (`bm_labor`) and multi-agent (`olg_analytic_6`); a Lagrangian path with explicit multipliers for KKT.

See `docs/site/models/implementing.md` § 2 for the hand-derived path this is meant to eventually replace.
