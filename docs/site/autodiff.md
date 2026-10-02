# Autodiff-synthesized equations (design note)

> Goal: a researcher writes down a Lagrangian, the state variables and the law of motion. The framework derives the equilibrium residuals by automatic differentiation and passes them to the DEQN trainer, with no hand-derived FOCs.

This note describes how that path works in DEQN-JAX today and what remains. The framework helper `euler_from_period_return` (`src/deqn_jax/training/autodiff.py`, exported from `deqn_jax.api`) builds the residuals. Two proof-of-concept models use it: `brock_mirman_autodiff` has the same economics as `brock_mirman`, and `bm_labor_autodiff` the same as `bm_labor`, but their residuals come from a single scalar function instead of being derived by hand.

## What the researcher writes

For a representative-agent problem with capital `K` as the intertemporal state, the minimal input is one function (simplified from `models/brock_mirman_autodiff/equations.py`):

```python
def period_return(K, K_next, z, policy, constants):
    """Pi(K_t, K_{t+1}, z_t, policy_t) = u(C_t)."""
    alpha, delta, gamma = constants["alpha"], constants["delta"], constants["gamma"]
    Z = jnp.exp(z[0])
    y = Z * K ** alpha
    c = y - (K_next - (1 - delta) * K)          # budget constraint baked in
    return jnp.log(c) if gamma == 1.0 else (c ** (1 - gamma) - 1) / (1 - gamma)
```

The production function, the budget identity and the utility form are all in this one expression. The researcher never writes `u'(c) - β E[u'(c')(1 + r' - δ)]`.

## What the framework synthesizes

Differentiating `Π(K, K', z) = u(C(K, K', z))` with `jax.grad` gives:

- `∂Π/∂K_{t+1}` at `(K_t, K_{t+1}, z_t)`: the cost today of investing one more unit.
- `∂Π/∂K_t` at `(K_{t+1}, K_{t+2}, z_{t+1})`: the benefit tomorrow of having that unit.

The Euler condition is their sum, in expectation over `z_{t+1}`: `0 = ∂Π/∂K_{t+1} + β·E[∂Π/∂K_t]`. This is the residual the trainer receives. `K_{t+2}` is rebuilt from `next_state` and `next_policy` by the model's own `step_fn` at zero shock, the one place the dynamics reappear inside the residual. The expectation over shocks is handled by the loss module, as for any model.

Check: with log utility and Cobb-Douglas production this reduces algebraically to `-u'(C_t) + β · u'(C_{t+1})·(1 + r_{t+1} − δ)`, the hand-derived form up to sign. The autodiff models match the hand-derived residuals of `brock_mirman` and `bm_labor` to float32 noise on a random batch of policy-consistent transitions.

## The helper

The model's `equations.py` passes the period return and its own `step` function to the helper:

```python
from deqn_jax.training.autodiff import euler_from_period_return

equations = euler_from_period_return(
    period_return_fn=period_return,
    step_fn=step,
    capital_idx=0,  # state = (k, z); capital is dim 0
    exog_idx=(1,),  # z is dim 1
    n_shocks=1,
)
```

The result is an ordinary `equations_fn`. For a Brock-Mirman-class model the subpackage then consists of:

- `variables.py`: SPEC, constants
- `equations.py`: the scalar Π function (the objective) and the one helper call above
- `dynamics.py`: the step function (the law of motion)
- `steady_state.py`, `__init__.py`: steady state and assembly, as for any model

The helper covers three generalizations beyond the single-agent case:

1. Multi-policy models. With labor or other intratemporal choices there is a second class of FOC, `∂Π/∂L = 0`. Pass `intratemporal_policy_idx` (and optionally `intratemporal_equation_names`); each listed policy index gets a pointwise residual `−∂Π/∂policy[j]` with no expectation. `bm_labor_autodiff` uses this for its labor FOC.
2. Multi-agent Euler. OLG-style models have one Euler equation per savings-choosing agent. Pass `capital_indices` and `equation_names` (one per agent); Π then takes an extra `agent_index` keyword, and the helper returns one Euler residual per agent. This mode is tested on a toy two-cohort OLG; no registered model uses it yet.
3. Any number of exogenous state dimensions, through `exog_idx`.

Not supported: constraints that need explicit multipliers. Borrowing constraints with Lagrange multipliers (KKT) do not come out of autodiff on Π alone; they need the full Lagrangian including the multiplier. Simon Scheidegger's OLG benchmark uses Fischer-Burmeister residuals here. A generalized helper should accept extra constraint residuals alongside the autodiff Euler equations.

## Where an LLM fits in

Scheidegger's framing: the researcher writes the Lagrangian in something close to paper notation, and Claude (or any LLM) transcribes it into the framework's `period_return`, state schema and dynamics. The mechanical part (autodiff FOCs, architecture search, loss reweighting, curriculum) is then framework work with no per-paper plumbing.

A Claude-authored `bring_your_own_paper` tool would need:

- a parse of the problem statement (state variables, controls, objective, constraints);
- a translation into `period_return_fn`, `step_fn` and the variable and shock schema;
- a round-trip check that the autodiff residuals are zero at a declared or solved steady state.

The last item checks the transcription. It is cheap, automatic, and catches most transcription errors before any training run.

## Current status

- Helper: `euler_from_period_return` in `src/deqn_jax/training/autodiff.py`.
- Models: `brock_mirman_autodiff` (`src/deqn_jax/models/brock_mirman_autodiff/`) and `bm_labor_autodiff` (`src/deqn_jax/models/bm_labor_autodiff/`).
- Tests: `tests/test_autodiff_equations.py` (residual parity with the hand-derived models, zero residuals at the steady state, registration, envelope behaviour; 9 tests) and `tests/test_autodiff_multi_agent.py` (multi-agent mode; 5 tests).
- Still to build: a registered multi-agent model (for example `olg_analytic_6`) on the helper, and a Lagrangian path with explicit multipliers for KKT constraints.

Section 2 of `docs/site/models/implementing.md` describes the hand-derived path this is meant to replace eventually.
