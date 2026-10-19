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

## What exists

- `euler_from_period_return` in `deqn_jax.training.autodiff` builds an `equations_fn` from Π: one Euler residual per capital column, optional intratemporal first-order conditions for listed policy columns, and a multi-agent mode where Π takes an `agent_index`. The models `brock_mirman_autodiff` and `bm_labor_autodiff` use it. It is a special case of the Lagrangian path below and delegates to it.
- `residuals_from_lagrangian` in `deqn_jax.training.lagrangian` adds constraints with multipliers, Fischer–Burmeister residuals for inequality constraints and prices taken as given. The model `olg_analytic_6_autodiff` uses it.
- Tests: `tests/test_autodiff_equations.py`, `tests/test_autodiff_multi_agent.py`, `tests/test_lagrangian_autodiff.py`, `tests/test_olg_analytic_6_autodiff.py`.

## Where Claude fits in

Simon's framing: researcher writes down the Lagrangian in something close to paper notation, Claude (or any LLM) transcribes it into the framework's `period_return` + state schema + dynamics. The mechanical part — autodiff FOCs, neural architecture search, loss reweighting, curriculum — is then framework work with no per-paper plumbing.

Concretely: a Claude-authored `bring_your_own_paper` tool would need

- a parse of the problem statement (state variables, controls, objective, constraints),
- translation into `period_return_fn` + `step_fn` + variable/shock schema,
- a round-trip check: autodiff residuals zero at a declared / solved steady state.

That last item is the "did I transcribe it right" gate. It's a cheap, automatic sanity check that catches most transcription errors without the user ever running training.

## Deriving every residual from a Lagrangian

`deqn_jax.training.lagrangian.residuals_from_lagrangian` builds a model's `equations_fn` from the period Lagrangian. The model author writes, for one sample,

- the period objective `F(state, x_next, policy, prices, constants)`, where `x_next` holds the next-period values of the endogenous states listed in `endogenous`;
- equality constraints `h(...) = 0` and inequality constraints `g(...) >= 0`, each as a `Constraint(name, fn, multiplier)` whose `multiplier` is the policy column that carries its Lagrange multiplier;
- optionally `prices_fn(state, policy, constants)`, the prices and aggregates the agents take as given;
- the discount factor, as a constants key, a number, or a function of the current exogenous state (calendar time, for example). A discount factor that depends on an endogenous state is not supported, because the derivative of future discounting is not part of the derived Euler.

The model's own `step_fn` is the law of motion. With `L = F + Σ λ_k h_k + Σ μ_k g_k`, the framework derives:

| Residual | Formula | Expectation |
|---|---|---|
| Euler, one per endogenous state `x_j` | `∂L_t/∂x'_j + β ∂L_{t+1}/∂x_j` | per-shock residual, averaged by the loss |
| static FOC, one per entry of `static_controls` | `−∂L_t/∂p_c`, holding `x'` fixed | none |
| equality constraint | `h` | none |
| inequality constraint | `FB(μ, g)`, the Fischer–Burmeister function from `models/_complementarity.py` | none |

The t+1 term is evaluated at the next state that the trainer computed for each shock, with `x_{t+2}` rebuilt by `step_fn` at zero shock. The loss module averages the per-shock residuals, which forms the conditional expectation. Prices enter `L` as an argument that is never differentiated, so each agent's first-order conditions hold prices fixed, as in a competitive equilibrium. Summing the objectives of all agents into one `F` then yields each agent's own conditions, provided every endogenous state and static control enters only its owner's terms. Any effect of one agent's choice on another agent has to pass through `prices_fn`; otherwise the summed derivative is a planner's first-order condition. Because the multipliers are policy outputs, the complementarity residuals involve period-t quantities only, and these models do not need the two-stage `inside_fn`/`combine_fn` loss that `olg_lifecycle` uses for a constraint on an expectation.

Two Euler forms are available. With `euler_form="raw"` the residual is `−(∂L_t/∂x'_j + β ∂L_{t+1}/∂x_j)` in units of the objective. With `euler_form="ratio"` (the default) it is divided by the period-t marginal cost `A_j = −∂(F + λ·h)/∂x'_j`, which gives `1 − (M_j + β ∂L_{t+1}/∂x_j)/A_j`, where `M_j` collects the inequality-multiplier terms. For consumption-saving problems `A_j` is the marginal utility of consumption, so the ratio form is the relative Euler error used by the hand-written OLG models. The divisor is a period-t quantity, so averaging over shocks still gives the expectation without a Jensen bias. The ratio form requires `A_j > 0` on the training support. Static FOCs and constraint residuals are returned in the units the author writes them in.

The period-return helper `euler_from_period_return` is the special case with one scalar capital state per agent, no prices, no constraints and the raw form, and it now delegates to `residuals_from_lagrangian`. `brock_mirman_autodiff` and `bm_labor_autodiff` produce bit-identical residuals before and after this change (`tests/test_lagrangian_autodiff.py`).

### Example: the 6-agent OLG

`olg_analytic_6_autodiff` is `olg_analytic_6` with its five Euler equations derived from the Lagrangian. The state is `(k², …, k⁶, η, δ)` and the policy is the vector of next-period holdings `k'^{h+1}`. The objective sums the six cohorts' utilities with the budget substituted in:

```python
def prices(state, policy, constants):
    alpha = constants["alpha"]
    K = jnp.sum(state[:5])
    eta, delta = state[5], state[6]
    r = alpha * eta * K ** (alpha - 1) + 1 - delta
    w = (1 - alpha) * eta * K ** alpha
    return {"r": r, "w": w}


def objective(state, k_next, policy, prices, constants):
    k = jnp.concatenate([jnp.zeros(1), state[:5]])          # k¹ = 0
    savings = jnp.concatenate([k_next, jnp.zeros(1)])       # the oldest cohort saves nothing
    labor = jnp.zeros(6).at[0].set(1.0)                     # only the youngest cohort works
    c = prices["r"] * k + prices["w"] * labor - savings
    return jnp.sum(u(c))


equations = residuals_from_lagrangian(
    objective, step, endogenous=range(5), euler_names=EQUATION_NAMES,
    n_shocks=2, prices_fn=prices, euler_form="ratio",
)
```

The holding `k'^{h+1}` appears in cohort h's consumption today and in cohort h+1's consumption tomorrow, so the derived condition is `1 − β r' u'(c'^{h+1})/u'(c^h) = 0`, the form the hand-written model uses. The utility `u` is continued linearly below `c = 10⁻³`, which reproduces the hand-written model's cap on `u'(c)` in infeasible regions. The tests in `tests/test_olg_analytic_6_autodiff.py` check that the derived residuals vanish at the Krueger–Kübler closed form on a grid of 972 states at every shock node, that they equal the hand-written residuals at random states and policies, and that two incorrect variants fail the closed-form check: one that differentiates through the prices and one with a perturbed discount factor.
