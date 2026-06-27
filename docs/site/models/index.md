# Models & the ModelSpec contract

A model is one object: a **`ModelSpec`**. You supply the states, the equilibrium
conditions in residual form, the transition law, the calibration, and a steady
state; the framework supplies the network, the solver, and the diagnostics. The
[Method Zoo](../method-zoo/index.md) parts plug into any conforming `ModelSpec`
unchanged.

```python
from deqn_jax.api import ModelSpec, register_model, TrainConfig, train_from_config

MODEL = ModelSpec(name="my_model", ...)   # states, equations, dynamics, SS, calibration
register_model(MODEL, description="my custom model")
state, history = train_from_config(TrainConfig(model="my_model", episodes=2000))
```

## Two ways in

- **Prose-first walkthrough** -- [Implementing a model](implementing.md) ports
  stochastic Brock-Mirman end to end (one Euler equation, two states, one
  shock): every moving part a larger model has, small enough to read in one
  sitting. Start here if you are hand-writing a model.
- **Type-signature-first contract** -- the
  [ModelSpec reference](../REFERENCE.md) is the complete `deqn_jax.api`
  surface: every field of `ModelSpec`, the programmatic `register_model(...)`
  path (no edits to the registry), config schema, and the evaluation /
  verification gates. This is the contract codegen and plugin packages target.

## Letting autodiff write the FOCs

For models where you would rather differentiate a payoff than hand-derive Euler
equations, the [autodiff path](../autodiff.md) synthesizes residuals from a
utility/payoff via `jax.grad` (POC models `brock_mirman_autodiff`,
`bm_labor_autodiff`).

## Models that ship in-tree

Ten are registered (`uv run deqn-jax list`); the [gallery](../gallery/index.md)
is the worked tour. Two carry full reference pages:

- [Brock-Mirman](brock_mirman.md) -- the canonical smoke test (1 eq, 2 states, 1 policy).
- [Disaster (NK-DSGE)](disaster.md) -- financial frictions and disaster risk (11 eq, 13 states, 11 policies).

The rest (deterministic and labor Brock-Mirman variants incl. two autodiff POCs,
two OLG models, the 2-country IRBC) are documented through their gallery
notebooks and the autodiff page.

!!! note "Maturity"
    deqn-jax is **alpha**. The `ModelSpec` fields surfaced through
    `deqn_jax.api` are the stable contract; everything imported from internal
    submodules may be refactored without notice.

## Automating it

If your model lives in a paper rather than your head, the
[deqn-agent stack](../ecosystem/deqn-agent.md) turns a `ModelSpec`-conforming
`model.py` (or a full paper) into a trained, verified policy.
