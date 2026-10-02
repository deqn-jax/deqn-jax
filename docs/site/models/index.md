# Models & the ModelSpec contract

A model is one object, a `ModelSpec`. You supply the states, the equilibrium
conditions in residual form, the transition law, the calibration and a steady
state. The framework supplies the network, the solver and the diagnostics. The
[Method Zoo](../method-zoo/index.md) components work with any conforming
`ModelSpec` without changes.

```python
from deqn_jax.api import ModelSpec, register_model, TrainConfig, train_from_config

MODEL = ModelSpec(name="my_model", ...)   # states, equations, dynamics, SS, calibration
register_model(MODEL, description="my custom model")
state, history = train_from_config(TrainConfig(model="my_model", episodes=2000))
```

## Two ways in

- [Implementing a model](implementing.md) is a walkthrough that ports
  stochastic Brock-Mirman end to end (one Euler equation, two states, one
  shock). It has every part a larger model has and is short enough to read in
  one sitting. Start here if you are writing a model by hand.
- The [ModelSpec reference](../REFERENCE.md) lists the full `deqn_jax.api`
  surface by type signature: every field of `ModelSpec`, the programmatic
  `register_model(...)` path (no edits to the registry), the config schema,
  and the evaluation and verification gates. Code generators and plugin
  packages target this contract.

## Letting autodiff write the FOCs

If you would rather differentiate a payoff than derive Euler equations by
hand, the [autodiff path](../autodiff.md) builds the residuals from a utility
or payoff function with `jax.grad`. Proof-of-concept models:
`brock_mirman_autodiff`, `bm_labor_autodiff`.

## Models in the tree

Twelve models are registered (`uv run deqn-jax list`). The
[gallery](../gallery/index.md) walks through most of them. Two have their own
reference pages:

- [Brock-Mirman](brock_mirman.md): the standard smoke test (1 equation, 2 states, 1 policy).
- [Disaster (NK-DSGE)](disaster.md): financial frictions and disaster risk (11 equations, 13 states, 11 policies).

The deterministic and labor Brock-Mirman variants (including the two autodiff
proofs of concept), the 6-agent analytic OLG, the 6-generation life-cycle OLG
and the 2-country IRBC are covered by their gallery notebooks and the autodiff
page. `olg_lifecycle_56` (56 generations) and `rss_trade_ez_ref` (three-country
trade model) have no page yet; `deqn-jax info <model>` shows their details.

The `ModelSpec` fields exposed through `deqn_jax.api` are the stable contract.
Anything imported from internal submodules may change without notice.

## Automating it

If your model is written down in a paper, the
[deqn-agent stack](../ecosystem/deqn-agent.md) turns a `ModelSpec`-conforming
`model.py`, or a full paper, into a trained and verified policy.
