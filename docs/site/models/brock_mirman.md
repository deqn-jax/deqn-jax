# Brock-Mirman

The canonical RBC smoke test. Two states (capital, productivity), one
policy (savings rate), one equilibrium equation (Euler), one shock.
Analytical steady state.

## Use it

```bash
deqn-jax train brock_mirman -n 1000 --warm-start
```

Or via YAML: see [`configs/brock_mirman.yaml`](https://github.com/deqn-jax/deqn-jax/blob/master/configs/brock_mirman.yaml).

## Source

| File                                               | Purpose                                       |
|----------------------------------------------------|-----------------------------------------------|
| `models/brock_mirman/variables.py`                 | `SPEC`, `CONSTANTS`, `N_SHOCKS`, `DESCRIPTION`|
| `models/brock_mirman/equations.py`                 | Euler equation, definitions                   |
| `models/brock_mirman/dynamics.py`                  | State transition `step()`                     |
| `models/brock_mirman/steady_state.py`              | Analytical SS solver                          |

Use this as the template for new models — see [Implementing a model](implementing.md).
