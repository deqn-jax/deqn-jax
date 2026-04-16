# Adding a new model

A model is four files in `src/deqn_jax/models/your_model/`:

| File              | Defines                                                           |
|-------------------|-------------------------------------------------------------------|
| `variables.py`    | `SPEC` (state/policy names), `CONSTANTS` dict, `N_SHOCKS`, `DESCRIPTION` |
| `equations.py`    | `equations(state, policy, next_state, next_policy, constants) -> Dict[str, Array]` |
| `dynamics.py`     | `step(state, policy, shock, constants) -> next_state`             |
| `steady_state.py` | `steady_state(constants) -> (ss_state, ss_policy)`                |

Then in `__init__.py` for your model, build a `ModelSpec`:

```python
from deqn_jax.types import ModelSpec
from deqn_jax.models.your_model.variables import SPEC, CONSTANTS, N_SHOCKS
from deqn_jax.models.your_model.equations import equations, EQUATION_NAMES
from deqn_jax.models.your_model.dynamics import step
from deqn_jax.models.your_model.steady_state import steady_state

MODEL = ModelSpec(
    name="your_model",
    n_states=len(SPEC.state_names),
    n_policies=len(SPEC.policy_names),
    n_shocks=N_SHOCKS,
    state_names=SPEC.state_names,
    policy_names=SPEC.policy_names,
    equation_names=EQUATION_NAMES,
    constants=CONSTANTS,
    equations_fn=equations,
    step_fn=step,
    steady_state_fn=steady_state,
)
```

Register the model in `src/deqn_jax/models/__init__.py`:

```python
from deqn_jax.models.your_model import MODEL as your_model_spec

_MODELS = {
    "brock_mirman": brock_mirman_spec,
    "disaster": disaster_spec,
    "your_model": your_model_spec,   # add here
}
```

Add a smoke test in `tests/test_basic.py`:

```python
def test_your_model_trains():
    config = TrainConfig(model="your_model", episodes=20, ...)
    params, history = train_from_config(config)
    assert history["loss"][-1] < history["loss"][0]
```

See `models/brock_mirman/` for the minimal reference and `models/disaster/`
for a full-scale DSGE.
