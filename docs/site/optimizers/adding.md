# Adding an optimizer

1. Create `src/deqn_jax/optimizers/your_opt.py`.
2. Either return an `optax.GradientTransformation` or implement a
   custom class with `.init(params)` and `.update(...)` methods.
3. Register with `@register_optimizer("name", kind=OptimizerKind.STANDARD)`.
4. Import in `src/deqn_jax/optimizers/__init__.py` so registration runs.

```python
import optax
from deqn_jax.optimizers.registry import register_optimizer, OptimizerKind

@register_optimizer("your_opt", kind=OptimizerKind.STANDARD)
def your_opt_factory(config):
    return optax.chain(
        optax.scale_by_adam(),
        optax.scale(-config.learning_rate),
    )
```

## OptimizerKind

Choose the right kind for your optimizer's signature. Add a new kind
only if you genuinely need a new train-step variant.

| Kind     | Train-step signature                                            |
|----------|------------------------------------------------------------------|
| STANDARD | `opt.update(grads, opt_state, params)`                          |
| PCGRAD   | per-equation grads → projection → `opt.update(grads, ...)`      |
| MAO      | per-equation Jacobian → `opt.update(eq_jac, opt_state, params)` |
| LBFGS    | `opt.update(grads, opt_state, params, value=v, value_fn=f)`     |
| GN       | residual Jacobian → custom step                                 |

See `optimizers/registry.py` for the `make_train_step` dispatch.
