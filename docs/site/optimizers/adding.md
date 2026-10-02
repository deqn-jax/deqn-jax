# Adding an optimizer

1. Create `src/deqn_jax/optimizers/your_opt.py`.
2. Return an `optax.GradientTransformation`, or implement a custom class with
   `.init(params)` and `.update(...)` methods.
3. Register it with `@register_optimizer("name", kind=OptimizerKind.STANDARD)`.
4. Import it in `src/deqn_jax/optimizers/__init__.py` so the registration runs.

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

Pick the kind that matches your optimizer's update signature. Add a new kind
only if you need a new train-step variant.

| Kind     | Train-step signature                                            |
|----------|------------------------------------------------------------------|
| STANDARD | `opt.update(grads, opt_state, params)`                          |
| PCGRAD   | per-equation grads → projection → `opt.update(grads, ...)`      |
| MAO      | per-equation Jacobian → `opt.update(eq_jac, opt_state, params)` |
| LBFGS    | `opt.update(grads, opt_state, params, value=v, value_fn=f)`     |
| GN       | residual Jacobian → custom step                                 |

PCGRAD is selected by `gradient_surgery: pcgrad` on a STANDARD optimizer, not
by a registered kind. The dispatch is `make_train_step` in
`src/deqn_jax/training/state_init.py`.
