# Optimizers

Nine optimizers are registered. Each has an `OptimizerKind` (STANDARD, MAO,
LBFGS or GN); together with the PCGrad variant of STANDARD that gives five
train-step variants, chosen at construction time, before JIT. Each variant
has its own grad-step factory in `optimizers/<variant>.py`; the generic one is
`make_grad_step_standard`.

| Variant | Names | Step shape |
| --- | --- | --- |
| STANDARD | `adam`, `muon`, `ngd`, `shampoo` | `jax.grad → opt.update(grads, state, params)` |
| PCGRAD | (`gradient_surgery: pcgrad`) | Per-equation grads with conflict projection |
| MAO | `mao` | Per-equation Jacobian via `jax.jacrev` → MAO update |
| LBFGS | `lbfgs` | Optax LBFGS with line search (needs `value`, `grad`, `value_fn`) |
| GN | `gn`, `ign`, `lm` | Gauss-Newton / Levenberg-Marquardt: `Δθ = −(JᵀJ)⁻¹ Jᵀr` |

Optimizers register with the `@register_optimizer(name, kind)` decorator, in
`registry.py` or in their own module; `optimizers/__init__.py` imports every
module so registration runs. `create_optimizer(config)` looks the name up and,
for STANDARD optimizers with `grad_clip` set, chains
`optax.clip_by_global_norm` in front.

MAO uses `_MAOFactory` to defer resolving `n_tasks`, because the model's
equation count is known only at `create_train_state` time.

Composite loss is rejected with `mao`, `gn`, `ign` and `lm`
(`training.state_init._validate_train_config` enforces this): those update
paths differentiate only the base residuals, so the auxiliary terms would be
logged but would not reach the update. `lbfgs` and PCGrad work with composite
loss; PCGrad then requires unit `loss_weights` (or none).

To add an optimizer, see [Adding an optimizer](../optimizers/adding.md).

::: deqn_jax.optimizers.registry

::: deqn_jax.optimizers.ngd

::: deqn_jax.optimizers.mao

::: deqn_jax.optimizers.shampoo

::: deqn_jax.optimizers.lbfgs

::: deqn_jax.optimizers.gauss_newton
