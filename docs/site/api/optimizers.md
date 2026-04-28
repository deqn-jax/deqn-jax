# Optimizers

13 built-in optimizers, dispatched into 5 families at construction time
(before JIT). Each family owns its own grad-step factory in
`optimizers/<family>.py`; the generic step is `make_grad_step_standard`.

| Family | Names | Step shape |
| --- | --- | --- |
| **STANDARD** | `adam`, `sgd`, `adamw`, `lion`, `muon`, `ngd`, `shampoo` | `jax.grad → opt.update(grads, state, params)` |
| **PCGRAD** | (`gradient_surgery: pcgrad`) | Per-equation grads with conflict projection |
| **MAO** | `mao`, `mao_kfac` | Per-equation Jacobian via `jax.jacrev` → MAO update |
| **LBFGS** | `lbfgs` | Optax LBFGS with line search (needs `value`, `grad`, `value_fn`) |
| **GN** | `gn`, `ign`, `lm` | Gauss-Newton / Levenberg-Marquardt: `Δθ = −(JᵀJ)⁻¹ Jᵀr` |

Registration uses the `@register_optimizer(name, kind)` decorator in
each module; `optimizers/__init__.py` imports every module to trigger
registration. `create_optimizer(config)` looks up the registry and
chains `optax.clip_by_global_norm` for STANDARD optimizers
automatically when `grad_clip` is set.

MAO uses `_MAOFactory` for deferred `n_tasks` resolution (the model's
equation count is known only at `create_train_state` time).

Composite loss is rejected with MAO/GN/IGN/LM/LBFGS and PCGrad
(`TrainConfig._validate_ranges` enforces this); on those paths the
optimizer's update doesn't see the auxiliary terms.

For adding a new optimizer, see [Adding an optimizer](../optimizers/adding.md).

::: deqn_jax.optimizers.registry

::: deqn_jax.optimizers.ngd

::: deqn_jax.optimizers.mao

::: deqn_jax.optimizers.shampoo

::: deqn_jax.optimizers.lbfgs

::: deqn_jax.optimizers.gauss_newton
