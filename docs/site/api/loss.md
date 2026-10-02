# Loss

There are two loss paths, selected by `TrainConfig.loss_type`.

## Base MSE (`compute_loss`)

`loss_type: "mse"` (default). Per batch element:

1. Per-shock residuals from `equations_fn`.
2. Shock expectation: the weighted mean over MC samples (uniform weights) or
   Gauss-Hermite nodes (Hermite weights).
3. Square the mean: `(E_shock[r])²`. This is correct under MC for E[r]=0
   conditions and biased for the Jensen-unsafe forms (see [Choosing the residual
   form](../models/implementing.md#choosing-the-residual-form)).
4. Aggregate across the batch: mean (Huber if `loss_choice="huber"`).
5. Aggregate across equations: mean, not sum (the DEQN-MAO convention).

`loss_choice="aio"` replaces step 3 with the all-in-one estimator: the product
of two independent shock-group means, which is unbiased for `(E[r])²` under MC.

`compute_residuals` is the inner helper for one shock realization.
`sample_antithetic_shocks` handles MC variance reduction, and
`gauss_hermite_nd` builds the quadrature grid (cached with `lru_cache`).
Auxiliary losses keyed `aux_*` are excluded from adaptive reweighting by
`eq_losses_to_array`.

## Composite (`make_composite_loss`)

`loss_type: "composite"` adds anchor, Jacobian, barrier and Newton auxiliary
terms on top of the base MSE. The anchor sample points and the
Blanchard-Kahn `P` matrix are computed once at setup, and each term is logged
under its own `aux_*` key.

The math, decay schedules and configuration options are in
[Composite loss](../training/composite_loss.md).

::: deqn_jax.training.loss

::: deqn_jax.training.composite_loss
