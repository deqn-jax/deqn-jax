# Loss

Two loss paths. Switch via `TrainConfig.loss_type`.

## Base MSE (`compute_loss`)

`loss_type: "mse"` (default). Per batch element:

1. **Per-shock residuals** via `equations_fn`.
2. **Shock-expectation**: weighted mean across MC samples (uniform) or
   Gauss-Hermite nodes (Hermite weights).
3. **Square the mean**: `(E_shock[r])²` — MC-correct for E[r]=0
   conditions; biased under the Jensen-unsafe forms (see [the residual
   trap](../models/implementing.md#choosing-the-right-residual-form--this-is-a-trap)).
4. **Aggregate across batch**: mean (or Huber if `loss_choice="huber"`).
5. **Aggregate across equations**: mean (DEQN-MAO convention; not sum).

`compute_residuals` is the inner per-shock-realization helper;
`sample_antithetic_shocks` handles MC variance reduction;
`gauss_hermite_nd` constructs the quadrature grid (lru_cached).
Aux losses keyed `aux_*` are filtered out of adaptive reweighting via
`eq_losses_to_array`.

## Composite (`make_composite_loss`)

`loss_type: "composite"`. Adds anchor + Jacobian + barrier + Newton aux
terms layered on the base MSE. Pre-computes anchor sample points and
the Blanchard-Kahn `P` matrix at setup time; every term is logged
under its own `aux_*` key.

For the math, decay schedules, and configuration knobs see
[Composite loss](../training/composite_loss.md).

::: deqn_jax.training.loss

::: deqn_jax.training.composite_loss
