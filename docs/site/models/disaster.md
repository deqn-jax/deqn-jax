# Disaster (NK-DSGE with financial frictions)

Christiano-Motto-Rostagno (CMR)-style New Keynesian DSGE with banking
sector and an optional disaster block.

| Quantity        | Count |
|-----------------|------:|
| States          | 13    |
| Policies        | 11    |
| Equations       | 11    |
| Shocks          | 5     |
| Steady state    | numerical |

## Calibrations

### Baseline (`p_disaster = 0`)

Plain CMR — no disaster code path activates. Configured in
[`configs/disaster.yaml`](https://github.com/mechanicpanic/deqn-jax/blob/master/configs/disaster.yaml).

### Disaster risk (`p_disaster > 0`)

Discrete mixture over disaster realisations:

$$
\mathbb{E}_t[x'] = (1 - p)\,\mathbb{E}_t[x'\mid \text{no disaster}]
                 + p\,\mathbb{E}_t[x'\mid \text{disaster}]
$$

In disaster, capital is destroyed by factor $\exp(-\theta_{\text{disaster}})$.

When `p_disaster > 0`, the trainer automatically swaps to the
**risky steady state** (`risky_steady_state`) for composite-loss anchor
and Blanchard-Kahn linearization. This uses a Gourio-style
locally-flat policy approximation.

Example config: [`configs/disaster_pdis.yaml`](https://github.com/mechanicpanic/deqn-jax/blob/master/configs/disaster_pdis.yaml).

## Validated stack

The disaster model is sensitive — plain MLP + bare residual loss finds
degenerate self-referential fixed points. The **validated stack** is:

- Network: `LinearPlusMLP` (residual over Blanchard-Kahn linearization)
- Loss: `composite` (anchor + Jacobian + barrier + Newton)
- Expectations: Gauss-Hermite quadrature, 3 points per shock
- Optimizer: Adam with cosine LR schedule

See [Composite loss](../training/composite_loss.md) for why this matters.

## Calvo validity edge

The price-dispersion formula

$$
K_p^{inner} = \frac{1 - \xi_p (\pi_{\text{tilda}}/\pi)^{-5}}{1 - \xi_p}
$$

requires $\pi < \sim 1.1\,\pi_{\text{tilda}}$ for $K_p^{inner} > 0$.
With `xi_p = 0.6` and `lambda_f = 1.2`, the policy `pi` upper bound
is **pinned** at the Calvo validity edge — widening it triggers
gradient explosions through the soft floor at 0.01.

See `models/disaster/variables.py` for the bound spec and rationale.
