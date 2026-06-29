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

!!! note "Experimental research example"
    The disaster model is included as an **experimental research target** for
    reproduction and method development — not a validated or turnkey result.
    Treat its outputs accordingly.

## Calibrations

### Baseline (`p_disaster = 0`)

Plain CMR — no disaster code path activates. Configured in
[`configs/disaster.yaml`](https://github.com/deqn-jax/deqn-jax/blob/master/configs/disaster.yaml).

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

Example config: [`configs/disaster_pdis.yaml`](https://github.com/deqn-jax/deqn-jax/blob/master/configs/disaster_pdis.yaml).

## Training configuration

The disaster model is sensitive to the network and loss choice. The
configuration used here is:

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

## Calibration coupling

`xi_p = 0.6` is the price-stickiness value used here. Lowering it requires
recalibrating the rest of the Phillips block at the same time, and the `pi`
upper bound (above) is derived against this value — so any change to `xi_p`
must re-derive the bound.

## Aggregator residuals: ratio form, not log form

Residuals on the Calvo aggregator equations (`eq2b` and friends in
`models/disaster/equations.py`) are written in **ratio** form:

```python
residuals["eq2b"] = eq2_rhs / (p.K_p + eps) - 1.0
```

…rather than the log form:

```python
# DON'T DO THIS on aggregator equations
residuals["eq2b"] = log(eq2_rhs) - log(p.K_p)
```

Under stochastic averaging, the log form enforces the **geometric**
mean of the aggregator (Jensen's inequality), not the arithmetic mean
the equations actually call for. For small Gaussian shocks, the bias is
tiny and you'd never notice. For disaster jumps it's huge and silently
biases the solution.

**Don't switch back to log-form residuals on aggregator equations
without thinking through the Jensen implications.** The general principle
of "ratio residuals on aggregators under non-Gaussian shocks" applies to
any future model that mixes large jumps with multiplicative aggregation.
