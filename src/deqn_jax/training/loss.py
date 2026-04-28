"""Loss computation with Monte Carlo or Gauss-Hermite quadrature expectations.

The DEQN loss is the mean squared residual of equilibrium equations:

    L = E_s[ E_ε[ r(s, π(s), s', π(s'))² ] ]

where the expectation is over:
1. States s drawn from episode trajectories
2. Shocks ε determining next state s' = step(s, π(s), ε)

Expectation methods:
- **MC**: Antithetic variates (pair each ε with -ε for variance reduction)
- **Quadrature**: Gauss-Hermite tensor-product nodes (exact for polynomial integrands)

Residual aggregation uses (E[r])² (average THEN square):
- Correct loss for E[r]=0 equilibrium conditions
- Robust to outlier residuals (averages first, tames singularities)
- With quadrature weights: weighted mean then square
"""

import inspect
import math
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from deqn_jax.types import ModelSpec

# ---------------------------------------------------------------------------
# Shock sampling: Monte Carlo
# ---------------------------------------------------------------------------


def sample_antithetic_shocks(
    key: Array,
    n_samples: int,
    batch_size: int,
    shock_dim: int,
    shock_scale: float | Array = 1.0,
) -> Array:
    """Generate Monte Carlo shocks with antithetic variates.

    Antithetic sampling pairs each shock ε with -ε, reducing variance
    for symmetric distributions (like standard normal).

    Args:
        key: JAX PRNG key
        n_samples: Number of MC samples (will be rounded to even)
        batch_size: Batch size
        shock_dim: Dimension of shock vector
        shock_scale: Curriculum scaling for shocks (0→1 ramp)

    Returns:
        Shocks array [n_samples, batch_size, shock_dim]
    """
    if n_samples <= 0 or shock_dim <= 0:
        return jnp.zeros((1, batch_size, shock_dim))

    half = n_samples // 2
    base = jax.random.normal(key, (half, batch_size, shock_dim))

    # Pair each shock with its antithetic twin
    shocks = jnp.concatenate([base, -base], axis=0)

    # Handle odd n_samples
    if n_samples % 2 == 1:
        key, subkey = jax.random.split(key)
        extra = jax.random.normal(subkey, (1, batch_size, shock_dim))
        shocks = jnp.concatenate([shocks, extra], axis=0)

    return shocks * shock_scale


# ---------------------------------------------------------------------------
# Shock sampling: Gauss-Hermite quadrature
# ---------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _hermgauss_1d(n_points: int):
    """Cached 1D Gauss-Hermite nodes/weights for exp(-x²)."""
    return np.polynomial.hermite.hermgauss(n_points)


def gauss_hermite_nd(
    n_points: int,
    dim: int,
    max_points: int = 4096,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Tensor-product Gauss-Hermite nodes/weights for standard normal.

    Transforms from Hermite basis (weight exp(-x²)) to standard normal:
    - Nodes: x' = sqrt(2) * x
    - Weights: w' = w / sqrt(π)

    Args:
        n_points: Quadrature points per dimension
        dim: Number of shock dimensions
        max_points: Safety cap on total grid points

    Returns:
        Tuple of (nodes [n_nodes, dim], weights [n_nodes]), or None if too many.
    """
    if dim <= 0 or n_points <= 0:
        return None

    n_nodes = n_points**dim
    if n_nodes > max_points:
        return None

    x, w = _hermgauss_1d(n_points)
    # Convert to standard normal: x' = sqrt(2)*x, w' = w/sqrt(pi)
    x = x * math.sqrt(2.0)
    w = w / math.sqrt(math.pi)

    if dim == 1:
        return x.reshape(-1, 1), w

    # Tensor product grid
    grids = np.array(np.meshgrid(*([x] * dim), indexing="ij"))
    nodes = grids.reshape(dim, -1).T  # [n_nodes, dim]
    w_grids = np.array(np.meshgrid(*([w] * dim), indexing="ij"))
    weights = np.prod(w_grids, axis=0).reshape(-1)  # [n_nodes]

    return nodes, weights


# ---------------------------------------------------------------------------
# Equations-signature introspection (actor-critic value passthrough)
# ---------------------------------------------------------------------------

# Optional kwargs the framework supplies to ``equations_fn`` when actor-critic
# is enabled and the model's equations function declares them. Lifting the
# tuple keeps the contract documented in one place.
_VALUE_KWARG_NAMES: Tuple[str, ...] = ("value_now", "value_next", "value_grad_next")


def equations_accepts_value(eq_fn: Callable) -> Tuple[str, ...]:
    """Which actor-critic value kwargs does ``eq_fn`` declare?

    Mirror of ``shocks.step_accepts_disaster``: introspects the function
    signature once at setup time (outside JIT). Returns a tuple of
    accepted kwarg names — empty tuple if none.

    The loss pipeline computes only the requested quantities and passes
    only those as keyword arguments to ``eq_fn``. This keeps backward
    compat for existing models (empty tuple → no value passing) and
    lets new models opt in to whichever subset they need (e.g. just
    ``value_now``/``value_next`` for an EZ Bellman without ever computing
    the autodiff gradient).

    Returns an empty tuple on functions whose signature can't be
    inspected — the equations contract requires a plain Python def.
    """
    try:
        params = inspect.signature(eq_fn).parameters
    except (ValueError, TypeError):
        return ()
    return tuple(name for name in _VALUE_KWARG_NAMES if name in params)


# ---------------------------------------------------------------------------
# Residual computation
# ---------------------------------------------------------------------------


def compute_residuals(
    model: ModelSpec,
    policy_fn: Callable[[Array], Array],
    train_batch: Array,
    shock: Array,
    target_policy_fn: Optional[Callable[[Array], Array]] = None,
    value_fn: Optional[Callable[[Array], Array]] = None,
    detach_value_in_policy_grad: bool = False,
) -> Dict[str, Array]:
    """Compute equilibrium equation residuals for a single shock realization.

    Handles both MLP [B, D] and sequence [B, H, D] inputs:
    - For [B, D]: standard MLP path, policy_fn(states)
    - For [B, H, D]: extract current state from last timestep,
      compute next_state, shift history window for next_policy

    The ndim check resolves at JAX trace time (no runtime branching).

    If target_policy_fn is provided (target network mode), next_policy is
    computed from the frozen target network with stop_gradient. This breaks
    the self-referential gradient loop where the network must simultaneously
    satisfy today's equations and be consistent with its own future outputs.

    If ``value_fn`` is provided (actor-critic) AND the model's equations
    function declares any of the value kwargs (``value_now``, ``value_next``,
    ``value_grad_next``), the framework computes V at the current and next
    states plus ∂V/∂s' and passes them in as keyword arguments. ``value_fn``
    must be a per-sample function ``[n_states] -> scalar`` so ``jax.grad``
    is well-defined.

    Args:
        model: Model specification
        policy_fn: Policy network (states -> policies) or (history -> policies)
        train_batch: Current states [batch, n_states] or history windows [batch, H, n_states]
        shock: Shock realization [batch, n_shocks]
        target_policy_fn: Frozen policy for next_policy (None = use policy_fn)
        value_fn: Per-sample value head [n_states] -> scalar (None = no critic)
        detach_value_in_policy_grad: If True, stop_gradient is applied to V
            and ∂V/∂s' before they are passed to equations(). Matches
            actor-critic "critic provides target" semantics.

    Returns:
        Dict mapping equation names to residuals [batch]
    """
    # Choose which function computes next_policy
    next_fn = target_policy_fn if target_policy_fn is not None else policy_fn

    # Resolve once outside the inner _branch (compile-time tuple). Empty
    # tuple = no AC value passing for this model; otherwise the named
    # subset is what the equations function accepts.
    accepted_value_kwargs: Tuple[str, ...] = (
        equations_accepts_value(model.equations_fn) if value_fn is not None else ()
    )

    # Disaster probability — discrete mixture over disaster realisation:
    # E_t[x'] = (1-p) E_t[x' | no disaster] + p E_t[x' | disaster]
    # We compute both branches and combine residuals at the end.
    # p_disaster = 0 (default) skips the disaster branch entirely so models
    # whose step_fn doesn't accept d_disaster still work.
    p_disaster = float(model.constants.get("p_disaster", 0.0))

    def _branch(d_disaster):
        """Compute equation residuals under a given disaster indicator.

        When d_disaster is None, call step_fn without the kwarg (baseline
        models like brock_mirman whose step_fn has no disaster path).
        """
        step_kwargs = {} if d_disaster is None else {"d_disaster": d_disaster}
        if train_batch.ndim == 3:
            states_local = train_batch[:, -1, :]
            policy_local = policy_fn(train_batch)
            next_state_local = model.step_fn(
                states_local,
                policy_local,
                shock,
                model.constants,
                **step_kwargs,
            )
            next_batch_local = jnp.concatenate(
                [train_batch[:, 1:, :], next_state_local[:, None, :]], axis=1
            )
            next_policy_local = next_fn(next_batch_local)
        else:
            states_local = train_batch
            policy_local = policy_fn(states_local)
            next_state_local = model.step_fn(
                states_local,
                policy_local,
                shock,
                model.constants,
                **step_kwargs,
            )
            next_policy_local = next_fn(next_state_local)
        if target_policy_fn is not None:
            next_policy_local = jax.lax.stop_gradient(next_policy_local)

        # Compute only the value quantities the equations function asks
        # for. The membership checks resolve at trace time (Python-level
        # tuple), so the unused branches are never traced by JAX.
        value_kwargs: Dict[str, Array] = {}
        if "value_now" in accepted_value_kwargs:
            v_now = jax.vmap(value_fn)(states_local)
            if detach_value_in_policy_grad:
                v_now = jax.lax.stop_gradient(v_now)
            value_kwargs["value_now"] = v_now
        if "value_next" in accepted_value_kwargs:
            v_next = jax.vmap(value_fn)(next_state_local)
            if detach_value_in_policy_grad:
                v_next = jax.lax.stop_gradient(v_next)
            value_kwargs["value_next"] = v_next
        if "value_grad_next" in accepted_value_kwargs:
            dv_next = jax.vmap(jax.grad(value_fn))(next_state_local)
            if detach_value_in_policy_grad:
                dv_next = jax.lax.stop_gradient(dv_next)
            value_kwargs["value_grad_next"] = dv_next

        return model.equations_fn(
            states_local,
            policy_local,
            next_state_local,
            next_policy_local,
            model.constants,
            **value_kwargs,
        )

    if p_disaster <= 0.0:
        # Call without d_disaster kwarg so baseline models work unchanged.
        return _branch(None)

    r_normal = _branch(jnp.array(0.0))
    r_disaster = _branch(jnp.array(1.0))
    return {
        k: (1.0 - p_disaster) * r_normal[k] + p_disaster * r_disaster[k]
        for k in r_normal
    }


# ---------------------------------------------------------------------------
# Loss computation (unified MC + quadrature)
# ---------------------------------------------------------------------------


def _make_value_fn(
    policy_fn: Callable[[Array], Array],
    aux_params: Optional[Any],
) -> Optional[Callable[[Array], Array]]:
    """Build a per-sample value function ``[n_states] -> scalar``.

    Three cases:

    * Separate-mode actor-critic (``aux_params`` is a callable critic
      module with scalar output): wrap to squeeze the trailing singleton
      so ``jax.grad`` is well-defined.
    * Shared-mode actor-critic (``policy_fn`` is an ``ActorCriticMLP``
      with a ``.value`` attribute): use it directly. ``ActorCriticMLP.value``
      already returns a scalar for ``[n_states]`` input.
    * No actor-critic: returns ``None`` and the caller skips value
      passthrough entirely (existing models train unchanged).
    """
    if aux_params is not None:

        def value_fn(state: Array) -> Array:
            # Critic MLP returns [1] for [n_states] input. Squeeze to scalar.
            return aux_params(state).squeeze()

        return value_fn
    if hasattr(policy_fn, "value") and callable(getattr(policy_fn, "value")):
        return policy_fn.value
    return None


def huber(x: Array, delta: float) -> Array:
    """Huber function: quadratic near 0, linear beyond |x| = delta.

    ``huber(x, δ) = 0.5·x²`` for ``|x| ≤ δ``
    ``huber(x, δ) = δ·(|x| - 0.5·δ)`` for ``|x| > δ``

    Matches DEQN_MAO's Huber_loss convention. Gradient saturates at
    ±δ for large residuals, which limits the influence of outlier
    batch elements on parameter updates — useful when a few ZLB-binding
    or extreme-shock states produce residuals ≫ typical.
    """
    abs_x = jnp.abs(x)
    return jnp.where(
        abs_x <= delta,
        0.5 * x**2,
        delta * (abs_x - 0.5 * delta),
    )


def compute_loss(
    model: ModelSpec,
    policy_fn: Callable[[Array], Array],
    states: Array,
    key: Array,
    mc_samples: int = 5,
    weights: Optional[Array] = None,
    shock_scale: float | Array = 1.0,
    quad_nodes: Optional[Array] = None,
    quad_weights: Optional[Array] = None,
    barrier_weight: float | Array = 0.0,
    target_policy_fn: Optional[Callable[[Array], Array]] = None,
    loss_choice: str = "mse",
    huber_delta: float | Array = 1.0,
    aux_params: Optional[Any] = None,
) -> Tuple[Array, Dict[str, Array]]:
    """Compute DEQN loss with MC or quadrature expectations.

    Aggregation: (E[r])² — square the weighted mean residual per batch element.
    This is the correct loss for E[r]=0 equilibrium conditions and is robust
    to outlier residuals (averages first, then squares).

    For MC:        shocks ~ N(0, shock_scale²), uniform weights 1/N
    For quadrature: shocks = nodes * shock_scale, Gauss-Hermite weights

    Handles both MLP [batch, n_states] and sequence [batch, H, n_states] inputs
    transparently (dispatched inside compute_residuals via ndim check).

    Args:
        model: Model specification
        policy_fn: Policy network (states -> policies) or (history -> policies)
        states: State batch [batch, n_states] or history windows [batch, H, n_states]
        key: PRNG key for MC shock sampling (ignored for quadrature)
        mc_samples: Number of Monte Carlo samples (ignored for quadrature)
        weights: Per-equation loss weights [n_eq] (default: uniform)
        shock_scale: Curriculum scaling for shocks (0→1 ramp)
        quad_nodes: Quadrature nodes [n_nodes, shock_dim] (None -> use MC)
        quad_weights: Quadrature weights [n_nodes] (None -> use MC)
        barrier_weight: Weight for state barrier penalty (0 = off)

    Returns:
        Tuple of (scalar loss, dict of per-equation losses)
    """
    batch_size = states.shape[0]
    use_quadrature = quad_nodes is not None and quad_weights is not None

    if use_quadrature:
        n_nodes = quad_nodes.shape[0]
        # Broadcast nodes to [n_nodes, batch_size, shock_dim] and apply curriculum
        shocks = (
            jnp.broadcast_to(
                quad_nodes[:, None, :],
                (n_nodes, batch_size, model.n_shocks),
            )
            * shock_scale
        )
        sample_weights = quad_weights  # [n_nodes]
    else:
        shocks = sample_antithetic_shocks(
            key,
            mc_samples,
            batch_size,
            model.n_shocks,
            shock_scale,
        )
        n_samples = shocks.shape[0]
        sample_weights = jnp.ones(n_samples) / n_samples  # uniform

    # Build the per-sample value function from the available state.
    # None unless the user has actor-critic enabled (either shared trunk
    # via ActorCriticMLP or separate critic in aux_params).
    value_fn = _make_value_fn(policy_fn, aux_params)

    # Compute residuals for each shock/node
    def compute_sample_residuals(shock):
        return compute_residuals(
            model,
            policy_fn,
            states,
            shock,
            target_policy_fn=target_policy_fn,
            value_fn=value_fn,
        )

    # vmap over samples/nodes: Dict[str, [n_samples, batch]]
    all_residuals = jax.vmap(compute_sample_residuals)(shocks)

    # (E[r])² aggregation: weighted mean over samples, then square
    eq_losses = {}
    total_loss = 0.0

    for i, (eq_name, residuals) in enumerate(all_residuals.items()):
        # residuals: [n_samples, batch]
        # Weighted mean over samples: E[r] for each batch element
        mean_residual = jnp.einsum("s,sb->b", sample_weights, residuals)  # [batch]
        # Aggregate per-state mean residual over batch. Huber is safe HERE
        # (after the shock expectation) because it only reshapes the
        # batch-level contribution. Applying it per-shock would break the
        # stochastic-fixed-point equivalence via Jensen, same as the
        # log-form rewrite we fixed earlier. Default "mse" matches prior
        # behaviour exactly.
        if loss_choice == "huber":
            eq_loss = jnp.mean(huber(mean_residual, huber_delta))
        else:
            eq_loss = jnp.mean(mean_residual**2)
        eq_losses[eq_name] = eq_loss
        w = 1.0 if weights is None else weights[i]
        total_loss = total_loss + w * eq_loss

    # Cross-equation aggregation: mean over equations (DEQN-MAO convention),
    # not sum. With `sum`, total loss and per-LR gradient scale grow linearly
    # in equation count, making LR calibrations non-transferable across
    # models of different sizes (brock_mirman=1, bm_labor=2, olg=5,
    # disaster=11). `mean` decouples the loss magnitude from equation count
    # so the same LR is roughly comparable across models. Note: this is a
    # convention change from an earlier sum-based aggregation; multi-
    # equation runs calibrated against the old convention may need their
    # LR multiplied by n_equations to recover the same effective per-eq
    # gradient magnitude.
    n_eq = len(all_residuals)
    if n_eq > 1:
        total_loss = total_loss / n_eq

    # State barrier: penalize next_states outside plausible bounds
    if barrier_weight > 0 and model.state_barrier_fn is not None:
        current_states = states[:, -1, :] if states.ndim == 3 else states
        # policy_fn needs the full input (history window or plain states)
        policy = policy_fn(states)
        zero_shock = jnp.zeros((batch_size, model.n_shocks))
        next_states = model.step_fn(current_states, policy, zero_shock, model.constants)
        barrier = jnp.mean(model.state_barrier_fn(next_states))
        total_loss = total_loss + barrier_weight * barrier
        eq_losses["aux_state_barrier"] = barrier

    # Declarative bound penalties (states and/or definitions). Matches the
    # DEQN-MAO ``penalty_bounds_policy`` soft-penalty pattern but driven
    # by per-variable specs on the model rather than hand-written code.
    if model.state_bounds or model.definition_bounds:
        current_states = states[:, -1, :] if states.ndim == 3 else states
        # Only evaluate policy/definitions if definition bounds are active,
        # to avoid a needless forward pass when only state bounds are set.
        policy_out = None
        defs_dict = None
        if model.definition_bounds and model.definitions_fn is not None:
            policy_out = policy_fn(states)
            defs_dict = model.definitions_fn(
                current_states, policy_out, model.constants
            )

        if model.state_bounds:
            state_vals = {
                name: current_states[:, i]
                for i, name in enumerate(model.state_names or ())
            }
            p_state = _compute_bound_penalty(state_vals, model.state_bounds)
            total_loss = total_loss + p_state
            eq_losses["aux_state_bounds"] = p_state

        if model.definition_bounds and defs_dict is not None:
            p_def = _compute_bound_penalty(defs_dict, model.definition_bounds)
            total_loss = total_loss + p_def
            eq_losses["aux_definition_bounds"] = p_def

    return total_loss, eq_losses


def _compute_bound_penalty(
    values: Dict[str, Array],
    bounds_spec: Dict[str, Dict[str, float]],
) -> Array:
    """Sum of soft-bound penalties across named variables.

    For each variable with a ``lower`` bound, adds
    ``penalty_lower * mean(relu(lower - value) ** 2)`` to the total.
    Analogous for ``upper``. Missing ``penalty_*`` defaults to
    ``1/bound**2`` (matching DEQN-MAO's policy_bounds_hard convention).

    Values whose names don't appear in ``bounds_spec`` are skipped.
    Values whose arrays are not scalar-per-batch are handled via
    ``jnp.mean`` over all axes.
    """
    penalty = jnp.array(0.0, dtype=jnp.float32)
    for name, spec in bounds_spec.items():
        if name not in values:
            continue
        v = values[name]
        # Promote to array in case a definition returned a python scalar.
        v = jnp.asarray(v)
        if "lower" in spec:
            lo = float(spec["lower"])
            p_lo = float(spec.get("penalty_lower", 1.0 / (lo * lo + 1e-30)))
            violation = jnp.maximum(0.0, lo - v)
            penalty = penalty + p_lo * jnp.mean(violation**2)
        if "upper" in spec:
            hi = float(spec["upper"])
            p_hi = float(spec.get("penalty_upper", 1.0 / (hi * hi + 1e-30)))
            violation = jnp.maximum(0.0, v - hi)
            penalty = penalty + p_hi * jnp.mean(violation**2)
    return penalty


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def eq_losses_to_array(eq_losses: Dict[str, Array]) -> Array:
    """Convert per-equation loss dict to stacked array [n_eq].

    Filters out aux_ prefixed keys so that adaptive reweighting
    (lr_annealing, relobralo) and per-equation gradient surgery
    (PCGrad, MAO) only see base equilibrium equation losses.
    """
    return jnp.stack([v for k, v in eq_losses.items() if not k.startswith("aux_")])


def compute_loss_for_grad(
    params,
    model: ModelSpec,
    states: Array,
    key: Array,
    mc_samples: int = 5,
) -> Array:
    """Loss function signature suitable for jax.grad."""
    loss, _ = compute_loss(model, params, states, key, mc_samples)
    return loss


def make_loss_fn(
    model: ModelSpec,
    mc_samples: int = 5,
) -> Callable:
    """Create a loss function closed over model spec.

    Returns a function (params, states, key) -> (loss, eq_losses)
    suitable for use with jax.value_and_grad.
    """

    def loss_fn(params, states: Array, key: Array):
        return compute_loss(model, params, states, key, mc_samples)

    return loss_fn
