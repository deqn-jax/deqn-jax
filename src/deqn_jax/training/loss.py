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

import math
from functools import lru_cache
from typing import Callable, Dict, Optional, Tuple

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
    # Split up front so `base` and the odd "extra" sample draw from independent
    # subkeys, never the parent key (audit JAX-SILENT-07). Previously `base`
    # used the parent key and `extra` used a split-child of it; JAX's PRNG
    # independence guarantee is between split children, not parent-vs-child.
    base_key, extra_key = jax.random.split(key)
    base = jax.random.normal(base_key, (half, batch_size, shock_dim))

    # Pair each shock with its antithetic twin
    shocks = jnp.concatenate([base, -base], axis=0)

    # Handle odd n_samples
    if n_samples % 2 == 1:
        extra = jax.random.normal(extra_key, (1, batch_size, shock_dim))
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
# Residual computation
# ---------------------------------------------------------------------------


def compute_residuals(
    model: ModelSpec,
    policy_fn: Callable[[Array], Array],
    train_batch: Array,
    shock: Array,
    target_policy_fn: Optional[Callable[[Array], Array]] = None,
    residual_fn: Optional[Callable[..., Dict[str, Array]]] = None,
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

    Args:
        model: Model specification
        policy_fn: Policy network (states -> policies) or (history -> policies)
        train_batch: Current states [batch, n_states] or history windows [batch, H, n_states]
        shock: Shock realization [batch, n_shocks]
        target_policy_fn: Frozen policy for next_policy (None = use policy_fn)

    Returns:
        Dict mapping equation names to residuals [batch]
    """
    # Choose which function computes next_policy
    next_fn = target_policy_fn if target_policy_fn is not None else policy_fn

    # Two-stage models pass their `inside_fn` here (the shock-dependent terms to
    # be expectation-averaged); standard models default to `equations_fn`.
    resid_fn = residual_fn if residual_fn is not None else model.equations_fn

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
        return resid_fn(
            states_local,
            policy_local,
            next_state_local,
            next_policy_local,
            model.constants,
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
    transition_matrix = getattr(model, "transition_matrix", None)
    z_state_idx = getattr(model, "z_state_idx", None)
    use_discrete = transition_matrix is not None and z_state_idx is not None

    # All-in-one (AiO) estimator (Maliar-Maliar-Winant 2021): replaces the
    # biased (sample mean)^2 aggregation with the product of TWO independent
    # group means, which is exactly unbiased for (E[r])^2. Only meaningful
    # under MC: the quadrature/discrete paths weight nodes exactly and have
    # no stochastic aggregation bias to remove.
    use_aio = loss_choice == "aio"
    if use_aio and (use_quadrature or use_discrete):
        raise ValueError(
            "loss_choice='aio' requires Monte Carlo expectations; the "
            "quadrature/discrete paths are exact and have no MC bias for "
            "aio to remove. Use expectation_type='mc' or loss_choice='mse'."
        )
    if use_aio and mc_samples < 2:
        raise ValueError(
            "loss_choice='aio' needs mc_samples >= 2 to form two "
            f"independent shock groups, got {mc_samples}."
        )

    # Sample weights are always [n_samples, batch] post-construction so the
    # discrete branch's per-batch weights and the GH/MC branches' broadcast
    # weights share the same einsum aggregation downstream.
    n_aio = 0  # group-1 size when loss_choice='aio' (0 otherwise)
    if use_aio:
        # Two INDEPENDENT antithetic groups: independence between the groups
        # is what makes E[rbar_1 * rbar_2] = (E[r])^2 hold exactly. A single
        # antithetic stream split in half would correlate the halves
        # (mirrored pairs) and reintroduce bias.
        key1, key2 = jax.random.split(key)
        n2 = mc_samples // 2
        n1 = mc_samples - n2
        shocks = jnp.concatenate(
            [
                sample_antithetic_shocks(
                    key1, n1, batch_size, model.n_shocks, shock_scale
                ),
                sample_antithetic_shocks(
                    key2, n2, batch_size, model.n_shocks, shock_scale
                ),
            ],
            axis=0,
        )
        n_aio = n1
        n_samples = shocks.shape[0]
        sample_weights = jnp.broadcast_to(
            (jnp.ones(n_samples) / n_samples)[:, None], (n_samples, batch_size)
        )
    elif use_discrete:
        Π = jnp.asarray(transition_matrix)
        K = Π.shape[0]
        # Read current z-index per batch element from state (works for both
        # MLP states [batch, n_states] and history windows [batch, H,
        # n_states] — for the latter, the policy is evaluated against the
        # most-recent slice).
        cur = states[:, -1, :] if states.ndim == 3 else states
        current_z = cur[:, int(z_state_idx)].astype(jnp.int32)  # [batch]
        # Enumerate next-z values: shocks[k, b] = k for all b. step_fn
        # treats the integer shock as the next-period categorical index.
        shocks = jnp.broadcast_to(
            jnp.arange(K, dtype=jnp.int32)[:, None], (K, batch_size)
        )
        # Per-batch weights: weights[k, b] = Π[current_z[b], k]
        sample_weights = Π[current_z, :].T  # [K, batch]
    elif use_quadrature:
        n_nodes = quad_nodes.shape[0]
        # Broadcast nodes to [n_nodes, batch_size, shock_dim] and apply curriculum
        shocks = (
            jnp.broadcast_to(
                quad_nodes[:, None, :],
                (n_nodes, batch_size, model.n_shocks),
            )
            * shock_scale
        )
        # Lift uniform-over-batch weights to [n_nodes, batch] via broadcast.
        sample_weights = jnp.broadcast_to(quad_weights[:, None], (n_nodes, batch_size))
    else:
        shocks = sample_antithetic_shocks(
            key,
            mc_samples,
            batch_size,
            model.n_shocks,
            shock_scale,
        )
        n_samples = shocks.shape[0]
        sample_weights = jnp.broadcast_to(
            (jnp.ones(n_samples) / n_samples)[:, None], (n_samples, batch_size)
        )

    # Two-stage (expectation-inside-residual) models compute their `inside_fn`
    # terms per shock; standard models compute the residual directly.
    _two_stage = model.combine_fn is not None
    _sample_fn = model.inside_fn if _two_stage else None

    # Compute residuals (or inside terms) for each shock/node
    def compute_sample_residuals(shock):
        return compute_residuals(
            model,
            policy_fn,
            states,
            shock,
            target_policy_fn=target_policy_fn,
            residual_fn=_sample_fn,
        )

    # vmap over samples/nodes: Dict[str, [n_samples, batch]]
    all_residuals = jax.vmap(compute_sample_residuals)(shocks)

    # Aggregate per equation. Two paths, BRANCHED so the standard path is the
    # original code verbatim -- identical XLA graph, hence bit-identical results.
    # (A restructure that merely *looks* equivalent can shift the last bit via
    # different fusion and bifurcate a chaotic run like disaster.) Cross-equation
    # aggregation is the mean over equations (DEQN-MAO convention): the loss
    # magnitude is decoupled from equation count so one LR transfers across
    # model sizes (brock_mirman=1, bm_labor=2, olg=5, disaster=11).
    if use_aio:
        # AiO aggregation: per equation, average each independent group
        # separately and take the batch mean of the PRODUCT of group means.
        # Unbiased for (E[r])^2; the loss (and per-eq losses) can be
        # transiently negative -- that is sampling noise around a
        # non-negative population value, not an error. On the two-stage
        # path the combine_fn is applied per group BEFORE the product,
        # which removes the outer squaring bias (the Jensen bias of a
        # nonlinear combine_fn itself remains O(1/mc_samples), as on the
        # mse path).
        if _two_stage:
            e1 = {k: jnp.mean(v[:n_aio], axis=0) for k, v in all_residuals.items()}
            e2 = {k: jnp.mean(v[n_aio:], axis=0) for k, v in all_residuals.items()}
            cur_states = states[:, -1, :] if states.ndim == 3 else states
            pol = policy_fn(states)
            r1 = model.combine_fn(cur_states, pol, e1, model.constants)
            r2 = model.combine_fn(cur_states, pol, e2, model.constants)
            group_means = {k: (r1[k], r2[k]) for k in r1}
        else:
            group_means = {
                k: (jnp.mean(v[:n_aio], axis=0), jnp.mean(v[n_aio:], axis=0))
                for k, v in all_residuals.items()
            }

        eq_losses = {}
        total_loss = 0.0
        for i, (eq_name, (m1, m2)) in enumerate(group_means.items()):
            eq_loss = jnp.mean(m1 * m2)
            eq_losses[eq_name] = eq_loss
            w = 1.0 if weights is None else weights[i]
            total_loss = total_loss + w * eq_loss
        n_eq = len(group_means)
        if n_eq > 1:
            total_loss = total_loss / n_eq
    elif _two_stage:
        # Average the inside terms -> E[inside], then apply the nonlinear
        # combine_fn (the expectation lives INSIDE the residual). MC-correct for
        # e.g. a Fischer-Burmeister constraint on an intertemporal Euler, where
        # E[fb(.)] != fb(E[.]).
        expectations = {
            k: jnp.einsum("sb,sb->b", sample_weights, v)
            for k, v in all_residuals.items()
        }
        cur_states = states[:, -1, :] if states.ndim == 3 else states
        residuals_by_eq = model.combine_fn(
            cur_states, policy_fn(states), expectations, model.constants
        )
        eq_losses = {}
        total_loss = 0.0
        for i, (eq_name, mean_residual) in enumerate(residuals_by_eq.items()):
            if loss_choice == "huber":
                eq_loss = jnp.mean(huber(mean_residual, huber_delta))
            else:
                eq_loss = jnp.mean(mean_residual**2)
            eq_losses[eq_name] = eq_loss
            w = 1.0 if weights is None else weights[i]
            total_loss = total_loss + w * eq_loss
        n_eq = len(residuals_by_eq)
        if n_eq > 1:
            total_loss = total_loss / n_eq
    else:
        # Standard (E[residual])^2 path -- VERBATIM original code so the XLA
        # graph (and thus the result to the last bit) is unchanged.
        eq_losses = {}
        total_loss = 0.0
        for i, (eq_name, residuals) in enumerate(all_residuals.items()):
            mean_residual = jnp.einsum("sb,sb->b", sample_weights, residuals)
            if loss_choice == "huber":
                eq_loss = jnp.mean(huber(mean_residual, huber_delta))
            else:
                eq_loss = jnp.mean(mean_residual**2)
            eq_losses[eq_name] = eq_loss
            w = 1.0 if weights is None else weights[i]
            total_loss = total_loss + w * eq_loss
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
