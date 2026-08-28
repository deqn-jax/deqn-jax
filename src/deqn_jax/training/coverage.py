"""EWM coverage sampling: rollout helpers + a compute_loss wrapper.

Coverage imposes the existing residual loss on a mixture measure
(base + stress + local) instead of the base batch alone. State
generation is never differentiated (every pool is stop_gradient'd);
gradient flows only through the policy re-evaluated at the fixed pool
states inside compute_loss. Generated states are repaired (clipped)
into a feasible box before the residual is evaluated -- the repair
bounds the simulation only, never penalizes the residual.

Reference: Scheidegger & Schaab (2026), "Equilibrium World Models",
arXiv:2606.23463 -- the surrogate-free coverage arm.

Variant note (2026-08-27): two stress-seed modes. 'box' (historical, the
default): SS-filled seeds with stress dims uniform in a config box, raw seed
excluded from the pool — an OPERATOR-faithful variant whose stress MEASURE
differs from the paper's (non-stressed coordinates sit at SS, not at
realistic joint values). 'path' (the paper's measure): seeds are visited
batch states with only the stress dims overridden, and the seed joins the
pool. Results are labeled by mode; comparisons to the paper's coverage
numbers are exact only for 'path'.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.training.episode import run_episode
from deqn_jax.types import ModelSpec


def sample_stress_seeds(
    key: Array,
    n: int,
    n_states: int,
    ss_state: Array,
    stress_idx: Array,
    lows: Array,
    highs: Array,
) -> Array:
    """Draw n stress seeds: SS-filled, with stress dims uniform in their box.

    The historical 'box' mode. Note the joint-structure caveat in the module
    docstring: every non-stressed coordinate sits at its SS value.
    """
    seeds = jnp.broadcast_to(ss_state, (n, n_states))
    u = jax.random.uniform(key, (n, stress_idx.shape[0]), minval=lows, maxval=highs)
    return seeds.at[:, stress_idx].set(u)


def sample_stress_seeds_from_path(
    key: Array,
    n: int,
    batch_states: Array,
    stress_idx: Array,
    lows: Array,
    highs: Array,
) -> Array:
    """Draw n stress seeds the paper's way: visited batch states with ONLY the
    stress dims overridden by draws in their box — every other coordinate keeps
    its realistic joint value ("never sampled from a hypercube")."""
    k_idx, k_u = jax.random.split(key)
    idx = jax.random.randint(k_idx, (n,), 0, batch_states.shape[0])
    seeds = batch_states[idx]
    u = jax.random.uniform(k_u, (n, stress_idx.shape[0]), minval=lows, maxval=highs)
    return jax.lax.stop_gradient(seeds.at[:, stress_idx].set(u))


def roll_states(
    model: ModelSpec,
    policy_fn: Callable[[Array], Array],
    seeds: Array,
    key: Array,
    horizon: int,
    shock_scale=1.0,
    lo: Optional[Array] = None,
    hi: Optional[Array] = None,
    include_seed: bool = False,
) -> Array:
    """Roll seeds horizon steps through the EXACT transition; return the
    projected landings s_1..s_H, repaired (clipped to [lo, hi] when given),
    stop-gradient'd.

    include_seed=False (box mode): the raw seed is excluded — box seeds sit on
    an SS-slice and are not realistic joint states. include_seed=True (path
    mode): the (repaired) seed s_0 joins the pool, matching the paper, where
    seeds are visited states and belong to the coverage measure.

    Reuses run_episode (MC shocks even under quadrature: state generation
    is a different object from the loss expectation). trajectory holds
    pre-transition states, so trajectory[0] is the raw seed and final_state
    is s_H; landings = trajectory[1:] ++ final_state.
    """
    trajectory, final_state = run_episode(
        model, policy_fn, seeds, key, episode_length=horizon, shock_scale=shock_scale
    )
    start = 0 if include_seed else 1
    landings = jnp.concatenate([trajectory[start:], final_state[None]], axis=0)
    landings = landings.reshape(-1, model.n_states)
    if lo is not None:
        landings = jnp.clip(landings, lo, hi)
    return jax.lax.stop_gradient(landings)


def make_local_pool(
    states: Array,
    key: Array,
    n: int,
    sigma: float,
    lo: Optional[Array] = None,
    hi: Optional[Array] = None,
) -> Array:
    """n locally-perturbed copies of base states (sampled with replacement),
    plus Gaussian noise; repaired (clipped) when a box is given; stop-gradient'd."""
    k_idx, k_noise = jax.random.split(key)
    idx = jax.random.randint(k_idx, (n,), 0, states.shape[0])
    base = states[idx]
    noise = jax.random.normal(k_noise, base.shape) * sigma
    out = base + noise
    if lo is not None:
        out = jnp.clip(out, lo, hi)
    return jax.lax.stop_gradient(out)


def make_coverage_loss(
    base_compute_loss: Callable,
    model: ModelSpec,
    cfg,
    base_pool_loss: Optional[Callable] = None,
) -> Callable:
    """Wrap base_compute_loss to impose it on a mixture measure.

    Returns a fn with the exact compute_loss signature. The base pool is
    the batch passed in; stress + local pools are generated internally
    (repaired + stop-gradient'd). All expectation kwargs are forwarded to
    every pool so all pools share the identical operator (e.g. quadrature
    on irbc). Zero-weight pools are skipped at BUILD time (Python-level,
    pre-JIT), so rho_stress=rho_local=0 collapses to exactly the plain
    base_compute_loss (the paper's kappa=0 => DEQN identity).

    ``base_pool_loss`` (optional): a different loss for the BASE pool only —
    the EWM world arm scores the path pool with the surrogate Ŵ while the
    stress/local pools keep the exact expectation (``surrogate.exact_in_coverage``).
    Loss fns that declare an ``aux_params`` kwarg receive it.
    """
    import inspect

    _base_pool = base_pool_loss or base_compute_loss
    _aux_ok = {
        id(f): "aux_params" in inspect.signature(f).parameters
        for f in (base_compute_loss, _base_pool)
    }
    # Box-mode stress seeds are SS-filled, so they need a steady state; path
    # mode seeds from visited states and works for ergodic-only models too.
    _path_mode = getattr(cfg, "stress_seed_mode", "box") == "path"
    if model.steady_state_fn is not None:
        ss_state, _ = model.steady_state_fn(model.constants)
        ss_state = jnp.asarray(ss_state)
    elif _path_mode or not (cfg.rho_stress > 0 and cfg.n_stress > 0):
        ss_state = None
    else:
        raise ValueError(
            f"coverage stress_seed_mode='box' needs model.steady_state_fn, and "
            f"'{model.name}' has none; use stress_seed_mode='path'."
        )
    n_states = model.n_states
    name_to_idx = {n: i for i, n in enumerate(model.state_names)}

    use_stress = cfg.rho_stress > 0 and cfg.n_stress > 0
    use_local = cfg.rho_local > 0 and cfg.n_local > 0

    if use_stress:
        names = list(cfg.stress_ranges.keys())
        stress_idx = jnp.array([name_to_idx[n] for n in names], dtype=jnp.int32)
        lows = jnp.array([cfg.stress_ranges[n][0] for n in names])
        highs = jnp.array([cfg.stress_ranges[n][1] for n in names])

    # Repair box (the paper's clip-to-feasible step): +-inf where unspecified.
    if cfg.repair_ranges:
        lo = jnp.full(n_states, -jnp.inf)
        hi = jnp.full(n_states, jnp.inf)
        for nme, rng in cfg.repair_ranges.items():
            i = name_to_idx[nme]
            lo = lo.at[i].set(rng[0])
            hi = hi.at[i].set(rng[1])
    else:
        lo = hi = None

    # Mixture weights over the INCLUDED pools, normalized to sum to 1.
    norm = cfg.rho_base
    norm += cfg.rho_stress if use_stress else 0.0
    norm += cfg.rho_local if use_local else 0.0
    w_base = cfg.rho_base / norm
    w_stress = (cfg.rho_stress / norm) if use_stress else 0.0
    w_local = (cfg.rho_local / norm) if use_local else 0.0

    n_stress = int(cfg.n_stress)
    n_local = int(cfg.n_local)
    horizon = int(cfg.rollout_horizon)
    local_sigma = float(cfg.local_sigma)
    path_seeded = getattr(cfg, "stress_seed_mode", "box") == "path"

    def coverage_loss_fn(
        model_,
        policy_fn,
        states,
        key,
        mc_samples: int = 5,
        weights=None,
        shock_scale=1.0,
        quad_nodes=None,
        quad_weights=None,
        target_policy_fn=None,
        loss_choice: str = "mse",
        huber_delta: float = 1.0,
        aux_params=None,
    ):
        # One independent subkey per random consumer: the three pools' loss
        # expectations must not share a shock stream under MC (identical keys
        # would correlate gradient noise across mixture components; under
        # quadrature the loss keys are unused and this is a no-op).
        (
            k_base,
            k_stress_loss,
            k_local_loss,
            k_seed,
            k_roll,
            k_local,
        ) = jax.random.split(key, 6)

        def _loss(pool_states, k, fn=base_compute_loss):
            return fn(
                model_,
                policy_fn,
                pool_states,
                k,
                mc_samples,
                weights=weights,
                shock_scale=shock_scale,
                quad_nodes=quad_nodes,
                quad_weights=quad_weights,
                target_policy_fn=target_policy_fn,
                loss_choice=loss_choice,
                huber_delta=huber_delta,
                **({"aux_params": aux_params} if _aux_ok[id(fn)] else {}),
            )

        l_base, eq = _loss(states, k_base, _base_pool)
        total = w_base * l_base
        eq["aux_cov_base"] = l_base

        if use_stress:
            if path_seeded:
                seeds = sample_stress_seeds_from_path(
                    k_seed, n_stress, states, stress_idx, lows, highs
                )
            else:
                seeds = sample_stress_seeds(
                    k_seed, n_stress, n_states, ss_state, stress_idx, lows, highs
                )
            stress = roll_states(
                model_,
                policy_fn,
                seeds,
                k_roll,
                horizon,
                shock_scale,
                lo,
                hi,
                include_seed=path_seeded,
            )
            l_stress, _ = _loss(stress, k_stress_loss)
            total = total + w_stress * l_stress
            eq["aux_cov_stress"] = l_stress

        if use_local:
            local = make_local_pool(states, k_local, n_local, local_sigma, lo, hi)
            l_local, _ = _loss(local, k_local_loss)
            total = total + w_local * l_local
            eq["aux_cov_local"] = l_local

        return total, eq

    return coverage_loss_fn
