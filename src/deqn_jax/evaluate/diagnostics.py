"""Accuracy diagnostics: Euler errors, market clearing, simulated moments, stability."""

from typing import Any, Dict, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from deqn_jax.evaluate.simulate import (
    _model_uses_discrete_chain,
    eval_p_disaster,
    eval_rollout,
)
from deqn_jax.training.loss import (
    QUADRATURE_EXPECTATION_TYPES,
    build_quadrature,
)


def _sim_seed_state(model, key):
    """Simulation seed for the ergodic diagnostics.

    Models with a closed-form steady state start there (the historical
    behavior). Models with ``steady_state_fn=None`` (olg_lifecycle family)
    start from one ``init_state_fn`` draw — the burn-in then carries the
    trajectory to the ergodic set. Returns ``(state0 [1, n_states],
    ss_state | None, ss_policy | None)``.
    """
    if model.steady_state_fn is not None:
        ss_state, ss_policy = model.steady_state_fn(model.constants)
        return ss_state[None, :], ss_state, ss_policy
    state0 = model.init_state_fn(key, 1, model.constants)
    return state0, None, None


# ---------------------------------------------------------------------------
# 1. Euler Equation Errors
# ---------------------------------------------------------------------------


def euler_equation_errors(
    policy_net: eqx.Module,
    model,
    n_periods: int = 10_000,
    seed: int = 123,
    burn_in: Optional[int] = None,
    expectation_type: str = "gauss_hermite",
    n_quadrature_points: Optional[int] = None,
) -> Dict[str, Any]:
    """Simulate a long stochastic path and compute Euler residuals everywhere.

    This is the gold standard for DEQN accuracy (Azinovic et al. 2022).
    Reports log10(|residual|) distribution.

    Args:
        policy_net: Trained policy network
        model: ModelSpec
        n_periods: Length of simulation (default: 10,000)
        seed: Random seed for shock draws
        burn_in: Discard first N periods (reach ergodic distribution). If
            None, uses min(500, n_periods // 5) so short simulations still
            produce some output.
        expectation_type: Expectation rule for two-stage models. Deterministic
            rules (`gauss_hermite`/`quadrature`/`gh`, `monomial`) are honored so
            a monomial-trained checkpoint is evaluated under its own operator;
            `mc` (the training default) keeps the evaluator's deterministic
            Gauss-Hermite default — replicating training-time Monte Carlo noise
            is a worse estimate of the same expectation, not parity.
        n_quadrature_points: Points per shock dimension for Gauss-Hermite. If
            omitted, the historical evaluator defaults apply (16 in one
            dimension, otherwise 3). Ignored by monomial.

    Returns:
        Dict with:
            "residuals": [n_periods - burn_in, n_equations] array of residuals
            "equation_names": list of equation names
            "states": [n_periods - burn_in, n_states] simulated states
    """
    if burn_in is None:
        burn_in = min(500, max(0, n_periods // 5))
    elif burn_in >= n_periods:
        # Guard against caller passing a too-large burn_in. Leave at least
        # one sample so downstream stack()/mean() don't crash.
        burn_in = max(0, n_periods - 1)
    constants = model.constants
    n_shocks = model.n_shocks

    key = jax.random.PRNGKey(seed)
    key, seed_key = jax.random.split(key)
    state, _ss_state, _ss_policy = _sim_seed_state(model, seed_key)  # [1, n_states]

    eq_names = list(model.equation_names) if model.equation_names else []

    # Detect disaster support once, outside JIT. If present, we draw
    # Bernoulli(p_disaster) each period so the ergodic accuracy report
    # actually visits disaster states -- without this, evaluating a
    # disaster-calibrated model produces a normal-shock-only path and
    # the reported accuracy excludes the disaster branch entirely.
    p_disaster = eval_p_disaster(model)

    # Detect discrete-chain support once. When set, residual reporting uses
    # exact enumeration over Π row at each visited state (not the single-
    # sample residual the Gaussian path produces) so the verifier's
    # expectation matches the trainer's.
    use_discrete = _model_uses_discrete_chain(model)
    if use_discrete:
        Π = jnp.asarray(model.transition_matrix)
        K = Π.shape[0]
        z_idx = int(model.z_state_idx)

    # JIT-compile the simulation step for speed
    @eqx.filter_jit
    def _sim_step_no_d(state, shock):
        policy = policy_net(state)  # pyright: ignore[reportCallIssue]  # ty: ignore[call-non-callable]
        if policy.ndim == 1:
            policy = policy[None, :]
        next_state = model.step_fn(state, policy, shock, constants)
        next_policy = policy_net(next_state)  # pyright: ignore[reportCallIssue]  # ty: ignore[call-non-callable]
        if next_policy.ndim == 1:
            next_policy = next_policy[None, :]
        residuals = model.equations_fn(
            state, policy, next_state, next_policy, constants
        )
        row = jnp.stack(
            [
                residuals[name][0] if residuals[name].ndim > 0 else residuals[name]
                for name in eq_names
            ]
        )
        return next_state, row, state[0]

    @eqx.filter_jit
    def _sim_step_with_d(state, shock, d_disaster):
        policy = policy_net(state)  # pyright: ignore[reportCallIssue]  # ty: ignore[call-non-callable]
        if policy.ndim == 1:
            policy = policy[None, :]
        next_state = model.step_fn(
            state, policy, shock, constants, d_disaster=d_disaster
        )
        next_policy = policy_net(next_state)  # pyright: ignore[reportCallIssue]  # ty: ignore[call-non-callable]
        if next_policy.ndim == 1:
            next_policy = next_policy[None, :]
        residuals = model.equations_fn(
            state, policy, next_state, next_policy, constants
        )
        row = jnp.stack(
            [
                residuals[name][0] if residuals[name].ndim > 0 else residuals[name]
                for name in eq_names
            ]
        )
        return next_state, row, state[0]

    @eqx.filter_jit
    def _sim_step_discrete(state, advance_shock):
        """Discrete-chain step.

        ``advance_shock`` is the categorical next-z used to roll trajectory
        forward. Residual is the *exact expectation* over the K possible
        next-z values weighted by Π[z_t]. Returned ``next_state`` is the
        rollout step (using ``advance_shock``).
        """
        policy = policy_net(state)  # pyright: ignore[reportCallIssue]  # ty: ignore[call-non-callable]
        if policy.ndim == 1:
            policy = policy[None, :]
        # Trajectory rollout: one categorical step
        next_state = model.step_fn(state, policy, advance_shock, constants)
        # Residual expectation: enumerate over all K candidate next-z
        current_z = state[:, z_idx].astype(jnp.int32)  # [1]

        def _residual_at_k(k):
            shock_k = jnp.array([k], dtype=jnp.int32)
            ns_k = model.step_fn(state, policy, shock_k, constants)
            np_k = policy_net(ns_k)
            if np_k.ndim == 1:
                np_k = np_k[None, :]
            r = model.equations_fn(state, policy, ns_k, np_k, constants)
            return jnp.stack(
                [r[name][0] if r[name].ndim > 0 else r[name] for name in eq_names]
            )

        all_r = jax.vmap(_residual_at_k)(jnp.arange(K, dtype=jnp.int32))  # [K, n_eq]
        # Π[z_t, :] weighting; current_z has shape [1] so take row 0
        weights = Π[current_z[0], :]  # [K]
        row = jnp.einsum("k,kn->n", weights, all_r)  # [n_eq]
        return next_state, row, state[0]

    # Two-stage models (inside_fn/combine_fn, e.g. the olg_lifecycle family):
    # the FB nonlinearity wraps an EXPECTATION, so a single-draw residual is
    # biased (E[fb] != fb(E)). Mirror the trainer's configured expectation of
    # inside first, then combine_fn — rollout still advances on a drawn shock.
    use_two_stage = (
        not use_discrete
        and p_disaster == 0.0
        and model.inside_fn is not None
        and model.combine_fn is not None
    )
    if use_two_stage:
        nodes_per_dim = n_quadrature_points
        if nodes_per_dim is None:
            nodes_per_dim = 16 if n_shocks == 1 else 3
        if expectation_type not in QUADRATURE_EXPECTATION_TYPES:
            # 'mc' and anything else non-deterministic: keep the evaluator's
            # own deterministic operator (see the docstring).
            expectation_type = "gauss_hermite"
        quadrature = build_quadrature(expectation_type, n_shocks, nodes_per_dim)
        if quadrature is None:
            # The Gauss-Hermite grid exceeds its safety cap; the linear-cost
            # degree-3 monomial rule is the deterministic fallback here (the
            # trainer falls back to Monte Carlo instead).
            quadrature = build_quadrature("monomial", n_shocks, nodes_per_dim)
        qn, qw = quadrature
        qn, qw = jnp.asarray(qn), jnp.asarray(qw)

        @eqx.filter_jit
        def _sim_step_two_stage(state, shock):
            policy = policy_net(state)  # pyright: ignore[reportCallIssue]  # ty: ignore[call-non-callable]
            if policy.ndim == 1:
                policy = policy[None, :]
            next_state = model.step_fn(state, policy, shock, constants)

            def _inside_at(node):
                ns = model.step_fn(state, policy, node[None, :], constants)
                npol = policy_net(ns)  # pyright: ignore[reportCallIssue]  # ty: ignore[call-non-callable]
                if npol.ndim == 1:
                    npol = npol[None, :]
                return model.inside_fn(state, policy, ns, npol, constants)

            ins_all = jax.vmap(_inside_at)(qn)  # dict of [n_nodes, 1]
            expectations = {
                k2: jnp.sum(qw[:, None] * v, axis=0) for k2, v in ins_all.items()
            }
            resid = model.combine_fn(state, policy, expectations, constants)
            row = jnp.stack(
                [
                    resid[name][0] if resid[name].ndim > 0 else resid[name]
                    for name in eq_names
                ]
            )
            return next_state, row, state[0]

    # Pick the per-step function once; the rollout loop itself lives in
    # evaluate/simulate.py (one loop for every eval primitive).
    if use_two_stage:

        def step(state, shock, _d):
            return _sim_step_two_stage(state, shock)

    elif use_discrete:

        def step(state, shock, _d):
            return _sim_step_discrete(state, shock)

    elif p_disaster > 0.0:

        def step(state, shock, d):
            return _sim_step_with_d(state, shock, d)

    else:

        def step(state, shock, _d):
            return _sim_step_no_d(state, shock)

    all_residuals = []
    all_states = []

    def record(t, outputs):
        _next_state, row, st = outputs
        if t >= burn_in:
            all_residuals.append(row)
            all_states.append(st)

    eval_rollout(
        model,
        state,
        key,
        n_periods,
        step,
        record,
    )

    residuals_array = jnp.stack(all_residuals)  # [T, n_eq]
    states_array = jnp.stack(all_states)  # [T, n_states]

    return {
        "residuals": residuals_array,
        "equation_names": eq_names,
        "states": states_array,
    }


def print_euler_errors(result: Dict, label: str = ""):
    """Print Euler equation error table in the standard format."""
    residuals = result["residuals"]  # [T, n_eq]
    eq_names = result["equation_names"]

    # log10(|residual|), clamp to avoid log(0)
    log_errors = jnp.log10(jnp.maximum(jnp.abs(residuals), 1e-20))

    header = "Euler Equation Errors (log10)"
    if label:
        header += f" — {label}"
    print(f"\n{header}")
    print("=" * 100)
    print(
        f"{'Equation':>30s}  {'Mean':>7s}  {'p50':>7s}  {'p95':>7s}  {'p99':>7s}  {'p99.9':>7s}  {'Max':>7s}  {'Grade':>12s}"
    )
    print("-" * 100)

    for i, name in enumerate(eq_names):
        col = log_errors[:, i]
        mean_val = float(jnp.mean(col))
        p50 = float(jnp.percentile(col, 50))
        p95 = float(jnp.percentile(col, 95))
        p99 = float(jnp.percentile(col, 99))
        p999 = float(jnp.percentile(col, 99.9))
        max_val = float(jnp.max(col))

        # Grade based on mean
        if mean_val < -4:
            grade = "Very good"
        elif mean_val < -3:
            grade = "Good"
        elif mean_val < -2:
            grade = "Acceptable"
        else:
            grade = "POOR"

        print(
            f"{name:>30s}  {mean_val:>7.2f}  {p50:>7.2f}  {p95:>7.2f}  "
            f"{p99:>7.2f}  {p999:>7.2f}  {max_val:>7.2f}  {grade:>12s}"
        )

    # Overall summary
    all_log = log_errors.flatten()
    print("-" * 100)
    print(
        f"{'OVERALL':>30s}  {float(jnp.mean(all_log)):>7.2f}  "
        f"{float(jnp.percentile(all_log, 50)):>7.2f}  "
        f"{float(jnp.percentile(all_log, 95)):>7.2f}  "
        f"{float(jnp.percentile(all_log, 99)):>7.2f}  "
        f"{float(jnp.percentile(all_log, 99.9)):>7.2f}  "
        f"{float(jnp.max(all_log)):>7.2f}"
    )
    print()

    # Interpretation
    overall_mean = float(jnp.mean(all_log))
    overall_max = float(jnp.max(all_log))
    print(
        f"  Mean log10 error: {overall_mean:.2f} → {10**overall_mean:.1e} "
        f"({'<0.1% Good' if overall_mean < -3 else '<1% Acceptable' if overall_mean < -2 else 'POOR >1%'})"
    )
    print(
        f"  Max  log10 error: {overall_max:.2f} → {10**overall_max:.1e} "
        f"({'<1% Good' if overall_max < -2 else '<10% Acceptable' if overall_max < -1 else 'POOR >10%'})"
    )


# ---------------------------------------------------------------------------
# 2. Market Clearing (resource constraint)
# ---------------------------------------------------------------------------


def market_clearing_errors(
    policy_net: eqx.Module,
    model,
    n_periods: int = 10_000,
    seed: int = 123,
    burn_in: int = 500,
    expectation_type: str = "gauss_hermite",
    n_quadrature_points: Optional[int] = None,
) -> Dict[str, Any]:
    """Check resource constraint satisfaction along simulated path.

    The resource equation is found by name (``"resource"`` in the equation
    name); e.g. the disaster model's ``eq9_resource_constraint``
    (Y = C + I + G + monitoring costs).

    Returns dict with ``equation``, ``mean_abs``, ``max_abs``, ``mean_log10``,
    ``max_log10`` — or ``{"error": ...}`` when no resource equation exists.
    """
    result = euler_equation_errors(
        policy_net,
        model,
        n_periods,
        seed,
        burn_in,
        expectation_type=expectation_type,
        n_quadrature_points=n_quadrature_points,
    )
    residuals = result["residuals"]
    eq_names = result["equation_names"]

    # Find resource constraint equation
    rc_idx = None
    for i, name in enumerate(eq_names):
        if "resource" in name.lower():
            rc_idx = i
            break

    if rc_idx is None:
        return {"error": "No resource constraint equation found"}

    rc_residuals = residuals[:, rc_idx]
    return {
        "equation": eq_names[rc_idx],
        "mean_abs": float(jnp.mean(jnp.abs(rc_residuals))),
        "max_abs": float(jnp.max(jnp.abs(rc_residuals))),
        "mean_log10": float(
            jnp.mean(jnp.log10(jnp.maximum(jnp.abs(rc_residuals), 1e-20)))
        ),
        "max_log10": float(
            jnp.max(jnp.log10(jnp.maximum(jnp.abs(rc_residuals), 1e-20)))
        ),
    }


# ---------------------------------------------------------------------------
# 3. Simulated Moments
# ---------------------------------------------------------------------------


def simulated_moments(
    policy_net: eqx.Module,
    model,
    n_periods: int = 10_000,
    seed: int = 123,
    burn_in: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute ergodic moments from long simulation.

    Returns moments for each state and policy variable:
    mean, std, min, max, and deviation from steady state.

    Args:
        burn_in: Discard first N periods. If None, auto-clamps to
            ``min(500, n_periods // 5)`` so short simulations still produce
            non-empty results.
    """
    if burn_in is None:
        burn_in = min(500, max(0, n_periods // 5))
    elif burn_in >= n_periods:
        burn_in = max(0, n_periods - 1)

    constants = model.constants
    key = jax.random.PRNGKey(seed)
    key, seed_key = jax.random.split(key)
    state, ss_state, ss_policy = _sim_seed_state(model, seed_key)

    state_names = list(model.state_names)
    policy_names = list(model.policy_names)

    # Disaster support: mirror euler_equation_errors so the ergodic moments
    # actually visit disaster states. Without this, a disaster-calibrated
    # model (p_disaster > 0) reports moments from a no-disaster-only path,
    # silently understating dispersion and tail behaviour.
    p_disaster = eval_p_disaster(model)

    @eqx.filter_jit
    def _sim_step(state, shock):
        policy = policy_net(state)  # pyright: ignore[reportCallIssue]  # ty: ignore[call-non-callable]
        if policy.ndim == 1:
            policy = policy[None, :]
        next_state = model.step_fn(state, policy, shock, constants)
        return next_state, state[0], policy[0]

    @eqx.filter_jit
    def _sim_step_with_d(state, shock, d_disaster):
        policy = policy_net(state)  # pyright: ignore[reportCallIssue]  # ty: ignore[call-non-callable]
        if policy.ndim == 1:
            policy = policy[None, :]
        next_state = model.step_fn(
            state, policy, shock, constants, d_disaster=d_disaster
        )
        return next_state, state[0], policy[0]

    if p_disaster > 0.0:

        def step(state, shock, d):
            return _sim_step_with_d(state, shock, d)

    else:

        def step(state, shock, _d):
            return _sim_step(state, shock)

    all_states = []
    all_policies = []

    def record(t, outputs):
        _next_state, st, pol = outputs
        if t >= burn_in:
            all_states.append(st)
            all_policies.append(pol)

    eval_rollout(
        model,
        state,
        key,
        n_periods,
        step,
        record,
    )

    states = jnp.stack(all_states)  # [T, n_states]
    policies = jnp.stack(all_policies)  # [T, n_policies]

    moments = {}
    for i, name in enumerate(state_names):
        col = states[:, i]
        ss_val = float(ss_state[i]) if ss_state is not None else float("nan")
        moments[name] = {
            "mean": float(jnp.mean(col)),
            "std": float(jnp.std(col)),
            "min": float(jnp.min(col)),
            "max": float(jnp.max(col)),
            "ss": ss_val,
            "mean_dev_pct": float((jnp.mean(col) - ss_val) / abs(ss_val) * 100)
            if abs(ss_val) > 0.01
            else 0.0,
        }

    for i, name in enumerate(policy_names):
        col = policies[:, i]
        ss_val = float(ss_policy[i]) if ss_policy is not None else float("nan")
        moments[name] = {
            "mean": float(jnp.mean(col)),
            "std": float(jnp.std(col)),
            "min": float(jnp.min(col)),
            "max": float(jnp.max(col)),
            "ss": ss_val,
            "mean_dev_pct": float((jnp.mean(col) - ss_val) / abs(ss_val) * 100)
            if abs(ss_val) > 0.01
            else 0.0,
        }

    return moments


def print_moments(
    moments: Dict[str, Dict[str, float]],
    label: str = "",
    n_periods: Optional[int] = None,
):
    """Print simulated moments table."""
    period_str = f"{n_periods:,}" if n_periods is not None else "simulated"
    header = f"Simulated Moments ({period_str} periods)"
    if label:
        header += f" — {label}"
    print(f"\n{header}")
    print("=" * 95)
    print(
        f"{'Variable':>20s}  {'SS':>8s}  {'Mean':>8s}  {'Std':>8s}  "
        f"{'Min':>8s}  {'Max':>8s}  {'Dev%':>7s}"
    )
    print("-" * 95)

    for name, m in moments.items():
        dev = m["mean_dev_pct"]
        flag = " !" if abs(dev) > 10 else "  " if abs(dev) > 5 else ""
        print(
            f"{name:>20s}  {m['ss']:>8.4f}  {m['mean']:>8.4f}  {m['std']:>8.4f}  "
            f"{m['min']:>8.4f}  {m['max']:>8.4f}  {dev:>+6.1f}%{flag}"
        )


# ---------------------------------------------------------------------------
# 4. Stability check — does the economy survive?
# ---------------------------------------------------------------------------


def stability_check(
    policy_net: eqx.Module,
    model,
    n_periods: int = 10_000,
    seed: int = 123,
) -> Dict[str, bool]:
    """Check if the simulated economy remains stable.

    Returns:
        ``nan_free`` (bool), ``bound_hit_pct`` (% of policy outputs within
        1e-4 of a bound), ``max_ss_deviation_pct`` (max relative state
        deviation from the reference — SS, or mid-trajectory for ergodic-only
        models), ``stable`` (nan_free and deviation < 500%).
    """
    constants = model.constants
    key = jax.random.PRNGKey(seed)
    key, seed_key = jax.random.split(key)
    state, ss_state, _ss_policy = _sim_seed_state(model, seed_key)
    # Divergence reference: the SS when the model has one. For ergodic-only
    # models the seed is an ARBITRARY init draw — the trajectory legitimately
    # travels far from it while converging to the ergodic set — so the
    # reference is instead the mid-simulation state (captured below): final
    # vs mid stays bounded for a stationary economy and explodes for a
    # divergent one.
    ref_state = ss_state
    mid_t = n_periods // 2

    policy_lower = model.policy_lower
    policy_upper = model.policy_upper

    # Bound-check margin — compute only for finite bounds so policies with
    # infinite upper (softplus-bounded) don't pollute the statistic.
    if policy_lower is not None and policy_upper is not None:
        finite = jnp.isfinite(policy_upper) & jnp.isfinite(policy_lower)
        span = jnp.where(finite, policy_upper - policy_lower, 0.0)
        margin = 0.01 * span
    else:
        margin = None
        finite = None

    # Disaster support: mirror euler_equation_errors / simulated_moments so the
    # stability check exercises disaster states. A disaster-calibrated model
    # that is stable only on the no-disaster branch would otherwise pass.
    p_disaster = eval_p_disaster(model)

    @eqx.filter_jit
    def _sim_step(state, shock):
        policy = policy_net(state)  # pyright: ignore[reportCallIssue]  # ty: ignore[call-non-callable]
        if policy.ndim == 1:
            policy = policy[None, :]
        next_state = model.step_fn(state, policy, shock, constants)
        return next_state, policy

    @eqx.filter_jit
    def _sim_step_with_d(state, shock, d_disaster):
        policy = policy_net(state)  # pyright: ignore[reportCallIssue]  # ty: ignore[call-non-callable]
        if policy.ndim == 1:
            policy = policy[None, :]
        next_state = model.step_fn(
            state, policy, shock, constants, d_disaster=d_disaster
        )
        return next_state, policy

    if p_disaster > 0.0:

        def step(state, shock, d):
            next_state, policy = _sim_step_with_d(state, shock, d)
            return next_state, policy, state

    else:

        def step(state, shock, _d):
            next_state, policy = _sim_step(state, shock)
            return next_state, policy, state

    bound_hits = 0
    total_outputs = 0
    has_nan = False

    def record(_t, outputs):
        nonlocal bound_hits, total_outputs, has_nan
        _next_state, policy, state = outputs

        # Check NaN
        if jnp.any(jnp.isnan(policy)) or jnp.any(jnp.isnan(state)):
            has_nan = True
            return True  # stop the rollout

        # Check bound hitting (within 1% of bounds) — only over policies
        # whose bounds are finite, so softplus-bounded (inf upper) policies
        # don't artificially inflate the count.
        if margin is not None and finite is not None:
            p = policy[0]
            lower_ok = finite & (p < policy_lower + margin)
            upper_ok = finite & (p > policy_upper - margin)
            bound_hits += int(jnp.sum(lower_ok) + jnp.sum(upper_ok))
            total_outputs += int(jnp.sum(finite))
        return False

    def after_step(t, state):
        nonlocal ref_state
        if ref_state is None and t == mid_t:
            ref_state = state[0]  # ergodic-only models: mid-trajectory reference

    state = eval_rollout(
        model,
        state,
        key,
        n_periods,
        step,
        record,
        after_step=after_step,
    )

    if ref_state is None:  # early NaN break before mid_t
        ref_state = state[0] if state.ndim == 2 else state

    # Check final state deviation from SS. Floor the normalisation at
    # 0.1 so states with SS = 0 (e.g. m_p, the monetary-policy shock) or
    # near zero don't produce spuriously large relative deviations from
    # tiny absolute moves.
    final_state = state[0] if state.ndim == 2 else state
    ss_dev = jnp.abs(final_state - ref_state) / jnp.maximum(jnp.abs(ref_state), 0.1)
    max_dev = float(jnp.max(ss_dev))

    bound_pct = bound_hits / max(total_outputs, 1) * 100

    return {
        "nan_free": not has_nan,
        "bound_hit_pct": bound_pct,
        "max_ss_deviation_pct": max_dev * 100,
        "stable": not has_nan and bound_pct < 20 and max_dev < 5,
    }
