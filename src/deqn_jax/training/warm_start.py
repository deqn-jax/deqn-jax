"""Warm start: L-BFGS initialization from steady state.

Fits the policy network to match steady state policy before training.
This gives a much better starting point than random initialization.

L-BFGS is ideal here because:
1. Fitting to steady state is a deterministic supervised problem
2. Second-order methods excel at this (converges in ~10-50 steps vs 500+ Adam)
3. No mini-batching needed (full batch fits in memory)
"""

from typing import Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
import optax
from jax import Array

from deqn_jax.types import ModelSpec


def _init_output_bias_to_ss(
    policy_net: eqx.Module,
    ss_policy: Array,
    policy_lower: Optional[Array],
    policy_upper: Optional[Array],
) -> eqx.Module:
    """Set the output layer bias so the network initially outputs near SS.

    Inverts the bounding function to find the raw values that produce ss_policy,
    then sets the last layer's bias to those values and zeros the weights.
    This ensures the network starts outputting SS values regardless of input.
    """
    n_policies = ss_policy.shape[0]

    if policy_lower is None:
        # No bounding — raw output = ss_policy
        target_raw = ss_policy
    else:
        lo = policy_lower
        target_raw = jnp.zeros(n_policies)

        for i in range(n_policies):
            if policy_upper is not None and jnp.isfinite(policy_upper[i]):
                # Sigmoid bounded: output = lo + (hi - lo) * sigmoid(raw)
                # raw = logit((output - lo) / (hi - lo))
                hi = float(policy_upper[i])
                lo_i = float(lo[i])
                frac = (float(ss_policy[i]) - lo_i) / (hi - lo_i)
                frac = max(0.01, min(0.99, frac))  # clamp to avoid inf
                raw_i = float(jnp.log(frac / (1 - frac)))
            else:
                # Softplus bounded: output = lo + softplus(raw)
                # raw = softplus_inverse(output - lo) = log(exp(output - lo) - 1)
                diff = float(ss_policy[i]) - float(lo[i])
                diff = max(1e-4, diff)
                raw_i = float(jnp.log(jnp.exp(diff) - 1))
            target_raw = target_raw.at[i].set(raw_i)

    # Set the last layer's bias and zero its weights
    last_layer = policy_net.layers[-1]
    new_bias = target_raw
    new_weight = jnp.zeros_like(last_layer.weight)

    new_last = eqx.tree_at(
        lambda l: (l.weight, l.bias),
        last_layer,
        (new_weight, new_bias),
    )
    policy_net = eqx.tree_at(lambda net: net.layers[-1], policy_net, new_last)

    return policy_net


def _adam_minimize(
    loss_fn: Callable,
    init_params,
    n_steps: int = 2000,
    lr: float = 3e-3,
    verbose: bool = True,
) -> eqx.Module:
    """Run Adam optimization on an Equinox module.

    More robust than L-BFGS for networks with bounded outputs
    (softplus/sigmoid), which create loss landscapes where L-BFGS overshoots.
    """
    opt = optax.adam(lr)
    params_arrays = eqx.filter(init_params, eqx.is_array)
    opt_state = opt.init(params_arrays)

    @jax.jit
    def step(params, opt_state):
        params_arr = eqx.filter(params, eqx.is_array)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
        grads_arr = eqx.filter(grads, eqx.is_array)
        updates, new_opt_state = opt.update(grads_arr, opt_state, params_arr)
        new_params_arr = optax.apply_updates(params_arr, updates)
        new_params = eqx.combine(new_params_arr, params)
        return new_params, new_opt_state, loss

    params = init_params
    for i in range(n_steps):
        params, opt_state, loss = step(params, opt_state)
        if verbose and (i + 1) % 500 == 0:
            print(f"    Adam step {i + 1}/{n_steps}: loss={float(loss):.2e}")

    return params


def _lbfgs_minimize(
    loss_fn: Callable,
    init_params,
    max_iter: int = 100,
    tol: float = 1e-6,
    memory_size: int = 10,
) -> Tuple:
    """Run L-BFGS optimization on a pytree of parameters.

    Uses optax.lbfgs with a flat-parameter loop.

    Args:
        loss_fn: Scalar loss function taking the pytree params
        init_params: Initial parameter pytree
        max_iter: Maximum iterations
        tol: Convergence tolerance (on loss value)
        memory_size: L-BFGS history size

    Returns:
        Tuple of (optimized_params, n_iters, final_loss)
    """
    flat, unravel = jax.flatten_util.ravel_pytree(init_params)

    def flat_loss(x):
        return loss_fn(unravel(x))

    opt = optax.lbfgs(memory_size=memory_size)
    opt_state = opt.init(flat)

    @jax.jit
    def step(x, opt_state):
        val, g = jax.value_and_grad(flat_loss)(x)
        updates, new_opt_state = opt.update(
            g,
            opt_state,
            x,
            value=val,
            grad=g,
            value_fn=flat_loss,
        )
        new_x = optax.apply_updates(x, updates)
        return new_x, new_opt_state, val

    n_iters = 0
    for i in range(max_iter):
        flat, opt_state, val = step(flat, opt_state)
        n_iters = i + 1
        if float(val) < tol:
            break

    return unravel(flat), n_iters, float(val)


def warm_start_network(
    policy_net: eqx.Module,
    model: ModelSpec,
    n_points: int = 256,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = True,
    key: Optional[Array] = None,
    linearize: bool = False,
) -> eqx.Module:
    """Warm-start policy network to match steady state or linearized solution.

    If linearize=True, computes the first-order policy rule via Blanchard-Kahn
    and fits the network to that state-dependent mapping. Otherwise fits to
    constant steady-state policy.

    Args:
        policy_net: Equinox policy network to initialize
        model: Model specification (must have steady_state_fn)
        n_points: Number of fitting points
        max_iter: Maximum L-BFGS iterations
        tol: Convergence tolerance
        verbose: Print progress
        key: PRNG key (uses default if None)
        linearize: Use linearized (state-dependent) warm start

    Returns:
        Warm-started policy network
    """
    if model.steady_state_fn is None:
        if verbose:
            print("No steady_state_fn, skipping warm start")
        return policy_net

    if key is None:
        key = jax.random.PRNGKey(0)

    # Get steady state
    ss_state, ss_policy = model.steady_state_fn(model.constants)

    if verbose:
        print("Warm starting from steady state...")
        print(f"  SS state: {ss_state}")
        print(f"  SS policy: {ss_policy}")

    # Sample points around steady state
    noise = jax.random.uniform(key, (n_points, model.n_states), minval=-0.2, maxval=0.2)
    states = ss_state * (1 + noise)

    if linearize:
        # Compute linearized policy rule
        from deqn_jax.training.linearize import linear_policy_fn, linearize_model

        if verbose:
            print("  Computing linearized policy rule (Blanchard-Kahn)...")
        P, Q = linearize_model(model, verbose=verbose)
        lin_fn = linear_policy_fn(P, ss_state, ss_policy)

        # Target: linearized policy at each sampled state
        targets = jax.vmap(lin_fn)(states)

        # Clip targets to policy bounds (linear approx can go out of bounds)
        if model.policy_lower is not None:
            targets = jnp.maximum(targets, model.policy_lower + 1e-4)
        if model.policy_upper is not None:
            targets = jnp.minimum(targets, model.policy_upper - 1e-4)

        if verbose:
            # Show how much the linear policy varies
            target_std = jnp.std(targets, axis=0)
            print("  Linear policy std across samples:")
            names = model.policy_names or [f"p{i}" for i in range(model.n_policies)]
            for i, name in enumerate(names):
                print(f"    {name:>12s}: {float(target_std[i]):.6f}")
    else:
        # Target: constant steady state policy
        targets = jnp.tile(ss_policy, (n_points, 1))

    # Loss function: MSE between network output and targets
    def loss_fn(params):
        pred = jax.vmap(params)(states)
        return jnp.mean((pred - targets) ** 2)

    # Run L-BFGS optimization
    final_params, n_iters, final_loss = _lbfgs_minimize(
        loss_fn,
        policy_net,
        max_iter=max_iter,
        tol=tol,
    )

    if verbose:
        mode = "linearized" if linearize else "constant-SS"
        print(f"  Warm start ({mode}): loss={final_loss:.2e}, iters={n_iters}")

    return final_params


def warm_start_from_dynare(
    policy_net: eqx.Module,
    model: ModelSpec,
    dynare_dir: str,
    n_points: int = 1024,
    max_iter: int = 200,
    tol: float = 1e-8,
    verbose: bool = True,
    key: Optional[Array] = None,
) -> eqx.Module:
    """Warm-start from Dynare's first-order perturbation solution.

    Reads ghx (policy response to states) and ghu (response to shocks)
    from Dynare CSV output and constructs the linear policy function:
        policy(state) = policy_ss + J @ (state - state_ss)

    This gives the network correct SLOPES, not just the SS intercept.
    The economy stays near SS under this policy, so subsequent training
    sees states in a useful region.

    Args:
        policy_net: Equinox policy network to initialize
        model: ModelSpec (must have steady_state_fn)
        dynare_dir: Path to directory with dynare_ghx.csv, dynare_ghu.csv
        n_points: Number of fitting points
        max_iter: Maximum L-BFGS iterations
        tol: Convergence tolerance
        verbose: Print progress
        key: PRNG key

    Returns:
        Warm-started policy network
    """
    import csv
    from pathlib import Path

    if key is None:
        key = jax.random.PRNGKey(42)

    assert model.steady_state_fn is not None, (
        "warm_start_network requires a model with steady_state_fn defined"
    )
    ss_state, ss_policy = model.steady_state_fn(model.constants)
    constants = model.constants

    # --- Parse Dynare CSVs ---
    dynare_path = Path(dynare_dir)

    def read_csv_matrix(fname):
        with open(dynare_path / fname) as f:
            reader = csv.reader(f)
            header = next(reader)
            col_names = header[1:]  # skip 'variable'
            rows = {}
            for row in reader:
                rows[row[0]] = [float(x) for x in row[1:]]
        return col_names, rows

    ghx_cols, ghx_rows = read_csv_matrix("dynare_ghx.csv")
    ghu_cols, ghu_rows = read_csv_matrix("dynare_ghu.csv")

    # --- Build Jacobian: J[n_policies, 13 DEQN states] ---
    # DEQN state ordering
    deqn_state_names = list(model.state_names)

    # DEQN policy ordering (omega_bar eliminated analytically)
    deqn_policy_names = list(model.policy_names)

    # Map DEQN policy names to Dynare variable names.
    # Keep aliases for renamed variables; default to identity when possible.
    policy_aliases = {
        "i": "i_var",
    }

    # Map DEQN state names to Dynare state column names (lagged endogenous)
    # Dynare ghx columns: R(-1), w_tilda(-1), L(-1), k(-1), eps(-1),
    #   mu_ups(-1), g(-1), c(-1), i_var(-1), pi(-1), q(-1), mu_z(-1)
    state_col_map = {
        "pi_lag": "pi(-1)",
        "k_lag": "k(-1)",
        "c_lag": "c(-1)",
        "q_lag": "q(-1)",
        "i_lag": "i_var(-1)",
        "R_lag": "R(-1)",
        "w_tilda_lag": "w_tilda(-1)",
        "L_lag": "L(-1)",
        "eps": "eps(-1)",
        "mu_ups": "mu_ups(-1)",
        "g": "g(-1)",
        "mu_z": "mu_z(-1)",
    }

    n_policies = model.n_policies
    n_states = model.n_states
    J = jnp.zeros((n_policies, n_states))

    for pi, pname in enumerate(deqn_policy_names):
        dynare_var = policy_aliases.get(pname, pname)
        if dynare_var not in ghx_rows or dynare_var not in ghu_rows:
            available = sorted(set(ghx_rows.keys()) & set(ghu_rows.keys()))
            raise KeyError(
                f"Policy '{pname}' maps to Dynare variable '{dynare_var}', "
                f"but that row was not found in dynare_ghx/ghu.csv. "
                f"Available rows: {available}"
            )
        ghx_row = ghx_rows[dynare_var]
        ghu_row = ghu_rows[dynare_var]

        for si, sname in enumerate(deqn_state_names):
            if sname == "m_p":
                # m_p is i.i.d.: m_p = sigma_mp * e_mp
                # response to m_p_t = ghu_mp / sigma_mp * m_p_t
                sigma_mp = constants["sigma_mp"]
                mp_col = ghu_cols.index("e_mp")
                J = J.at[pi, si].set(ghu_row[mp_col] / sigma_mp)
            elif sname in state_col_map:
                ghx_col_name = state_col_map[sname]
                col_idx = ghx_cols.index(ghx_col_name)
                coeff = ghx_row[col_idx]

                # For exogenous states (eps, mu_ups, g, mu_z):
                # Dynare ghx has response to x_{t-1}, but DEQN state has x_t.
                # x_t ≈ x_ss + rho*(x_{t-1} - x_ss) + sigma*e
                # Treating deviation as persistent: x_{t-1} - x_ss ≈ (x_t - x_ss)
                # (ghx already gives the right order of magnitude)
                # No adjustment needed for endogenous lags.
                J = J.at[pi, si].set(coeff)

    if verbose:
        print("Warm starting from Dynare linear policy...")
        print(f"  Jacobian J: [{n_policies} x {n_states}]")
        print(f"  Max |J|: {float(jnp.max(jnp.abs(J))):.4f}")
        # Show per-policy response magnitude
        for pi, pname in enumerate(deqn_policy_names):
            row_norm = float(jnp.sqrt(jnp.sum(J[pi] ** 2)))
            print(f"    {pname:>12s}: ||J_row|| = {row_norm:.4f}")

    # --- Build linear target function ---
    def linear_policy(state):
        dev = state - ss_state
        policy = ss_policy + J @ dev
        # Clip to bounds (linear approx can exceed bounds)
        if model.policy_lower is not None:
            policy = jnp.maximum(policy, model.policy_lower + 1e-4)
        if model.policy_upper is not None:
            has_upper = jnp.isfinite(model.policy_upper)
            upper_clip = jnp.where(has_upper, model.policy_upper - 1e-4, policy)
            policy = jnp.where(has_upper, jnp.minimum(policy, upper_clip), policy)
        return policy

    # --- Step 0: Set output bias to SS (prevents bound-collapse) ---
    policy_net = _init_output_bias_to_ss(
        policy_net,
        ss_policy,
        model.policy_lower,
        model.policy_upper,
    )
    if verbose:
        test_out = policy_net(ss_state)  # pyright: ignore[reportCallIssue]  # ty: ignore[call-non-callable]
        ss_err = float(jnp.max(jnp.abs(test_out - ss_policy)))
        print(f"  After bias init: max SS error = {ss_err:.2e}")

    # --- Generate sample states ---
    # +/-10% range — wider causes linear policy to hit bounds
    key, sample_key = jax.random.split(key)
    noise = jax.random.uniform(
        sample_key, (n_points, n_states), minval=-0.1, maxval=0.1
    )
    states = ss_state * (1 + noise)

    # --- Compute targets ---
    targets = jax.vmap(linear_policy)(states)

    # --- Fit with Adam (L-BFGS fails with mixed softplus/sigmoid bounding) ---
    def loss_fn(params):
        pred = jax.vmap(params)(states)
        return jnp.mean((pred - targets) ** 2)

    policy_net = _adam_minimize(
        loss_fn,
        policy_net,
        n_steps=max_iter,
        lr=3e-3,
        verbose=verbose,
    )

    final_loss = float(loss_fn(policy_net))
    if verbose:
        print(f"  Warm start (Dynare): final loss={final_loss:.2e}")

    return policy_net


def warm_start_to_function(
    policy_net: eqx.Module,
    target_fn: Callable[[Array], Array],
    sample_states: Array,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = True,
) -> eqx.Module:
    """Warm-start policy network to match an arbitrary target function.

    More general version - fits network to match target_fn(state) for each state.

    Args:
        policy_net: Network to initialize
        target_fn: Target function (state -> policy)
        sample_states: States to fit on [n_points, n_states]
        max_iter: Maximum L-BFGS iterations
        tol: Convergence tolerance
        verbose: Print progress

    Returns:
        Warm-started network
    """
    # Compute targets
    targets = jax.vmap(target_fn)(sample_states)

    def loss_fn(params):
        pred = jax.vmap(params)(sample_states)
        return jnp.mean((pred - targets) ** 2)

    final_params, n_iters, final_loss = _lbfgs_minimize(
        loss_fn,
        policy_net,
        max_iter=max_iter,
        tol=tol,
    )

    if verbose:
        print(f"Warm start: loss={final_loss:.2e}, iters={n_iters}")

    return final_params
