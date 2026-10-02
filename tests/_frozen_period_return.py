"""Frozen copy of ``euler_from_period_return`` before it was re-expressed
through ``residuals_from_lagrangian``.

Kept only as the reference for the bit-identity test in
``tests/test_lagrangian_autodiff.py``; the arithmetic is the original's,
argument validation is dropped.
"""

import jax
import jax.numpy as jnp


def frozen_euler_from_period_return(
    period_return_fn,
    step_fn,
    capital_idx=None,
    exog_idx=(1,),
    n_shocks=1,
    equation_name=None,
    intratemporal_policy_idx=(),
    intratemporal_equation_names=(),
    *,
    capital_indices=None,
    equation_names=None,
):
    exog_idx = tuple(exog_idx)
    intratemporal_policy_idx = tuple(intratemporal_policy_idx)
    intratemporal_equation_names = tuple(intratemporal_equation_names)
    if capital_indices is not None:
        capital_indices = tuple(int(i) for i in capital_indices)
        equation_names = tuple(equation_names)
        is_multi_agent = True
    else:
        capital_indices = (int(capital_idx or 0),)
        equation_names = (equation_name or "euler",)
        is_multi_agent = False
    if not intratemporal_equation_names:
        intratemporal_equation_names = tuple(
            f"intratemporal_j{j}" for j in intratemporal_policy_idx
        )

    def _grads(agent_idx):
        if is_multi_agent:

            def pi_i(K, K_next, z, policy, c):
                return period_return_fn(K, K_next, z, policy, c, agent_index=agent_idx)
        else:
            pi_i = period_return_fn
        dK_next = jax.vmap(jax.grad(pi_i, argnums=1), in_axes=(0, 0, 0, 0, None))
        dK = jax.vmap(jax.grad(pi_i, argnums=0), in_axes=(0, 0, 0, 0, None))
        return dK_next, dK

    grads = [_grads(i) for i in range(len(capital_indices))]

    if is_multi_agent:

        def pi_intra(K, K_next, z, policy, c):
            return period_return_fn(K, K_next, z, policy, c, agent_index=0)
    else:
        pi_intra = period_return_fn
    dPi_dp_v = jax.vmap(jax.grad(pi_intra, argnums=3), in_axes=(0, 0, 0, 0, None))

    def equations_fn(state, policy, next_state, next_policy, constants):
        z_t = jnp.take(state, jnp.asarray(exog_idx), axis=1)
        z_tp1 = jnp.take(next_state, jnp.asarray(exog_idx), axis=1)
        next_policy_frozen = jax.lax.stop_gradient(next_policy)
        zero_shock = jnp.zeros((state.shape[0], n_shocks))
        next_next_state = step_fn(next_state, next_policy_frozen, zero_shock, constants)
        out = {}
        for i, (cap, name) in enumerate(zip(capital_indices, equation_names)):
            dPi1 = grads[i][0](
                state[:, cap], next_state[:, cap], z_t, policy, constants
            )
            dPi2 = grads[i][1](
                next_state[:, cap],
                next_next_state[:, cap],
                z_tp1,
                next_policy_frozen,
                constants,
            )
            out[name] = -(dPi1 + constants["beta"] * dPi2)
        if intratemporal_policy_idx:
            cap0 = capital_indices[0]
            dPi_dp = dPi_dp_v(
                state[:, cap0], next_state[:, cap0], z_t, policy, constants
            )
            for j, name in zip(intratemporal_policy_idx, intratemporal_equation_names):
                out[name] = -dPi_dp[:, j]
        return out

    return equations_fn
