"""Shared eval-rollout helpers: discrete-chain detection + shock draw."""


import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Eval-time shock dispatcher (shared by all eval primitives)
# ---------------------------------------------------------------------------


def _model_uses_discrete_chain(model) -> bool:
    """True iff the model declares a discrete Markov-chain shock.

    Mirrors the trainer-side check in ``training/loss.py`` and
    ``training/shocks.py`` so the verifier visits the same support the
    trainer trained on.
    """
    return (
        getattr(model, "transition_matrix", None) is not None
        and getattr(model, "z_state_idx", None) is not None
    )


def _draw_eval_shock(model, key, state):
    """Draw one shock for a single-batch eval step.

    Continuous case: ``[1, n_shocks]`` Gaussian (legacy behavior).
    Discrete case:   ``[1]`` int32 sampled from ``Π[z_t]``, where
    ``z_t = state[:, z_state_idx]``. The shock IS the next-period
    categorical index — ``step_fn`` is responsible for embedding it
    into next-state.
    """
    if _model_uses_discrete_chain(model):
        from deqn_jax.training.shocks import draw_discrete_shocks

        z_idx = int(model.z_state_idx)
        current_z = state[:, z_idx].astype(jnp.int32)
        return draw_discrete_shocks(
            key, current_z, jnp.asarray(model.transition_matrix)
        )
    return jax.random.normal(key, (1, model.n_shocks))
