"""BK pin (network.bk_pin): Blanchard-Kahn selection by construction.

The pin subtracts the MLP delta's value and tangent at s*, so the policy
level AND Jacobian at the steady state equal the BK linearization for
EVERY parameter value — the selection cannot be unlearned by training.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from deqn_jax.models import load_model  # noqa: E402
from deqn_jax.models.disaster.network import create_disaster_policy_net  # noqa: E402


def _nets():
    model = load_model("disaster")
    ss_state, _ = model.steady_state_fn(model.constants)
    key = jax.random.PRNGKey(7)
    # pinned net with a LARGE delta (init_scale far from 0)
    net_pin = create_disaster_policy_net(
        model, hidden_sizes=(16, 16), init_scale=0.5, bk_pin=True, key=key
    )
    # reference: same seed, no pin, zero delta = exact BK policy
    net_ref = create_disaster_policy_net(
        model, hidden_sizes=(16, 16), init_scale=0.0, bk_pin=False, key=key
    )
    return model, jnp.asarray(ss_state), net_pin, net_ref


def test_pin_exact_at_ss():
    _, ss, net_pin, net_ref = _nets()
    np.testing.assert_allclose(
        np.asarray(net_pin(ss[None, :])[0]),
        np.asarray(net_ref(ss[None, :])[0]),
        rtol=1e-10,
    )
    J_pin = jax.jacobian(lambda s: net_pin(s[None, :])[0])(ss)
    J_ref = jax.jacobian(lambda s: net_ref(s[None, :])[0])(ss)
    np.testing.assert_allclose(np.asarray(J_pin), np.asarray(J_ref), atol=1e-9)


def test_pin_expressive_off_ss():
    _, ss, net_pin, net_ref = _nets()
    s_off = ss * 1.05
    diff = float(jnp.max(jnp.abs(net_pin(s_off[None, :]) - net_ref(s_off[None, :]))))
    assert diff > 1e-6, f"pinned delta must still act away from SS, diff={diff}"


def test_pin_survives_parameter_perturbation():
    import equinox as eqx

    _, ss, net_pin, net_ref = _nets()
    # brutalize every MLP parameter (what training updates; the BK buffers
    # ss_state/ss_policy/P are stop-gradient'd constants) — the pin must
    # hold by construction
    mlp_arrays = eqx.filter(net_pin.mlp, eqx.is_array)
    shaken_mlp = jax.tree.map(lambda a: a + 0.1, mlp_arrays)
    net_shaken = eqx.tree_at(
        lambda n: n.mlp, net_pin, eqx.combine(shaken_mlp, net_pin.mlp)
    )
    np.testing.assert_allclose(
        np.asarray(net_shaken(ss[None, :])[0]),
        np.asarray(net_ref(ss[None, :])[0]),
        rtol=1e-10,
    )
    J_shaken = jax.jacobian(lambda s: net_shaken(s[None, :])[0])(ss)
    J_ref = jax.jacobian(lambda s: net_ref(s[None, :])[0])(ss)
    np.testing.assert_allclose(np.asarray(J_shaken), np.asarray(J_ref), atol=1e-9)


def test_gradient_flows_through_pin():
    import equinox as eqx

    _, ss, net_pin, _ = _nets()
    s_off = ss[None, :] * 1.03

    def loss(net):
        return jnp.sum(net(s_off) ** 2)

    grads = eqx.filter_grad(loss)(net_pin)
    gnorm = float(
        jnp.sqrt(
            sum(jnp.sum(g**2) for g in jax.tree.leaves(eqx.filter(grads, eqx.is_array)))
        )
    )
    assert gnorm > 1e-8, "pin must not block gradients off-SS"
