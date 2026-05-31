"""Tests for per-policy output_links (linear vs log) on residual networks."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

jax.config.update("jax_enable_x64", True)

from deqn_jax.models import load_model
from deqn_jax.models.disaster.network import create_disaster_policy_net
from deqn_jax.models.disaster.steady_state import _ss_cache
from deqn_jax.networks.linear_plus_mlp import create_linear_plus_mlp


@pytest.fixture(autouse=True)
def _clear_disaster_ss_cache():
    _ss_cache.clear()
    yield
    _ss_cache.clear()


# ---------------------------------------------------------------------------
# Generic LinearPlusMLP
# ---------------------------------------------------------------------------


def test_linear_plus_mlp_default_unchanged():
    """No output_links arg → all-linear behavior, byte-equivalent to legacy."""
    m = load_model("disaster")
    s_ss, ss_policy = m.steady_state_fn(m.constants)
    net = create_linear_plus_mlp(
        m, hidden_sizes=(32, 32), init_scale=0.0, key=jr.PRNGKey(0)
    )
    out = net(s_ss[None, :])[0]
    # init_scale=0 + all-linear at SS → exact ss_policy
    assert float(jnp.max(jnp.abs(out - ss_policy))) == 0.0


def test_linear_plus_mlp_log_recovers_ss_at_init():
    """All-log + init_scale=0 + state at SS → exact ss_policy via exp(0)=1."""
    m = load_model("disaster")
    s_ss, ss_policy = m.steady_state_fn(m.constants)
    net = create_linear_plus_mlp(
        m,
        hidden_sizes=(32, 32),
        init_scale=0.0,
        output_links=["log"] * m.n_policies,
        key=jr.PRNGKey(0),
    )
    out = net(s_ss[None, :])[0]
    assert float(jnp.max(jnp.abs(out - ss_policy))) == 0.0


def test_linear_plus_mlp_log_positive_under_perturbation():
    """Log output stays positive at perturbed states even with random MLP weights."""
    m = load_model("disaster")
    s_ss, _ = m.steady_state_fn(m.constants)
    net = create_linear_plus_mlp(
        m,
        hidden_sizes=(32, 32),
        init_scale=0.5,  # nonzero, so MLP is not ≈ 0
        output_links=["log"] * m.n_policies,
        key=jr.PRNGKey(7),
    )
    key = jr.PRNGKey(42)
    for sigma in (0.05, 0.1, 0.2):
        key, sub = jr.split(key)
        noise = jr.normal(sub, s_ss.shape) * sigma * jnp.abs(s_ss + 1e-3)
        out = net((s_ss + noise)[None, :])[0]
        assert bool(jnp.all(jnp.isfinite(out))), f"NaN/Inf at sigma={sigma}"
        # The clip post-exp may set some entries to 0 if policy_lower=0; for
        # disaster all policy_lower > 0, so output strictly positive.
        assert bool(jnp.all(out > 0)), f"non-positive output at sigma={sigma}: {out}"


def test_linear_plus_mlp_log_first_order_matches_linear():
    """At small ε: log and linear forms agree to O(ε²) (first-order Taylor).

    log: ss · exp(P_log · ε) ≈ ss · (1 + P_log · ε)
    linear: ss + P_level · ε
    With P_log = P_level / ss, both reduce to ss + P_level · ε at first order.
    """
    m = load_model("disaster")
    s_ss, _ = m.steady_state_fn(m.constants)
    net_lin = create_linear_plus_mlp(
        m, hidden_sizes=(32, 32), init_scale=0.0, key=jr.PRNGKey(0)
    )
    net_log = create_linear_plus_mlp(
        m,
        hidden_sizes=(32, 32),
        init_scale=0.0,
        output_links=["log"] * m.n_policies,
        key=jr.PRNGKey(0),
    )
    eps = 1e-5
    perturb = jnp.zeros_like(s_ss).at[1].set(eps)  # k_lag perturbation
    out_lin = net_lin((s_ss + perturb)[None, :])[0]
    out_log = net_log((s_ss + perturb)[None, :])[0]
    # Difference between log and linear forms is O(ε²); for ε=1e-5 should be
    # at most ~1e-9 (with safety margin for fp64 numerical noise).
    assert float(jnp.max(jnp.abs(out_lin - out_log))) < 1e-8


def test_linear_plus_mlp_invalid_link_raises():
    """Unknown output_link string raises a clear error."""
    m = load_model("disaster")
    with pytest.raises(ValueError, match="unknown output_link"):
        create_linear_plus_mlp(
            m,
            hidden_sizes=(32, 32),
            init_scale=0.0,
            output_links=["log"] * (m.n_policies - 1) + ["sigmoid"],
            key=jr.PRNGKey(0),
        )


def test_linear_plus_mlp_log_requires_positive_ss():
    """Log link with non-positive ss_policy raises (defensive — disaster ss > 0)."""
    # Construct a synthetic model spec with a zero-valued ss_policy entry to
    # trigger the validation. Easier: just call __init__ directly.
    from deqn_jax.networks.linear_plus_mlp import LinearPlusMLP

    n_states, n_pol = 2, 2
    P = jnp.zeros((n_pol, n_states))
    ss_state = jnp.array([1.0, 1.0])
    ss_policy = jnp.array([1.0, 0.0])  # second is zero → log-link invalid
    with pytest.raises(ValueError, match="output_link='log' requires ss_policy > 0"):
        LinearPlusMLP(
            n_states=n_states,
            n_policies=n_pol,
            hidden_sizes=(8,),
            activation="tanh",
            P=P,
            ss_state=ss_state,
            ss_policy=ss_policy,
            output_links=["log", "log"],
            init_scale=0.0,
            key=jr.PRNGKey(0),
        )


# ---------------------------------------------------------------------------
# DisasterPolicyNet
# ---------------------------------------------------------------------------


def test_disaster_policy_net_default_link_from_model():
    """Disaster ModelSpec.default_output_links left None — existing configs
    keep legacy additive-linear behavior. Log ansatz opt-in via YAML."""
    m = load_model("disaster")
    assert m.default_output_links is None


def test_disaster_policy_net_log_recovers_ss_at_init():
    m = load_model("disaster")
    s_ss, ss_policy = m.steady_state_fn(m.constants)
    net = create_disaster_policy_net(
        m,
        hidden_sizes=(32, 32),
        init_scale=0.0,
        kf_names=("F_p", "K_p", "F_w", "K_w"),
        output_links=["log"] * 11,
        key=jr.PRNGKey(0),
    )
    out = net(s_ss[None, :])[0]
    assert float(jnp.max(jnp.abs(out - ss_policy))) == 0.0


def test_disaster_policy_net_reparam_log_conflict_raises():
    """reparam_pi_as_kp_inner=True + output_links[pi_idx]='log' is incompatible."""
    m = load_model("disaster")
    with pytest.raises(ValueError, match="reparam_pi_as_kp_inner.*forces.*linear"):
        create_disaster_policy_net(
            m,
            hidden_sizes=(32, 32),
            init_scale=0.0,
            kf_names=("F_p", "K_p", "F_w", "K_w"),
            reparam_pi_as_kp_inner=True,
            output_links=["log"] * 11,
            key=jr.PRNGKey(0),
        )


def test_disaster_policy_net_mixed_links_with_reparam():
    """log everywhere except reparam-affected slot (linear) — should work."""
    m = load_model("disaster")
    s_ss, ss_policy = m.steady_state_fn(m.constants)
    pi_idx = list(m.policy_names).index("pi")
    links = ["log"] * 11
    links[pi_idx] = "linear"
    net = create_disaster_policy_net(
        m,
        hidden_sizes=(32, 32),
        init_scale=0.0,
        kf_names=("F_p", "K_p", "F_w", "K_w"),
        reparam_pi_as_kp_inner=True,
        output_links=links,
        key=jr.PRNGKey(0),
    )
    out = net(s_ss[None, :])[0]
    # Non-zero numerical residual expected from K_p_inner round-trip but bounded
    assert float(jnp.max(jnp.abs(out - ss_policy))) < 1e-4
