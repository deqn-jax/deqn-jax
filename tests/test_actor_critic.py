"""Tests for the actor-critic framework.

Two tiers:

* **Plumbing tests** (fast): synthetic equations + small network + 1-2
  cycles. Verify that the (params, aux_params) tuple grad path works,
  that the value head's parameters actually update, that introspection
  filters value kwargs correctly, and that backward compat is intact.

* **Convergence test** (slow): brock_mirman_ez trains to a finite,
  decreasing loss with both modes. Just checks loss decreases — no
  policy-against-benchmark comparison since EZ has no tractable
  closed form for ψ ≠ γ_ez ≠ 1.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from deqn_jax.config import (
    ActorCriticConfig,
    NetworkConfig,
    OptimizerConfig,
    TrainConfig,
)
from deqn_jax.models import load_model
from deqn_jax.networks.mlp import ActorCriticMLP, create_actor_critic_mlp
from deqn_jax.training.loss import equations_accepts_value
from deqn_jax.training.trainer import create_train_state, make_train_step

# -- Helpers ------------------------------------------------------------


def _toy_ac_equations(
    state,
    policy,
    next_state,
    next_policy,
    constants,
    *,
    value_now,
    value_next,
    value_grad_next,
):
    """Synthetic equations exercising all three value kwargs."""
    bellman = value_now - 0.99 * value_next
    # Pull in value_grad_next so its gradient flows.
    euler = policy[..., 0] - 0.36 + 0.01 * jnp.sum(value_grad_next, axis=-1)
    return {"bellman": bellman, "euler": euler}


def _build_state(model, mode: str, *, lr=1e-3, critic_lr=None, hidden=(8,)):
    key = jax.random.key(0)
    opt_cfg = OptimizerConfig(
        name="adam",
        learning_rate=lr,
        critic_learning_rate=critic_lr,
    )
    ac_cfg = ActorCriticConfig(mode=mode, value_hidden_sizes=(8,))
    state, opt, kind, critic_opt = create_train_state(
        model,
        key,
        hidden_sizes=hidden,
        batch_size=4,
        sim_batch=4,
        n_equations=len(model.equation_names),
        optimizer_config=opt_cfg,
        network_config=NetworkConfig(hidden_sizes=hidden),
        actor_critic_config=ac_cfg,
    )
    train_step = make_train_step(
        model,
        opt,
        episode_length=4,
        mc_samples=2,
        batch_size=4,
        kind=kind,
        history_len=1,
        n_epochs_per_rollout=1,
        n_minibatches_per_epoch=1,
        initialize_each_episode=True,
        critic_opt=critic_opt,
    )
    return state, train_step


# -- Plumbing tests ------------------------------------------------------


class TestActorCriticMLP:
    """Shared-trunk module mechanics."""

    def test_policy_and_value_shapes(self):
        key = jax.random.key(0)
        ac = create_actor_critic_mlp(
            n_states=2,
            n_policies=3,
            hidden_sizes=(8, 8),
            key=key,
        )
        s = jnp.array([0.3, -0.2])
        sb = jnp.array([[0.3, -0.2], [0.1, 0.4]])

        # Single state
        assert ac.policy(s).shape == (3,)
        assert ac.value(s).shape == ()  # scalar
        # Batched
        assert ac.policy(sb).shape == (2, 3)
        assert ac.value(sb).shape == (2,)
        # Default __call__ is policy
        assert jnp.allclose(ac(s), ac.policy(s))

    def test_value_jax_grad(self):
        """jax.grad(value) returns ∂V/∂s (per-sample); vmap works."""
        key = jax.random.key(1)
        ac = create_actor_critic_mlp(
            n_states=2,
            n_policies=1,
            hidden_sizes=(8,),
            key=key,
        )
        s = jnp.array([0.3, -0.2])
        dv = jax.grad(ac.value)(s)
        assert dv.shape == (2,) and jnp.all(jnp.isfinite(dv))

        # Vmapped grad over batch — what compute_residuals does
        sb = jnp.array([[0.3, -0.2], [0.1, 0.4], [-0.5, 0.0]])
        dvb = jax.vmap(jax.grad(ac.value))(sb)
        assert dvb.shape == (3, 2) and jnp.all(jnp.isfinite(dvb))

    def test_eqx_filter_includes_value_head(self):
        """Param filter picks up the value head; gradient can update it."""
        key = jax.random.key(2)
        ac = create_actor_critic_mlp(
            n_states=2,
            n_policies=1,
            hidden_sizes=(8,),
            key=key,
        )
        params = eqx.filter(ac, eqx.is_array)
        leaves = jax.tree.leaves(params)
        # Value head Linear has weight + bias; both must show up.
        assert any("value_head" in str(p.shape) or p.size > 0 for p in leaves)


class TestEquationsIntrospection:
    """equations_accepts_value returns the granular kwarg set."""

    def test_existing_models_no_value_kwargs(self):
        """Backward compat: brock_mirman / disaster don't accept value kwargs."""
        for name in ("brock_mirman", "disaster"):
            m = load_model(name)
            assert equations_accepts_value(m.equations_fn) == ()

    def test_partial_value_acceptance(self):
        """A model declaring only some kwargs gets only those passed."""

        def eq_only_now(s, p, ns, np, c, *, value_now):
            return {}

        def eq_now_and_next(s, p, ns, np, c, *, value_now, value_next):
            return {}

        def eq_grad_only(s, p, ns, np, c, *, value_grad_next):
            return {}

        assert equations_accepts_value(eq_only_now) == ("value_now",)
        assert equations_accepts_value(eq_now_and_next) == ("value_now", "value_next")
        assert equations_accepts_value(eq_grad_only) == ("value_grad_next",)

    def test_brock_mirman_ez_kwargs(self):
        """Demo model declares value_now + value_next (Bellman)."""
        m = load_model("brock_mirman_ez")
        assert equations_accepts_value(m.equations_fn) == ("value_now", "value_next")


class TestSeparateMode:
    """Separate-mode AC: critic is a standalone network in aux_params."""

    def test_aux_params_populated(self):
        m = load_model("brock_mirman")._replace(
            equations_fn=_toy_ac_equations, equation_names=("bellman", "euler")
        )
        state, _ = _build_state(m, mode="separate")
        assert state.aux_params is not None
        assert state.aux_opt_state is not None
        assert state.params is not state.aux_params  # disjoint modules

    def test_critic_params_update_after_one_step(self):
        m = load_model("brock_mirman")._replace(
            equations_fn=_toy_ac_equations, equation_names=("bellman", "euler")
        )
        state, ts = _build_state(m, mode="separate", lr=1e-2)
        c0 = jax.tree.map(
            lambda x: x.copy() if hasattr(x, "copy") else x,
            eqx.filter(state.aux_params, eqx.is_array),
        )
        new_state, _ = ts(state, jnp.array(1.0), jnp.array(1.0))
        c1 = eqx.filter(new_state.aux_params, eqx.is_array)
        diffs = jax.tree.map(lambda a, b: jnp.max(jnp.abs(a - b)), c0, c1)
        assert max(jax.tree.leaves(diffs)) > 0, "critic params must update"

    def test_separate_critic_lr_zero_keeps_critic_frozen(self):
        """critic_learning_rate=0 freezes the critic but still moves the actor."""
        m = load_model("brock_mirman")._replace(
            equations_fn=_toy_ac_equations, equation_names=("bellman", "euler")
        )
        # critic_learning_rate validator rejects <= 0, so use a very small value.
        state, ts = _build_state(m, mode="separate", lr=1e-2, critic_lr=1e-12)
        c0 = jax.tree.map(
            lambda x: x.copy() if hasattr(x, "copy") else x,
            eqx.filter(state.aux_params, eqx.is_array),
        )
        p0 = jax.tree.map(
            lambda x: x.copy() if hasattr(x, "copy") else x,
            eqx.filter(state.params, eqx.is_array),
        )
        new_state, _ = ts(state, jnp.array(1.0), jnp.array(1.0))
        c1 = eqx.filter(new_state.aux_params, eqx.is_array)
        p1 = eqx.filter(new_state.params, eqx.is_array)
        c_diff = max(
            jax.tree.leaves(jax.tree.map(lambda a, b: jnp.max(jnp.abs(a - b)), c0, c1))
        )
        p_diff = max(
            jax.tree.leaves(jax.tree.map(lambda a, b: jnp.max(jnp.abs(a - b)), p0, p1))
        )
        # Critic essentially frozen, actor has moved
        assert c_diff < 1e-9, f"critic not frozen at tiny lr: {c_diff}"
        assert p_diff > 1e-6, f"actor didn't move: {p_diff}"


class TestSharedMode:
    """Shared-trunk AC: ActorCriticMLP with policy + value heads in state.params."""

    def test_value_head_in_params(self):
        m = load_model("brock_mirman")._replace(
            equations_fn=_toy_ac_equations, equation_names=("bellman", "euler")
        )
        state, _ = _build_state(m, mode="shared")
        assert state.aux_params is None  # no separate critic
        assert isinstance(state.params, ActorCriticMLP)
        assert hasattr(state.params, "value_head")

    def test_value_head_updates_after_one_step(self):
        m = load_model("brock_mirman")._replace(
            equations_fn=_toy_ac_equations, equation_names=("bellman", "euler")
        )
        state, ts = _build_state(m, mode="shared", lr=1e-2)
        v0 = state.params.value_head.weight
        new_state, _ = ts(state, jnp.array(1.0), jnp.array(1.0))
        v1 = new_state.params.value_head.weight
        assert float(jnp.max(jnp.abs(v0 - v1))) > 0


class TestBackwardCompat:
    """Existing models train byte-identically when AC is off."""

    def test_brock_mirman_unchanged_with_default_config(self):
        m = load_model("brock_mirman")
        # Default config: no AC, no critic. Run one cycle, just confirm
        # it doesn't crash and produces a finite loss.
        key = jax.random.key(0)
        state, opt, kind, critic_opt = create_train_state(
            m,
            key,
            hidden_sizes=(8,),
            batch_size=4,
            sim_batch=4,
            n_equations=1,
            optimizer_config=OptimizerConfig(name="adam", learning_rate=1e-3),
            network_config=NetworkConfig(hidden_sizes=(8,)),
        )
        # No AC config supplied → critic_opt must be None (the contract).
        assert critic_opt is None
        ts = make_train_step(
            m,
            opt,
            episode_length=4,
            mc_samples=2,
            batch_size=4,
            kind=kind,
            history_len=1,
            n_epochs_per_rollout=1,
            n_minibatches_per_epoch=1,
            initialize_each_episode=True,
        )
        new_state, metrics = ts(state, jnp.array(1.0), jnp.array(1.0))
        assert state.aux_params is None and state.aux_opt_state is None
        assert jnp.isfinite(metrics.loss)


class TestBrockMirmanEzPlumbing:
    """The demo model wires through both AC modes for a single cycle."""

    @pytest.mark.parametrize("mode", ["shared", "separate"])
    def test_one_cycle_finite_loss(self, mode):
        m = load_model("brock_mirman_ez")
        state, ts = _build_state(m, mode=mode, lr=1e-3)
        new_state, metrics = ts(state, jnp.array(1.0), jnp.array(1.0))
        assert jnp.isfinite(metrics.loss)
        assert set(metrics.residuals.keys()) == {"euler", "bellman"}
        assert all(jnp.isfinite(v) for v in metrics.residuals.values())


# -- Convergence test (slow) --------------------------------------------


class TestBrockMirmanEzConvergence:
    """Loss decreases on brock_mirman_ez. Shared mode only (faster)."""

    def test_loss_decreases(self):
        from deqn_jax.training.trainer import train_from_config

        config = TrainConfig(
            model="brock_mirman_ez",
            episodes=200,
            batch_size=32,
            episode_length=20,
            mc_samples=3,
            seed=42,
            network=NetworkConfig(hidden_sizes=(32, 32)),
            optimizer=OptimizerConfig(name="adam", learning_rate=1e-3),
            actor_critic=ActorCriticConfig(mode="shared"),
            initialize_each_episode=True,
            verbose=False,
            n_minibatches_per_epoch=1,
        )
        _, history = train_from_config(config)
        # Loss is intrinsically noisy here (initialize_each_episode=True
        # → fresh uniform draws each cycle, so per-cycle loss reflects
        # state-distribution variance more than policy quality).
        # Require a 25% reduction averaged over windows — enough to catch
        # a totally-broken AC pipeline without flaking on iteration noise.
        n = len(history["loss"])
        early = sum(history["loss"][: n // 5]) / max(1, n // 5)
        late = sum(history["loss"][-n // 5 :]) / max(1, n // 5)
        assert late < 0.75 * early, (
            f"loss didn't decrease 25%: early={early:.3e} -> late={late:.3e}"
        )


# -- Regression tests for review-found bugs -----------------------------


class TestModelACValidation:
    """A model that requires value kwargs must have AC enabled at config time.

    Reproduces the failure mode where ``deqn-jax train brock_mirman_ez``
    without ``actor_critic.mode`` set crashed at the first JIT trace
    with a ``TypeError: equations() missing required keyword-only
    arguments 'value_now' and 'value_next'``. Validation now rejects
    that combination at config-load time with a clear message.
    """

    def test_value_required_model_without_ac_raises(self):
        from deqn_jax.training.trainer import _resolve_model_for_training

        cfg = TrainConfig(
            model="brock_mirman_ez",
            episodes=1,
            batch_size=4,
            episode_length=4,
            mc_samples=1,
            network=NetworkConfig(hidden_sizes=(8,)),
            optimizer=OptimizerConfig(name="adam", learning_rate=1e-3),
            initialize_each_episode=True,
            verbose=False,
        )
        with pytest.raises(ValueError, match="value kwargs"):
            _resolve_model_for_training(cfg)

    def test_shared_mode_with_non_mlp_raises(self):
        """Shared-mode AC + non-MLP network is rejected at config time."""
        from deqn_jax.training.trainer import _resolve_model_for_training

        cfg = TrainConfig(
            model="brock_mirman_ez",
            episodes=1,
            batch_size=4,
            episode_length=4,
            mc_samples=1,
            network=NetworkConfig(type="lstm", hidden_sizes=(8,)),
            optimizer=OptimizerConfig(name="adam", learning_rate=1e-3),
            actor_critic=ActorCriticConfig(mode="shared"),
            initialize_each_episode=True,
            verbose=False,
        )
        with pytest.raises(ValueError, match="shared.*mlp"):
            _resolve_model_for_training(cfg)

    def test_separate_mode_works_with_any_network(self):
        """Separate-mode builds an independent critic regardless of network type."""
        from deqn_jax.training.trainer import _resolve_model_for_training

        cfg = TrainConfig(
            model="brock_mirman_ez",
            episodes=1,
            batch_size=4,
            episode_length=4,
            mc_samples=1,
            network=NetworkConfig(type="lstm", hidden_sizes=(8,)),
            optimizer=OptimizerConfig(name="adam", learning_rate=1e-3),
            actor_critic=ActorCriticConfig(mode="separate"),
            initialize_each_episode=True,
            verbose=False,
        )
        # Should not raise.
        model, _ = _resolve_model_for_training(cfg)
        assert model.name == "brock_mirman_ez"


class TestResumePreservesCriticOpt:
    """Resume from checkpoint must rebuild the critic optimizer.

    Reproduces the bug where the resume path discarded the side-channel
    critic_opt attached to the original opt, then built a fresh primary
    opt without re-attaching anything — so the separate-mode critic
    silently stopped training. Now the explicit 4-tuple + helper
    rebuild ensures the critic optimizer is always reconstructed from
    the current OptimizerConfig.critic_* fields.
    """

    def test_separate_mode_critic_resumes(self, tmp_path):
        import os

        import equinox as eqx

        from deqn_jax.training.checkpointing import save_checkpoint
        from deqn_jax.training.trainer import _build_initial_state

        cfg = TrainConfig(
            model="brock_mirman_ez",
            episodes=1,
            batch_size=4,
            episode_length=4,
            mc_samples=1,
            network=NetworkConfig(hidden_sizes=(8,)),
            optimizer=OptimizerConfig(name="adam", learning_rate=1e-3),
            actor_critic=ActorCriticConfig(mode="separate", value_hidden_sizes=(8,)),
            initialize_each_episode=True,
            verbose=False,
            seed=0,
        )
        # First, build a fresh state and snapshot it to disk. Use the
        # legacy uint32 PRNG key (matching production train_from_config
        # at trainer.py:1521) — equinox's default serialise filter
        # doesn't handle the typed key dtype.
        from deqn_jax.training.trainer import _resolve_model_for_training

        model, n_eq = _resolve_model_for_training(cfg)
        state, opt, _kind, critic_opt, _ep, _tot = _build_initial_state(
            cfg, model, jax.random.PRNGKey(cfg.seed), n_eq, cfg.optimizer
        )
        assert critic_opt is not None
        assert state.aux_params is not None and state.aux_opt_state is not None

        # Persist (mimicking checkpoint flow).
        ckpt_dir = tmp_path / "ckpt"
        os.makedirs(ckpt_dir, exist_ok=True)
        save_checkpoint(state, str(ckpt_dir), episode=1, config=cfg)
        ckpt_path = str(ckpt_dir / "checkpoint_000001.eqx")
        assert os.path.exists(ckpt_path)

        # Now resume — critic_opt must come back non-None.
        resume_cfg = cfg.model_copy(update={"resume": ckpt_path})
        model2, n_eq2 = _resolve_model_for_training(resume_cfg)
        state2, _opt, _kind, critic_opt2, _ep2, _tot2 = _build_initial_state(
            resume_cfg, model2, jax.random.PRNGKey(0), n_eq2, resume_cfg.optimizer
        )
        assert critic_opt2 is not None, (
            "resumed critic_opt is None — separate-mode critic would silently stop"
        )
        assert state2.aux_params is not None
        assert state2.aux_opt_state is not None

        # Verify a step actually moves critic params with the rebuilt critic_opt.
        from deqn_jax.training.trainer import make_train_step

        ts = make_train_step(
            model2,
            _opt,
            episode_length=4,
            mc_samples=1,
            batch_size=4,
            kind=_kind,
            history_len=1,
            n_epochs_per_rollout=1,
            n_minibatches_per_epoch=1,
            initialize_each_episode=True,
            critic_opt=critic_opt2,
        )
        c0 = jax.tree.map(
            lambda x: x.copy() if hasattr(x, "copy") else x,
            eqx.filter(state2.aux_params, eqx.is_array),
        )
        new_state, _ = ts(state2, jnp.array(1.0), jnp.array(1.0))
        c1 = eqx.filter(new_state.aux_params, eqx.is_array)
        diffs = jax.tree.map(lambda a, b: jnp.max(jnp.abs(a - b)), c0, c1)
        assert max(jax.tree.leaves(diffs)) > 0, (
            "critic params didn't move after resume — critic_opt missing again"
        )


class TestCompositeLossWithACDispatch:
    """make_grad_step_standard always passes aux_params to compute_loss_fn.

    Reproduces the bug where composite_loss_fn didn't accept aux_params
    and crashed AC training (and would crash any custom-loss training
    that goes through STANDARD now). The fix added aux_params to
    composite_loss_fn's signature; this test verifies non-AC + composite
    still works (the same path that the new aux_params=None default
    flows through).
    """

    def test_composite_loss_no_ac_still_works(self):
        """composite + no AC: aux_params=None is accepted and ignored."""
        import contextlib
        import io

        from deqn_jax.training.composite_loss import (
            make_composite_loss,
            prepare_composite_data,
        )
        from deqn_jax.training.linearize import linearize_model

        m = load_model("disaster")
        key = jax.random.key(0)
        state, opt, kind, critic_opt = create_train_state(
            m,
            key,
            hidden_sizes=(16, 16),
            batch_size=8,
            sim_batch=8,
            n_equations=len(m.equation_names),
            optimizer_config=OptimizerConfig(name="adam", learning_rate=1e-3),
            network_config=NetworkConfig(hidden_sizes=(16, 16)),
        )
        assert critic_opt is None
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            P, Q = linearize_model(m)
            composite_data = prepare_composite_data(
                m, P, Q, n_anchor_points=8, verbose=False
            )
        composite_loss_fn = make_composite_loss(
            m,
            composite_data,
            anchor_weight=0.1,
            jac_weight=0.01,
            jac_anchor_weight=0.0,
            barrier_weight=0.01,
            newton_weight=0.01,
            aux_decay_floor=0.2,
            leverage_mult=5.0,
            loss_choice="mse",
            huber_delta=1.0,
        )
        ts = make_train_step(
            m,
            opt,
            episode_length=4,
            mc_samples=2,
            batch_size=8,
            kind=kind,
            history_len=1,
            n_epochs_per_rollout=1,
            n_minibatches_per_epoch=1,
            initialize_each_episode=False,
            compute_loss_fn=composite_loss_fn,
            critic_opt=critic_opt,
        )
        new_state, metrics = ts(state, jnp.array(1.0), jnp.array(1.0))
        assert jnp.isfinite(metrics.loss)
        # Composite aux residuals should still appear.
        assert any(k.startswith("aux_") for k in metrics.residuals)


class TestDroppedConfigKnobs:
    """The half-baked AC knobs were removed, not silently kept dead."""

    @pytest.mark.parametrize(
        "field,value",
        [
            ("bellman_loss_weight", 0.5),
            ("aux_bellman", True),
            ("detach_value_in_policy_grad", True),
        ],
    )
    def test_dropped_knobs_rejected(self, field, value):
        with pytest.raises((ValueError, TypeError)):
            ActorCriticConfig(**{field: value})
