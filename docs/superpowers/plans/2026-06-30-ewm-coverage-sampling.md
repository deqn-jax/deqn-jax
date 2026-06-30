# EWM Coverage Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Equilibrium-World-Model coverage sampling — impose the existing residual loss on a broader measure (base ∪ stress ∪ local pools, off-path seeds rolled through the exact transition) — to the trainer as a config-gated `compute_loss` wrapper, validated on `irbc`.

**Architecture:** A loss *wrapper* (`make_coverage_loss`) installed inside `_build_custom_loss_fn` evaluates the unchanged `compute_loss` on three state pools and returns a mixture-weighted sum. Stress seeds are drawn from a config box and rolled H steps through the exact transition via `run_episode` (reused, takes `policy_fn`), then `stop_gradient`'d. Zero changes to `compute_loss` or the five train-step variants; bit-identical to baseline when disabled.

**Tech Stack:** JAX, Equinox, Optax, Pydantic v2, pytest, uv.

## Global Constraints

- **Branch:** `ewm-coverage` (private/local). NEVER push without an explicit user ask.
- **uv only:** run everything via `uv run ...`; never activate the venv.
- **Single JIT boundary:** the coverage wrapper is built before JIT (inside `_build_custom_loss_fn`). Do NOT introduce a second `@jax.jit` or break the train step.
- **Pydantic v2 config:** `model_config = ConfigDict(extra="forbid")`; use `_coerce_float` / `_coerce_int` from `deqn_jax.config._base`.
- **v1 scope (enforced by validators):** coverage requires a STANDARD optimizer (`adam`/`sgd`/`adamw`/`lion`/`muon`/`ngd`/`shampoo`, `gradient_surgery="none"`) and `loss_type: mse`; it is mutually exclusive with `composite`, `barrier_weight>0`, `loss_choice!="mse"`, and `moment_matching.enabled`.
- **Bit-identical when disabled** is a hard requirement (exact equality, not allclose).
- **Mixture weights** are normalized to sum to 1 inside the wrapper (faithful to the paper; avoids an LR shift).
- **Stop-gradient** every generated pool (`roll_states`, `make_local_pool`): state generation is never differentiated.
- **No VariableSpec on the loaded model** — resolve `stress_ranges` names via `model.state_names` (a `Tuple[str, ...]`), never `model.variable_spec`.
- **Commit messages:** clean and descriptive; do NOT include a `Claude-Session:` line (user preference). A `Co-Authored-By:` line is fine.
- **ruff pre-edit hook** strips unused imports — make sure every import a test adds is referenced in the test body (it will be, since tests use the symbols).
- Spec: `docs/superpowers/specs/2026-06-29-ewm-coverage-sampling-design.md`.

---

### Task 1: `CoverageConfig` + wire into `TrainConfig`

**Files:**
- Create: `src/deqn_jax/config/coverage.py`
- Modify: `src/deqn_jax/config/__init__.py` (export)
- Modify: `src/deqn_jax/config/train.py` (field + `from_dict` wiring)
- Test: `tests/test_config_coverage.py`

**Interfaces:**
- Produces: `CoverageConfig` with fields `enabled: bool`, `rho_base: float`, `rho_stress: float`, `rho_local: float`, `n_stress: int`, `n_local: int`, `rollout_horizon: int`, `local_sigma: float`, `stress_ranges: Dict[str, Tuple[float, float]]`. Available as `TrainConfig(...).coverage`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_coverage.py
"""Validation tests for CoverageConfig (EWM coverage sampling)."""

import pytest

from deqn_jax.config import CoverageConfig, TrainConfig


def test_defaults_disabled():
    c = CoverageConfig()
    assert c.enabled is False
    assert c.rho_base == 1.0 and c.rho_stress == 1.0 and c.rho_local == 1.0
    assert c.n_stress == 128 and c.n_local == 128
    assert c.rollout_horizon == 8
    assert c.local_sigma == 0.01
    assert c.stress_ranges == {}


def test_extra_forbidden():
    with pytest.raises(Exception):
        CoverageConfig(notafield=1)


def test_negative_weight_rejected():
    with pytest.raises(Exception):
        CoverageConfig(rho_stress=-1.0)


def test_all_zero_weights_rejected():
    with pytest.raises(Exception):
        CoverageConfig(rho_base=0.0, rho_stress=0.0, rho_local=0.0)


def test_enabled_empty_stress_ranges_rejected():
    # rho_stress>0 but no stress box → cannot build the stress pool
    with pytest.raises(Exception):
        CoverageConfig(enabled=True, rho_stress=1.0, stress_ranges={})


def test_enabled_empty_pool_with_weight_rejected():
    # weighted pool with zero seeds → would NaN at jnp.mean over 0 rows
    with pytest.raises(Exception):
        CoverageConfig(
            enabled=True, rho_stress=1.0, n_stress=0,
            stress_ranges={"z_0": (-0.5, -0.2)},
        )


def test_bad_range_rejected():
    with pytest.raises(Exception):
        CoverageConfig(
            enabled=True, stress_ranges={"z_0": (0.2, -0.5)},  # low > high
        )


def test_valid_enabled_config():
    c = CoverageConfig(
        enabled=True,
        stress_ranges={"z_0": (-0.5, -0.2), "k_0": (1.05, 1.20)},
    )
    assert c.enabled is True
    assert c.stress_ranges["z_0"] == (-0.5, -0.2)


def test_trainconfig_has_coverage_default():
    cfg = TrainConfig()
    assert isinstance(cfg.coverage, CoverageConfig)
    assert cfg.coverage.enabled is False


def test_trainconfig_from_dict_coverage():
    cfg = TrainConfig.from_dict(
        {"model": "irbc", "coverage": {"enabled": True,
         "stress_ranges": {"z_0": [-0.5, -0.2]}}}
    )
    assert cfg.coverage.enabled is True
    assert cfg.coverage.stress_ranges["z_0"] == (-0.5, -0.2)


def test_trainconfig_from_dict_unknown_coverage_key():
    with pytest.raises(Exception):
        TrainConfig.from_dict({"coverage": {"nope": 1}})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_coverage.py -q`
Expected: FAIL — `ImportError: cannot import name 'CoverageConfig'`.

- [ ] **Step 3: Create `CoverageConfig`**

```python
# src/deqn_jax/config/coverage.py
"""CoverageConfig: EWM coverage-sampling settings.

Off by default. When enabled, the residual loss is imposed on a mixture
measure (base + stress + local pools) instead of the base batch alone.
Stress seeds are drawn from `stress_ranges` and rolled `rollout_horizon`
steps through the exact transition; the learned continuation surrogate is
out of scope in v1 (irbc's expectation is cheap quadrature).
"""

from __future__ import annotations

from typing import Dict, Tuple

from pydantic import ConfigDict, Field, field_validator, model_validator

from deqn_jax.config._base import _coerce_float, _coerce_int, _ConfigBase


class CoverageConfig(_ConfigBase):
    """EWM coverage-sampling configuration (v1: coverage-only, irbc)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Master switch. When False, training is byte-identical to no coverage.",
    )
    rho_base: float = Field(
        default=1.0, description="Mixture weight on the base (init-rect / on-policy) pool."
    )
    rho_stress: float = Field(
        default=1.0, description="Mixture weight on the stress pool."
    )
    rho_local: float = Field(
        default=1.0, description="Mixture weight on the local-perturbation pool. Weights are normalized to sum to 1 inside the wrapper.",
    )
    n_stress: int = Field(
        default=128, description="Number of stress seeds drawn per step (before rollout)."
    )
    n_local: int = Field(
        default=128, description="Number of local perturbations per step."
    )
    rollout_horizon: int = Field(
        default=8, description="H: steps to roll stress seeds through the exact transition.",
    )
    local_sigma: float = Field(
        default=0.01, description="Std of Gaussian local perturbations, in state units.",
    )
    stress_ranges: Dict[str, Tuple[float, float]] = Field(
        default_factory=dict,
        description="Per-state-name uniform box for stress seeds. Keys are state names (validated against model.state_names at model resolution). Empty is an error when enabled with rho_stress>0.",
    )

    @field_validator(
        "rho_base", "rho_stress", "rho_local", "local_sigma", mode="before"
    )
    @classmethod
    def _coerce_float_reject_bool(cls, v, info):
        return _coerce_float(v, f"coverage.{info.field_name}")

    @field_validator("n_stress", "n_local", "rollout_horizon", mode="before")
    @classmethod
    def _coerce_int_reject_bool(cls, v, info):
        return _coerce_int(v, f"coverage.{info.field_name}")

    @model_validator(mode="after")
    def _validate(self):
        for name in ("rho_base", "rho_stress", "rho_local"):
            if getattr(self, name) < 0:
                raise ValueError(f"coverage.{name} must be >= 0")
        if self.rho_base + self.rho_stress + self.rho_local <= 0:
            raise ValueError("coverage weights must not all be zero")
        if self.local_sigma < 0:
            raise ValueError("coverage.local_sigma must be >= 0")
        for n in ("n_stress", "n_local", "rollout_horizon"):
            if getattr(self, n) < 0:
                raise ValueError(f"coverage.{n} must be >= 0")
        for k, rng in self.stress_ranges.items():
            if rng[0] > rng[1]:
                raise ValueError(
                    f"coverage.stress_ranges[{k!r}] must be [low <= high], got {rng}"
                )
        if self.enabled:
            if self.rho_stress > 0:
                if self.n_stress <= 0:
                    raise ValueError("coverage.rho_stress>0 requires n_stress>0")
                if self.rollout_horizon < 1:
                    raise ValueError("coverage.rho_stress>0 requires rollout_horizon>=1")
                if not self.stress_ranges:
                    raise ValueError("coverage.rho_stress>0 requires non-empty stress_ranges")
            if self.rho_local > 0 and self.n_local <= 0:
                raise ValueError("coverage.rho_local>0 requires n_local>0")
        return self
```

- [ ] **Step 4: Export from the config package**

In `src/deqn_jax/config/__init__.py`, add the import and `__all__` entry:

```python
from deqn_jax.config.coverage import CoverageConfig  # add near the other config imports
```

and add `"CoverageConfig",` to the `__all__` list.

- [ ] **Step 5: Add the field + from_dict wiring in `train.py`**

In `src/deqn_jax/config/train.py`:

Add the import (near the other config imports, ~line 25):
```python
from deqn_jax.config.coverage import CoverageConfig
```

Add the field (after `replay_buffer`, ~line 88):
```python
    coverage: CoverageConfig = Field(
        default_factory=CoverageConfig,
        description="EWM coverage sampling; only active when coverage.enabled=true.",
    )
```

In `from_dict` (~line 588) add the pop:
```python
        cov_dict = d.pop("coverage", {})
```
the unknown-key check (~after line 625):
```python
        cov_fields = set(CoverageConfig.model_fields.keys())
        _check_unknown_keys(set(cov_dict.keys()), cov_fields, "coverage")
```
and pass it to the constructor (~line 632):
```python
            coverage=CoverageConfig(**cov_dict),
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_config_coverage.py -q`
Expected: PASS (11 passed).

- [ ] **Step 7: Commit**

```bash
git add src/deqn_jax/config/coverage.py src/deqn_jax/config/__init__.py src/deqn_jax/config/train.py tests/test_config_coverage.py
git commit -m "feat(config): CoverageConfig for EWM coverage sampling

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Coverage rollout helpers

**Files:**
- Create: `src/deqn_jax/training/coverage.py`
- Test: `tests/test_coverage.py`

**Interfaces:**
- Consumes: `run_episode` from `deqn_jax.training.episode` (signature `run_episode(model, policy_fn, init_state, key, episode_length=100, shock_scale=1.0, shock_mask=None) -> (trajectory [T,B,D], final_state [B,D])`, trajectory holds PRE-transition states).
- Produces:
  - `sample_stress_seeds(key, n, n_states, ss_state, stress_idx, lows, highs) -> Array [n, n_states]`
  - `roll_states(model, policy_fn, seeds, key, horizon, shock_scale=1.0) -> Array [n*horizon, n_states]` (stop-gradient'd; landings s_1..s_H, raw seed excluded)
  - `make_local_pool(states, key, n, sigma) -> Array [n, n_states]` (stop-gradient'd)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coverage.py
"""Unit tests for EWM coverage rollout helpers + loss wrapper."""

import jax
import jax.numpy as jnp
import numpy as np

from deqn_jax.models import load_model
from deqn_jax.training.coverage import (
    make_local_pool,
    roll_states,
    sample_stress_seeds,
)


def _irbc():
    return load_model("irbc")


def test_sample_stress_seeds_shape_and_box():
    model = _irbc()
    ss_state, _ = model.steady_state_fn(model.constants)
    # stress only z_0 (index 2); leave others at SS
    stress_idx = jnp.array([2])
    lows = jnp.array([-0.5])
    highs = jnp.array([-0.2])
    seeds = sample_stress_seeds(
        jax.random.PRNGKey(0), 64, model.n_states, ss_state, stress_idx, lows, highs
    )
    assert seeds.shape == (64, model.n_states)
    # z_0 column inside the box
    assert np.all(np.asarray(seeds[:, 2]) >= -0.5 - 1e-6)
    assert np.all(np.asarray(seeds[:, 2]) <= -0.2 + 1e-6)
    # non-stress dims pinned at SS
    np.testing.assert_allclose(np.asarray(seeds[:, 0]), float(ss_state[0]))


def test_roll_states_shape_excludes_raw_seed():
    model = _irbc()
    policy = load_model("irbc")  # not used; need a callable policy
    # a real callable policy: build the network params from a tiny config path
    from deqn_jax.networks.factory import create_network

    net = create_network(model, type="mlp", hidden_sizes=(8,), key=jax.random.PRNGKey(1))
    seeds = jnp.tile(jnp.array([1.0, 1.0, -0.4, -0.4]), (16, 1))
    out = roll_states(model, net, seeds, jax.random.PRNGKey(2), horizon=8)
    assert out.shape == (16 * 8, model.n_states)
    # landings differ from the raw seed (the exact-Γ rollout moved them)
    assert not np.allclose(np.asarray(out[:16]), np.asarray(seeds))


def test_roll_states_stop_gradient():
    model = _irbc()
    from deqn_jax.networks.factory import create_network

    net = create_network(model, type="mlp", hidden_sizes=(8,), key=jax.random.PRNGKey(1))
    seeds = jnp.tile(jnp.array([1.0, 1.0, -0.4, -0.4]), (8, 1))

    def f(p):
        return jnp.sum(roll_states(model, p, seeds, jax.random.PRNGKey(2), horizon=4))

    import equinox as eqx

    grads = eqx.filter_grad(f)(net)
    leaves = [g for g in jax.tree.leaves(eqx.filter(grads, eqx.is_array))]
    # every gradient leaf is exactly zero: no grad flows through generated states
    assert all(np.all(np.asarray(g) == 0.0) for g in leaves)


def test_make_local_pool_shape_and_detach():
    model = _irbc()
    states = jnp.tile(jnp.array([1.0, 1.0, 0.0, 0.0]), (32, 1))
    local = make_local_pool(states, jax.random.PRNGKey(3), n=20, sigma=0.01)
    assert local.shape == (20, model.n_states)
    # perturbed (not identical to base rows)
    assert not np.allclose(np.asarray(local[:1]), np.asarray(states[:1]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_coverage.py -q`
Expected: FAIL — `ModuleNotFoundError: deqn_jax.training.coverage`.

- [ ] **Step 3: Implement the helpers**

```python
# src/deqn_jax/training/coverage.py
"""EWM coverage sampling: rollout helpers + a compute_loss wrapper.

Coverage imposes the existing residual loss on a mixture measure
(base + stress + local) instead of the base batch alone. State
generation is never differentiated (every pool is stop_gradient'd);
gradient flows only through the policy re-evaluated at the fixed pool
states inside compute_loss.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

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
    """Draw n stress seeds: SS-filled, with stress dims uniform in their box."""
    seeds = jnp.broadcast_to(ss_state, (n, n_states))
    u = jax.random.uniform(
        key, (n, stress_idx.shape[0]), minval=lows, maxval=highs
    )
    return seeds.at[:, stress_idx].set(u)


def roll_states(
    model: ModelSpec,
    policy_fn: Callable[[Array], Array],
    seeds: Array,
    key: Array,
    horizon: int,
    shock_scale=1.0,
) -> Array:
    """Roll seeds horizon steps through the EXACT transition; return the
    projected landings s_1..s_H (raw seed excluded), stop-gradient'd.

    Reuses run_episode (MC shocks even under quadrature: state generation
    is a different object from the loss expectation). trajectory holds
    pre-transition states, so trajectory[0] is the raw seed and final_state
    is s_H; landings = trajectory[1:] ++ final_state.
    """
    trajectory, final_state = run_episode(
        model, policy_fn, seeds, key, episode_length=horizon, shock_scale=shock_scale
    )
    landings = jnp.concatenate([trajectory[1:], final_state[None]], axis=0)
    landings = landings.reshape(-1, model.n_states)
    return jax.lax.stop_gradient(landings)


def make_local_pool(states: Array, key: Array, n: int, sigma: float) -> Array:
    """n locally-perturbed copies of base states (sampled with replacement),
    plus Gaussian noise; stop-gradient'd."""
    batch = states.shape[0]
    idx = jax.random.randint(key, (n,), 0, batch)
    base = states[idx]
    noise = jax.random.normal(key, base.shape) * sigma
    return jax.lax.stop_gradient(base + noise)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_coverage.py -q`
Expected: PASS (4 passed).

If `create_network`'s signature differs, adjust the test's network construction to the project's helper (check `src/deqn_jax/networks/factory.py` for the exact factory entry point); the helpers under test do not depend on it.

- [ ] **Step 5: Commit**

```bash
git add src/deqn_jax/training/coverage.py tests/test_coverage.py
git commit -m "feat(training): coverage rollout helpers (stress seeds, roll, local)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `make_coverage_loss` wrapper

**Files:**
- Modify: `src/deqn_jax/training/coverage.py` (add `make_coverage_loss`)
- Test: `tests/test_coverage.py` (add)

**Interfaces:**
- Consumes: `compute_loss` contract `(model, policy_fn, states, key, mc_samples=5, weights=None, shock_scale=1.0, quad_nodes=None, quad_weights=None, target_policy_fn=None) -> (Array, Dict[str, Array])`; the helpers from Task 2; `CoverageConfig`.
- Produces: `make_coverage_loss(base_compute_loss, model, cfg) -> coverage_loss_fn` whose signature mirrors `compute_loss` exactly and returns `(total, eq_losses)` with extra scalar keys `aux_cov_base`, `aux_cov_stress`, `aux_cov_local`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_coverage.py
from deqn_jax.config import CoverageConfig
from deqn_jax.training.coverage import make_coverage_loss


def _cov_cfg(**kw):
    base = dict(
        enabled=True, rho_base=2.0, rho_stress=1.0, rho_local=1.0,
        n_stress=8, n_local=8, rollout_horizon=2, local_sigma=0.01,
        stress_ranges={"z_0": (-0.5, -0.2), "k_0": (1.05, 1.20)},
    )
    base.update(kw)
    return CoverageConfig(**base)


def test_wrapper_forwards_quad_kwargs_to_all_pools():
    model = _irbc()
    calls = []

    def spy(model_, pf, states, key, mc_samples=5, weights=None, shock_scale=1.0,
            quad_nodes=None, quad_weights=None, target_policy_fn=None):
        calls.append({"quad_nodes": quad_nodes, "n": int(states.shape[0])})
        return jnp.array(float(len(calls)) * 10.0), {"euler": jnp.array(1.0)}

    fn = make_coverage_loss(spy, model, _cov_cfg())
    from deqn_jax.networks.factory import create_network

    net = create_network(model, type="mlp", hidden_sizes=(8,), key=jax.random.PRNGKey(1))
    states = jnp.tile(jnp.array([1.0, 1.0, 0.0, 0.0]), (16, 1))
    qn = jnp.zeros((4, model.n_shocks))
    qw = jnp.ones((4,)) / 4
    total, eq = fn(model, net, states, jax.random.PRNGKey(0),
                   quad_nodes=qn, quad_weights=qw)
    # all three pools were evaluated, each got the quadrature nodes
    assert len(calls) == 3
    assert all(c["quad_nodes"] is not None for c in calls)
    # mixture weights normalized: 2/1/1 -> 0.5/0.25/0.25; spy returns 10,20,30
    assert np.isclose(float(total), 0.5 * 10 + 0.25 * 20 + 0.25 * 30)
    for k in ("aux_cov_base", "aux_cov_stress", "aux_cov_local"):
        assert k in eq


def test_wrapper_real_compute_loss_runs():
    model = _irbc()
    from deqn_jax.networks.factory import create_network
    from deqn_jax.training.loss import compute_loss

    net = create_network(model, type="mlp", hidden_sizes=(8,), key=jax.random.PRNGKey(1))
    fn = make_coverage_loss(compute_loss, model, _cov_cfg())
    states = jnp.tile(jnp.array([1.0, 1.0, 0.0, 0.0]), (16, 1))
    qn = jnp.zeros((4, model.n_shocks))
    qw = jnp.ones((4,)) / 4
    total, eq = fn(model, net, states, jax.random.PRNGKey(0),
                   quad_nodes=qn, quad_weights=qw)
    assert np.isfinite(float(total))
    assert np.isfinite(float(eq["aux_cov_stress"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_coverage.py -q -k wrapper`
Expected: FAIL — `ImportError: cannot import name 'make_coverage_loss'`.

- [ ] **Step 3: Implement `make_coverage_loss`**

Append to `src/deqn_jax/training/coverage.py`:

```python
def make_coverage_loss(
    base_compute_loss: Callable,
    model: ModelSpec,
    cfg,
) -> Callable:
    """Wrap base_compute_loss to impose it on a mixture measure.

    Returns a fn with the exact compute_loss signature. The base pool is
    the batch passed in; stress + local pools are generated internally
    (stop-gradient'd). All expectation kwargs are forwarded to every pool
    so all three share the identical operator (e.g. quadrature on irbc).
    """
    ss_state, _ = model.steady_state_fn(model.constants)
    ss_state = jnp.asarray(ss_state)
    n_states = model.n_states
    name_to_idx = {n: i for i, n in enumerate(model.state_names)}
    names = list(cfg.stress_ranges.keys())
    stress_idx = jnp.array([name_to_idx[n] for n in names], dtype=jnp.int32)
    lows = jnp.array([cfg.stress_ranges[n][0] for n in names])
    highs = jnp.array([cfg.stress_ranges[n][1] for n in names])
    rho = jnp.array([cfg.rho_base, cfg.rho_stress, cfg.rho_local])
    rho = rho / jnp.sum(rho)
    n_stress = int(cfg.n_stress)
    n_local = int(cfg.n_local)
    horizon = int(cfg.rollout_horizon)
    local_sigma = float(cfg.local_sigma)

    def coverage_loss_fn(
        model_: ModelSpec,
        policy_fn: Callable[[Array], Array],
        states: Array,
        key: Array,
        mc_samples: int = 5,
        weights: Optional[Array] = None,
        shock_scale=1.0,
        quad_nodes: Optional[Array] = None,
        quad_weights: Optional[Array] = None,
        target_policy_fn: Optional[Callable[[Array], Array]] = None,
    ) -> Tuple[Array, Dict[str, Array]]:
        k_base, k_seed, k_roll, k_local = jax.random.split(key, 4)

        def _loss(pool_states, k):
            return base_compute_loss(
                model_, policy_fn, pool_states, k, mc_samples,
                weights=weights, shock_scale=shock_scale,
                quad_nodes=quad_nodes, quad_weights=quad_weights,
                target_policy_fn=target_policy_fn,
            )

        l_base, eq = _loss(states, k_base)

        seeds = sample_stress_seeds(
            k_seed, n_stress, n_states, ss_state, stress_idx, lows, highs
        )
        stress = roll_states(model_, policy_fn, seeds, k_roll, horizon, shock_scale)
        l_stress, _ = _loss(stress, k_base)

        local = make_local_pool(states, k_local, n_local, local_sigma)
        l_local, _ = _loss(local, k_base)

        total = rho[0] * l_base + rho[1] * l_stress + rho[2] * l_local
        eq["aux_cov_base"] = l_base
        eq["aux_cov_stress"] = l_stress
        eq["aux_cov_local"] = l_local
        return total, eq

    return coverage_loss_fn
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_coverage.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/deqn_jax/training/coverage.py tests/test_coverage.py
git commit -m "feat(training): make_coverage_loss mixture wrapper

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Wire into `_build_custom_loss_fn` + add validators

**Files:**
- Modify: `src/deqn_jax/training/composite_loss.py` (`_build_custom_loss_fn`)
- Modify: `src/deqn_jax/training/state_init.py` (`_validate_train_config`, `_resolve_model_for_training`)
- Test: `tests/test_coverage_wiring.py`

**Interfaces:**
- Consumes: `make_coverage_loss` (Task 3); `compute_loss`; `config.coverage`.
- Produces: when `config.coverage.enabled`, `_build_custom_loss_fn` returns the coverage wrapper; the validators reject unsupported combinations and unknown `stress_ranges` names.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coverage_wiring.py
"""Validators + bit-identical / divergence guards for coverage wiring."""

import numpy as np
import pytest

from deqn_jax.config import TrainConfig
from deqn_jax.training.state_init import _validate_train_config
from deqn_jax.training.trainer import train_from_config


def _cfg(**over):
    base = {
        "model": "brock_mirman",
        "episodes": 30,
        "episode_length": 20,
        "batch_size": 16,
        "sim_batch": 32,
        "mc_samples": 2,
        "verbose": False,
        "log_every": 1,
        "seed": 0,
        "network": {"type": "mlp", "hidden_sizes": [16, 16], "activation": "tanh"},
        "optimizer": {"name": "adam", "learning_rate": 1e-3},
    }
    base.update(over)
    return TrainConfig.from_dict(base)


def test_coverage_plus_composite_rejected():
    cfg = _cfg(loss_type="composite",
               coverage={"enabled": True, "stress_ranges": {"z": (-0.1, -0.05)}})
    with pytest.raises(ValueError):
        _validate_train_config(cfg)


def test_coverage_plus_mao_rejected():
    cfg = _cfg(optimizer={"name": "mao"},
               coverage={"enabled": True, "stress_ranges": {"z": (-0.1, -0.05)}})
    with pytest.raises(ValueError):
        _validate_train_config(cfg)


def test_coverage_plus_barrier_rejected():
    cfg = _cfg(barrier_weight=1.0,
               coverage={"enabled": True, "stress_ranges": {"z": (-0.1, -0.05)}})
    with pytest.raises(ValueError):
        _validate_train_config(cfg)


def test_coverage_unknown_state_name_rejected():
    # 'znope' is not an irbc state name → error at model resolution
    cfg = TrainConfig.from_dict({
        "model": "irbc", "episodes": 2, "batch_size": 16,
        "episode_length": 1, "initialize_each_episode": True,
        "expectation_type": "gauss_hermite", "n_quadrature_points": 3,
        "network": {"type": "mlp", "hidden_sizes": [8]},
        "coverage": {"enabled": True, "stress_ranges": {"znope": (-0.5, -0.2)}},
    })
    with pytest.raises(ValueError):
        train_from_config(cfg)


def test_bit_identical_when_disabled():
    # coverage block present but disabled == no coverage block at all
    _, h_off = train_from_config(_cfg())
    _, h_dis = train_from_config(
        _cfg(coverage={"enabled": False, "n_stress": 999,
                       "stress_ranges": {"z": (-0.1, -0.05)}})
    )
    a = np.asarray(h_off["loss"])
    b = np.asarray(h_dis["loss"])
    assert a.shape == b.shape
    np.testing.assert_array_equal(a, b)  # EXACT equality


def test_coverage_changes_trajectory_when_enabled():
    _, h_off = train_from_config(_cfg())
    _, h_on = train_from_config(
        _cfg(coverage={"enabled": True, "n_stress": 32, "n_local": 32,
                       "rollout_horizon": 4,
                       "stress_ranges": {"z": (-0.15, -0.05), "k": (7.0, 9.0)}})
    )
    a = np.asarray(h_off["loss"])
    b = np.asarray(h_on["loss"])
    assert np.abs(a - b).max() > 1e-8  # coverage actually feeds gradients
```

Note: `brock_mirman` state names are `("k", "z")` — adjust the stress box accordingly (the `k*≈6.367` SS makes `k∈[7,9]` a mild stress).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_coverage_wiring.py -q`
Expected: FAIL — validators don't exist yet (`_validate_train_config` does not raise; coverage not wired).

- [ ] **Step 3: Add the coverage gates in `_validate_train_config`**

In `src/deqn_jax/training/state_init.py`, inside `_validate_train_config`, after the existing composite/optimizer gates and before the `episode_length==1` check, add:

```python
    if config.coverage.enabled:
        if config.loss_type == "composite":
            raise ValueError(
                "coverage.enabled is not supported with loss_type='composite' "
                "in v1 (both are compute_loss wrappers). Use loss_type='mse'."
            )
        _cov_bad = {"mao", "lm", "gn", "ign", "lbfgs"}
        if config.optimizer.name.lower() in _cov_bad or config.gradient_surgery == "pcgrad":
            raise ValueError(
                f"coverage.enabled requires a STANDARD optimizer; '{config.optimizer.name}'"
                + (" + gradient_surgery='pcgrad'" if config.gradient_surgery == "pcgrad" else "")
                + " differentiates the per-equation/residual vector, so the stress/"
                "local pools (folded into the scalar total) would be silently dropped "
                "from the gradient. Use adam/sgd/adamw/lion/muon/ngd/shampoo."
            )
        if (
            config.barrier_weight > 0
            or config.loss_choice != "mse"
            or config.moment_matching.enabled
        ):
            raise ValueError(
                "coverage.enabled wraps plain MSE compute_loss in v1; disable "
                "barrier_weight / loss_choice!='mse' / moment_matching (they would "
                "be silently dropped on the coverage path)."
            )
```

- [ ] **Step 4: Add the stress-name check in `_resolve_model_for_training`**

In `src/deqn_jax/training/state_init.py`, inside `_resolve_model_for_training`, after the `shock_mask` length check and before `return model, n_equations`, add:

```python
    if config.coverage.enabled:
        unknown = set(config.coverage.stress_ranges) - set(model.state_names)
        if unknown:
            raise ValueError(
                f"coverage.stress_ranges names {sorted(unknown)} are not in "
                f"model.state_names {model.state_names!r} (model={model.name})."
            )
```

- [ ] **Step 5: Install the wrapper in `_build_custom_loss_fn`**

In `src/deqn_jax/training/composite_loss.py`, at the TOP of `_build_custom_loss_fn` (right after the docstring / `from functools import partial`), add:

```python
    if getattr(config, "coverage", None) is not None and config.coverage.enabled:
        from deqn_jax.training.coverage import make_coverage_loss

        if config.verbose:
            print("  Coverage sampling: base + stress + local pools")
        return make_coverage_loss(compute_loss, model, config.coverage)
```

(`compute_loss` is already imported at module top in composite_loss.py — verify; if not, add `from deqn_jax.training.loss import compute_loss`.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_coverage_wiring.py -q`
Expected: PASS (6 passed).

- [ ] **Step 7: Run the full suite (no regressions)**

Run: `uv run pytest tests/ -q`
Expected: PASS (existing count + new tests; no failures).

- [ ] **Step 8: Commit**

```bash
git add src/deqn_jax/training/composite_loss.py src/deqn_jax/training/state_init.py tests/test_coverage_wiring.py
git commit -m "feat(training): wire coverage into loss builder + validators

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: irbc configs + integration smoke

**Files:**
- Create: `configs/irbc_plain.yaml`
- Create: `configs/irbc_ewm.yaml`
- Test: `tests/test_coverage_smoke.py`

**Interfaces:**
- Consumes: the full coverage path (Tasks 1-4); `train_from_config`.
- Produces: two runnable configs; an end-to-end smoke proving the irbc+quadrature+coverage path runs.

- [ ] **Step 1: Write `configs/irbc_plain.yaml`** (the failing-recipe baseline)

```yaml
# 2-country IRBC — PLAIN recipe (reconstructed historical baseline).
# Plain MLP + bare MSE residual, Gauss-Hermite quadrature. No composite
# anchor. This is the recipe that trains to small on-measure residuals but
# lands on a closed-loop UNSTABLE policy (rho(SS)~1.23). The EWM coverage
# run (irbc_ewm.yaml) adds ONLY a coverage block on top of this.
model: irbc

episodes: 4001
batch_size: 128
episode_length: 1
mc_samples: 1
expectation_type: gauss_hermite
n_quadrature_points: 3

initialize_each_episode: true
n_epochs_per_rollout: 1
n_minibatches_per_epoch: 1

loss_type: mse

network:
  type: mlp
  hidden_sizes: [64, 64]
  activation: tanh
  init: xavier_uniform

optimizer:
  name: adam
  learning_rate: 1.0e-3
  lr_schedule: constant

loss_reweight: none
warm_start: false
log_every: 200
```

- [ ] **Step 2: Write `configs/irbc_ewm.yaml`** (plain + coverage, the only delta)

```yaml
# 2-country IRBC — EWM coverage recipe = irbc_plain.yaml + a coverage block.
# Coverage is the ONLY change vs irbc_plain.yaml. Tests whether imposing the
# same residual on base + stress + local pools (stress = deep recession with
# binding irreversibility, rolled through the exact transition) yields a
# stable policy (rho(SS) < 1) WITHOUT the BK-anchor composite recipe.
model: irbc

episodes: 4001
batch_size: 128
episode_length: 1
mc_samples: 1
expectation_type: gauss_hermite
n_quadrature_points: 3

initialize_each_episode: true
n_epochs_per_rollout: 1
n_minibatches_per_epoch: 1

loss_type: mse

network:
  type: mlp
  hidden_sizes: [64, 64]
  activation: tanh
  init: xavier_uniform

optimizer:
  name: adam
  learning_rate: 1.0e-3
  lr_schedule: constant

loss_reweight: none
warm_start: false
log_every: 200

coverage:
  enabled: true
  rho_base: 1.0
  rho_stress: 1.0
  rho_local: 1.0
  n_stress: 128
  n_local: 128
  rollout_horizon: 8
  local_sigma: 0.01
  stress_ranges:
    z_0: [-0.5, -0.2]   # deep recession TFP, country 0
    z_1: [-0.5, -0.2]   # deep recession TFP, country 1
    k_0: [1.05, 1.20]   # high capital -> desired disinvestment -> i>=0 binds
    k_1: [1.05, 1.20]
```

- [ ] **Step 3: Write the smoke test**

```python
# tests/test_coverage_smoke.py
"""End-to-end smoke: irbc + Gauss-Hermite quadrature + coverage runs."""

import numpy as np

from deqn_jax.config import TrainConfig
from deqn_jax.training.trainer import train_from_config


def test_irbc_ewm_smoke_runs():
    cfg = TrainConfig.from_yaml("configs/irbc_ewm.yaml").with_overrides(
        {"episodes": 20, "log_every": 1, "verbose": False}
    )
    _params, history = train_from_config(cfg)
    losses = np.asarray(history["loss"])
    assert len(losses) >= 5
    assert np.all(np.isfinite(losses)), f"non-finite losses: {losses}"


def test_irbc_plain_smoke_runs():
    cfg = TrainConfig.from_yaml("configs/irbc_plain.yaml").with_overrides(
        {"episodes": 20, "log_every": 1, "verbose": False}
    )
    _params, history = train_from_config(cfg)
    losses = np.asarray(history["loss"])
    assert np.all(np.isfinite(losses))
```

- [ ] **Step 4: Run the smoke test**

Run: `uv run pytest tests/test_coverage_smoke.py -q`
Expected: PASS (2 passed). If `with_overrides` doesn't accept `episodes`, build the config via `from_dict` of the YAML dict merged with the overrides instead (see `tests/test_replay_smoke.py` for the `model_validate` pattern).

- [ ] **Step 5: Commit**

```bash
git add configs/irbc_plain.yaml configs/irbc_ewm.yaml tests/test_coverage_smoke.py
git commit -m "feat(configs): irbc_plain + irbc_ewm coverage configs + smoke

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Research validation (manual / DGX — reported finding, not a pytest)

This produces the three-way table; it trains to completion and computes a closed-loop spectral radius, so it runs on the DGX, not in CI. Not a Definition-of-Done gate.

1. **Train all three** (DGX, host uv or container):
   ```bash
   uv run deqn-jax train --config configs/irbc_plain.yaml   # ρ(SS)≈1.23 baseline
   uv run deqn-jax train --config configs/irbc_ewm.yaml     # coverage (the bet)
   uv run deqn-jax train --config configs/irbc.yaml         # BK-anchor (ρ=0.981, known)
   ```
2. **ρ(SS) for each** via the existing evaluator `scripts/evidence_report.py::_rho_ss(policy_net, model)` (gitignored, local-only): load each run's final checkpoint into the irbc model and call `_rho_ss`. Target: `irbc_ewm` gives ρ(SS) < 1.
3. **Stress-region residual:** sample the `stress_ranges` box once with a fixed seed, roll through the exact Γ, and report per-equation `mean (E[r])²` for `fb_0`, `fb_1`, `arc` on `irbc_plain` vs `irbc_ewm` (reuse `compute_residuals` from `training/loss.py` with the quadrature nodes). Headline: % reduction in the max of those three, coverage vs plain.
4. **Record** the result in the spec's "Research finding" table. A ρ(SS) ≥ 1 is a valid negative result, not an engineering failure.

---

## Self-Review

**Spec coverage:** CoverageConfig (Task 1) ✓; roll/seed/local helpers + stop-gradient (Task 2) ✓; mixture wrapper + exact signature + quad-kwarg forwarding + key-split + aux_cov_* (Task 3) ✓; `_build_custom_loss_fn` install + all validators + name check + bit-identical + divergence (Task 4) ✓; two configs + integration smoke (Task 5) ✓; ρ(SS) + stress-grid research validation (Research section) ✓. Non-goals are enforced by Task 4 validators ✓.

**Placeholder scan:** every code step has complete code; commands have expected output. Two spots flag a graceful fallback (network factory entry point in Task 2 Step 4; `with_overrides` shape in Task 5 Step 4) with the concrete alternative — not placeholders.

**Type consistency:** helper signatures in Task 2 Interfaces match their use in Task 3; `coverage_loss_fn` mirrors `compute_loss` (verified against loss.py:264 and composite_loss.py:240); `make_coverage_loss(base_compute_loss, model, cfg)` matches the Task 4 install call `make_coverage_loss(compute_loss, model, config.coverage)`; `aux_cov_base/stress/local` keys consistent across Task 3 and the diagnostics claim.
