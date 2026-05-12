# Mech Interp Pedagogical Walkthrough Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a single narrated `examples/interp_brock_mirman.ipynb` notebook backed by a small `src/deqn_jax/interp.py` module that teaches mechanistic interpretability on a `LinearPlusMLP` trained on `brock_mirman`, with a γ-sweep ∈ {1.0, 2.0, 5.0} as the intensity dial.

**Architecture:** Five pure-function primitives in `src/deqn_jax/interp.py` (`branch_decompose`, `forward_with_activations`, `neuron_contributions`, `linear_probe`, `ablate_neuron`), unit-tested against a fixture `LinearPlusMLP`, then exercised in a 6-chapter notebook on three networks trained at different γs.

**Tech Stack:** JAX + Equinox (existing project stack); `optax` for training; `matplotlib` for figures (already promoted to required dep); `pytest` for tests.

**Spec:** `docs/superpowers/specs/2026-05-11-mech-interp-deqn-design.md`

**Convention note:** the spec mentioned `notebooks/` for the deliverable; the actual codebase uses `examples/` (existing notebooks: `examples/brock_mirman.ipynb`, etc.). The plan uses `examples/` to match the existing pattern. The supporting module path (`src/deqn_jax/interp.py`) is unchanged from the spec.

---

## File Structure

**New files:**
- `src/deqn_jax/interp.py` — the five primitives. Single file, ~250–400 lines. Mirrors the layout of `src/deqn_jax/active_subspace.py`.
- `tests/test_interp.py` — sanity tests for each primitive. ~150 lines.
- `examples/interp_brock_mirman.ipynb` — the narrated teaching artifact.
- `docs/dev/figures/interp/` — directory for saved PNGs from the notebook.

**No existing files are modified.** The notebook reads from the existing public API (`deqn_jax.api`, `deqn_jax.config`, `deqn_jax.training.trainer.train_from_config`, `deqn_jax.models.brock_mirman`, etc.).

---

## Task 1: Scaffold `interp.py` and `test_interp.py` with fixture builder

**Files:**
- Create: `src/deqn_jax/interp.py`
- Create: `tests/test_interp.py`

The interp module needs a stable test fixture: a deterministically-built `LinearPlusMLP` whose forward output is reproducible. Every later task tests against this fixture.

- [ ] **Step 1: Create the empty `interp.py` module with header and section banner**

Create `src/deqn_jax/interp.py`:

```python
"""Mechanistic interpretability primitives for DEQN networks.

Five pure functions for inspecting a trained ``LinearPlusMLP``:

1. ``branch_decompose`` — split the policy into Blanchard-Kahn
   linearization and MLP correction.
2. ``forward_with_activations`` — capture per-layer post-activations.
3. ``neuron_contributions`` — per-neuron attribution to downstream units.
4. ``linear_probe`` — regress concept scalars on hidden activations.
5. ``ablate_neuron`` — zero out a chosen post-activation and rerun.

Companion narrated notebook: ``examples/interp_brock_mirman.ipynb``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from deqn_jax.networks.linear_plus_mlp import LinearPlusMLP
from deqn_jax.networks.mlp import MLP

# ---------------------------------------------------------------------------
# Primitives — populated by subsequent tasks
# ---------------------------------------------------------------------------
```

- [ ] **Step 2: Create `tests/test_interp.py` with the fixture builder**

Create `tests/test_interp.py`:

```python
"""Sanity tests for ``deqn_jax.interp``.

Fixture: a small, deterministic ``LinearPlusMLP`` matching brock_mirman's
shape (2 states, 1 policy) but with hand-set linearization and a default-
initialized MLP under a fixed seed.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from deqn_jax.networks.linear_plus_mlp import LinearPlusMLP


def _make_fixture_net(
    hidden_sizes=(4,),
    seed: int = 0,
    output_link: str = "linear",
) -> LinearPlusMLP:
    """Build a deterministic LinearPlusMLP for brock_mirman-like shape."""
    key = jax.random.PRNGKey(seed)
    return LinearPlusMLP(
        n_states=2,
        n_policies=1,
        hidden_sizes=hidden_sizes,
        activation="tanh",
        P=jnp.array([[0.5, 0.3]]),
        ss_state=jnp.array([1.0, 0.0]),
        ss_policy=jnp.array([0.5]),
        output_links=(output_link,),
        # Wide bounds so clipping never triggers in tests:
        policy_lower=jnp.array([-1e6]),
        policy_upper=jnp.array([1e6]),
        key=key,
    )


def _sample_states(n: int = 32, seed: int = 1) -> jnp.ndarray:
    """Random (k, z) state samples near the fixture SS."""
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(key, (n, 2)) * 0.1 + jnp.array([1.0, 0.0])


def test_fixture_builds_and_evaluates():
    net = _make_fixture_net()
    states = _sample_states()
    out = net(states)
    assert out.shape == (32, 1)
    assert jnp.all(jnp.isfinite(out))
```

- [ ] **Step 3: Run the fixture test to verify it passes**

```bash
uv run pytest tests/test_interp.py::test_fixture_builds_and_evaluates -v
```

Expected: 1 passed.

- [ ] **Step 4: Commit**

```bash
git add src/deqn_jax/interp.py tests/test_interp.py
git commit -m "interp: scaffold module + test fixture builder"
```

---

## Task 2: `branch_decompose`

**Files:**
- Modify: `src/deqn_jax/interp.py` (add `branch_decompose`)
- Modify: `tests/test_interp.py` (add tests)

The function splits `LinearPlusMLP`'s forward into BK linearization, MLP delta, and final policy. Closes-numerically only when no clipping was active (the spec requires this on the ergodic grid).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_interp.py`:

```python
from deqn_jax.interp import branch_decompose


def test_branch_decompose_shapes_and_keys():
    net = _make_fixture_net()
    states = _sample_states()
    out = branch_decompose(net, states)
    assert set(out.keys()) == {"bk", "mlp_delta", "policy", "closes_numerically"}
    assert out["bk"].shape == (32, 1)
    assert out["mlp_delta"].shape == (32, 1)
    assert out["policy"].shape == (32, 1)
    assert isinstance(bool(out["closes_numerically"]), bool)


def test_branch_decompose_closes_numerically_linear_link():
    net = _make_fixture_net(output_link="linear")
    states = _sample_states()
    out = branch_decompose(net, states)
    # bk + mlp_delta should match policy exactly (no clipping)
    reconstructed = out["bk"] + out["mlp_delta"]
    assert jnp.allclose(reconstructed, out["policy"], atol=1e-6)
    assert bool(out["closes_numerically"])


def test_branch_decompose_closes_numerically_log_link():
    # ss_policy > 0 required for log link
    net = _make_fixture_net(output_link="log")
    states = _sample_states()
    out = branch_decompose(net, states)
    # For log link: policy = ss_policy * exp(bk_corr + delta)
    # We return bk in *level* space here: bk = ss_policy * exp(bk_corr)
    # and mlp_delta in *level* space: policy - bk
    # So bk + mlp_delta == policy by construction.
    reconstructed = out["bk"] + out["mlp_delta"]
    assert jnp.allclose(reconstructed, out["policy"], atol=1e-6)


def test_branch_decompose_clip_disables_closure():
    # Tight bounds force clipping; numerical closure should report False.
    net_tight = _make_fixture_net()
    net_tight = eqx.tree_at(
        lambda n: (n.policy_lower, n.policy_upper),
        net_tight,
        (tuple([float(0.49)]), tuple([float(0.51)])),
    )
    states = _sample_states()
    out = branch_decompose(net_tight, states)
    assert not bool(out["closes_numerically"])
```

- [ ] **Step 2: Run to verify the tests fail with "branch_decompose not defined"**

```bash
uv run pytest tests/test_interp.py -v
```

Expected: 4 failures with `ImportError: cannot import name 'branch_decompose'`.

- [ ] **Step 3: Implement `branch_decompose`**

Append to `src/deqn_jax/interp.py`:

```python
def branch_decompose(net: LinearPlusMLP, states: Array) -> Dict[str, Any]:
    """Split a ``LinearPlusMLP``'s policy into BK and MLP components.

    Returns a dict with arrays for ``bk`` (Blanchard-Kahn baseline in
    level space), ``mlp_delta`` (the residual the MLP contributes,
    in level space), ``policy`` (the final clipped output), and a
    boolean ``closes_numerically`` that is true iff
    ``bk + mlp_delta == policy`` to ``1e-6`` everywhere — i.e. no
    clipping was active.

    For log-link outputs we deliberately compute ``bk`` as
    ``ss_policy * exp(P @ (s - ss_state))`` (the BK *level* prediction),
    and ``mlp_delta`` as ``policy - bk``. This keeps the additive
    decomposition meaningful in plot units even when the underlying
    forward composes multiplicatively.

    Args:
        net: A trained ``LinearPlusMLP``.
        states: Array of shape ``[batch, n_states]``.

    Returns:
        Dict with keys ``"bk"``, ``"mlp_delta"``, ``"policy"``,
        ``"closes_numerically"``.
    """
    if states.ndim != 2:
        raise ValueError(
            f"branch_decompose expects states of shape [batch, n_states], "
            f"got shape {states.shape}"
        )

    ss_state = net.ss_state
    ss_policy = net.ss_policy
    P = net.P

    # bk_corr is in the *natural* link space per row (level for linear-link
    # rows, log for log-link rows), since P was pre-converted in the factory.
    bk_corr = (states - ss_state[None, :]) @ P.T  # [batch, n_policies]

    is_log = jnp.asarray(net.output_links, dtype=jnp.int8) == 1  # [n_policies]

    # BK in level space:
    #   linear rows: ss + bk_corr
    #   log rows:    ss * exp(bk_corr)
    bk_linear = ss_policy[None, :] + bk_corr
    bk_log = ss_policy[None, :] * jnp.exp(bk_corr)
    bk = jnp.where(is_log[None, :], bk_log, bk_linear)

    # Final policy via the model's actual forward (handles clipping).
    policy = net(states)

    # mlp_delta is the level-space residual.
    mlp_delta = policy - bk

    closes = jnp.allclose(bk + mlp_delta, policy, atol=1e-6)

    return {
        "bk": bk,
        "mlp_delta": mlp_delta,
        "policy": policy,
        "closes_numerically": closes,
    }
```

- [ ] **Step 4: Run the tests; verify all pass**

```bash
uv run pytest tests/test_interp.py -v
```

Expected: 5 passed (fixture test + 4 new).

- [ ] **Step 5: Commit**

```bash
git add src/deqn_jax/interp.py tests/test_interp.py
git commit -m "interp: branch_decompose (BK + MLP delta + closes_numerically)"
```

---

## Task 3: `forward_with_activations`

**Files:**
- Modify: `src/deqn_jax/interp.py`
- Modify: `tests/test_interp.py`

Mirror `MLP._forward_single` but record every post-activation. The fixture's MLP uses `tanh` activations.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_interp.py`:

```python
from deqn_jax.interp import forward_with_activations


def test_forward_with_activations_keys_and_shapes():
    net = _make_fixture_net(hidden_sizes=(4,))
    states = _sample_states()
    acts = forward_with_activations(net.mlp, states)
    # 1 hidden layer + 1 output
    assert set(acts.keys()) == {"h0", "out"}
    assert acts["h0"].shape == (32, 4)
    assert acts["out"].shape == (32, 1)


def test_forward_with_activations_two_hidden():
    net = _make_fixture_net(hidden_sizes=(4, 3))
    states = _sample_states()
    acts = forward_with_activations(net.mlp, states)
    assert set(acts.keys()) == {"h0", "h1", "out"}
    assert acts["h0"].shape == (32, 4)
    assert acts["h1"].shape == (32, 3)
    assert acts["out"].shape == (32, 1)


def test_forward_with_activations_out_matches_call():
    net = _make_fixture_net(hidden_sizes=(4,))
    states = _sample_states()
    acts = forward_with_activations(net.mlp, states)
    direct = net.mlp(states)  # pre-bounds MLP output
    assert jnp.allclose(acts["out"], direct, atol=1e-6)
```

- [ ] **Step 2: Run to verify the tests fail**

```bash
uv run pytest tests/test_interp.py -v
```

Expected: 3 new failures with `ImportError: cannot import name 'forward_with_activations'`.

- [ ] **Step 3: Implement `forward_with_activations`**

Append to `src/deqn_jax/interp.py`:

```python
def forward_with_activations(mlp: MLP, states: Array) -> Dict[str, Array]:
    """Run ``mlp`` and capture every post-activation along the way.

    Mirrors ``MLP._forward_single`` but yields each hidden layer's
    post-activation. Output keys are ``"h{i}"`` for hidden layer ``i``
    (post-activation) and ``"out"`` for the pre-bounds final output.

    Args:
        mlp: The MLP module (e.g. ``linear_plus_mlp_net.mlp``).
        states: Array of shape ``[batch, n_states]``.

    Returns:
        Dict mapping layer name to activation array. Each hidden layer
        contributes ``"h{i}"``; the final pre-bounds output is ``"out"``.
    """

    def _single(state: Array) -> Dict[str, Array]:
        # Mirror MLP._forward_single's input normalization
        from deqn_jax.networks.mlp import _normalize_input  # local: tiny helper

        x = _normalize_input(state, mlp.input_shift, mlp.input_scale)

        captures: Dict[str, Array] = {}
        for i, layer in enumerate(mlp.layers[:-1]):
            x = mlp.activations[i](layer(x))
            captures[f"h{i}"] = x

        out = mlp.layers[-1](x)
        captures["out"] = out  # pre-bounds output
        return captures

    return jax.vmap(_single)(states)
```

- [ ] **Step 4: Run; verify all pass**

```bash
uv run pytest tests/test_interp.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deqn_jax/interp.py tests/test_interp.py
git commit -m "interp: forward_with_activations (per-layer post-activation capture)"
```

---

## Task 4: `neuron_contributions`

**Files:**
- Modify: `src/deqn_jax/interp.py`
- Modify: `tests/test_interp.py`

For each hidden layer, return `W_next[j, i] * h[i]` per `(batch, hidden_unit_i, downstream_unit_j)`. The contribution explicitly excludes the next-layer bias.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_interp.py`:

```python
from deqn_jax.interp import neuron_contributions


def test_neuron_contributions_shapes_one_hidden():
    net = _make_fixture_net(hidden_sizes=(4,))
    states = _sample_states()
    contribs = neuron_contributions(net.mlp, states)
    # One hidden layer => one entry; shape [batch, hidden, downstream]
    assert set(contribs.keys()) == {0}
    assert contribs[0].shape == (32, 4, 1)  # 4 hidden -> 1 output


def test_neuron_contributions_sum_equals_pre_bias():
    net = _make_fixture_net(hidden_sizes=(4,))
    states = _sample_states()
    contribs = neuron_contributions(net.mlp, states)
    # Sum across hidden neurons + bias of last layer == pre-bounds output
    summed = contribs[0].sum(axis=1)  # [batch, downstream=out]
    bias = net.mlp.layers[-1].bias
    out = forward_with_activations(net.mlp, states)["out"]
    assert jnp.allclose(summed + bias[None, :], out, atol=1e-6)


def test_neuron_contributions_two_hidden_layers():
    net = _make_fixture_net(hidden_sizes=(4, 3))
    states = _sample_states()
    contribs = neuron_contributions(net.mlp, states)
    assert set(contribs.keys()) == {0, 1}
    assert contribs[0].shape == (32, 4, 3)  # hidden0 -> hidden1
    assert contribs[1].shape == (32, 3, 1)  # hidden1 -> output
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_interp.py -v
```

Expected: 3 new failures with `ImportError`.

- [ ] **Step 3: Implement `neuron_contributions`**

Append to `src/deqn_jax/interp.py`:

```python
def neuron_contributions(
    mlp: MLP, states: Array
) -> Dict[int, Array]:
    """Per-neuron contribution to the next layer's pre-activation.

    For hidden layer ``ℓ`` with output activation ``h_ℓ`` of shape
    ``[batch, H_ℓ]``, and the next layer's weight ``W_{ℓ+1}`` of shape
    ``[H_{ℓ+1}, H_ℓ]``, the per-neuron contribution to downstream unit
    ``j`` from hidden unit ``i`` is ``W_{ℓ+1}[j, i] * h_ℓ[batch, i]``.

    Returns a dict keyed by hidden layer index ``ℓ ∈ {0, …, L-1}`` (where
    ``L`` is the number of hidden layers), each value of shape
    ``[batch, H_ℓ, H_{ℓ+1}]`` (or ``[batch, H_ℓ, n_outputs]`` for the last
    hidden layer).

    Bias of the downstream layer is *not* included; the caller can add
    ``mlp.layers[ℓ+1].bias`` if they want full reconstruction.

    Args:
        mlp: The MLP module.
        states: Array of shape ``[batch, n_states]``.

    Returns:
        Dict mapping ``layer_idx -> Array[batch, H_layer, H_downstream]``.
    """
    acts = forward_with_activations(mlp, states)
    out: Dict[int, Array] = {}
    n_hidden = len(mlp.layers) - 1
    for layer_idx in range(n_hidden):
        h = acts[f"h{layer_idx}"]  # [batch, H_layer]
        w = mlp.layers[layer_idx + 1].weight  # [H_downstream, H_layer]
        # Per (b, i, j): w[j, i] * h[b, i]
        # Result shape [batch, H_layer, H_downstream]
        out[layer_idx] = h[:, :, None] * w.T[None, :, :]
    return out
```

- [ ] **Step 4: Run; verify all pass**

```bash
uv run pytest tests/test_interp.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deqn_jax/interp.py tests/test_interp.py
git commit -m "interp: neuron_contributions (per-neuron output attribution)"
```

---

## Task 5: `linear_probe`

**Files:**
- Modify: `src/deqn_jax/interp.py`
- Modify: `tests/test_interp.py`

Per-(neuron, concept) 1-D linear regression. No regularization, no joint fitting.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_interp.py`:

```python
from deqn_jax.interp import linear_probe


def test_linear_probe_perfect_fit():
    n = 100
    key = jax.random.PRNGKey(42)
    c = jax.random.normal(key, (n, 1))
    activations = 3.0 * c + 1.0  # perfect linear relationship, shape [n, 1]
    out = linear_probe(activations, c)
    assert out["r2"].shape == (1, 1)
    assert jnp.isclose(out["r2"][0, 0], 1.0, atol=1e-5)
    assert jnp.isclose(out["coef"][0, 0], 3.0, atol=1e-4)


def test_linear_probe_no_fit():
    n = 1000
    key = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(key)
    activations = jax.random.normal(k1, (n, 4))
    concepts = jax.random.normal(k2, (n, 3))
    out = linear_probe(activations, concepts)
    assert out["r2"].shape == (4, 3)
    # All R² should be near zero (random pairs)
    assert jnp.all(out["r2"] < 0.05)


def test_linear_probe_constant_activation_handled():
    # Constant activations have zero variance — R² should be 0, not NaN.
    n = 50
    activations = jnp.ones((n, 1)) * 2.7
    concepts = jnp.arange(n, dtype=jnp.float32).reshape(n, 1)
    out = linear_probe(activations, concepts)
    assert jnp.all(jnp.isfinite(out["r2"]))
    assert jnp.isclose(out["r2"][0, 0], 0.0, atol=1e-5)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_interp.py -v
```

Expected: 3 new failures with `ImportError`.

- [ ] **Step 3: Implement `linear_probe`**

Append to `src/deqn_jax/interp.py`:

```python
def linear_probe(activations: Array, concepts: Array) -> Dict[str, Array]:
    """Per-(neuron, concept) univariate linear regression.

    For each pair ``(i, j)`` of neuron ``i`` and concept ``j``, fits
    ``activations[:, i] ≈ a * concepts[:, j] + b`` and returns the slope
    ``a``, the coefficient of determination ``R²``, and the residual
    variance.

    R² uses the standard formula
    ``1 - SS_res / SS_tot`` where ``SS_tot`` is the sample variance of
    the activation (with denominator ``n``). When ``SS_tot`` is zero
    (constant activation) R² is defined as 0.

    No regularization, no joint regression across concepts. Concepts
    should be pre-scaled by the caller if they want comparable
    coefficients.

    Args:
        activations: Array of shape ``[batch, n_neurons]``.
        concepts: Array of shape ``[batch, n_concepts]``.

    Returns:
        Dict with:
          - ``"coef"``: ``Array[n_neurons, n_concepts]`` — slope per pair.
          - ``"r2"``: ``Array[n_neurons, n_concepts]`` — coefficient of
            determination per pair.
          - ``"residual_var"``: ``Array[n_neurons, n_concepts]`` —
            variance of the residual per pair.
    """
    if activations.ndim != 2 or concepts.ndim != 2:
        raise ValueError(
            f"activations and concepts must both be 2-D; got "
            f"activations.shape={activations.shape}, concepts.shape={concepts.shape}"
        )
    if activations.shape[0] != concepts.shape[0]:
        raise ValueError(
            f"batch size mismatch: activations has {activations.shape[0]} rows, "
            f"concepts has {concepts.shape[0]}"
        )

    n = activations.shape[0]
    # Center: simpler closed-form (b = mean(a) - slope*mean(c))
    a_mean = activations.mean(axis=0, keepdims=True)  # [1, n_neurons]
    c_mean = concepts.mean(axis=0, keepdims=True)  # [1, n_concepts]
    a_c = activations - a_mean
    c_c = concepts - c_mean

    # cov[i, j] = mean over batch of a_c[:, i] * c_c[:, j]
    # = (a_c.T @ c_c) / n
    cov = (a_c.T @ c_c) / n  # [n_neurons, n_concepts]
    c_var = (c_c**2).mean(axis=0)  # [n_concepts]
    a_var = (a_c**2).mean(axis=0)  # [n_neurons]

    # slope_ij = cov_ij / c_var_j
    eps = 1e-12
    coef = cov / (c_var[None, :] + eps)

    # SS_res_ij / n = a_var_i - 2*slope_ij*cov_ij + slope_ij^2*c_var_j
    #               = a_var_i - cov_ij^2 / c_var_j
    residual_var = a_var[:, None] - cov**2 / (c_var[None, :] + eps)

    # R²_ij = 1 - residual_var_ij / a_var_i; zero when a_var_i == 0.
    r2_raw = 1.0 - residual_var / (a_var[:, None] + eps)
    r2 = jnp.where(a_var[:, None] > eps, r2_raw, jnp.zeros_like(r2_raw))

    return {
        "coef": coef,
        "r2": r2,
        "residual_var": residual_var,
    }
```

- [ ] **Step 4: Run; verify all pass**

```bash
uv run pytest tests/test_interp.py -v
```

Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deqn_jax/interp.py tests/test_interp.py
git commit -m "interp: linear_probe (per-neuron-per-concept R² regression)"
```

---

## Task 6: `ablate_neuron`

**Files:**
- Modify: `src/deqn_jax/interp.py`
- Modify: `tests/test_interp.py`

Forward pass with a chosen post-activation forced to zero. Returns the perturbed policy through the full `LinearPlusMLP` (so clipping and link-type are respected).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_interp.py`:

```python
from deqn_jax.interp import ablate_neuron


def test_ablate_neuron_predicted_diff():
    net = _make_fixture_net(hidden_sizes=(4,))
    states = _sample_states()
    baseline = net(states)
    # Predict the diff via neuron_contributions: zeroing neuron i removes
    # contribution W_last[:, i] * h_last[:, i] from the MLP's pre-bounds out.
    contribs = neuron_contributions(net.mlp, states)
    predicted_mlp_out_drop = contribs[0][:, 1, :]  # [batch, downstream=1]
    # The LinearPlusMLP composition (linear link, no clip) is additive in
    # delta, so the change in final policy equals the change in mlp output:
    ablated = ablate_neuron(net, layer_idx=0, neuron_idx=1, states=states)
    actual_diff = baseline - ablated  # [batch, 1]
    assert jnp.allclose(actual_diff, predicted_mlp_out_drop, atol=1e-6)


def test_ablate_neuron_shape_matches_policy():
    net = _make_fixture_net(hidden_sizes=(4, 3))
    states = _sample_states()
    ablated = ablate_neuron(net, layer_idx=1, neuron_idx=0, states=states)
    assert ablated.shape == (32, 1)


def test_ablate_neuron_zero_idx_is_idempotent_if_h_is_zero():
    # If the activation we ablate is already ~0, the output should be ~unchanged.
    net = _make_fixture_net(hidden_sizes=(4,))
    states = _sample_states()
    acts = forward_with_activations(net.mlp, states)
    # Find the neuron with the smallest |activation| sum (closest to dead).
    sums = jnp.abs(acts["h0"]).sum(axis=0)  # [4]
    neuron_idx = int(jnp.argmin(sums))
    baseline = net(states)
    ablated = ablate_neuron(net, layer_idx=0, neuron_idx=neuron_idx, states=states)
    # Diff should be small but not necessarily zero; bound it generously.
    assert float(jnp.max(jnp.abs(baseline - ablated))) < 5.0
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_interp.py -v
```

Expected: 3 new failures with `ImportError`.

- [ ] **Step 3: Implement `ablate_neuron`**

Append to `src/deqn_jax/interp.py`:

```python
def ablate_neuron(
    net: LinearPlusMLP,
    layer_idx: int,
    neuron_idx: int,
    states: Array,
) -> Array:
    """Run the network with a chosen post-activation forced to zero.

    Mirrors ``LinearPlusMLP._forward_single`` and ``MLP._forward_single``
    exactly, except that after computing the post-activation of hidden
    layer ``layer_idx``, we zero entry ``neuron_idx`` before passing on.

    Args:
        net: The ``LinearPlusMLP`` to inspect (unchanged).
        layer_idx: Which hidden layer to ablate in (0-indexed).
        neuron_idx: Which neuron within that layer to zero.
        states: Array of shape ``[batch, n_states]``.

    Returns:
        The policy with the chosen post-activation forced to zero, shape
        ``[batch, n_policies]``. Full clipping + link-type semantics from
        ``LinearPlusMLP._forward_single`` are preserved.
    """
    from deqn_jax.networks.mlp import _normalize_input

    n_hidden = len(net.mlp.layers) - 1
    if not 0 <= layer_idx < n_hidden:
        raise ValueError(
            f"layer_idx {layer_idx} out of range for {n_hidden} hidden layer(s)"
        )
    hidden_size_at_layer = net.mlp.layers[layer_idx].weight.shape[0]
    if not 0 <= neuron_idx < hidden_size_at_layer:
        raise ValueError(
            f"neuron_idx {neuron_idx} out of range for layer {layer_idx} "
            f"with size {hidden_size_at_layer}"
        )

    mlp = net.mlp

    def _single(state: Array) -> Array:
        # MLP forward with ablation
        x = _normalize_input(state, mlp.input_shift, mlp.input_scale)
        for i, layer in enumerate(mlp.layers[:-1]):
            x = mlp.activations[i](layer(x))
            if i == layer_idx:
                x = x.at[neuron_idx].set(0.0)
        delta = mlp.layers[-1](x)
        # NB: MLP bounds intentionally not applied — LinearPlusMLP uses MLP
        # without output_lower/upper, matching its own forward pass.

        # LinearPlusMLP composition (mirror _forward_single):
        ss_state = jax.lax.stop_gradient(net.ss_state)
        ss_policy = jax.lax.stop_gradient(net.ss_policy)
        P = jax.lax.stop_gradient(net.P)
        bk_corr = P @ (state - ss_state)

        if all(code == 0 for code in net.output_links):
            raw = ss_policy + bk_corr + delta
        elif all(code == 1 for code in net.output_links):
            raw = ss_policy * jnp.exp(bk_corr + delta)
        else:
            is_log = jnp.asarray(net.output_links, dtype=jnp.int8) == 1
            raw_linear = ss_policy + bk_corr + delta
            raw_log = ss_policy * jnp.exp(bk_corr + delta)
            raw = jnp.where(is_log, raw_log, raw_linear)

        if net.policy_lower is not None:
            lower = jax.lax.stop_gradient(jnp.asarray(net.policy_lower))
            raw = jnp.maximum(raw, lower)
        if net.policy_upper is not None:
            upper = jax.lax.stop_gradient(jnp.asarray(net.policy_upper))
            safe_upper = jnp.where(jnp.isinf(upper), jnp.array(1e10), upper)
            raw = jnp.minimum(raw, safe_upper)
        return raw

    return jax.vmap(_single)(states)
```

- [ ] **Step 4: Run; verify all pass**

```bash
uv run pytest tests/test_interp.py -v
```

Expected: 17 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deqn_jax/interp.py tests/test_interp.py
git commit -m "interp: ablate_neuron (zero a chosen post-activation, rerun)"
```

---

## Task 7: End-to-end wiring test

**Files:**
- Modify: `tests/test_interp.py`

A single integration test that trains a tiny brock_mirman LinearPlusMLP for a handful of steps and runs every primitive on it. Not a correctness test — wiring only.

- [ ] **Step 1: Write the test**

Append to `tests/test_interp.py`:

```python
@pytest.mark.slow  # ~5–10 seconds; opt-in via -m slow if filtering elsewhere
def test_end_to_end_wiring_on_trained_brock_mirman():
    from deqn_jax.config import (
        NetworkConfig,
        OptimizerConfig,
        TrainConfig,
    )
    from deqn_jax.training.trainer import train_from_config

    cfg = TrainConfig(
        model="brock_mirman",
        n_episodes=5,
        episode_length=32,
        batch_size=16,
        mc_samples=2,
        seed=0,
        network=NetworkConfig(
            kind="linear_plus_mlp",
            hidden_sizes=(8,),
            activation="tanh",
        ),
        optimizer=OptimizerConfig(name="adam", learning_rate=1e-3),
    )
    trained, _history = train_from_config(cfg)

    # Sample a small state batch and exercise every primitive.
    states = _sample_states(n=8)
    bd = branch_decompose(trained, states)
    assert bd["policy"].shape == (8, 1)

    acts = forward_with_activations(trained.mlp, states)
    assert acts["out"].shape == (8, 1)

    contribs = neuron_contributions(trained.mlp, states)
    assert contribs[0].shape == (8, 8, 1)

    probe = linear_probe(acts["h0"], states)
    assert probe["r2"].shape == (8, 2)

    ablated = ablate_neuron(trained, 0, 0, states)
    assert ablated.shape == (8, 1)
```

- [ ] **Step 2: Run the wiring test**

```bash
uv run pytest tests/test_interp.py -v
```

Expected: 18 passed.

If the `TrainConfig` field names differ from what's shown, consult `src/deqn_jax/config.py` and adjust the test inline (do not mock — the point is real wiring).

- [ ] **Step 3: Commit**

```bash
git add tests/test_interp.py
git commit -m "interp: end-to-end wiring test on trained brock_mirman"
```

---

## Task 8: Notebook scaffold + Chapter 0 (setup, training, baseline plot)

**Files:**
- Create: `examples/interp_brock_mirman.ipynb`
- Create: `docs/dev/figures/interp/` (empty directory; `.gitkeep` if needed)

The notebook trains three networks at γ ∈ {1.0, 2.0, 5.0} and plots the γ=2.0 policy. Hidden size starts at (16, 16); reduce/adjust if Chapter 2 reveals all-dead or all-generic neurons.

- [ ] **Step 1: Create the figures directory**

```bash
mkdir -p docs/dev/figures/interp
touch docs/dev/figures/interp/.gitkeep
```

- [ ] **Step 2: Create the notebook with Chapter 0 cells**

Create `examples/interp_brock_mirman.ipynb` as a Jupyter notebook with the following cells. Use `nbformat` or open Jupyter directly; here is the cell content.

**Cell 0 (markdown):**

```markdown
# Mech Interp on DEQNs: Brock-Mirman Walkthrough

A six-chapter narrated example. We train tiny `LinearPlusMLP` networks
on Brock-Mirman at three risk-aversion settings, then peel them open
chapter by chapter.

Each chapter introduces one mech-interp move on a substrate where
every claim is checkable against a known economic solution.

**Outline:**
- Ch 0 — Setup: train the networks, see what they compute
- Ch 1 — Output decomposition (BK linearization vs MLP correction)
- Ch 2 — Per-neuron contributions inside the MLP correction
- Ch 3 — Linear probes: what do live neurons encode?
- Ch 4 — Ablation: causation vs correlation
- Ch 5 — The intensity dial: γ ∈ {1, 2, 5}
- Ch 6 — Honest limits and pointers forward
```

**Cell 1 (code):**

```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from deqn_jax.config import NetworkConfig, OptimizerConfig, TrainConfig
from deqn_jax.training.trainer import train_from_config
from deqn_jax.interp import (
    ablate_neuron,
    branch_decompose,
    forward_with_activations,
    linear_probe,
    neuron_contributions,
)

FIGDIR = "../docs/dev/figures/interp"
GAMMAS = (1.0, 2.0, 5.0)
HIDDEN = (16, 16)
SEED = 0
N_EPISODES = 300  # adjust if loss hasn't converged
```

**Cell 2 (markdown):**

```markdown
## Chapter 0 — Setup

We train three Brock-Mirman policies, one per γ. Each is a
`LinearPlusMLP` with hidden sizes `(16, 16)` and the same seed. The
linear branch is the Blanchard-Kahn first-order solution of the model
(fixed, not trained). The MLP branch is initialized to ≈0 (last-layer
weights scaled by 0.01, biases zero), so at init the network *is*
Blanchard-Kahn. Training carves the nonlinear correction.

Below we train one network and plot its learned policy. Chapters 1–4
focus on the γ=2 network; γ=1 and γ=5 come back in Chapter 5.
```

**Cell 3 (code):**

```python
def train_one(gamma: float) -> "LinearPlusMLP":
    cfg = TrainConfig(
        model="brock_mirman",
        model_constants={"gamma": gamma},
        n_episodes=N_EPISODES,
        episode_length=128,
        batch_size=64,
        mc_samples=8,
        seed=SEED,
        network=NetworkConfig(
            kind="linear_plus_mlp",
            hidden_sizes=HIDDEN,
            activation="tanh",
        ),
        optimizer=OptimizerConfig(name="adam", learning_rate=1e-3),
    )
    trained, history = train_from_config(cfg)
    final_loss = history["loss"][-1] if "loss" in history else float("nan")
    print(f"γ={gamma}  final loss={final_loss:.3e}")
    return trained


# Train all three; keep handles for later chapters.
nets = {g: train_one(g) for g in GAMMAS}
net = nets[2.0]  # primary network for Ch 1–4
```

Note: if `model_constants` is not the actual override field on
`TrainConfig`, check `src/deqn_jax/config.py` and substitute the correct
field name (e.g., `constants_override` or similar). Do not pass γ via
an unsupported field — fix the call here rather than ignoring the warning.

**Cell 4 (code):**

```python
def state_grid(net, n=50):
    """Grid over ±2σ of the ergodic support around the network's SS."""
    k_ss = float(net.ss_state[0])
    z_ss = float(net.ss_state[1])
    sigma_z = 0.04
    rho = 0.9
    z_std = sigma_z / np.sqrt(1.0 - rho**2)
    ks = np.linspace(0.7 * k_ss, 1.3 * k_ss, n)
    zs = np.linspace(z_ss - 2 * z_std, z_ss + 2 * z_std, n)
    K, Z = np.meshgrid(ks, zs)
    states = jnp.stack([K.ravel(), Z.ravel()], axis=-1)
    return states, K, Z


states, K, Z = state_grid(net)
policy = np.asarray(net(states)).reshape(K.shape)

fig, ax = plt.subplots(figsize=(5, 4))
pcm = ax.pcolormesh(K, Z, policy, shading="auto", cmap="viridis")
ax.set_xlabel("capital k")
ax.set_ylabel("TFP z")
ax.set_title("Learned savings rate s(k, z) — γ=2")
fig.colorbar(pcm, ax=ax, label="s")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/ch0_policy_gamma2.png", dpi=150)
plt.show()
```

- [ ] **Step 3: Execute the notebook end-to-end**

```bash
uv run jupyter nbconvert --to notebook --execute examples/interp_brock_mirman.ipynb --output examples/interp_brock_mirman.ipynb
```

Expected: completes without error; one PNG saved to `docs/dev/figures/interp/`; the policy heatmap should look smooth and increasing in `z`.

- [ ] **Step 4: Commit**

```bash
git add examples/interp_brock_mirman.ipynb docs/dev/figures/interp/
git commit -m "interp: notebook Ch 0 (train 3 nets, plot γ=2 policy)"
```

---

## Task 9: Notebook Chapter 1 — Output decomposition

**Files:**
- Modify: `examples/interp_brock_mirman.ipynb`

Visualize BK baseline, MLP correction, and combined policy side-by-side on the γ=2 grid.

- [ ] **Step 1: Append Chapter 1 cells to the notebook**

**Markdown:**

```markdown
## Chapter 1 — Output decomposition

The first mech-interp move on any model is the same: try to write the
output as a sum of separately-interpretable components. `LinearPlusMLP`
gives us this for free.

```
policy(s) = π_BK(s) + δ_θ(s)         (linear-link case; clipping aside)
           ↑          ↑
       perturbation   the only thing
       theory's       the neural net adds
       prediction
```

Macro people: π_BK is what you'd get from a Blanchard-Kahn solver. δ_θ
is the network's nonlinear correction.

ML people: this is a residual network with a hand-derived skip connection.
```

**Code:**

```python
bd = branch_decompose(net, states)
print(f"Closes numerically on grid: {bool(bd['closes_numerically'])}")

bk = np.asarray(bd["bk"]).reshape(K.shape)
mlp_delta = np.asarray(bd["mlp_delta"]).reshape(K.shape)
combined = np.asarray(bd["policy"]).reshape(K.shape)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, arr, title in zip(
    axes,
    (bk, mlp_delta, combined),
    ("π_BK (linearization)", "δ_θ (MLP correction)", "policy = clip(π_BK + δ_θ)"),
):
    pcm = ax.pcolormesh(K, Z, arr, shading="auto", cmap="viridis")
    ax.set_xlabel("k"); ax.set_ylabel("z"); ax.set_title(title)
    fig.colorbar(pcm, ax=ax)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/ch1_decomposition_gamma2.png", dpi=150)
plt.show()

# Magnitudes
print(f"|π_BK|_max = {np.abs(bk).max():.4f}")
print(f"|δ_θ|_max  = {np.abs(mlp_delta).max():.4f}")
print(f"ratio δ/π  = {np.abs(mlp_delta).max() / np.abs(bk).max():.3%}")
```

**Markdown (closer):**

```markdown
Two things to notice:
- `closes_numerically` is true — `π_BK + δ_θ` reconstructs the policy
  exactly on the grid. No clipping fires inside the ergodic support.
- The MLP correction is a small fraction of the BK baseline in
  magnitude, but its spatial pattern is *structured*. The next chapter
  opens that pattern up.
```

- [ ] **Step 2: Execute the notebook end-to-end**

```bash
uv run jupyter nbconvert --to notebook --execute examples/interp_brock_mirman.ipynb --output examples/interp_brock_mirman.ipynb
```

Expected: completes without error; `closes_numerically` prints `True`; three side-by-side heatmaps render and save.

- [ ] **Step 3: Commit**

```bash
git add examples/interp_brock_mirman.ipynb docs/dev/figures/interp/
git commit -m "interp: notebook Ch 1 (BK vs MLP delta decomposition)"
```

---

## Task 10: Notebook Chapter 2 — Per-neuron contributions

**Files:**
- Modify: `examples/interp_brock_mirman.ipynb`

Identify dead / generic / selective neurons in the last hidden layer.

- [ ] **Step 1: Append Chapter 2 cells**

**Markdown:**

```markdown
## Chapter 2 — Per-neuron contributions

The MLP correction is computed by a small network — at this size,
`(16, 16)` hidden units. Not every neuron is doing the same amount of
work. The simplest interp move is to ask, for each neuron in the last
hidden layer, how much it actually contributes to the output.

Per-neuron contribution to the network's output is `W_last[j, i] · h_last[i]`
— the weight times the post-activation. Summed across `i`, this plus the
last-layer bias reconstructs the pre-bounds MLP output.

We separate three archetypes you can spot in any small MLP:
- **dead** — near-zero activation everywhere.
- **generic** — smooth, roughly-linear-in-state activation; carries a low-order trend.
- **selective** — strong only in part of state space; the interesting case for interp.
```

**Code:**

```python
contribs = neuron_contributions(net.mlp, states)
last_layer_idx = max(contribs.keys())  # 1 if hidden=(16,16), 0 if hidden=(16,)
contrib_last = np.asarray(contribs[last_layer_idx])  # [batch, H, 1]
# Mean absolute contribution per neuron over the ergodic grid:
mean_abs = np.abs(contrib_last[..., 0]).mean(axis=0)  # [H]
order = np.argsort(-mean_abs)  # descending

fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(range(len(mean_abs)), mean_abs[order])
ax.set_xlabel("neuron (ranked)")
ax.set_ylabel("mean |contribution|")
ax.set_title(f"Last-hidden-layer (idx {last_layer_idx}) contribution magnitude — γ=2")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/ch2_contribution_bar_gamma2.png", dpi=150)
plt.show()
print("ranked neuron indices (descending):", order.tolist())
```

**Code:**

```python
# Heatmaps for top-3 and bottom-3 ranked neurons.
H_last = contrib_last.shape[1]
picks = list(order[:3]) + list(order[-3:])
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for ax, i in zip(axes.flat, picks):
    grid = contrib_last[:, i, 0].reshape(K.shape)
    pcm = ax.pcolormesh(K, Z, grid, shading="auto", cmap="RdBu_r",
                         vmin=-np.abs(grid).max(), vmax=np.abs(grid).max())
    ax.set_title(f"neuron {i}  (mean |·| = {mean_abs[i]:.4f})")
    fig.colorbar(pcm, ax=ax)
fig.suptitle("Top-3 (top row) and bottom-3 (bottom row) contributors — γ=2")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/ch2_contribution_heatmaps_gamma2.png", dpi=150)
plt.show()
```

**Markdown (closer):**

```markdown
**Sanity check:** for any neuron `i` in the last hidden layer, summing
its contribution across all neurons gives back the network's
pre-bias MLP output. Verify:
```

**Code:**

```python
acts = forward_with_activations(net.mlp, states)
summed = contrib_last.sum(axis=1)  # [batch, 1]
bias = np.asarray(net.mlp.layers[-1].bias)
out = np.asarray(acts["out"])
print(f"max |summed + bias - out| = {np.abs(summed + bias - out).max():.2e}")
assert np.allclose(summed + bias[None, :], out, atol=1e-5)
```

**Markdown:**

```markdown
If the bar chart shows most contribution concentrated in a few neurons
and the rest near zero, the network learned a sparse solution and the
analysis ahead has a clear target. If the bar is uniform-looking, the
network spread the work — more honest probes will be needed.
```

- [ ] **Step 2: Execute the notebook end-to-end**

```bash
uv run jupyter nbconvert --to notebook --execute examples/interp_brock_mirman.ipynb --output examples/interp_brock_mirman.ipynb
```

Expected: bar chart and 2×3 heatmap grid render; sanity print < 1e-5.

If all 16 last-hidden neurons end up uniformly dead (max mean |contribution| < 1e-4), bump `N_EPISODES` in Cell 1 or check the loss trajectory — the network probably didn't train. If everything is generic-looking, drop hidden size to `(8,)` in Cell 1 to coax sparsity.

- [ ] **Step 3: Commit**

```bash
git add examples/interp_brock_mirman.ipynb docs/dev/figures/interp/
git commit -m "interp: notebook Ch 2 (per-neuron contribution maps)"
```

---

## Task 11: Notebook Chapter 3 — Linear probes

**Files:**
- Modify: `examples/interp_brock_mirman.ipynb`

Regress hand-chosen concept scalars onto last-hidden activations.

- [ ] **Step 1: Append Chapter 3 cells**

**Markdown:**

```markdown
## Chapter 3 — Linear probes

A neuron's post-activation is a scalar function of state. We can ask:
*is that function "explained" by some interpretable scalar we name*?

The procedure: pick a list of candidate concepts (in state-space units
that mean something to a domain expert), and for each (neuron, concept)
pair, fit a 1-D linear regression. The R² of that fit tells us how
much of the neuron's variation we can read off as a single named
quantity.

Caveat — and Chapter 4 will press on this — probes are correlational.
A high R² says "this neuron tracks concept X." It does not say "the
network is using this neuron to compute X."
```

**Code:**

```python
# Concept basis on the grid.
k = states[:, 0]
z = states[:, 1]
alpha = 0.36  # brock_mirman constant
k_ss = float(net.ss_state[0])
z_ss = float(net.ss_state[1])

concepts = jnp.stack([
    k,
    z,
    k * z,
    k**2,
    z**2,
    jnp.log(k),
    z,  # log(z) ill-defined for z<0 — use z itself again
    k - k_ss,
    z - z_ss,
    z * jnp.power(k, alpha),  # y
    alpha * z * jnp.power(k, alpha - 1.0),  # mpk
], axis=-1)

concept_names = [
    "k", "z", "k·z", "k²", "z²", "log k",
    "z (=log z proxy)", "k-k_ss", "z-z_ss", "y = z·k^α", "mpk",
]

probe = linear_probe(acts["h0"] if last_layer_idx == 0 else acts[f"h{last_layer_idx}"],
                      concepts)
r2 = np.asarray(probe["r2"])  # [n_neurons, n_concepts]

fig, ax = plt.subplots(figsize=(9, 6))
pcm = ax.imshow(r2[order, :], vmin=0.0, vmax=1.0, cmap="magma", aspect="auto")
ax.set_xticks(range(len(concept_names)))
ax.set_xticklabels(concept_names, rotation=40, ha="right")
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"n{int(i)}" for i in order])
ax.set_xlabel("concept")
ax.set_ylabel("neuron (ranked by mean |contribution|)")
ax.set_title("Linear-probe R² (per neuron × per concept) — γ=2")
fig.colorbar(pcm, ax=ax, label="R²")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/ch3_probe_r2_gamma2.png", dpi=150)
plt.show()
```

**Markdown (closer):**

```markdown
Reading the heatmap:
- A row that's mostly black is a neuron that *no listed concept* explains.
  Either it encodes a nonlinear combination of state, or it encodes
  noise.
- A row with one bright cell is a neuron well-explained by that single
  concept. Pick the brightest example and report it as "this neuron
  tracks X."
- A row with two-or-three medium cells is a neuron mixing concepts —
  much harder to read.

Print the cleanest claim:
```

**Code:**

```python
best_neuron, best_concept = np.unravel_index(np.argmax(r2), r2.shape)
print(f"Highest R² = {r2[best_neuron, best_concept]:.3f}")
print(f"  neuron {best_neuron} ≈ concept '{concept_names[best_concept]}'")
```

- [ ] **Step 2: Execute the notebook end-to-end**

```bash
uv run jupyter nbconvert --to notebook --execute examples/interp_brock_mirman.ipynb --output examples/interp_brock_mirman.ipynb
```

Expected: heatmap renders; "highest R²" print produces a sensible-looking claim.

- [ ] **Step 3: Commit**

```bash
git add examples/interp_brock_mirman.ipynb docs/dev/figures/interp/
git commit -m "interp: notebook Ch 3 (linear probes on hidden activations)"
```

---

## Task 12: Notebook Chapter 4 — Ablation

**Files:**
- Modify: `examples/interp_brock_mirman.ipynb`

Ablate each live neuron, compare to its probe R²: build the correlation × causation 2×2.

- [ ] **Step 1: Append Chapter 4 cells**

**Markdown:**

```markdown
## Chapter 4 — Ablation (causation, not just correlation)

Probes are correlational. To ask *whether the network uses* a neuron,
we run a causal intervention: force that neuron's post-activation to
zero, recompute the policy, and measure the change.

A neuron with high probe-R² *and* large ablation-effect is a strong
candidate for "real feature encoding concept X."
A neuron with high R² and small ablation-effect is redundant.
A neuron with low R² and large ablation-effect is interesting — it
matters for the policy in ways we can't read off from our concept list.
A neuron with low R² and small ablation-effect is genuinely dead.
```

**Code:**

```python
baseline = np.asarray(net(states))
H_last = contrib_last.shape[1]
necessity = np.zeros(H_last)
for i in range(H_last):
    ablated = np.asarray(ablate_neuron(net, last_layer_idx, i, states))
    necessity[i] = float(np.linalg.norm(baseline - ablated))

fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(range(H_last), necessity[order])
ax.set_xlabel("neuron (ranked by contribution)")
ax.set_ylabel("‖Δpolicy‖ on ablation")
ax.set_title("Necessity score per neuron — γ=2")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/ch4_necessity_gamma2.png", dpi=150)
plt.show()
```

**Code:**

```python
# Scatter: probe-R² (max over concepts) vs necessity, per neuron
r2_max_per_neuron = r2.max(axis=1)
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(r2_max_per_neuron, necessity)
for i in range(H_last):
    ax.annotate(str(i), (r2_max_per_neuron[i], necessity[i]),
                 fontsize=8, alpha=0.6)
ax.set_xlabel("max probe R² (correlational)")
ax.set_ylabel("‖Δpolicy‖ (causal)")
ax.set_title("Correlation × causation per neuron — γ=2")
ax.set_xlim(-0.05, 1.05)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/ch4_corr_vs_causal_gamma2.png", dpi=150)
plt.show()
```

**Markdown (closer):**

```markdown
The four quadrants of the scatter give the 2×2 from the chapter
intro. The most interesting story to find is a neuron in the
*upper-left*: low probe R² yet high necessity — the network needs it
but our concept list doesn't read it. That's the gap where deeper
interp earns its money.
```

- [ ] **Step 2: Execute the notebook end-to-end**

```bash
uv run jupyter nbconvert --to notebook --execute examples/interp_brock_mirman.ipynb --output examples/interp_brock_mirman.ipynb
```

Expected: bar chart + scatter render; scatter labels visible.

- [ ] **Step 3: Commit**

```bash
git add examples/interp_brock_mirman.ipynb docs/dev/figures/interp/
git commit -m "interp: notebook Ch 4 (ablation + correlation-vs-causation scatter)"
```

---

## Task 13: Notebook Chapter 5 (γ-sweep) + Chapter 6 (limits)

**Files:**
- Modify: `examples/interp_brock_mirman.ipynb`

Compact panel comparing all three networks. Then honest-limits prose.

- [ ] **Step 1: Append Chapter 5 cells**

**Markdown:**

```markdown
## Chapter 5 — The intensity dial: γ ∈ {1, 2, 5}

Risk aversion γ controls how curved the household's utility is. Higher
γ means the optimal policy bends more away from the linearization. So
γ is an *intensity dial* for how much nonlinear correction the MLP
branch needs to encode.

We run Chapters 1–4 compactly on all three networks.
```

**Code:**

```python
fig, axes = plt.subplots(3, 3, figsize=(13, 11))

for row, g in enumerate(GAMMAS):
    net_g = nets[g]
    states_g, K_g, Z_g = state_grid(net_g)
    bd = branch_decompose(net_g, states_g)
    bk_g = np.asarray(bd["bk"]).reshape(K_g.shape)
    delta_g = np.asarray(bd["mlp_delta"]).reshape(K_g.shape)
    policy_g = np.asarray(bd["policy"]).reshape(K_g.shape)
    for ax, arr, title in zip(
        axes[row],
        (bk_g, delta_g, policy_g),
        ("π_BK", "δ_θ", "policy"),
    ):
        pcm = ax.pcolormesh(K_g, Z_g, arr, shading="auto", cmap="viridis")
        ax.set_title(f"γ={g}  {title}")
        fig.colorbar(pcm, ax=ax)
        if row == 2:
            ax.set_xlabel("k")
        if title == "π_BK":
            ax.set_ylabel("z")
fig.suptitle("Branch decomposition across γ")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/ch5_decomposition_gamma_sweep.png", dpi=150)
plt.show()
```

**Code:**

```python
# Necessity bar charts side-by-side.
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
for ax, g in zip(axes, GAMMAS):
    net_g = nets[g]
    states_g, _, _ = state_grid(net_g, n=30)  # coarser grid for speed
    baseline_g = np.asarray(net_g(states_g))
    H_last_g = net_g.mlp.layers[-2].weight.shape[0]
    necessity_g = np.zeros(H_last_g)
    for i in range(H_last_g):
        ab = np.asarray(ablate_neuron(net_g, len(net_g.mlp.layers) - 2, i, states_g))
        necessity_g[i] = float(np.linalg.norm(baseline_g - ab))
    ax.bar(range(H_last_g), np.sort(necessity_g)[::-1])
    ax.set_title(f"γ={g}  necessity (sorted)")
    ax.set_xlabel("neuron (rank)")
    if g == GAMMAS[0]:
        ax.set_ylabel("‖Δpolicy‖")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/ch5_necessity_gamma_sweep.png", dpi=150)
plt.show()
```

**Markdown:**

```markdown
What to look for:
- γ=1: necessity bars near zero, decomposition shows tiny δ_θ. The MLP
  branch barely fires — close to our "null condition" for what
  "no signal" looks like.
- γ=2: a handful of necessary neurons; clear δ_θ pattern.
- γ=5: more neurons necessary, larger δ_θ in magnitude, more spatial
  structure to interpret.

If the actual picture diverges from this expectation, *that* is the
finding worth recording. The notebook doesn't pretend an expected
pattern was observed when it wasn't.
```

- [ ] **Step 2: Append Chapter 6 cells**

**Markdown:**

```markdown
## Chapter 6 — Honest limits and pointers

What we did:
- Decomposed the policy into linear (BK) and nonlinear (MLP) components.
- Mapped per-neuron contribution to the output.
- Probed neuron activations against a hand-chosen concept basis.
- Causally ablated each neuron and cross-tabbed with the probe results.
- Swept γ to see how the picture changes with task difficulty.

What this *doesn't* cover:
- **Superposition.** A neuron can encode a linear combination of two
  features that no single concept in our basis matches. Linear probes
  miss this. Sparse autoencoders (SAEs) are the standard fix; the
  Brock-Mirman MLP is too small to benefit much, but on the disaster
  model this would be the next step.
- **Off-manifold ablation.** Zeroing a neuron's activation drives the
  input to the next layer outside the training distribution. The
  ablation diff is real but the magnitude can over-state importance.
  Sharper alternatives: mean-ablation, activation patching across
  states (= deferred approach B), or training a sparse coding layer.
- **Disaster regimes.** Disaster has natural regime structure (ZLB
  binding, disaster shock active, normal). Our tools transfer; the
  concept basis needs to change. That's the natural sequel.

Suggested next moves: port the probe + ablation toolkit to the
disaster model; replace the concept basis with regime indicators and
shock-block scalars; add an SAE pass on the disaster MLP's last hidden
layer.
```

- [ ] **Step 3: Execute the notebook end-to-end**

```bash
uv run jupyter nbconvert --to notebook --execute examples/interp_brock_mirman.ipynb --output examples/interp_brock_mirman.ipynb
```

Expected: full notebook executes without error; 3×3 decomposition panel and 3-up necessity-sweep both render and save. Total runtime should be a few minutes (training dominates).

- [ ] **Step 4: Commit**

```bash
git add examples/interp_brock_mirman.ipynb docs/dev/figures/interp/
git commit -m "interp: notebook Ch 5–6 (γ-sweep + honest limits)"
```

---

## Self-Review

**Spec coverage check:**

| Spec section | Task(s) |
|---|---|
| 2. Scope (in scope: train 3 nets at γ) | Task 8 |
| 2. Scope (5 primitives) | Tasks 2–6 |
| 2. Scope (narrated notebook 6 chapters) | Tasks 8–13 |
| 2. Scope (figures under docs/dev/figures/interp/) | Tasks 8–13 (each chapter saves) |
| 2. Scope (sanity tests in tests/test_interp.py) | Tasks 2–7 |
| 3. Background facts (BK + MLP decomp, init ≈0) | Task 2 (decomposition); referenced in notebook Ch 0–1 prose |
| 4.1 `branch_decompose` API + closes_numerically | Task 2 |
| 4.2 `forward_with_activations` API | Task 3 |
| 4.3 `neuron_contributions` API (bias excluded) | Task 4 |
| 4.4 `linear_probe` API (per-pair univariate) | Task 5 |
| 4.5 `ablate_neuron` API | Task 6 |
| 5. Notebook chapters 0–6 | Tasks 8–13 |
| 6. Tests (closes_numerically, log link, perfect-fit probe, no-fit probe, ablate predicted diff, end-to-end wiring) | Tasks 2–7 |

All spec sections have at least one task covering them. The `test_branch_decompose_respects_log_link` from the spec is covered by `test_branch_decompose_closes_numerically_log_link` in Task 2 (same coverage, simpler assertion).

**Placeholder scan:** no "TBD"/"TODO"/"similar to" in the plan. Two soft-decision points are flagged with concrete fallback actions: hidden-size in Task 8 ("drop to (8,) if everything generic") and `model_constants` config-field name in Task 8 ("substitute correct field if different"). Both are actionable, not deferred.

**Type consistency:** `branch_decompose` returns `Dict[str, Any]` consistently across Task 2 implementation and Task 7/8 usage. `forward_with_activations` returns `Dict[str, Array]` (keys `"h{i}"` and `"out"`). `neuron_contributions` returns `Dict[int, Array]` (integer-keyed by hidden-layer index). `linear_probe` returns `Dict[str, Array]` with keys `"coef"`, `"r2"`, `"residual_var"`. `ablate_neuron` returns `Array[batch, n_policies]`. All consistent across the plan.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-11-mech-interp-deqn.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
