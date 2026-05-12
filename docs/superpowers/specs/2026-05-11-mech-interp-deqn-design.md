# Mech Interp on DEQNs: A Pedagogical Walkthrough on Brock-Mirman

**Status:** design, awaiting implementation plan
**Date:** 2026-05-11
**Author:** Anna Smirnova
**Scope owner:** open-ended exploration; one bounded artifact

---

## 1. Motivation

DEQNs train neural policy networks to satisfy a model's equilibrium equations on
its ergodic distribution. The networks are unusual for ML — small, with
domain-meaningful inputs and outputs, and (in the canonical `LinearPlusMLP`
architecture) built around an explicit Blanchard-Kahn linearization that the MLP
augments with a nonlinear correction. That structure is a near-ideal substrate
for *learning* mechanistic interpretability:

- Inputs are economic state variables, not opaque tokens or pixels.
- Outputs are economic policies that we already have intuition about.
- The architecture itself provides a free decomposition into "what perturbation
  methods give you" and "what the network adds."
- For Brock-Mirman, the analytical solution is close enough at hand that every
  interp claim can be sanity-checked.

This spec defines **one pedagogical artifact**: a narrated Jupyter notebook,
backed by a small reusable module, that walks one mech-interp concept per
chapter on a single trained `LinearPlusMLP` for `brock_mirman` (plus a γ-sweep
in the final chapter as the "intensity dial").

The artifact serves a dual audience:

- DEQN / macro researchers learning what mech interp *is*, on a model they
  already understand.
- ML readers learning what's distinctive about DEQN networks (tiny,
  equation-grounded, with a known linear baseline).

It is **not** a research result. It is **not** a CLI tool. It is **not** an
attempt to find novel features in disaster-model networks. Those are deferred.

## 2. Scope

### In scope

- Train **three** `LinearPlusMLP` networks on `brock_mirman` (one per γ ∈ {1.0,
  2.0, 5.0}), from scratch, in the notebook.
- Build a thin **`src/deqn_jax/interp.py`** module with five primitives:
  `branch_decompose`, `forward_with_activations`, `neuron_contributions`,
  `linear_probe`, `ablate_neuron`.
- Write a narrated notebook `notebooks/interp_brock_mirman.ipynb` covering six
  chapters: setup → output decomposition → per-neuron contributions → linear
  probes → ablation → γ-sweep → honest limits.
- Generate inline figures in the notebook. Save the most useful ones as PNGs
  under `docs/dev/figures/interp/` for reuse.
- Sanity tests in `tests/test_interp.py`: branch decomposition closes
  numerically, ablation runs, probe regression handles edge cases.

### Out of scope (deferred)

- Disaster model. The disaster model has regime structure (ZLB, disaster shock)
  worth interp work, but it complicates every chapter without changing what's
  taught.
- Sparse autoencoders. Too heavy for a 1–2 hidden layer MLP; meaningful only
  when superposition matters.
- Activation patching across (k, z) anchor points. A natural sequel to ablation;
  defer until the notebook is real.
- Cross-seed universality grids. Worth doing eventually; not part of the first
  pedagogical pass.
- CLI subcommand (`deqn-jax interp …`). Reconsider once the primitives have
  been used in anger.

## 3. Background and pedagogically useful facts

Two structural facts about `LinearPlusMLP` (verified in
`src/deqn_jax/networks/linear_plus_mlp.py`) drive the entire pedagogy:

1. **The linear branch is the Blanchard-Kahn first-order solution**, not a
   trained linear layer:

   ```
   policy(s) = clip(π_BK(s) + δ_θ(s), lower, upper)
   π_BK(s) = ss_policy + P · (s − ss_state)        # fixed, from linearization
   δ_θ(s)  = mlp(s)                                # trainable MLP correction
   ```

   `P`, `ss_state`, `ss_policy` are wrapped in `stop_gradient` and held fixed
   throughout training.

2. **The MLP branch is initialized to ≈0**: the constructor scales the final
   linear layer's weights by `init_scale=0.01` and zeroes its biases. The
   network at init *is* the Blanchard-Kahn linearization. Training only moves
   `δ_θ` away from zero to reduce residuals.

These two facts let us frame the central question of the notebook as:

> The Blanchard-Kahn linearization is what macroeconomic perturbation methods
> already give you. The MLP branch is the only thing the neural network adds.
> What did it add, and how?

`brock_mirman` parameters (from `src/deqn_jax/models/brock_mirman/variables.py`):

| symbol | value | role |
|--------|-------|------|
| α | 0.36 | capital share |
| β | 0.99 | discount factor |
| γ | 1.0 (default) | CRRA risk aversion |
| δ | 0.10 | depreciation rate |
| ρ_z | 0.9 | TFP persistence |
| σ_z | 0.04 | TFP shock std |

Because δ < 1 and ρ_z > 0, the optimal policy is *mildly nonlinear* even at
γ=1 — there are no degenerate cases in the γ-sweep.

## 4. Module: `src/deqn_jax/interp.py`

Single file, mirroring the shape of `src/deqn_jax/active_subspace.py`. Five
top-level functions, all pure in their arguments. No mutable state, no
classes (the `LinearPlusMLP` *is* the state).

### 4.1 `branch_decompose(net, states) -> dict`

Inputs:

- `net: LinearPlusMLP`
- `states: Array[batch, n_states]`

Returns dict with keys:

- `"bk"`: Array[batch, n_policies] — `ss_policy + (states − ss_state) @ P.T`,
  with the link-type conversion (`linear` vs `log`) applied per output the same
  way `LinearPlusMLP._forward_single` does it.
- `"mlp_delta"`: Array[batch, n_policies] — raw `net.mlp(states)`.
- `"policy"`: Array[batch, n_policies] — the final clipped policy
  (`net(states)`).
- `"closes_numerically"`: bool — true iff `bk + mlp_delta` matches `policy`
  within 1e-6 on every grid point (i.e. no clipping was active). The notebook
  asserts this on the ergodic grid.

Implementation note: this function mirrors `LinearPlusMLP._forward_single`'s
linearization step rather than re-deriving it, so any future change to how
log-link outputs are combined stays in one place.

### 4.2 `forward_with_activations(mlp, states) -> dict`

Inputs:

- `mlp: MLP` (the `net.mlp` from a LinearPlusMLP)
- `states: Array[batch, n_states]`

Parallels `MLP._forward_single` but records every post-activation. Returns:

- `"h0"`, `"h1"`, … `"h{L-1}"`: Array[batch, hidden_size_i] — post-activation
  outputs of each hidden layer.
- `"out"`: Array[batch, n_outputs] — final pre-clip MLP output.

For Brock-Mirman the typical MLP has 1–2 hidden layers, so this dict is small.

### 4.3 `neuron_contributions(mlp, states) -> dict`

Inputs same as `forward_with_activations`.

Returns a dict keyed by hidden-layer index. For layer `ℓ` with hidden size
`H_ℓ` and `H_{ℓ+1}` units in the next layer (or `n_outputs` if `ℓ` is the last
hidden layer), the value is `Array[batch, H_ℓ, H_{ℓ+1}]` containing
`W_{ℓ+1}[j, i] · h_ℓ[i]` per (batch, neuron `i`, downstream-unit `j`).

The most-used slice is the final hidden layer's contribution to outputs;
intermediate layers are returned for completeness on 2-hidden-layer nets.

### 4.4 `linear_probe(activations, concepts) -> dict`

Inputs:

- `activations: Array[batch, n_neurons]`
- `concepts: Array[batch, n_concepts]` — caller-supplied concept basis; the
  notebook constructs this from `(k, z, k², z², k·z, log k, log z, k − k_ss,
  z − z_ss, y, mpk)` etc.

Returns:

- `"coef"`: Array[n_neurons, n_concepts] — per-pair regression slope (fit one
  concept at a time, no joint regression).
- `"r2"`: Array[n_neurons, n_concepts] — coefficient of determination.
- `"residual_var"`: Array[n_neurons, n_concepts] — variance of the residual,
  useful when activations are near-constant (degenerate R²).

Each `(neuron, concept)` regression is a 1-D least-squares fit of
`activation = a · concept + b`. Computed with `jnp.linalg.lstsq` vectorized
over neurons; vmapped over concepts. No regularization. Caller is responsible
for centering / scaling concepts if they want comparable coefficients.

### 4.5 `ablate_neuron(net, layer_idx, neuron_idx, states) -> Array`

Inputs:

- `net: LinearPlusMLP`
- `layer_idx: int` — which hidden layer's activation to zero
- `neuron_idx: int` — which neuron within that layer
- `states: Array[batch, n_states]`

Returns:

- `Array[batch, n_policies]` — the policy with the chosen post-activation
  forced to zero everywhere.

Implementation: a custom forward pass that mirrors `_forward_single` but
masks the indicated post-activation. Implementing this via an in-place modify
or `eqx.tree_at` on the weights doesn't work cleanly (we want to zero an
*activation*, not a *parameter*). The cleanest implementation is a small
parallel function that takes the same `mlp` apart layer-by-layer.

Caller computes the diff vs. baseline themselves; we don't bundle it.

### 4.6 Notes on what the module *doesn't* do

- No batching helpers beyond what `jax.vmap` gives for free.
- No figure generation. Figures are notebook concerns; this module returns
  arrays.
- No I/O. The notebook owns checkpoint loading and figure saving.
- No GPU/TPU-specific paths. Everything is JAX numpy, runs anywhere a
  trained Brock-Mirman fits (which is "anywhere").

## 5. Notebook: `notebooks/interp_brock_mirman.ipynb`

One narrated notebook, six chapters. Heavy on inline matplotlib. State at
the top: three trained `LinearPlusMLP`s (γ ∈ {1, 2, 5}), kept in memory as
checkpoints / Equinox modules; one set of ergodic states sampled from the
γ=2 network's simulation; one (k, z) grid for the heatmaps.

### Chapter 0 — Setup

- Train the three networks. Use the existing
  `deqn_jax.training.trainer.train_from_config` with a minimal config dict
  per γ. Hidden sizes `(16, 16)` (small enough to interpret, large enough to
  show distribution-of-work).
- Plot the learned `sav_rate(k, z)` policy on a (k, z) grid covering ±2σ of
  the ergodic support.
- Plot per-state Euler residual to show the model was solved.
- Sanity-check that policies sit inside `POLICY_LOWER, POLICY_UPPER`.

### Chapter 1 — Output decomposition

- Call `branch_decompose` on the grid. Plot three side-by-side heatmaps:
  `π_BK(k, z)`, `δ_θ(k, z)`, `policy(k, z)`.
- Show the MLP correction is *much smaller* in magnitude than the BK
  baseline but spatially structured.
- Assert `closes_numerically` is true on the grid (no clipping triggered
  in-sample).
- *Concept taught:* additive decomposition as the simplest "circuit."

### Chapter 2 — Per-neuron contributions

- Call `forward_with_activations` and `neuron_contributions`.
- Bar chart of `mean(|contribution|)` per last-hidden-layer neuron over the
  ergodic samples — which neurons matter.
- For the top-3 contributors and the top-3 non-contributors, heatmap of
  contribution as a function of (k, z).
- Name the three archetypes inline: **dead** (near-zero everywhere),
  **generic** (smooth, near-linear over the grid), **selective** (activates
  in a sub-region of state space).
- *Concept taught:* networks distribute work unevenly; spatial selectivity
  is a real, visible thing in DEQN nets.

### Chapter 3 — Linear probes

- Build a concept basis matrix on ergodic samples:
  - Linear: `k`, `z`
  - Polynomial: `k²`, `z²`, `k · z`
  - Log: `log k`, `log z`
  - SS-centered: `k − k_ss`, `z − z_ss`
  - Economic: `y = z · k^α`, `mpk = α · z · k^(α−1)`
- Call `linear_probe(last_hidden_activations, concept_matrix)`.
- Plot R² heatmap (rows: neurons, columns: concepts).
- Highlight: one clean case ("neuron 7 has 0.94 R² on log k") and one
  mysterious case ("neuron 12 has R² < 0.1 on every listed concept" — maybe
  it encodes a nonlinear combination, maybe noise).
- *Concept taught:* probes are correlational; high R² ≠ "the network uses
  this neuron to compute the concept"; low R² ≠ "the neuron is doing
  nothing."

### Chapter 4 — Ablation

- For each live neuron, call `ablate_neuron`, compute
  `Δpolicy = baseline − ablated`, plot `‖Δpolicy‖` per neuron (necessity) and
  the spatial pattern for top-3.
- Cross-tabulate probe R² vs. ablation magnitude into a 2×2:
  - high-R² + high-ablation-effect → likely real feature.
  - high-R² + low-ablation-effect → redundancy (multiple neurons encode it).
  - low-R² + high-ablation-effect → the interesting mystery.
  - low-R² + low-ablation-effect → genuinely dead.
- *Concept taught:* correlation vs causation in mech interp.

### Chapter 5 — γ-sweep

- Run chapters 1–4 (compact, side-by-side panels) on the γ=1.0, 2.0, 5.0
  networks.
- Expected pattern, with honest reporting of whatever we actually find:
  - γ=1: small but non-zero δ; mostly dead neurons; weak features.
  - γ=2: moderate δ; mixed selectivity.
  - γ=5: large δ; clearer features, more selective neurons.
- The cross-γ comparison is the central figure of the notebook.
- *Concept taught:* what an "interp null result" looks like (γ=1), how
  features sharpen with task difficulty.

### Chapter 6 — Limits and pointers

- Linear probes miss nonlinear features; superposition is invisible to
  single-neuron analysis. Pointer to SAEs (deferred).
- Ablation can break things via off-manifold inputs — a flagged caveat,
  not a fatal flaw.
- Disaster-model regime structure (ZLB, disaster shock) is where this
  toolkit would next be applied. Pointer to follow-on work.

## 6. Tests: `tests/test_interp.py`

Focused, fast (≈seconds), no training inside tests. Build a fixture
`LinearPlusMLP` with hand-set weights / SS values so we can predict every
output exactly.

- **`test_branch_decompose_closes_numerically`** — on a fixture, assert
  `bk + mlp_delta == policy` to 1e-6 (clip bounds set wide enough to be
  inactive).
- **`test_branch_decompose_respects_log_link`** — on a fixture with one
  `linear` and one `log` output, assert per-policy combination matches
  `LinearPlusMLP._forward_single`.
- **`test_forward_with_activations_matches_call`** — `forward_with_activations`'s
  `"out"` equals `mlp(states)` (pre-bounds).
- **`test_neuron_contributions_sum_to_pre_bias`** — for the last hidden layer,
  sum of contributions across neurons equals `mlp.layers[-1](h_last) −
  layers[-1].bias`.
- **`test_linear_probe_perfect_fit`** — set `activations[:, 0] = 3·concepts[:, 0] + 1`
  and assert R²[0, 0] == 1 and coef[0, 0] ≈ 3.
- **`test_linear_probe_no_fit`** — random activations vs random concepts
  yield R² near zero (use a generous tolerance and a fixed seed).
- **`test_ablate_neuron_zero_changes_output`** — ablate a known-nonzero
  neuron, assert the output differs from baseline by the predicted amount
  (`W_last[:, i] · h_last[:, i]`).

Plus one end-to-end sanity test that trains a tiny brock_mirman net for ~5
steps and runs every primitive on it without erroring. Not a correctness
test, a wiring test.

## 7. Risks and open questions

- **Brock-Mirman's MLP correction might be uninteresting even at γ=5.** δ_θ
  could be uniformly tiny relative to BK. If so, the γ-sweep panel becomes
  itself the lesson ("here is what 'almost no signal' looks like"); we don't
  pretend otherwise. The notebook's tone explicitly allows this.
- **MLP hidden size choice.** `(16, 16)` is a guess: small enough to inspect
  every neuron by eye, large enough to leave room for dead/redundant
  archetypes. If most neurons end up dead at this size, drop to `(8,)`;
  if everything is generic, bump to `(32, 32)`. Decided empirically in
  Chapter 0.
- **Log-link branch composition.** Brock-Mirman's `sav_rate` is a single
  output bounded in (0, 1) — it uses the linear link by default. The
  `branch_decompose` function still needs to handle log-link outputs for
  future disaster-model use; tested in fixtures even though the notebook
  itself only exercises linear-link.
- **Probe basis is hand-picked.** R²s are only informative within the chosen
  basis. The notebook is explicit about this and lists candidate concepts
  the reader could add.
- **Ablation off-manifold.** Setting a neuron to zero pushes the input to the
  next layer outside the training distribution. The notebook flags this as a
  known caveat, not a methodological fix.

## 8. Out of scope (explicit non-goals, restated)

- No disaster-model interp in this artifact.
- No SAEs.
- No activation patching across anchor states.
- No cross-seed universality grid.
- No CLI command.
- No paper-style writeup; the notebook is the deliverable.

## 9. Success criteria

This artifact succeeds when:

- A reader who has never done mech interp can read the notebook end-to-end
  and identify the *operations* used (decomposition, contribution, probe,
  ablation) and what each tells you.
- A reader can rerun the notebook on a fresh checkout and reproduce the
  figures.
- The five primitives in `interp.py` are reusable on a `LinearPlusMLP`
  trained on a different model (e.g., the disaster model) without code
  changes — only the concept-basis and grid construction need to change.
- The tests pass and run in under ~30 seconds.

If any of those fail, the artifact has shipped scope it shouldn't have.
