# Running Experiments

Once a model is implemented and training works on a basic config (see [Implementing a model](models/implementing.md)), this doc covers everything *after*: launching runs, persisting checkpoints, resuming, logging to TensorBoard/W&B, comparing runs, and tuning. Anchor-linked so you can jump to the specific operation.

- [CLI quickstart](#cli-quickstart)
- [YAML config patterns](#yaml-config-patterns)
- [Warm start](#warm-start)
- [Checkpointing and resuming](#checkpointing-and-resuming)
- [TensorBoard](#tensorboard)
- [Weights & Biases](#weights-biases)
- [Comparing runs](#comparing-runs)
- [Tuning](#tuning) (outline — fleshed out once enough models are ported to state empirical tradeoffs)

---

## CLI quickstart

```bash
# list what's available
uv run deqn-jax list               # models
uv run deqn-jax optimizers         # optimizers

# train a model — all defaults
uv run deqn-jax train brock_mirman

# train from a YAML config
uv run deqn-jax train --config configs/brock_mirman.yaml

# override anything with --set (dot notation)
uv run deqn-jax train --config configs/brock_mirman.yaml \
    --set optimizer.learning_rate=1e-4 \
    --set episodes=5000

# short sanity-check run
uv run deqn-jax train brock_mirman -n 500 -q

# use fp64 (slower, for tight numerics)
uv run deqn-jax train brock_mirman --fp64

# post-training diagnostics
uv run deqn-jax evaluate <checkpoint.eqx>
uv run deqn-jax irf <checkpoint.eqx> --shock-name eps_z --horizon 40

# introspection
uv run deqn-jax info brock_mirman   # model details
uv run deqn-jax check                # installation sanity check
uv run deqn-jax init-config          # generate a default YAML
```

Always use `uv run`; never activate the venv by hand. `uv run` makes the invocation reproducible from a clean shell, which is what the CI and the DGX jobs also do.

### Override precedence

```
--set overrides  >  CLI flags  >  YAML file  >  dataclass defaults
```

Dot-notation works for any depth: `--set network.hidden_sizes='[128, 128]'`, `--set composite_loss.anchor_weight=0.01`, etc. Repeat `--set` as many times as needed.

---

## YAML config patterns

Minimal YAML for a stochastic model:

```yaml
model: brock_mirman
episodes: 20001
batch_size: 128
episode_length: 1          # 1 = exogenous-rect sampling (with initialize_each_episode: true)
mc_samples: 5
initialize_each_episode: true
n_epochs_per_rollout: 1
n_minibatches_per_epoch: 1

network:
  type: mlp
  hidden_sizes: [50, 50]
  activation: relu
  init: xavier_uniform

optimizer:
  name: adam
  learning_rate: 3.0e-4
  lr_schedule: cosine
  lr_min_factor: 0.1

warm_start: false
log_every: 1000
```

### Sampling patterns

- **Exogenous rect** (`episode_length: 1` + `initialize_each_episode: true`): fresh uniform draws from the rect specified by `init_state_fn`, one gradient step, repeat. Required for strongly-attracting systems (deterministic, low-dimensional) and for models with closed-form benchmarks.
- **Rollout ergodic** (`episode_length: N` + `initialize_each_episode: false`): simulate `N` periods from the last cycle's terminal state and use those as training points. Concentrates training density on the ergodic support — good for accuracy on simulated moments, bad for extrapolation.
- **Hybrid** (`episode_length: N` + `initialize_each_episode: true`): fresh rect start, then `N` rollout steps. Fills out both the rect and the attractor.
- **Minibatch sweep** (`n_epochs_per_rollout > 1`, `n_minibatches_per_epoch > 1`): after simulating, take multiple gradient steps over the same data before re-rolling. Raises sample efficiency; risks overfitting to a single rollout.

### Sim batch vs minibatch batch

Post the upstream parity work, `sim_batch` (number of trajectories simulated) and `batch_size` (gradient minibatch size) are independent. Simulate 1024 trajectories, do gradient on chunks of 128: set `sim_batch: 1024, batch_size: 128`. If `sim_batch` is omitted it defaults to `batch_size` (the simple case).

### Composite loss

For models with a known linearization and where you want the auxiliary anchor / Jacobian / barrier / Newton losses:

```yaml
loss_type: composite
composite_loss:
  anchor_weight: 0.01
  jacobian_weight: 0.01
  barrier_weight: 0.001
  newton_weight: 0.01
  aux_decay_floor: 0.1     # set to 1.0 to keep aux terms fully active
```

See `src/deqn_jax/training/composite_loss.py` for the full field list.

---

## Warm start

```yaml
warm_start: true
```

Runs an L-BFGS pre-fit of the network to the deterministic steady-state policy before gradient-based training starts. 10-50 L-BFGS steps; no effect on the main training loop.

**When to use it:**
- Models with a non-trivial `steady_state_fn` where a cold random init wastes the first few hundred gradient steps drifting toward the fixed point.
- High-dimensional models where the unwarmed initial loss is so large it dominates gradient direction for a long time.

**When to skip it:**
- Debugging a freshly ported model — warm start can mask a bug by starting the network at a hand-computed steady state regardless of whether the Euler equation is correct.
- Small/closed-form models where the rect is tiny and the cold init is cheap.

Implementation lives in `src/deqn_jax/training/warm_start.py` — it is a thin wrapper around `optax.lbfgs` with a flat-parameter loop.

---

## Checkpointing and resuming

### Write checkpoints during training

```bash
uv run deqn-jax train brock_mirman \
    --config configs/brock_mirman.yaml \
    --checkpoint-dir runs/brock_mirman_2026_04 \
    --checkpoint-every 1000 \
    --max-checkpoints 5
```

Emits:
- `runs/brock_mirman_2026_04/checkpoint_<episode>.eqx` — periodic
- `runs/brock_mirman_2026_04/checkpoint_best.eqx` — overwritten whenever a new best loss is seen
- `runs/brock_mirman_2026_04/checkpoint_best.meta` — episode + loss for the best
- `runs/brock_mirman_2026_04/config.yaml` — the full resolved config used for the run

`--max-checkpoints N` trims the periodic checkpoints to the most recent N (the best checkpoint is never deleted).

### Resume

```bash
uv run deqn-jax train \
    --config runs/brock_mirman_2026_04/config.yaml \
    --resume runs/brock_mirman_2026_04/checkpoint_20000.eqx \
    --checkpoint-dir runs/brock_mirman_2026_04
```

Reuses the exact config (hence the re-pointing of `--config` to the saved one) and continues from the checkpointed episode. Combine with `-n`/`--episodes` to set a new termination horizon.

### Resume gotchas

- **Config must match.** Training state (optimizer moments, reweighting stats) is pytree-shaped by config; loading a checkpoint into a different config will error out or silently produce garbage. Always resume against the saved `config.yaml`.
- **PRNG continuity.** Seeds are not restored verbatim; the resumed run re-seeds from the checkpoint's step count, so it diverges from a single-run trajectory even with the same seed. Reproducibility across resume boundaries is not a framework guarantee.
- **Precision.** fp32 checkpoints can't be loaded into fp64 training and vice versa. Match the flag.

### Evaluate or IRF from a checkpoint

No training required:

```bash
uv run deqn-jax evaluate runs/brock_mirman_2026_04/checkpoint_best.eqx
uv run deqn-jax irf runs/brock_mirman_2026_04/checkpoint_best.eqx \
    --shock-name eps_z --horizon 40 --csv runs/brock_mirman_2026_04/irf.csv
```

Config is auto-detected from the checkpoint's sibling `config.yaml` unless `--config` is passed explicitly.

---

## TensorBoard

```bash
uv run deqn-jax train brock_mirman \
    --config configs/brock_mirman.yaml \
    --tensorboard runs/brock_mirman_2026_04/tb
```

Logs:
- **Scalars**: total loss, per-equation losses, gradient norm, learning rate, wall-clock episodes/sec.
- **Histograms** (every `log_every` episodes): each variable in `definitions()`, each equation residual, each policy output. This is what lets you diagnose "is the policy going out of bounds" or "is a definition collapsing to zero" without writing plot code.
- **Aux losses**: every entry prefixed `aux_` in the `eq_losses` dict (barrier, anchor, Jacobian, bound penalties) as scalars.

View:

```bash
uv run tensorboard --logdir runs/
```

The framework's logger lives in `src/deqn_jax/metrics.py` (class `TensorBoardLogger`). All scalar/histogram calls go through a shared `MetricLogger` interface so TB, W&B, and the null logger are swappable.

---

## Weights & Biases

```bash
uv run deqn-jax train brock_mirman \
    --config configs/brock_mirman.yaml \
    --wandb my-deqn-project
```

Logs the same scalar/histogram surface as TensorBoard plus the full resolved config as the W&B run's `config` field (searchable/filterable in the UI).

Combine freely: `--tensorboard runs/.../tb --wandb my-project` writes to both.

Authentication: `wandb login` once; the CLI picks up the token from `~/.netrc`. No env-var plumbing required.

---

## Comparing runs

The `deqn_jax.plots.compare` module reads TensorBoard event files (or the text logs from `-q`-less runs) and produces aligned multi-run plots.

```python
from deqn_jax.plots.compare import parse_log, plot_multi_run_loss

runs = {
    "adam-3e-4": parse_log("runs/brock_mirman_adam_3e4/train.log"),
    "adam-1e-3": parse_log("runs/brock_mirman_adam_1e3/train.log"),
    "mao":        parse_log("runs/brock_mirman_mao/train.log"),
}
plot_multi_run_loss(runs, log_y=True)
```

For schedule alignment (e.g. comparing LR curves across runs with different schedules):

```python
from deqn_jax.plots.compare import plot_schedule_alignment
plot_schedule_alignment(runs, metric="learning_rate")
```

Full text-log parsing lives in `plots.compare.parse_log`; schema is stable across framework versions.

---

## Tuning

> Sketch only. Will be fleshed out with empirical tradeoffs once more models are ported and cross-model patterns emerge. For now, treat as a menu of knobs.

### Optimizer choice

Baseline Adam is fine for most models. Reach for specialized optimizers when:
- **NGD** — when the equilibrium conditions have wildly different scales across equations and Adam's diagonal preconditioner under-corrects.
- **MAO** — when gradient conflict across equations is measurable (per-equation gradients point in different directions).
- **Shampoo** — large networks (100k+ params) where Kronecker-factored preconditioning outweighs its per-step cost.
- **L-BFGS / GN / LM** — low-noise regimes near convergence (after Adam has done the bulk of the work).

### LR schedule

- `constant` — debugging only.
- `cosine` — default for single-phase training. `lr_min_factor: 0.1` retains meaningful gradient pressure at the end.
- `reduce_on_plateau` — single-phase runs where convergence stalls unpredictably.

### Reweighting

- `none` — single-equation models.
- `lr_annealing` — inverse-EMA weighting; stable, low-maintenance.
- `relobralo` — softmax of loss ratios; reacts faster to regime changes but can thrash.

### Batch and sampling

- `batch_size`: start 128; raise if gradient noise dominates late-stage training.
- `mc_samples`: 5 is standard; increase if Euler residuals are dominated by shock variance (check by comparing per-shock std to mean in `definitions`).
- `initialize_each_episode`: true for edge-case robustness, false for simulated-moment accuracy.

### Composite loss

- Toggle on only when a linearization is trusted. A wrong linearization poisons the anchor and Jacobian terms and makes training worse than `mse`.
- `aux_decay_floor: 1.0` keeps the curriculum aux terms active all the way through (no late-stage decay).

### Warm start

See the dedicated section above.

---

## Cross-references

- [Overview](why.md) — positioning / when to use the framework at all.
- [Implementing a model](models/implementing.md) — how to add a new model.
- [Composite loss](training/composite_loss.md) — the composite-loss system in detail.
- `CLAUDE.md` — framework architecture at a glance.
- `src/deqn_jax/config.py` — canonical source for every config field's type, default, and validation. (Until the config reference doc lands, this is the ground truth.)
