# Running Experiments

This page assumes a model is implemented and trains on a basic config (see [Implementing a model](models/implementing.md)). It covers launching runs, checkpoints, resuming, logging to TensorBoard and W&B, comparing runs, and tuning.

- [CLI quickstart](#cli-quickstart)
- [YAML config patterns](#yaml-config-patterns)
- [Warm start](#warm-start)
- [Checkpointing and resuming](#checkpointing-and-resuming)
- [TensorBoard](#tensorboard)
- [Weights & Biases](#weights-biases)
- [Comparing runs](#comparing-runs)
- [Tuning](#tuning) (an outline for now)

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
uv run deqn-jax irf <checkpoint.eqx> --shock eps_z --horizon 40

# introspection
uv run deqn-jax info brock_mirman   # model details
uv run deqn-jax check                # installation sanity check
uv run deqn-jax init-config          # generate a default YAML
```

### Override precedence

```
--set overrides  >  CLI flags  >  YAML file  >  dataclass defaults
```

Dot notation reaches any depth, for example `--set network.hidden_sizes='[128, 128]'` or `--set composite_loss.anchor_weight=0.01`. `--set` can be repeated.

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

- Exogenous rect (`episode_length: 1`, `initialize_each_episode: true`): fresh uniform draws from the rect given by `init_state_fn`, one gradient step, repeat. Required for strongly attracting systems (deterministic, low-dimensional) and for models with closed-form benchmarks.
- Rollout ergodic (`episode_length: N`, `initialize_each_episode: false`): simulate `N` periods from the last cycle's terminal state and train on those points. Training density concentrates on the ergodic support. This helps accuracy on simulated moments and hurts extrapolation.
- Hybrid (`episode_length: N`, `initialize_each_episode: true`): a fresh rect start, then `N` rollout steps. Covers both the rect and the attractor.
- Minibatch sweep (`n_epochs_per_rollout > 1`, `n_minibatches_per_epoch > 1`): after simulating, take several gradient steps over the same data before the next rollout. Raises sample efficiency at the risk of overfitting one rollout.

### Sim batch vs minibatch batch

`sim_batch` (number of simulated trajectories) and `batch_size` (gradient minibatch size) are independent. To simulate 1024 trajectories and take gradients on chunks of 128, set `sim_batch: 1024, batch_size: 128`. If `sim_batch` is omitted it defaults to `batch_size`.

### Composite loss

For models with a known linearization, the composite loss adds auxiliary anchor, Jacobian, barrier and Newton terms:

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

Before gradient training starts, an L-BFGS pre-fit (10-50 steps) fits the network to the deterministic steady-state policy. The main training loop is unchanged.

Use it for:

- models with a non-trivial `steady_state_fn`, where a random init spends the first few hundred gradient steps drifting toward the fixed point;
- high-dimensional models whose initial loss without warm start is large enough to dominate the gradient direction for a long time.

Skip it for:

- debugging a newly ported model, since starting at a hand-computed steady state can hide a wrong Euler equation;
- small or closed-form models, where the rect is small and a cold init is cheap.

The implementation, in `src/deqn_jax/training/warm_start.py`, wraps `optax.lbfgs` in a flat-parameter loop.

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

This writes:

- `runs/brock_mirman_2026_04/checkpoint_<episode>.eqx`: periodic checkpoints;
- `runs/brock_mirman_2026_04/checkpoint_best.eqx`: overwritten at each new best loss;
- `runs/brock_mirman_2026_04/checkpoint_best.meta`: episode and loss of the best checkpoint;
- `runs/brock_mirman_2026_04/config.yaml`: the full resolved config of the run.

`--max-checkpoints N` keeps only the N most recent periodic checkpoints. The best checkpoint is never deleted.

### Resume

```bash
uv run deqn-jax train \
    --config runs/brock_mirman_2026_04/config.yaml \
    --resume runs/brock_mirman_2026_04/checkpoint_20000.eqx \
    --checkpoint-dir runs/brock_mirman_2026_04
```

Pointing `--config` at the saved file reuses the exact config. Training continues from the checkpointed episode; `-n`/`--episodes` sets a new end point.

### What resume preserves, and what it doesn't

The full `TrainState` deserialises from the `.eqx` file: params, optimizer state, episode-state batch, PRNG key, step and episode counters, loss weights, reweighting running statistics, target params and aux params. Resume is deterministic across the boundary. Running N episodes straight through equals running K and then N-K with a checkpoint at K, up to JAX-wide non-determinism (device, precision).

These must match the original run, because they set pytree shapes:

- Network architecture (`hidden_sizes`, `activation`, `type`). A change alters the params pytree and deserialisation fails.
- Number of equations. `loss_weights` and `reweight_state` are shaped by `n_equations`.
- `sim_batch`, which shapes the saved `episode_state`.
- Precision. An fp32 checkpoint cannot be loaded into fp64 training, and vice versa.

On resume the framework loads the `config.yaml` next to the checkpoint to rebuild the template, so keep that file beside the `.eqx`.

These can change on resume:

- Learning rate, LR schedule, episodes, log frequency and checkpoint frequency. None of them affects the checkpointed tree shape.
- The optimizer. The new optimizer's state is initialised from the resumed params and the old moments are discarded. For a handoff during training (for example Adam to L-BFGS near convergence) use `--switch-optimizer` and `--switch-episode`.

### Evaluate or IRF from a checkpoint

```bash
uv run deqn-jax evaluate runs/brock_mirman_2026_04/checkpoint_best.eqx
uv run deqn-jax irf runs/brock_mirman_2026_04/checkpoint_best.eqx \
    --shock eps_z --horizon 40 --output runs/brock_mirman_2026_04/irf
```

The config is read from the `config.yaml` next to the checkpoint unless `--config` is given.

---

## TensorBoard

```bash
uv run deqn-jax train brock_mirman \
    --config configs/brock_mirman.yaml \
    --tensorboard runs/brock_mirman_2026_04/tb
```

It logs:

- scalars: total loss, per-equation losses, gradient norm, learning rate, episodes per second;
- histograms, every `log_every` episodes: each variable in `definitions()`, each equation residual, each policy output. These show a policy leaving its bounds or a definition collapsing to zero without extra plotting code;
- aux losses: every `aux_`-prefixed entry of the `eq_losses` dict (barrier, anchor, Jacobian, bound penalties), as scalars.

To view:

```bash
uv run tensorboard --logdir runs/
```

The logger is `TensorBoardLogger` in `src/deqn_jax/training/metrics.py`. Scalar and histogram calls go through the `MetricLogger` interface, so TensorBoard, W&B and the null logger are interchangeable.

---

## Weights & Biases

```bash
uv run deqn-jax train brock_mirman \
    --config configs/brock_mirman.yaml \
    --wandb my-deqn-project
```

This logs the same scalars and histograms as TensorBoard, plus the full resolved config as the W&B run's `config` field, which the UI can search and filter.

`--tensorboard runs/.../tb --wandb my-project` writes to both.

Run `wandb login` once; the CLI reads the token from `~/.netrc`, so no environment variables are needed.

---

## Comparing runs

The `deqn_jax.plots.compare` module parses the text logs of runs made without `-q` (the per-episode `loss=… | grad=…` lines) and overlays several runs in one plot.

```python
from deqn_jax.plots.compare import parse_log_single, plot_multi_run_loss

runs = {
    **parse_log_single("runs/brock_mirman_adam_3e4/train.log", "adam-3e-4"),
    **parse_log_single("runs/brock_mirman_adam_1e3/train.log", "adam-1e-3"),
    **parse_log_single("runs/brock_mirman_mao/train.log", "mao"),
}
plot_multi_run_loss(runs, log_y=True)
```

`parse_log_single(path, name)` returns `{name: history}` for a log holding one run. `parse_log(path)` splits a log holding several runs at `==== <name> starting ...` / `==== <name> finished ...` marker lines and returns `{name: history}` for each. A history has the keys `episodes`, `loss`, `grad_norm` and `best`.

Runs that take different numbers of gradient updates per cycle compare better against total gradient updates. Pass each history with its updates per cycle:

```python
from deqn_jax.plots.compare import plot_schedule_alignment
plot_schedule_alignment({
    "adam-3e-4": (runs["adam-3e-4"], 1),
    "mao": (runs["mao"], 4),
})
```

---

## Tuning

> An outline, to be filled in with measured tradeoffs as more models are ported. For now it lists the settings to try.

### Optimizer choice

Adam works for most models. The other optimizers fit specific cases:

- NGD: equations whose scales differ widely, so that Adam's diagonal preconditioner under-corrects.
- MAO: measurable gradient conflict across equations (per-equation gradients point in different directions).
- Shampoo: large networks (100k+ params), where Kronecker-factored preconditioning repays its per-step cost.
- L-BFGS, GN, LM: low-noise regimes near convergence, after Adam has done most of the work.

### LR schedule

- `constant`: debugging only.
- `cosine`: the usual choice for single-phase training. `lr_min_factor: 0.1` keeps a useful learning rate at the end.

### Reweighting

- `none`: single-equation models.
- `lr_annealing`: inverse-EMA weighting; stable and needs little tuning.
- `relobralo`: softmax of loss ratios; reacts faster to regime changes but can oscillate.

### Batch and sampling

- `batch_size`: start at 128; raise it if gradient noise dominates late training.
- `mc_samples`: 5 is standard; increase it if Euler residuals are dominated by shock variance (compare per-shock std to the mean in `definitions`).
- `initialize_each_episode`: true for robustness at the edges of the state space, false for accuracy on simulated moments.

### Composite loss

- Turn it on only when the linearization is trusted. A wrong linearization corrupts the anchor and Jacobian terms, and training ends up worse than with `mse`.
- `aux_decay_floor: 1.0` keeps the curriculum aux terms at full weight to the end.

### Warm start

See [Warm start](#warm-start) above.

---

## Cross-references

- [Overview](why.md): when to use the framework.
- [Implementing a model](models/implementing.md): adding a new model.
- [Composite loss](training/composite_loss.md): the composite loss in detail.
- [Config field reference](config_reference.md), generated from `src/deqn_jax/config/`. The package is the source of truth for each field's type, default and validation: `TrainConfig` in `train.py`, the nested configs in their own submodules.
