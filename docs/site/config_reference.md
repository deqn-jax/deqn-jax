# Config reference

Every field on the four Pydantic config classes (``TrainConfig``, ``OptimizerConfig``, ``NetworkConfig``, ``CompositeLossConfig``) with its type, default, and a one-line description.

Generated from introspection by ``scripts/gen_config_reference.py`` — regenerate after any config change:

```bash
uv run python scripts/gen_config_reference.py
```

Fields with description ``—`` haven't had an explicit ``Field(description=...)`` added yet; the generator surfaces these as a TODO list for the docs effort. Start there when a user asks "what does X do."

For YAML / CLI usage patterns (override precedence, sampling conventions, checkpoint/resume rules, etc.) see [Running experiments](running_experiments.md). For building models with these configs, see [Implementing a model](models/implementing.md).

## `TrainConfig`

Top-level training configuration.

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `'brock_mirman'` | Name of the registered model to train; see `deqn-jax list` for valid choices. |
| `episodes` | `int` | `1000` | Number of outer training cycles (rollout + minibatch sweep). |
| `batch_size` | `int` | `64` | Minibatch size used for each gradient step. |
| `episode_length` | `int` | `100` | Trajectory length per rollout (T). With T=1 you must set `initialize_each_episode=True` (see validator). |
| `mc_samples` | `int` | `5` | Monte Carlo shock samples per state for the residual expectation. Ignored when `expectation_type='gauss_hermite'`. |
| `seed` | `int` | `42` | Top-level PRNG seed. Controls network init and the rollout/loss shock streams. |
| `network` | `NetworkConfig` | `PydanticUndefined` | Policy network architecture; see NetworkConfig. |
| `optimizer` | `OptimizerConfig` | `PydanticUndefined` | Optimizer and LR schedule; see OptimizerConfig. |
| `loss_type` | `str` | `'mse'` | `mse` = base residual MSE. `composite` = base + anchor + Jacobian + barriers + Newton (disaster-style). Composite is rejected at startup with MAO / GN / LM / LBFGS / PCGrad. |
| `composite_loss` | `CompositeLossConfig` | `PydanticUndefined` | Composite-loss weights; only active when `loss_type='composite'`. |
| `loss_choice` | `str` | `'mse'` | Residual aggregation over batch elements: `mse` or `huber`. Applied AFTER the shock expectation. Huber caps gradient at ±huber_delta and helps when rare pathological states dominate. |
| `huber_delta` | `float` | `1.0` | Cutoff for Huber loss (`loss_choice='huber'`). Ignored for `loss_choice='mse'`. |
| `warm_start` | `bool` | `False` | If True, run L-BFGS pre-fit of the network to the steady-state policy before gradient-based training. Speeds early convergence; can mask Euler-equation bugs. |
| `warm_start_linearize` | `bool` | `False` | If True, linearize the model around SS and use the Blanchard-Kahn P matrix to seed the network's Jacobian at SS. Advanced. |
| `warm_start_dynare` | `Union[str, None]` | `None` | Path to a Dynare output file to seed warm-start linearization. Rare. |
| `loss_weights` | `Union[list[float], None]` | `None` | Manual per-equation weight vector of length `n_equations`. Default None = uniform weight 1.0. |
| `loss_reweight` | `str` | `'none'` | Adaptive reweighting: `none` (default), `lr_annealing` (inverse-EMA), `relobralo` (softmax of loss ratios). |
| `reweight_alpha` | `float` | `0.9` | EMA decay for `lr_annealing` / `relobralo`. Higher = slower adaptation. |
| `log_every` | `int` | `100` | Episodes between console / TensorBoard scalar logs and cycle_hook invocations. |
| `verbose` | `bool` | `True` | If False, suppress console output (the CLI `-q` flag sets this). |
| `fp64` | `bool` | `False` | Enable JAX x64 mode for higher numerical precision. Applied at `train_from_config` entry. |
| `tensorboard_dir` | `Union[str, None]` | `None` | Directory for TensorBoard event files. None disables TB logging. |
| `wandb_project` | `Union[str, None]` | `None` | W&B project name. None disables W&B logging. |
| `checkpoint_dir` | `Union[str, None]` | `None` | Directory to save checkpoints (`checkpoint_<episode>.eqx` + `checkpoint_best.eqx` + `config.yaml`). None disables. |
| `checkpoint_every` | `Union[int, None]` | `None` | Episodes between periodic checkpoints. None = no periodic checkpoints (only best is saved). |
| `max_checkpoints` | `Union[int, None]` | `None` | Keep only the N most recent periodic checkpoints (best is never deleted). |
| `gradient_surgery` | `str` | `'none'` | Multi-equation gradient conflict resolution: `none` or `pcgrad` (projecting conflicting gradients). |
| `resume` | `Union[str, None]` | `None` | Path to a `.eqx` checkpoint to resume from. Reads the sibling `config.yaml` to rebuild the correct pytree template. |
| `switch_optimizer` | `Union[str, None]` | `None` | If set, switch to this optimizer name at `switch_episode`. Old optimizer state is discarded; new optimizer is initialized from resumed params. |
| `switch_episode` | `Union[int, None]` | `None` | Episode at which to activate `switch_optimizer` and `switch_lr`. |
| `switch_lr` | `Union[float, None]` | `None` | Learning rate for the switched optimizer. None = keep the original optimizer's LR. |
| `early_stop_patience` | `Union[int, None]` | `None` | Stop training if loss hasn't improved by `early_stop_min_delta` for this many episodes. None = no early stopping. |
| `early_stop_min_delta` | `float` | `1e-06` | Minimum absolute loss improvement counted against `early_stop_patience`. |
| `curriculum_episodes` | `int` | `0` | Ramp `shock_scale` linearly from `curriculum_start` to 1.0 over this many episodes. 0 = no curriculum. |
| `curriculum_start` | `float` | `0.1` | Initial `shock_scale` when curriculum is active. |
| `ss_reset_frac` | `float` | `0.0` | Fraction of batch re-initialized to SS-neighborhood each rollout (prevents trajectory drift). Orthogonal to `initialize_each_episode`. |
| `initialize_each_episode` | `bool` | `False` | If True, replace episode_state with a fresh `init_state_fn` draw at the start of every rollout cycle (non-ergodic training, matches DEQN-MAO's flag of the same name). False = continue trajectory across cycles (ergodic). Required True when `episode_length=1`. |
| `expectation_type` | `str` | `'mc'` | How to integrate over shocks in the residual: `mc` (antithetic Monte Carlo, uses `mc_samples`) or `quadrature`/`gh`/`gauss_hermite` (deterministic tensor-product grid, uses `n_quadrature_points`). |
| `n_quadrature_points` | `int` | `3` | Quadrature points per shock dimension when `expectation_type='gauss_hermite'`. Total nodes = n_quadrature_points^n_shocks. |
| `barrier_weight` | `float` | `0.0` | Legacy state-barrier penalty weight. 0 disables. Prefer `definition_bounds` on the ModelSpec for new models. |
| `shock_mask` | `Union[list[float], None]` | `None` | Per-dimension multiplicative mask over shocks (length must equal `model.n_shocks`). Values in [0, 1]; 0 zeroes that shock entirely. Applied to BOTH the residual expectation and the rollout state path. |
| `target_update_every` | `int` | `0` | Target-network update interval in episodes. 0 disables target network entirely. |
| `target_tau` | `float` | `1.0` | Polyak averaging coefficient for target-network update. 1.0 = hard copy, <1 = soft update toward current params. |
| `constants` | `dict[str, float]` | `PydanticUndefined` | Per-run override of model.constants (e.g. `{p_disaster: 0.02}`). Merges into the model's built-in calibration. |
| `use_risky_steady_state` | `bool` | `True` | If True and `p_disaster > 0`, anchor composite loss and linearization at the risky SS (E_d[F]=0) instead of deterministic SS. Set False to force deterministic SS anchor under disaster risk (for ablation). |
| `save_best_checkpoint` | `bool` | `True` | If True and `checkpoint_dir` is set, persist `checkpoint_best.eqx` on every loss improvement (after `curriculum_episodes` grace period). Guards against rare huge-gradient events corrupting the latest snapshot. |
| `n_epochs_per_rollout` | `int` | `1` | DEQN cycle: per outer iteration, 1 rollout fills a trajectory of (`sim_batch` × `episode_length`) states, then we do `n_epochs_per_rollout` sweeps over it. Default 1 matches DEQN-MAO's run_cycle. |
| `n_minibatches_per_epoch` | `Union[int, None]` | `None` | Minibatches per sweep. None = all available (full-trajectory sweep). Set to 1 for the legacy one-grad-per-rollout behavior. |
| `sorted_within_batch` | `bool` | `False` | Minibatch shuffle policy. False = IID shuffle across all (episode_length × sim_batch) samples. True = each minibatch is a contiguous temporal slice of a single trajectory (RL-style); batch order shuffled, intra-batch order preserved. MLP-only. |
| `sim_batch` | `Union[int, None]` | `None` | Number of parallel simulation trajectories in the rollout. None (default) = same as `batch_size`. Setting `sim_batch > batch_size` decouples trajectory count from gradient minibatch size — larger pool = more representative ergodic distribution per cycle. |

## `OptimizerConfig`

Optimizer choice and hyperparameters; nested under ``optimizer:`` in YAML.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `'adam'` | Optimizer name. Options: `adam`, `sgd`, `adamw`, `lion`, `muon`, `ngd`, `shampoo`, `lbfgs`, `mao`, `mao_kfac`, `gn`, `lm`. |
| `learning_rate` | `float` | `0.001` | Peak learning rate (or constant LR when `lr_schedule='constant'`). |
| `grad_clip` | `Union[float, None]` | `None` | Global gradient-norm clipping. None disables. |
| `weight_decay` | `float` | `0.0` | L2 weight decay (used by adamw / adam / sgd). |
| `beta1` | `float` | `0.9` | Adam / MAO first-moment decay. |
| `beta2` | `float` | `0.999` | Adam / MAO second-moment decay. |
| `epsilon` | `float` | `1e-08` | Adam / MAO numerical floor. |
| `damping` | `float` | `0.0001` | NGD preconditioner damping (adds to Fisher diagonal). |
| `decay` | `float` | `0.999` | NGD / Shampoo preconditioner EMA decay. |
| `block_size` | `int` | `64` | Shampoo Kronecker block size. |
| `precond_update_freq` | `int` | `10` | Shampoo preconditioner update frequency. |
| `memory_size` | `int` | `10` | L-BFGS history size. |
| `ns_steps` | `int` | `5` | Muon Newton-Schulz iteration count. |
| `lr_schedule` | `str` | `'constant'` | LR schedule: `constant`, `cosine`, or `reduce_on_plateau`. |
| `lr_warmup` | `int` | `0` | Linear warmup episodes before `lr_schedule` kicks in. |
| `lr_min_factor` | `float` | `0.0` | Minimum LR as a fraction of peak (cosine / reduce_on_plateau floor). |
| `lr_reduce_factor` | `float` | `0.5` | ReduceLROnPlateau: multiply LR by this factor on plateau. |
| `lr_reduce_patience` | `int` | `500` | ReduceLROnPlateau: episodes without improvement before decay. |
| `lr_reduce_cooldown` | `int` | `100` | ReduceLROnPlateau: episodes to wait after a decay before resuming monitoring. |
| `lr_reduce_min_delta` | `float` | `1e-06` | ReduceLROnPlateau: minimum loss drop that counts as improvement. |

## `NetworkConfig`

Policy network architecture; nested under ``network:`` in YAML.

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `str` | `'mlp'` | Network architecture: `mlp` (feedforward), `lstm`, `transformer`, or `linear_plus_mlp`. |
| `hidden_sizes` | `tuple[int, Ellipsis]` | `(64, 64)` | Hidden layer widths. E.g. `(64, 64)` = two 64-unit hidden layers. |
| `activation` | `str` | `'tanh'` | Per-layer activation: `tanh`, `relu`, `gelu`, `silu`, `sigmoid`, `softplus`. |
| `activations` | `Union[tuple[str, Ellipsis], None]` | `None` | Per-layer activations if different per layer. None = use `activation` uniformly. Length = `len(hidden_sizes)`. |
| `init` | `str` | `'default'` | Weight init scheme: `default` (Equinox default), `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`. |
| `multi_head` | `bool` | `False` | If True, use separate output heads per policy dimension (experimental). |
| `skip_connections` | `bool` | `False` | If True, add residual connections between matching-width hidden layers. |
| `history_len` | `int` | `1` | History window length for sequence policies. 1 = MLP (no history). >1 = LSTM / Transformer. |
| `num_heads` | `int` | `4` | Transformer: attention heads per layer. |
| `n_layers` | `int` | `2` | Transformer: number of transformer blocks. |
| `init_scale` | `float` | `0.0` | `linear_plus_mlp` only: init scale of the MLP delta's final layer. 0.0 = policy starts exactly at the linear solution. |
| `use_zlb_feature` | `bool` | `False` | `linear_plus_mlp` + disaster only: prepend `(R_lag - R_lb)` as an extra feature for the delta MLP. Experimental. |

## `CompositeLossConfig`

Composite-loss weights (only active when ``loss_type: composite``); nested under ``composite_loss:`` in YAML.

| Field | Type | Default | Description |
|---|---|---|---|
| `anchor_weight` | `float` | `0.1` | Weight on the anchor loss (\|\|π_net(x) - π_lin(x)\|\|² at sampled anchor points near SS). |
| `jac_weight` | `float` | `0.01` | Weight on the Jacobian-match loss (\|\|J_net(SS) - P\|\|² at the steady state). |
| `jac_anchor_weight` | `float` | `0.0` | Weight on the per-anchor Jacobian match (\|\|J_net(x_i) - P\|\|² averaged over anchors). 0 = off. ~d× more expensive than `jac_weight`. |
| `barrier_weight` | `float` | `0.01` | Weight on economic feasibility barriers (net worth, leverage, consumption positivity). |
| `newton_weight` | `float` | `0.01` | Weight on Newton-step auxiliary losses (condition number, residual) for kink-approximation stabilization. |
| `n_anchor_points` | `int` | `64` | Number of anchor points sampled near SS at setup time (deterministic). |
| `anchor_sigma` | `float` | `1.0` | Scale of the Gaussian spread around SS for anchor-point sampling. |
| `leverage_mult` | `float` | `5.0` | Leverage barrier fires when `L > leverage_mult * L_ss`. Higher = more permissive. |
| `aux_decay_floor` | `float` | `0.2` | Minimum retained weight of anchor+jac auxiliaries as curriculum progresses. Set to 1.0 to keep aux terms fully active throughout. |
