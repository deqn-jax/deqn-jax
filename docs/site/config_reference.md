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
| `mc_samples` | `int` | `5` | Monte Carlo shock samples per state for the residual expectation. Ignored when `expectation_type` is `quadrature`/`gh`/`gauss_hermite` or `discrete`. |
| `seed` | `int` | `42` | Top-level PRNG seed. Controls network init and the rollout/loss shock streams. |
| `network` | `NetworkConfig` | `PydanticUndefined` | Policy network architecture; see NetworkConfig. |
| `optimizer` | `OptimizerConfig` | `PydanticUndefined` | Optimizer and LR schedule; see OptimizerConfig. |
| `loss_type` | `str` | `'mse'` | `mse` = base residual MSE. `composite` = base + anchor + Jacobian + barriers + Newton (disaster-style). Composite is rejected at startup with MAO / GN / IGN / LM (LBFGS and PCGrad compose with it). |
| `composite_loss` | `CompositeLossConfig` | `PydanticUndefined` | Composite-loss weights; only active when `loss_type='composite'`. |
| `replay_buffer` | `ReplayBufferConfig` | `PydanticUndefined` | Prioritized state-replay buffer; only active when `replay_buffer.enabled=true`. |
| `coverage` | `CoverageConfig` | `PydanticUndefined` | EWM coverage sampling (base + stress + local pools); only active when `coverage.enabled=true`. |
| `moment_matching` | `MomentMatchingConfig` | `PydanticUndefined` | Aux loss biasing ergodic moments toward a Dynare reference; only active when `moment_matching.enabled=true`. |
| `loss_choice` | `str` | `'mse'` | Residual aggregation: `mse` (square the shock-mean residual), `huber` (Huber of the shock-mean; caps gradient at ±huber_delta when rare pathological states dominate), or `aio` (all-in-one, Maliar-Maliar-Winant 2021: product of two independent shock-group means -- unbiased for (E[r])², removing the Var(r̄)/N bias of `mse` under MC; requires expectation_type='mc' and mc_samples>=2; per-eq losses can be transiently negative, so prefer loss_reweight='none'). |
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
| `expectation_type` | `str` | `'mc'` | How to integrate over shocks in the residual: `mc` (antithetic Monte Carlo, uses `mc_samples`) or `quadrature`/`gh`/`gauss_hermite` (deterministic tensor-product grid, uses `n_quadrature_points`) or `discrete` (exact enumeration over a finite-state Markov chain; requires `model.transition_matrix` and `model.z_state_idx`). Trajectory rollout uses Gaussian draws for `mc`/`quadrature` and categorical draws from `Π[z_t]` for `discrete`. |
| `n_quadrature_points` | `int` | `3` | Quadrature points per shock dimension when `expectation_type` is `quadrature`/`gh`/`gauss_hermite`. Total nodes = n_quadrature_points^n_shocks. |
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
| `name` | `str` | `'adam'` | Optimizer name. Options: `adam`, `sgd`, `adamw`, `lion`, `muon`, `ngd`, `shampoo`, `lbfgs`, `mao`, `mao_kfac`, `gn`, `ign`, `lm`. |
| `learning_rate` | `float` | `0.001` | Peak learning rate (or constant LR when `lr_schedule='constant'`). |
| `grad_clip` | `Union[float, None]` | `None` | Global gradient-norm clipping. None disables. |
| `weight_decay` | `float` | `0.0` | L2 weight decay (used by adamw only). |
| `beta1` | `float` | `0.9` | Adam / MAO first-moment decay. |
| `beta2` | `float` | `0.999` | Adam / MAO second-moment decay. |
| `epsilon` | `float` | `1e-08` | Adam / MAO numerical floor. |
| `damping` | `float` | `0.0001` | Preconditioner damping for NGD / GN / IGN / LM. |
| `decay` | `float` | `0.999` | NGD preconditioner EMA decay. |
| `precond_update_freq` | `int` | `10` | Shampoo preconditioner update frequency. |
| `memory_size` | `int` | `10` | L-BFGS history size. |
| `ns_steps` | `int` | `5` | Muon Newton-Schulz iteration count. |
| `cg_iters` | `int` | `20` | Implicit Gauss-Newton conjugate-gradient iteration cap. |
| `cg_tol` | `float` | `1e-06` | Implicit Gauss-Newton relative conjugate-gradient residual tolerance. |
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
| `type` | `str` | `'mlp'` | Network architecture: `mlp` (feedforward), `lstm`, `transformer`, `linear_plus_mlp` (generic residual ansatz), `disaster_policy_net` (residual ansatz + disaster-specific shape priors), or `kf_anchored_mlp` (legacy K/F gauge elimination). |
| `hidden_sizes` | `tuple[int, Ellipsis]` | `(64, 64)` | Hidden layer widths. E.g. `(64, 64)` = two 64-unit hidden layers. |
| `activation` | `str` | `'tanh'` | Per-layer activation: `tanh`, `relu`, `gelu`, `silu`, `softplus`. |
| `activations` | `Union[tuple[str, Ellipsis], None]` | `None` | Per-layer activations if different per layer. None = use `activation` uniformly. Length = `len(hidden_sizes)`. |
| `init` | `str` | `'default'` | Weight init scheme: `default` (Equinox default), `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`. |
| `multi_head` | `bool` | `False` | If True, use separate output heads per policy dimension (experimental). |
| `skip_connections` | `bool` | `False` | If True, add residual connections between matching-width hidden layers. |
| `history_len` | `int` | `1` | History window length for sequence policies. 1 = MLP (no history). >1 = LSTM / Transformer. |
| `num_heads` | `int` | `4` | Transformer: attention heads per layer. |
| `n_layers` | `int` | `2` | Transformer: number of transformer blocks. |
| `init_scale` | `float` | `0.0` | `linear_plus_mlp` and `disaster_policy_net`: init scale of the MLP delta's final layer. 0.0 = policy starts exactly at the linear solution. |
| `use_zlb_feature` | `bool` | `False` | `disaster_policy_net` only: prepend `(R_lag - R_lb)` as an extra MLP input feature. |
| `bk_pin` | `bool` | `False` | `disaster_policy_net` only: Blanchard-Kahn selection by construction — subtract the MLP delta's value and tangent at the steady state, so pi(s*)=pi* and dpi/ds(s*)=P hold exactly for every parameter value. The residual loss then shapes only second-order-and-beyond deviations. |
| `zlb_feature_kind` | `Literal[raw, kink]` | `'raw'` | `disaster_policy_net` only, when use_zlb_feature=true: 'raw' = signed distance R_lag - R_lb; 'kink' = max(R_lag - R_lb, 0), PINN-style explicit kink at the floor. |
| `kf_names` | `tuple[str, Ellipsis]` | `('F_p', 'K_p', 'F_w', 'K_w')` | `kf_anchored_mlp` and `disaster_policy_net`: policy names whose MLP delta is masked to zero (gauge fix). Default targets the four CMR Calvo Phillips-curve auxiliaries. |
| `reparam_q_as_m` | `bool` | `False` | `disaster_policy_net` only: treat the network's `q` output as `M = q · 𝓑(x)` where 𝓑(x) = 1 - S(x) - x·S'(x) is the investment-Euler bracket; recover q = M/𝓑(x) post-MLP. Eliminates the eq 7 sign-flip pathology by parameterization. |
| `reparam_pi_as_kp_inner` | `bool` | `False` | `disaster_policy_net` only: treat the network's `pi` output as K_p_inner ∈ (0, 1/(1−ξ_p)); derive π via the inverse Calvo formula post-clip. Encodes the Calvo asymptote in the parameterization so the MLP only learns smooth K_p_inner. |
| `reparam_wtilda_as_kw_inner` | `bool` | `False` | `disaster_policy_net` only: treat the network's `w_tilda` output as K_w_inner ∈ (0, 1/(1−ξ_w)); derive w_tilda via the inverse eq 4a formula post-clip. Wage-side mirror of reparam_pi_as_kp_inner; combine with that flag for symmetric Calvo reparam. |
| `output_links` | `Union[tuple[str, Ellipsis], None]` | `None` | Per-policy output parameterization for residual networks. Each entry must be 'linear' (additive: π_i = ss_i + BK + MLP) or 'log' (multiplicative: π_i = ss_i·exp(BK_log + MLP), bakes in positivity). Length must equal n_policies. None = use the model's default_output_links (or all-linear if model doesn't specify). |

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
| `anchor_gate` | `bool` | `False` | Kink-aware anchor: down-weight anchor points where the model declares its linearization invalid (requires the model to define `anchor_gate_fn`, e.g. disaster's interest-rate floor). Off = legacy unweighted anchor, bit-identical. |
| `drift_weight` | `float` | `0.0` | Certificate-in-the-loop stability loss: penalize average per-period log growth of the deterministic closed loop from small SS perturbations (≈ log rho(SS)). 0 = off, bit-identical. Does not decay with curriculum. |
| `drift_horizon` | `int` | `20` | Closed-loop rollout length T for the drift loss (gradient flows through all T steps). |
| `drift_eps` | `float` | `0.001` | Scale of the ergodic-shaped SS perturbations used as drift probes. |
| `drift_n_probes` | `int` | `4` | Number of fixed probe directions (drawn once at build time, a priori). |
| `drift_target` | `float` | `0.99` | Growth-factor target: the hinge fires when the per-period growth exceeds log(drift_target). 0.99 asks for mild contraction. |
| `res_sobolev_weight` | `float` | `0.0` | Residual-Sobolev loss: penalize directional derivatives of the per-state EXPECTED residual toward zero (the true policy zeroes E[r] on a neighborhood; impostors keep values small with finite gradients). Adds selection information the value loss cannot see. 0 = off, bit-identical. Quadrature expectations + single-stage models only. |
| `res_sobolev_n_states` | `int` | `16` | Batch subsample size for the residual-Sobolev term (cost control). |
| `res_sobolev_n_dirs` | `int` | `2` | Number of fixed ergodic-shaped unit directions for the JVPs (drawn once at build time). |
| `anchor_sigma` | `float` | `1.0` | Scale of the Gaussian spread around SS for anchor-point sampling. |
| `leverage_mult` | `float` | `5.0` | Leverage barrier fires when `L > leverage_mult * L_ss`. Higher = more permissive. |
| `aux_decay_floor` | `float` | `0.2` | Minimum retained weight of anchor+jac auxiliaries as curriculum progresses. Set to 1.0 to keep aux terms fully active throughout. |
