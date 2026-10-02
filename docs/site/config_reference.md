# Config reference

Every field on the seven Pydantic config classes (``TrainConfig`` and its nested blocks ``OptimizerConfig``, ``NetworkConfig``, ``CompositeLossConfig``, ``ReplayBufferConfig``, ``CoverageConfig``, ``MomentMatchingConfig``) with its type, default, and a one-line description.

Generated from introspection by ``scripts/dev/gen_config_reference.py`` — regenerate after any config change:

```bash
uv run python scripts/dev/gen_config_reference.py
```

A field with description ``—`` has no explicit ``Field(description=...)`` yet.

For YAML / CLI usage patterns (override precedence, sampling conventions, checkpoint/resume rules, etc.) see [Running experiments](running_experiments.md). For building models with these configs, see [Implementing a model](models/implementing.md).

## `TrainConfig`

Top-level training configuration.

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `'brock_mirman'` | Name of the registered model to train; see `deqn-jax list` for valid choices. |
| `episodes` | `int` | `1000` | Number of outer training cycles (rollout + minibatch sweep). |
| `batch_size` | `int` | `64` | Minibatch size used for each gradient step. |
| `episode_length` | `int` | `100` | Trajectory length per rollout (T). With T=1 you must set `initialize_each_episode=True` (checked when training starts). |
| `mc_samples` | `int` | `5` | Monte Carlo shock samples per state for the residual expectation. Ignored when a deterministic rule is in use: `expectation_type` `quadrature`/`gh`/`gauss_hermite` (unless the grid exceeds 4096 nodes and falls back to MC) or `monomial`, and any model with a discrete shock chain. `expectation_type='discrete'` on a model without a chain runs MC with this sample count. |
| `seed` | `int` | `42` | Top-level PRNG seed. Controls network init and the rollout/loss shock streams. |
| `network` | `NetworkConfig` | `NetworkConfig()` | Policy network architecture; see NetworkConfig. |
| `optimizer` | `OptimizerConfig` | `OptimizerConfig()` | Optimizer and LR schedule; see OptimizerConfig. |
| `loss_type` | `str` | `'mse'` | `mse` = base residual MSE. `composite` = base + anchor + Jacobian + barriers + Newton (disaster-style). Composite is rejected at startup with MAO / GN / IGN / LM (LBFGS and PCGrad compose with it). |
| `composite_loss` | `CompositeLossConfig` | `CompositeLossConfig()` | Composite-loss weights; only active when `loss_type='composite'`. |
| `replay_buffer` | `ReplayBufferConfig` | `ReplayBufferConfig()` | Prioritized state-replay buffer; only active when `replay_buffer.enabled=true`. |
| `coverage` | `CoverageConfig` | `CoverageConfig()` | EWM coverage sampling (base + stress + local pools); only active when `coverage.enabled=true`. |
| `moment_matching` | `MomentMatchingConfig` | `MomentMatchingConfig()` | Aux loss biasing ergodic moments toward a Dynare reference; only active when `moment_matching.enabled=true`. |
| `loss_choice` | `str` | `'mse'` | Residual aggregation: `mse` (square the shock-mean residual), `huber` (Huber of the shock-mean; caps gradient at ±huber_delta when rare pathological states dominate), or `aio` (all-in-one, Maliar-Maliar-Winant 2021: product of two independent shock-group means -- unbiased for (E[r])², removing the Var(r̄)/N bias of `mse` under MC (on two-stage models the Jensen bias of a nonlinear `combine_fn` remains); requires expectation_type='mc' and mc_samples>=2; per-eq losses can be transiently negative, so prefer loss_reweight='none'). |
| `huber_delta` | `float` | `1.0` | Cutoff for Huber loss (`loss_choice='huber'`). Ignored for `loss_choice='mse'`. |
| `warm_start` | `bool` | `False` | If True, pre-fit the network to a target policy before gradient-based training: by default an L-BFGS fit to the constant steady-state policy (see `warm_start_linearize` and `warm_start_dynare` for the other targets). Skipped for networks anchored at the linear policy (`linear_plus_mlp`, `disaster_policy_net`). Speeds early convergence; can mask Euler-equation bugs. |
| `warm_start_linearize` | `bool` | `False` | With `warm_start=True`, fit the network to the Blanchard-Kahn linear policy (from linearizing the model around SS) at sampled states near SS instead of to the constant SS policy. Not used by sequence networks. |
| `warm_start_dynare` | `Union[str, None]` | `None` | With `warm_start=True`, directory holding Dynare's `dynare_ghx.csv` and `dynare_ghu.csv`; the network is fitted (with Adam) to the Dynare first-order policy instead. Takes precedence over `warm_start_linearize`. Not used by sequence networks. Rare. |
| `loss_weights` | `Union[list[float], None]` | `None` | Manual per-equation weight vector of length `n_equations`. Default None = uniform weight 1.0. |
| `loss_reweight` | `str` | `'none'` | Adaptive reweighting: `none` (default), `lr_annealing` (inverse-EMA), `relobralo` (softmax of loss ratios). |
| `reweight_alpha` | `float` | `0.9` | For `lr_annealing`: EMA decay of the per-equation losses (higher = slower adaptation). For `relobralo`: weight on the balancing term built from the ratio to the previous step's losses, against the ratio to the first step's losses. |
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
| `switch_optimizer` | `Union[str, None]` | `None` | If set, switch to this optimizer name at `switch_episode`. Old optimizer state is discarded; the new optimizer is initialized at the current params. Only the name, `switch_lr` and `optimizer.grad_clip` carry over (other optimizer fields take their defaults), and the LR schedule is not applied after the switch. |
| `switch_episode` | `Union[int, None]` | `None` | Episode at which to activate `switch_optimizer` and `switch_lr`. |
| `switch_lr` | `Union[float, None]` | `None` | Learning rate for the switched optimizer. None = keep the original optimizer's LR. |
| `early_stop_patience` | `Union[int, None]` | `None` | Stop training if loss hasn't improved by `early_stop_min_delta` for this many episodes. None = no early stopping. |
| `early_stop_min_delta` | `float` | `1e-06` | Minimum absolute loss improvement counted against `early_stop_patience`. |
| `curriculum_episodes` | `int` | `0` | Ramp `shock_scale` linearly from `curriculum_start` to 1.0 over this many episodes. 0 = no curriculum. |
| `curriculum_start` | `float` | `0.1` | Initial `shock_scale` when curriculum is active. |
| `ss_reset_frac` | `float` | `0.0` | Fraction of batch re-initialized to SS-neighborhood each rollout (prevents trajectory drift). Not applied when `initialize_each_episode` redraws the whole batch. |
| `initialize_each_episode` | `bool` | `False` | If True, replace episode_state with a fresh `init_state_fn` draw at the start of every rollout cycle (non-ergodic training, matches DEQN-MAO's flag of the same name). False = continue trajectory across cycles (ergodic). Required True when `episode_length=1`. Has no effect on a model without `init_state_fn`. |
| `expectation_type` | `str` | `'mc'` | How to integrate over shocks in the residual: `mc` (antithetic Monte Carlo, uses `mc_samples`) or `quadrature`/`gh`/`gauss_hermite` (deterministic tensor-product grid, uses `n_quadrature_points`) or `monomial` (degree-3, 2*n_shocks nodes, practical when n_shocks > 6) or `discrete`. A model that sets `transition_matrix` and `z_state_idx` always gets exact enumeration over its finite-state Markov chain, in the residual expectation and categorical draws from `Π[z_t]` in the rollout, whatever this field says; `discrete` names that case and on a model without a chain runs MC. Other models roll out with Gaussian draws. |
| `n_quadrature_points` | `int` | `3` | Quadrature points per shock dimension when `expectation_type` is `quadrature`/`gh`/`gauss_hermite` (total nodes = n_quadrature_points^n_shocks). Ignored by `monomial`, whose node count is always 2*n_shocks. |
| `barrier_weight` | `float` | `0.0` | Legacy state-barrier penalty weight. 0 disables. Applies only to models that define `state_barrier_fn`, and only under `loss_type='mse'` (rejected with `composite`). Prefer `definition_bounds` on the ModelSpec for new models. |
| `shock_mask` | `Union[list[float], None]` | `None` | Per-dimension multiplicative mask over shocks (length must equal `model.n_shocks`). Values in [0, 1]; 0 zeroes that shock entirely. Applied to BOTH the residual expectation and the rollout state path. |
| `target_update_every` | `int` | `0` | Target-network update interval in episodes. 0 disables target network entirely. |
| `target_tau` | `float` | `1.0` | Polyak averaging coefficient for target-network update. 1.0 = hard copy, <1 = soft update toward current params. |
| `constants` | `dict[str, float]` | `{}` | Per-run override of model.constants (e.g. `{p_disaster: 0.02}`). Merges into the model's built-in calibration. |
| `use_risky_steady_state` | `bool` | `True` | Read by the disaster model's `setup_fn`. If True and `p_disaster > 0`, anchor composite loss and linearization at the risky SS (E_d[F]=0) instead of deterministic SS. Set False to force deterministic SS anchor under disaster risk (for ablation). |
| `save_best_checkpoint` | `bool` | `True` | If True and `checkpoint_dir` is set, persist `checkpoint_best.eqx` on every loss improvement after a grace period of max(`curriculum_episodes`, `log_every`) episodes. Guards against rare huge-gradient events corrupting the latest snapshot. |
| `n_epochs_per_rollout` | `int` | `1` | DEQN cycle: per outer iteration, 1 rollout fills a trajectory of (`sim_batch` × `episode_length`) states, then we do `n_epochs_per_rollout` sweeps over it. Default 1 matches DEQN-MAO's run_cycle. |
| `n_minibatches_per_epoch` | `Union[int, None]` | `None` | Minibatches per sweep. None = all available (full-trajectory sweep). Set to 1 for the legacy one-grad-per-rollout behavior. |
| `sorted_within_batch` | `bool` | `False` | Minibatch shuffle policy. False = IID shuffle across all (episode_length × sim_batch) samples. True = each minibatch is a contiguous slice of the trajectory-major data (RL-style), which can span several trajectories when `episode_length` is not a multiple of `batch_size`; batch order shuffled, intra-batch order preserved. MLP-only. |
| `sim_batch` | `Union[int, None]` | `None` | Number of parallel simulation trajectories in the rollout. None (default) = same as `batch_size`. Setting `sim_batch > batch_size` decouples trajectory count from gradient minibatch size — larger pool = more representative ergodic distribution per cycle. |

## `OptimizerConfig`

Optimizer choice and hyperparameters; nested under ``optimizer:`` in YAML.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `'adam'` | Optimizer name. Options: `adam`, `muon`, `ngd`, `shampoo`, `lbfgs`, `mao`, `gn`, `ign`, `lm`. |
| `learning_rate` | `float` | `0.001` | Peak learning rate (or constant LR when `lr_schedule='constant'`). |
| `grad_clip` | `Union[float, None]` | `None` | Global-norm clipping threshold. None disables. Chained before the update for STANDARD optimizers; MAO clips the norm of its update instead. Rejected for `lbfgs`, `gn`, `ign` and `lm`, whose update paths do not apply it. |
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
| `lr_schedule` | `str` | `'constant'` | LR schedule: `constant` or `cosine`. |
| `lr_warmup` | `int` | `0` | With `lr_schedule='cosine'`: episodes of linear warmup from 0 to `learning_rate`, counted inside the schedule's `episodes` horizon. Ignored by `constant`. |
| `lr_min_factor` | `float` | `0.0` | Minimum LR as a fraction of peak (cosine floor). |

## `NetworkConfig`

Policy network architecture; nested under ``network:`` in YAML.

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `str` | `'mlp'` | Network architecture: `mlp` (feedforward), `lstm`, `transformer`, `linear_plus_mlp` (generic residual ansatz), `disaster_policy_net` (residual ansatz + disaster-specific shape priors), or `rss_market_clearing_net` (fixed RSS checkpoint-parity architecture). |
| `hidden_sizes` | `tuple[int, Ellipsis]` | `(64, 64)` | Hidden layer widths. E.g. `(64, 64)` = two 64-unit hidden layers. `lstm` stacks one cell per entry, `transformer` uses only the first entry as its model width, `rss_market_clearing_net` requires exactly two. |
| `activation` | `str` | `'tanh'` | Per-layer activation: `tanh`, `relu`, `gelu`, `silu`, `softplus`. Read by `mlp`, `linear_plus_mlp` and `disaster_policy_net`. |
| `activations` | `Union[tuple[str, Ellipsis], None]` | `None` | Per-layer activations if different per layer. None = use `activation` uniformly. Length = `len(hidden_sizes)`. Read by `mlp` only. |
| `init` | `str` | `'default'` | Weight init scheme: `default` (Equinox default), `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`. Read by `mlp`, `linear_plus_mlp` and `disaster_policy_net`. |
| `history_len` | `int` | `1` | History window length for the sequence networks (`lstm`, `transformer`); other network types ignore it. 1 = no history. |
| `num_heads` | `int` | `4` | Transformer: attention heads per layer. |
| `n_layers` | `int` | `2` | Transformer: number of transformer blocks. |
| `init_scale` | `float` | `0.0` | `linear_plus_mlp` and `disaster_policy_net`: init scale of the MLP delta's final layer. 0.0 = policy starts exactly at the linear solution. |
| `use_zlb_feature` | `bool` | `False` | `disaster_policy_net` only: append `(R_lag - R_lb)` as an extra MLP input feature. |
| `bk_pin` | `bool` | `False` | `disaster_policy_net` only: Blanchard-Kahn selection by construction — subtract the MLP delta's value and tangent at the steady state, so pi(s*)=pi* and dpi/ds(s*)=P hold exactly for every parameter value. The residual loss then shapes only second-order-and-beyond deviations. |
| `zlb_feature_kind` | `Literal[raw, kink]` | `'raw'` | `disaster_policy_net` only, when use_zlb_feature=true: 'raw' = signed distance R_lag - R_lb; 'kink' = max(R_lag - R_lb, 0), PINN-style explicit kink at the floor. |
| `kf_names` | `tuple[str, Ellipsis]` | `('F_p', 'K_p', 'F_w', 'K_w')` | `disaster_policy_net`: policy names whose MLP delta is masked to zero (a restriction that holds them linear; not a gauge fix — graph @aleph/deqn #34). Default targets the four CMR Calvo Phillips-curve auxiliaries. |
| `reparam_q_as_m` | `bool` | `False` | `disaster_policy_net` only: treat the network's `q` output as `M = q · 𝓑(x)` where 𝓑(x) = 1 - S(x) - x·S'(x) is the investment-Euler bracket; recover q = M/𝓑(x) post-MLP (𝓑 floored at 1e-3 in the division). Aimed at the eq 7 sign-flip pathology; 𝓑(x) itself is not constrained. |
| `reparam_pi_as_kp_inner` | `bool` | `False` | `disaster_policy_net` only: treat the network's `pi` output as K_p_inner ∈ (0, 1/(1−ξ_p)); derive π via the inverse Calvo formula post-clip. Encodes the Calvo asymptote in the parameterization so the MLP only learns smooth K_p_inner. |
| `reparam_wtilda_as_kw_inner` | `bool` | `False` | `disaster_policy_net` only: treat the network's `w_tilda` output as K_w_inner ∈ (0, 1/(1−ξ_w)); derive w_tilda via the inverse eq 4a formula post-clip. Wage-side mirror of reparam_pi_as_kp_inner; combine with that flag for symmetric Calvo reparam. |
| `output_links` | `Union[tuple[str, Ellipsis], None]` | `None` | Per-policy output parameterization for residual networks. Each entry must be 'linear' (additive: π_i = ss_i + BK + MLP) or 'log' (multiplicative: π_i = ss_i·exp(BK_log + MLP), bakes in positivity). Length must equal n_policies. None = use the model's default_output_links (or all-linear if model doesn't specify). |

## `CompositeLossConfig`

Composite-loss weights (only active when ``loss_type: composite``); nested under ``composite_loss:`` in YAML.

| Field | Type | Default | Description |
|---|---|---|---|
| `anchor_weight` | `float` | `0.1` | Weight on the anchor loss (mean squared difference π_net(x) - π_lin(x) over the anchor points sampled near SS and the policies). |
| `jac_weight` | `float` | `0.01` | Weight on the Jacobian-match loss (mean squared entry of J_net(SS) - P at the steady state). |
| `jac_anchor_weight` | `float` | `0.0` | Weight on the per-anchor Jacobian match (mean squared entry of J_net(x_i) - P, averaged over anchors). 0 = off. ~d× more expensive than `jac_weight`. |
| `barrier_weight` | `float` | `0.01` | Weight on economic feasibility barriers (net worth, leverage, consumption positivity). Read by the model's `composite_aux_fn`; only the disaster model defines one. |
| `newton_weight` | `float` | `0.01` | Weight on Newton-step auxiliary losses (condition number, residual) for kink-approximation stabilization. Read by the model's `composite_aux_fn`; only the disaster model defines one. |
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
| `leverage_mult` | `float` | `5.0` | Leverage barrier fires when `L > leverage_mult * L_ss`. Higher = more permissive. Read by the disaster model's `composite_aux_fn`. |
| `aux_decay_floor` | `float` | `0.2` | Minimum retained weight of anchor+jac auxiliaries as curriculum progresses. Set to 1.0 to keep aux terms fully active throughout. |

## `ReplayBufferConfig`

Prioritized state-replay buffer (only active when ``enabled: true``); nested under ``replay_buffer:`` in YAML.

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `False` | Master switch. When False, the cycle path is byte-identical to no-replay training. |
| `capacity` | `int` | `65536` | Number of past states retained in the ring buffer. Memory: capacity × n_states × 4B. |
| `mix_ratio` | `float` | `0.5` | Fraction of each minibatch dataset drawn from the buffer (0=none, 1=all-buffer). 0.5 is the natural default. |
| `min_fill_frac` | `float` | `0.25` | Buffer must reach this fraction of capacity before sampling activates. Until then, training uses current trajectory only. |
| `priority_alpha` | `float` | `0.6` | PER's α: sampling probability ∝ (priority + eps)^α. α=0 is uniform, α=1 is fully proportional. 0.6 is the original PER default. |
| `priority_eps` | `float` | `1e-06` | Floor added to priorities before exponentiation. Prevents zero-priority states from being completely starved. |

## `CoverageConfig`

EWM coverage sampling (only active when ``enabled: true``); nested under ``coverage:`` in YAML.

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `False` | Master switch. When False, training is byte-identical to no coverage. |
| `rho_base` | `float` | `1.0` | Mixture weight on the base (init-rect / on-policy) pool. |
| `rho_stress` | `float` | `0.5` | Mixture weight on the stress pool (paper's coverage-exact arm: 0.5 relative to path). |
| `rho_local` | `float` | `0.25` | Mixture weight on the local-perturbation pool (paper: 0.25). Weights are normalized over the included pools inside the wrapper. |
| `n_stress` | `int` | `128` | Number of stress seeds drawn per step (before rollout). |
| `n_local` | `int` | `128` | Number of local perturbations per step. |
| `rollout_horizon` | `int` | `5` | H: steps to roll stress seeds through the exact transition (paper: typically 3 or 5). |
| `local_sigma` | `float` | `0.02` | Std of Gaussian local perturbations, in state units. |
| `stress_ranges` | `dict[str, tuple[float, float]]` | `{}` | Per-state-name uniform box for stress seeds. Keys are state names (validated against model.state_names at model resolution). Empty is an error when enabled with rho_stress>0. |
| `repair_ranges` | `dict[str, tuple[float, float]]` | `{}` | Per-state-name feasible box; stress landings and local perturbations are clipped into it before the residual is evaluated (the paper's repair step). Empty = no clipping. |
| `stress_seed_mode` | `Literal[box, path]` | `'box'` | 'box' (historical variant): stress seeds are SS-filled states with the stress dims uniform in stress_ranges; the raw seed is excluded from the pool. 'path' (the paper's measure): seeds are visited batch states with ONLY the stress dims overridden — every other coordinate keeps its realistic joint value — and the seed itself joins the pool alongside its rollout landings. |

## `MomentMatchingConfig`

Moment-matching auxiliary loss (only active when ``enabled: true``); nested under ``moment_matching:`` in YAML.

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `False` | Master switch. When False, training behaviour is identical to the base loss. |
| `weight` | `float` | `0.1` | Multiplier on the aux loss term added to the total loss. |
| `mean_weight` | `float` | `1.0` | Within the aux, weight on the squared mean-deviation term. |
| `std_weight` | `float` | `1.0` | Within the aux, weight on the squared std-deviation term. |
| `dynare_dir` | `str` | `'dynare/results'` | Directory containing dynare_moments.csv (the target moments). |
| `scale_eps` | `float` | `0.001` | Floor on the per-variable scale used for relative comparison; prevents division blowup for variables with near-zero target. |
