"""TrainConfig: the top-level training configuration."""

from __future__ import annotations

import copy
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import (
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from deqn_jax.config._base import (
    _coerce_float,
    _coerce_int,
    _coerce_optional_float,
    _coerce_optional_int,
    _ConfigBase,
)
from deqn_jax.config.loss import CompositeLossConfig, MomentMatchingConfig
from deqn_jax.config.network import NetworkConfig
from deqn_jax.config.optimizer import OptimizerConfig
from deqn_jax.config.replay import ReplayBufferConfig

# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------


class TrainConfig(_ConfigBase):
    """Complete training configuration.

    Supports construction from:
    - Direct keyword arguments
    - YAML file via from_yaml()
    - Dictionary via from_dict()
    - Overrides via with_overrides()
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        default="brock_mirman",
        description="Name of the registered model to train; see `deqn-jax list` for valid choices.",
    )
    episodes: int = Field(
        default=1000,
        description="Number of outer training cycles (rollout + minibatch sweep).",
    )
    batch_size: int = Field(
        default=64, description="Minibatch size used for each gradient step."
    )
    episode_length: int = Field(
        default=100,
        description="Trajectory length per rollout (T). With T=1 you must set `initialize_each_episode=True` (see validator).",
    )
    mc_samples: int = Field(
        default=5,
        description="Monte Carlo shock samples per state for the residual expectation. Ignored when `expectation_type='gauss_hermite'`.",
    )
    seed: int = Field(
        default=42,
        description="Top-level PRNG seed. Controls network init and the rollout/loss shock streams.",
    )

    network: NetworkConfig = Field(
        default_factory=NetworkConfig,
        description="Policy network architecture; see NetworkConfig.",
    )
    optimizer: OptimizerConfig = Field(
        default_factory=OptimizerConfig,
        description="Optimizer and LR schedule; see OptimizerConfig.",
    )

    loss_type: str = Field(
        default="mse",
        description="`mse` = base residual MSE. `composite` = base + anchor + Jacobian + barriers + Newton (disaster-style). Composite is rejected at startup with MAO / GN / LM / LBFGS / PCGrad.",
    )
    composite_loss: CompositeLossConfig = Field(
        default_factory=CompositeLossConfig,
        description="Composite-loss weights; only active when `loss_type='composite'`.",
    )
    replay_buffer: ReplayBufferConfig = Field(
        default_factory=ReplayBufferConfig,
        description="Prioritized state-replay buffer; only active when `replay_buffer.enabled=true`.",
    )
    moment_matching: MomentMatchingConfig = Field(
        default_factory=MomentMatchingConfig,
        description="Aux loss biasing ergodic moments toward a Dynare reference; only active when `moment_matching.enabled=true`.",
    )

    loss_choice: str = Field(
        default="mse",
        description="Residual aggregation: `mse` (square the shock-mean residual), `huber` (Huber of the shock-mean; caps gradient at ±huber_delta when rare pathological states dominate), or `aio` (all-in-one, Maliar-Maliar-Winant 2021: product of two independent shock-group means -- unbiased for (E[r])², removing the Var(r̄)/N bias of `mse` under MC; requires expectation_type='mc' and mc_samples>=2; per-eq losses can be transiently negative, so prefer loss_reweight='none').",
    )
    huber_delta: float = Field(
        default=1.0,
        description="Cutoff for Huber loss (`loss_choice='huber'`). Ignored for `loss_choice='mse'`.",
    )

    warm_start: bool = Field(
        default=False,
        description="If True, run L-BFGS pre-fit of the network to the steady-state policy before gradient-based training. Speeds early convergence; can mask Euler-equation bugs.",
    )
    warm_start_linearize: bool = Field(
        default=False,
        description="If True, linearize the model around SS and use the Blanchard-Kahn P matrix to seed the network's Jacobian at SS. Advanced.",
    )
    warm_start_dynare: Optional[str] = Field(
        default=None,
        description="Path to a Dynare output file to seed warm-start linearization. Rare.",
    )
    loss_weights: Optional[List[float]] = Field(
        default=None,
        description="Manual per-equation weight vector of length `n_equations`. Default None = uniform weight 1.0.",
    )
    loss_reweight: str = Field(
        default="none",
        description="Adaptive reweighting: `none` (default), `lr_annealing` (inverse-EMA), `relobralo` (softmax of loss ratios).",
    )
    reweight_alpha: float = Field(
        default=0.9,
        description="EMA decay for `lr_annealing` / `relobralo`. Higher = slower adaptation.",
    )

    log_every: int = Field(
        default=100,
        description="Episodes between console / TensorBoard scalar logs and cycle_hook invocations.",
    )
    verbose: bool = Field(
        default=True,
        description="If False, suppress console output (the CLI `-q` flag sets this).",
    )
    fp64: bool = Field(
        default=False,
        description="Enable JAX x64 mode for higher numerical precision. Applied at `train_from_config` entry.",
    )

    tensorboard_dir: Optional[str] = Field(
        default=None,
        description="Directory for TensorBoard event files. None disables TB logging.",
    )
    wandb_project: Optional[str] = Field(
        default=None, description="W&B project name. None disables W&B logging."
    )
    checkpoint_dir: Optional[str] = Field(
        default=None,
        description="Directory to save checkpoints (`checkpoint_<episode>.eqx` + `checkpoint_best.eqx` + `config.yaml`). None disables.",
    )
    checkpoint_every: Optional[int] = Field(
        default=None,
        description="Episodes between periodic checkpoints. None = no periodic checkpoints (only best is saved).",
    )
    max_checkpoints: Optional[int] = Field(
        default=None,
        description="Keep only the N most recent periodic checkpoints (best is never deleted).",
    )

    gradient_surgery: str = Field(
        default="none",
        description="Multi-equation gradient conflict resolution: `none` or `pcgrad` (projecting conflicting gradients).",
    )
    resume: Optional[str] = Field(
        default=None,
        description="Path to a `.eqx` checkpoint to resume from. Reads the sibling `config.yaml` to rebuild the correct pytree template.",
    )
    switch_optimizer: Optional[str] = Field(
        default=None,
        description="If set, switch to this optimizer name at `switch_episode`. Old optimizer state is discarded; new optimizer is initialized from resumed params.",
    )
    switch_episode: Optional[int] = Field(
        default=None,
        description="Episode at which to activate `switch_optimizer` and `switch_lr`.",
    )
    switch_lr: Optional[float] = Field(
        default=None,
        description="Learning rate for the switched optimizer. None = keep the original optimizer's LR.",
    )

    early_stop_patience: Optional[int] = Field(
        default=None,
        description="Stop training if loss hasn't improved by `early_stop_min_delta` for this many episodes. None = no early stopping.",
    )
    early_stop_min_delta: float = Field(
        default=1e-6,
        description="Minimum absolute loss improvement counted against `early_stop_patience`.",
    )

    curriculum_episodes: int = Field(
        default=0,
        description="Ramp `shock_scale` linearly from `curriculum_start` to 1.0 over this many episodes. 0 = no curriculum.",
    )
    curriculum_start: float = Field(
        default=0.1, description="Initial `shock_scale` when curriculum is active."
    )
    ss_reset_frac: float = Field(
        default=0.0,
        description="Fraction of batch re-initialized to SS-neighborhood each rollout (prevents trajectory drift). Orthogonal to `initialize_each_episode`.",
    )

    initialize_each_episode: bool = Field(
        default=False,
        description=(
            "If True, replace episode_state with a fresh `init_state_fn` draw "
            "at the start of every rollout cycle (non-ergodic training, matches "
            "DEQN-MAO's flag of the same name). False = continue trajectory "
            "across cycles (ergodic). Required True when `episode_length=1`."
        ),
    )

    expectation_type: str = Field(
        default="mc",
        description=(
            "How to integrate over shocks in the residual: `mc` (antithetic "
            "Monte Carlo, uses `mc_samples`) or `quadrature`/`gh`/`gauss_hermite` "
            "(deterministic tensor-product grid, uses `n_quadrature_points`) "
            "or `discrete` (exact enumeration over a finite-state Markov chain; "
            "requires `model.transition_matrix` and `model.z_state_idx`). "
            "Trajectory rollout uses Gaussian draws for `mc`/`quadrature` and "
            "categorical draws from `Π[z_t]` for `discrete`."
        ),
    )
    n_quadrature_points: int = Field(
        default=3,
        description="Quadrature points per shock dimension when `expectation_type='gauss_hermite'`. Total nodes = n_quadrature_points^n_shocks.",
    )

    barrier_weight: float = Field(
        default=0.0,
        description="Legacy state-barrier penalty weight. 0 disables. Prefer `definition_bounds` on the ModelSpec for new models.",
    )
    shock_mask: Optional[List[float]] = Field(
        default=None,
        description=(
            "Per-dimension multiplicative mask over shocks (length must equal "
            "`model.n_shocks`). Values in [0, 1]; 0 zeroes that shock entirely. "
            "Applied to BOTH the residual expectation and the rollout state path."
        ),
    )

    target_update_every: int = Field(
        default=0,
        description="Target-network update interval in episodes. 0 disables target network entirely.",
    )
    target_tau: float = Field(
        default=1.0,
        description="Polyak averaging coefficient for target-network update. 1.0 = hard copy, <1 = soft update toward current params.",
    )

    constants: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-run override of model.constants (e.g. `{p_disaster: 0.02}`). Merges into the model's built-in calibration.",
    )

    use_risky_steady_state: bool = Field(
        default=True,
        description=(
            "If True and `p_disaster > 0`, anchor composite loss and "
            "linearization at the risky SS (E_d[F]=0) instead of deterministic SS. "
            "Set False to force deterministic SS anchor under disaster risk "
            "(for ablation)."
        ),
    )

    save_best_checkpoint: bool = Field(
        default=True,
        description=(
            "If True and `checkpoint_dir` is set, persist `checkpoint_best.eqx` "
            "on every loss improvement (after `curriculum_episodes` grace period). "
            "Guards against rare huge-gradient events corrupting the latest snapshot."
        ),
    )

    n_epochs_per_rollout: int = Field(
        default=1,
        description=(
            "DEQN cycle: per outer iteration, 1 rollout fills a trajectory of "
            "(`sim_batch` × `episode_length`) states, then we do `n_epochs_per_rollout` "
            "sweeps over it. Default 1 matches DEQN-MAO's run_cycle."
        ),
    )
    n_minibatches_per_epoch: Optional[int] = Field(
        default=None,
        description=(
            "Minibatches per sweep. None = all available (full-trajectory sweep). "
            "Set to 1 for the legacy one-grad-per-rollout behavior."
        ),
    )

    sorted_within_batch: bool = Field(
        default=False,
        description=(
            "Minibatch shuffle policy. False = IID shuffle across all "
            "(episode_length × sim_batch) samples. True = each minibatch is a "
            "contiguous temporal slice of a single trajectory (RL-style); batch "
            "order shuffled, intra-batch order preserved. MLP-only."
        ),
    )

    sim_batch: Optional[int] = Field(
        default=None,
        description=(
            "Number of parallel simulation trajectories in the rollout. None "
            "(default) = same as `batch_size`. Setting `sim_batch > batch_size` "
            "decouples trajectory count from gradient minibatch size — larger "
            "pool = more representative ergodic distribution per cycle."
        ),
    )

    VALID_LOSS_TYPES: ClassVar[frozenset] = frozenset({"mse", "composite"})
    VALID_LOSS_CHOICES: ClassVar[frozenset] = frozenset({"mse", "huber", "aio"})
    VALID_LOSS_REWEIGHTS: ClassVar[frozenset] = frozenset(
        {"none", "lr_annealing", "relobralo"}
    )
    VALID_GRADIENT_SURGERY: ClassVar[frozenset] = frozenset({"none", "pcgrad"})
    VALID_EXPECTATION_TYPES: ClassVar[frozenset] = frozenset(
        {"mc", "quadrature", "gh", "gauss_hermite", "discrete"}
    )

    # -- before-mode validators for type coercion --

    @field_validator("model", mode="before")
    @classmethod
    def _check_model_type(cls, v):
        if not isinstance(v, str):
            raise TypeError(
                f"TrainConfig.model: expected str, got {type(v).__name__} ({v!r})"
            )
        return v

    @field_validator(
        "episodes",
        "batch_size",
        "episode_length",
        "mc_samples",
        "seed",
        "log_every",
        "curriculum_episodes",
        "n_quadrature_points",
        "target_update_every",
        "n_epochs_per_rollout",
        mode="before",
    )
    @classmethod
    def _coerce_int_reject_bool(cls, v, info):
        return _coerce_int(v, info.field_name)

    @field_validator("n_minibatches_per_epoch", mode="before")
    @classmethod
    def _coerce_n_minibatches(cls, v):
        return _coerce_optional_int(v, "n_minibatches_per_epoch")

    @field_validator("sim_batch", mode="before")
    @classmethod
    def _coerce_sim_batch(cls, v):
        return _coerce_optional_int(v, "sim_batch")

    @field_validator(
        "reweight_alpha",
        "early_stop_min_delta",
        "curriculum_start",
        "ss_reset_frac",
        "barrier_weight",
        "target_tau",
        "huber_delta",
        mode="before",
    )
    @classmethod
    def _coerce_float_reject_bool(cls, v, info):
        return _coerce_float(v, info.field_name)

    @field_validator("switch_lr", mode="before")
    @classmethod
    def _coerce_switch_lr(cls, v, info):
        return _coerce_optional_float(v, info.field_name)

    @field_validator(
        "switch_episode",
        "checkpoint_every",
        "max_checkpoints",
        "early_stop_patience",
        mode="before",
    )
    @classmethod
    def _coerce_optional_int_fields(cls, v, info):
        return _coerce_optional_int(v, info.field_name)

    @field_validator(
        "verbose",
        "warm_start",
        "warm_start_linearize",
        "fp64",
        "use_risky_steady_state",
        "save_best_checkpoint",
        mode="before",
    )
    @classmethod
    def _check_bool_type(cls, v, info):
        if not isinstance(v, bool):
            raise TypeError(
                f"TrainConfig.{info.field_name}: expected bool, got {type(v).__name__} ({v!r})"
            )
        return v

    @field_validator("loss_weights", mode="before")
    @classmethod
    def _check_loss_weights_type(cls, v):
        if v is not None and not isinstance(v, list):
            raise TypeError(
                f"TrainConfig.loss_weights: expected Optional[List[float]], "
                f"got {type(v).__name__} ({v!r})"
            )
        return v

    @field_validator("constants", mode="before")
    @classmethod
    def _check_constants_type(cls, v):
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise TypeError(
                f"TrainConfig.constants: expected Dict[str, float], "
                f"got {type(v).__name__} ({v!r})"
            )
        for k, val in v.items():
            if not isinstance(k, str):
                raise TypeError(
                    f"TrainConfig.constants: keys must be str, got {type(k).__name__} ({k!r})"
                )
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(
                    f"TrainConfig.constants[{k!r}]: expected number, "
                    f"got {type(val).__name__} ({val!r})"
                )
        return v

    @field_validator("optimizer", mode="before")
    @classmethod
    def _coerce_optimizer_str(cls, v):
        """Allow `optimizer: "adam"` shorthand in YAML."""
        if isinstance(v, str):
            return OptimizerConfig(name=v)
        return v

    @field_validator("network", mode="before")
    @classmethod
    def _coerce_network_str(cls, v):
        """Allow `network: "mlp"` shorthand in YAML."""
        if isinstance(v, str):
            return NetworkConfig(type=v)
        return v

    @model_validator(mode="after")
    def _validate_ranges(self):
        if not self.model or not isinstance(self.model, str):
            raise ValueError(f"model must be a non-empty string, got {self.model!r}")
        if self.episodes <= 0:
            raise ValueError(f"episodes must be > 0, got {self.episodes}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.episode_length <= 0:
            raise ValueError(f"episode_length must be > 0, got {self.episode_length}")
        if self.mc_samples <= 0:
            raise ValueError(f"mc_samples must be > 0, got {self.mc_samples}")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        if self.loss_type not in self.VALID_LOSS_TYPES:
            raise ValueError(
                f"Unknown loss_type '{self.loss_type}'. "
                f"Valid: {sorted(self.VALID_LOSS_TYPES)}"
            )
        if self.loss_choice not in self.VALID_LOSS_CHOICES:
            raise ValueError(
                f"Unknown loss_choice '{self.loss_choice}'. "
                f"Valid: {sorted(self.VALID_LOSS_CHOICES)}"
            )
        if self.huber_delta <= 0:
            raise ValueError(f"huber_delta must be > 0, got {self.huber_delta}")
        if self.loss_choice == "aio":
            if self.expectation_type != "mc":
                raise ValueError(
                    "loss_choice='aio' requires expectation_type='mc': the "
                    "quadrature/discrete expectation paths are exact and have "
                    f"no MC bias to remove, got '{self.expectation_type}'."
                )
            if self.mc_samples < 2:
                raise ValueError(
                    "loss_choice='aio' needs mc_samples >= 2 to form two "
                    f"independent shock groups, got {self.mc_samples}."
                )
        if self.loss_reweight not in self.VALID_LOSS_REWEIGHTS:
            raise ValueError(
                f"Unknown loss_reweight '{self.loss_reweight}'. "
                f"Valid: {sorted(self.VALID_LOSS_REWEIGHTS)}"
            )
        if not (0 < self.reweight_alpha < 1):
            raise ValueError(
                f"reweight_alpha must be in (0, 1), got {self.reweight_alpha}"
            )
        if self.gradient_surgery not in self.VALID_GRADIENT_SURGERY:
            raise ValueError(
                f"Unknown gradient_surgery '{self.gradient_surgery}'. "
                f"Valid: {sorted(self.VALID_GRADIENT_SURGERY)}"
            )
        if self.expectation_type not in self.VALID_EXPECTATION_TYPES:
            raise ValueError(
                f"Unknown expectation_type '{self.expectation_type}'. "
                f"Valid: {sorted(self.VALID_EXPECTATION_TYPES)}"
            )
        if self.n_quadrature_points <= 0:
            raise ValueError(
                f"n_quadrature_points must be > 0, got {self.n_quadrature_points}"
            )
        if self.log_every <= 0:
            raise ValueError(f"log_every must be > 0, got {self.log_every}")
        if self.curriculum_episodes < 0:
            raise ValueError(
                f"curriculum_episodes must be >= 0, got {self.curriculum_episodes}"
            )
        if self.curriculum_episodes > 0 and not (0 < self.curriculum_start <= 1):
            raise ValueError(
                f"curriculum_start must be in (0, 1] when curriculum is active, "
                f"got {self.curriculum_start}"
            )
        if self.early_stop_min_delta < 0:
            raise ValueError(
                f"early_stop_min_delta must be >= 0, got {self.early_stop_min_delta}"
            )
        if self.switch_optimizer is not None and self.switch_episode is None:
            raise ValueError(
                "switch_episode must be set when switch_optimizer is specified"
            )
        if self.checkpoint_every is not None and self.checkpoint_every <= 0:
            raise ValueError(
                f"checkpoint_every must be > 0, got {self.checkpoint_every}"
            )
        if self.network is not None and self.network.history_len > self.episode_length:
            raise ValueError(
                f"history_len ({self.network.history_len}) must be <= episode_length "
                f"({self.episode_length}), otherwise no training windows can be formed"
            )
        if self.loss_weights is not None:
            if any(w < 0 for w in self.loss_weights):
                raise ValueError(
                    f"All loss_weights must be >= 0, got {self.loss_weights}"
                )
        if self.shock_mask is not None:
            if not all(0 <= m <= 1 for m in self.shock_mask):
                raise ValueError(
                    f"All shock_mask values must be in [0, 1], got {self.shock_mask}"
                )
        if self.target_update_every < 0:
            raise ValueError(
                f"target_update_every must be >= 0, got {self.target_update_every}"
            )
        if not (0 < self.target_tau <= 1):
            raise ValueError(f"target_tau must be in (0, 1], got {self.target_tau}")
        if self.n_epochs_per_rollout < 1:
            raise ValueError(
                f"n_epochs_per_rollout must be >= 1, got {self.n_epochs_per_rollout}"
            )
        if (
            self.n_minibatches_per_epoch is not None
            and self.n_minibatches_per_epoch < 1
        ):
            raise ValueError(
                f"n_minibatches_per_epoch must be >= 1 or None, got {self.n_minibatches_per_epoch}"
            )
        return self

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        """Create config from a flat or nested dictionary.

        Handles nested ``optimizer:`` and ``network:`` sub-dicts.
        """
        from deqn_jax.config.io import _check_unknown_keys

        d = copy.deepcopy(d)

        # Extract nested sub-configs
        opt_dict = d.pop("optimizer", {})
        net_dict = d.pop("network", {})
        comp_dict = d.pop("composite_loss", {})
        replay_dict = d.pop("replay_buffer", {})
        mom_dict = d.pop("moment_matching", {})

        # If optimizer is a plain string, treat as name
        if isinstance(opt_dict, str):
            opt_dict = {"name": opt_dict}
        if isinstance(net_dict, str):
            net_dict = {"type": net_dict}

        # Convert hidden_sizes list to tuple. The dict came from YAML
        # so its element type is Any; ty narrows it to ``str`` after
        # the ``isinstance(..., list)`` check on a *different* key,
        # which makes the tuple-of-Any assignment look invalid.
        # Pydantic re-validates on construction, so the runtime type
        # is checked there.
        if "hidden_sizes" in net_dict and isinstance(net_dict["hidden_sizes"], list):
            net_dict["hidden_sizes"] = tuple(net_dict["hidden_sizes"])  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-assignment]

        # Convert activations list to tuple
        if "activations" in net_dict and isinstance(net_dict["activations"], list):
            net_dict["activations"] = tuple(net_dict["activations"])  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-assignment]

        # Convert loss_weights list (YAML gives lists)
        if "loss_weights" in d and isinstance(d["loss_weights"], list):
            d["loss_weights"] = list(d["loss_weights"])

        # Validate: reject unknown keys (with did-you-mean suggestions)
        opt_fields = set(OptimizerConfig.model_fields.keys())
        net_fields = set(NetworkConfig.model_fields.keys())
        comp_fields = set(CompositeLossConfig.model_fields.keys())
        replay_fields = set(ReplayBufferConfig.model_fields.keys())
        mom_fields = set(MomentMatchingConfig.model_fields.keys())
        train_fields = set(TrainConfig.model_fields.keys())

        _check_unknown_keys(set(opt_dict.keys()), opt_fields, "optimizer")
        _check_unknown_keys(set(net_dict.keys()), net_fields, "network")
        _check_unknown_keys(set(comp_dict.keys()), comp_fields, "composite_loss")
        _check_unknown_keys(set(replay_dict.keys()), replay_fields, "replay_buffer")
        _check_unknown_keys(set(mom_dict.keys()), mom_fields, "moment_matching")
        _check_unknown_keys(set(d.keys()), train_fields, "config")

        return cls(
            optimizer=OptimizerConfig(**opt_dict),
            network=NetworkConfig(**net_dict),
            composite_loss=CompositeLossConfig(**comp_dict),
            replay_buffer=ReplayBufferConfig(**replay_dict),
            moment_matching=MomentMatchingConfig(**mom_dict),
            **{k: v for k, v in d.items() if k in train_fields},
        )

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        """Load config from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def with_overrides(self, overrides: Dict[str, Any]) -> "TrainConfig":
        """Return a new config with dot-notation overrides applied.

        Example:
            config.with_overrides({"optimizer.learning_rate": 0.01, "episodes": 500})
        """
        from deqn_jax.config.io import (
            _config_to_flat_dict,
            _flat_dict_to_config,
            _infer_type,
        )

        d = _config_to_flat_dict(self)
        for key, val in overrides.items():
            val = _infer_type(val)
            d[key] = val
        return _flat_dict_to_config(d)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested dictionary."""
        return self.model_dump()

    def to_yaml(self, path: str) -> None:
        """Write config to a YAML file."""
        import yaml

        d = self.to_dict()
        # Convert tuples to lists for YAML readability + safe_load compat:
        # PyYAML's default dumper writes tuples as `!!python/tuple` which
        # safe_load refuses; the round-trip path uses safe_load (correctly,
        # since trusting arbitrary-Python deserialization on a config is
        # bad). Coerce known-tuple fields to lists at write time.
        if "network" in d and "hidden_sizes" in d["network"]:
            d["network"]["hidden_sizes"] = list(d["network"]["hidden_sizes"])
        if "network" in d and d["network"].get("activations") is not None:
            d["network"]["activations"] = list(d["network"]["activations"])
        if "network" in d and d["network"].get("kf_names") is not None:
            d["network"]["kf_names"] = list(d["network"]["kf_names"])
        if "network" in d and d["network"].get("output_links") is not None:
            d["network"]["output_links"] = list(d["network"]["output_links"])

        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)
