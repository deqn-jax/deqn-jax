# Monolith Decomposition Plan — trainer.py / config.py / evaluate.py

Goal: no source file over ~600 lines. Behavior-preserving. Sequenced so the
parts that could perturb training are done first under **exact-loss-curve-equality**
guarding (you've repeatedly hit chaotic bifurcation on numerically-irrelevant
changes — this plan makes that detectable, not silent).

## Outcome at a glance (before → after line counts)

| File | Now | After | Becomes |
|------|-----|-------|---------|
| `training/trainer.py` | **1790** | ~450 | slim orchestrator: `train`, `train_from_config`, `_run_training_loop` |
| `config.py` | **1663** | — | becomes a package `config/` (largest module ~430) |
| `evaluate.py` | **1022** | — | becomes a package `evaluate/` (largest module ~350) |

No file above ~600 lines when done. Largest single new file: `config/train.py` (~430).

---

## 1. trainer.py (1790 → ~450) — extract by lifecycle phase

The 23 symbols cluster into five lifecycle groups. Four move out:

### → `training/state_init.py` (NEW, ~560) — "build everything before the loop"
| Symbol | Lines | Span |
|--------|-------|------|
| `create_train_state` | 90 | 276 |
| `make_train_step` | 366 | 140 |
| `_build_initial_state` | 613 | 141 |
| `_resolve_model_for_training` | 562 | 51 |
| `_validate_train_config` | 506 | 56 |

`create_train_state` (276 lines, the biggest function in the codebase) **sheds its
disaster/kf net-type branches** (trainer-02) into the factory below — drops to ~210.

### → `composite_loss.py` (existing, +129)
| `_build_custom_loss_fn` | 754 | 129 |

It builds the composite/custom loss object — it belongs beside `make_composite_loss`,
not in the trainer.

### → `networks/factory.py` (NEW, ~120) — the decoupling fix (trainer-02)
The two disaster-specific `net_type` branches currently inline in `create_train_state`
(lines 213-270: `disaster_policy_net`, `kf_anchored_mlp` + literal K/F defaults) move
behind a net-type registry: `build_policy_net(config, model) -> eqx.Module` that
dispatches `mlp/linear_plus_mlp/lstm/transformer` generically and looks up
model-supplied factories for anything else. After this, `create_train_state` contains
zero disaster knowledge.

### → `training/loop_control.py` (NEW, ~270) — per-episode runtime controllers
| Symbol | Lines | Span |
|--------|-------|------|
| `_OptimizerRuntime` / `_NanRollback` / `_SaveBestTracker` | 883/900/918 | ~40 |
| `_maybe_switch_optimizer` | 924 | 76 |
| `_episode_lr_scale` / `_episode_shock_scale` | 1000/1025 | 41 |
| `_maybe_update_target` | 1041 | 16 |
| `_handle_nan` / `_check_early_stop` | 1057/1098 | 73 |

### → `training/reporting.py` (existing 150, +151 → ~300)
| `_log_episode` | 1130 | 113 |
| `_print_episode_progress` | 1243 | 38 |

These call `print_header`/`print_residual_table` already living in reporting.py — they're the orchestration layer over those primitives.

### → `training/checkpointing.py` (existing 97, +98 → ~195)
| `_maybe_checkpoint` / `_maybe_save_best` / `_final_save_best_fallback` | 1281/1301/1322 | 98 |

Orchestration over the `save_checkpoint`/`resume_from` primitives already there.

**trainer.py keeps:** `train` (60), `train_from_config` (194), `_run_training_loop`
(158) + imports ≈ **450 lines** — a readable orchestrator that reads top-to-bottom
as "validate → build state → loop → save."

---

## 2. config.py (1663) → `config/` package

Pydantic configs split one-class-per-module cleanly. Backward compat preserved by
re-exporting everything from `config/__init__.py` (every `from deqn_jax.config import
TrainConfig` keeps working — there are ~40 such imports across the tree).

```
config/
  __init__.py     # re-export TrainConfig, *Config, load_config  (compat shim)
  _base.py    ~135  _ConfigBase, _coerce_*, _reraise_validation_error, _pydantic_type_to_name
  optimizer.py ~110  OptimizerConfig
  loss.py     ~135  CompositeLossConfig, MomentMatchingConfig
  replay.py    ~90  ReplayBufferConfig
  network.py  ~230  NetworkConfig
  train.py    ~430  TrainConfig (class only)
  io.py       ~180  load_config, from_dict, _config_to_flat_dict, _flat_dict_to_config,
                    _check_unknown_keys, _infer_type
```

Largest module ~430 (TrainConfig). Pydantic forward-refs resolve via `__init__`
import order (base → leaf configs → train → io).

**Folded-in fix (config-02):** today, adding a nested config block means editing
~7-8 hand-synced sites (`from_dict`, `_config_to_flat_dict`, `_flat_dict_to_config`,
`to_yaml`, `load_config`). In `io.py`, derive the nested-config set once:
`{name for name,f in TrainConfig.model_fields.items() if isinstance(f.annotation, type) and issubclass(f.annotation, _ConfigBase)}` and loop over it. Adding a config block
then touches one place (declare the field on TrainConfig).

---

## 3. evaluate.py (1022) → `evaluate/` package

```
evaluate/
  __init__.py        # re-export public diagnostics + print fns + run_evaluate_cli
  simulate.py  ~150  _model_uses_discrete_chain, _draw_eval_shock,
                     + NEW eval_simulation_step  (the eval-01/02 fix, see Phase 3)
  diagnostics.py ~350 euler_equation_errors, market_clearing_errors,
                      simulated_moments, stability_check
  dynare.py    ~250  compare_to_dynare_moments / _ghx / _irfs
  report.py    ~180  print_euler_errors, print_moments, print_dynare_comparison
  cli.py       ~116  run_evaluate_cli
```

The four diagnostics currently each hand-roll their own JIT'd step closure
(`_sim_step_no_d`, `_sim_step_with_d`, `_sim_step_discrete`, two more `_sim_step`s).
The split *enables* collapsing those into one `eval_simulation_step` in `simulate.py`
that routes through `training/shocks.py:simulation_step` — but that's a behavior
change (Phase 3), not part of the move.

---

## Phasing — risk-ordered, each phase independently shippable

### Phase 0 — unblock (5 min, do first)
Commit the untracked `models/disaster/network.py` atomically with the config flips
(audit trainer-01). The trainer.py refactor touches `create_train_state`, which
imports it — can't safely refactor around a file that isn't tracked.

### Phase 1 — PURE MOVES (zero behavior change)
All extractions above are cut → paste → re-export. **No logic edits, no reordering
of numerical ops inside any function.** config/ and evaluate/ package conversions
included.

Guardrail (this is the important part for you):
1. **Capture a reference curve before touching anything:**
   `uv run deqn-jax train brock_mirman -n 100 --seed 0` → save loss array.
2. After each extraction, re-run the same command and assert the loss curve is
   **bit-identical**. Pure moves don't change the execution order of numerical ops,
   so equality must be *exact* — any drift means a move silently became a behavior
   change, and you catch it at the commit that introduced it.
3. Full suite stays 490 pass / 1 skip.
4. `git diff -M` should render most hunks as pure relocations.

### Phase 2 — DECOUPLING (behavior-preserving, touches dispatch)
- `networks/factory.py` net-type registry (removes disaster branch from create_train_state).
- `config/io.py` programmatic nested-config derivation (config-02).
Same seeded bit-identical-curve check + a disaster smoke run (`train disaster -n 20`)
since dispatch paths moved.

### Phase 3 — THE FIXES THAT CHANGE NUMERICS (acknowledge, don't curve-match)
- `eval_simulation_step` shared step (eval-01/02): `simulated_moments` and
  `stability_check` **start visiting disaster states** — that's the bug fix, behavior
  *should* change. Verified by new tests asserting they now include the disaster
  branch, not by curve-identity.

This ordering means: everything that could perturb *training* is Phase 1–2 and is
locked to exact-curve-equality; the only intended behavior change (Phase 3) is in
*evaluation diagnostics*, isolated and explicitly tested.

---

## What this does NOT touch
- Disaster model internals (out of scope).
- Numerical algorithms in loss/optimizers/networks (no formula changes anywhere).
- Public API (`from deqn_jax.config import ...`, `from deqn_jax.evaluate import ...`,
  `from deqn_jax.training.trainer import train` all unchanged via re-export shims).

## Effort estimate
- Phase 0: 5 min. Phase 1: ~half day (mechanical, the curve-check is the slow part).
  Phase 2: ~half day. Phase 3: ~half day (real tests). Total ~1.5 days, fully
  reversible per-phase.
