# Disaster training: zombie paths, honest metrics, and the spectral lottery

**Date:** 2026-06-10. Companion artifacts:
`scripts/disaster_anchor_diagnostic.py`, `scripts/aio_bias_floor_diagnostic.py`,
fix commit `b22259e`, checkpoints `disaster_fresh_dgx/` (pre-fix run) and
`disaster_fixed_reset/` (post-fix run).

## TL;DR

1. The years-long "disaster training stalls at ~1e-1" was a **measurement
   artifact**: a fixed-prefix `ss_reset` left 55/64 training paths
   permanently absorbed at the soft-clip ceiling (k=100), so the training
   loss measured zombies while the actual policy converged to 3e-5 on its
   true ergodic set — better than the linearized policy. Every
   "functionally useless" checkpoint was a `checkpoint_best` artifact
   selected by the zombie-dominated metric. Fixed in `b22259e`.
2. With the metric honest, the **real disease** is visible: trained
   policies are a **stability lottery**. Nothing in the loss controls the
   closed-loop spectral radius ρ of `s ↦ step(s, π_net(s), 0)` at SS.
   Across runs, ρ lands either side of 1; ρ>1 policies drift to the state
   ceiling over a few hundred steps and their ergodic set is the clip
   boundary. This is the missing Blanchard-Kahn selection, measured.

## Evidence chain

- **Pre-fix run (DGX, canonical config):** final training loss 9.8e-2;
  but GH3 loss at ep3000 on the policy's own ergodic cloud = 4.2e-5
  (base 3.0e-5 vs linearized 3.3e-5). Unclipped simulation stable.
  GH3 loss on the *actual serialized `episode_state` buffer* = 1.011e-1,
  matching the training log per-equation to 3 digits — the buffer was
  the corruption. 55/64 paths pinned at exactly the soft-clip bounds
  (k=100, q=5, i=5) with zero recovery over 400 steps (absorbing).
- **Cause:** `ep_states.at[:n_reset].set(fresh)` — `ss_reset_frac=0.15`
  reset the same indices 0–8 every episode; indices 9–63 were immortal.
  (Introduced in `0955ca5`, survived the framework audit because the
  bit-identical refactor guards faithfully preserved it, and smoke tests
  run 3 episodes — zombies need hundreds.)
- **Post-fix run (same config, new key stream):** training loss again
  ~1e-1 — but now honestly: this run's policy itself drifts to the
  ceiling from SS within ~2000 steps (20-step episodes look fine,
  max k=35.6 — the drift is slow). The buffer is ceiling-heavy because
  the *policy* is unstable, not because paths are immortal.
- **Spectral diagnostic** (ρ of closed-loop Jacobian at SS, zero shock):

  | checkpoint | ρ | behavior |
  |---|---|---|
  | pre-fix final | 0.987 | stable, loss 3e-5 |
  | pre-fix best (ep1449) | 1.025 | mid-transient junk |
  | post-fix final | 1.067 | slow drift to ceiling |
  | post-fix best (ep2119) | 1.079 | worse |

  0.987 is the BK eigenvalue (present in all four — the linearized
  closed loop is stable by construction). The aux_jac anchor
  (`dπ/ds(SS) = P`, weight 0.1) failed to hold the slope in the
  unstable run: final jac loss 4.4e-3 vs 7.1e-5 in the stable run.

## Interpretation

The DEQN loss penalizes equilibrium residuals at *visited* states; local
closed-loop stability at SS is a property of the policy's *slope*, which
the residual loss constrains only weakly (multiple fixed points / BK
multiplicity: an explosive-root solution family also has small residuals
along its own trajectories until it hits the clip). Which basin a run
lands in depends on the training realization. The clip ceiling converts
"explosive" into "absorbing far region", so losing the lottery looks
like a plateau, not a NaN.

## Candidate cures (in increasing order of principle)

1. **`composite_loss.jac_weight: 1.0`** (config-only) — hold the SS
   slope to P harder. Probe run: `checkpoints/disaster_jacw1`.
2. **Tangent-pinned delta** (architectural, extends DisasterPolicyNet in
   the spirit of the K/F gauge mask): output
   `δ(s) − J_δ(ss)·(s − ss)` so `dπ/ds(SS) = P` *exactly*, making local
   BK stability structural rather than penalized. `J_δ(ss)` is one
   jacfwd per parameter update, not per batch element.
3. **Spectral penalty** — differentiable power iteration on the
   closed-loop Jacobian at SS (or at anchor points), penalize
   `relu(ρ − ρ_max)²`. Controls stability beyond the SS tangent;
   composable with 2 for off-SS eigenvalue drift.

Also worth doing regardless: **log ρ(SS) during training** (host-side at
`log_every`; it's one jacfwd + eigvals) so stability is observable live,
and add ρ to `disaster_anchor_diagnostic.py` output.

## Status / next steps

- [x] Zombie fix shipped (`b22259e`), regression-tested.
- [x] jac_weight=1.0 probe — **cure works on this draw**
      (`checkpoints/disaster_jacw1`): training loss 1.45e-5 with best at
      ep2823 (late, ≈ final), ρ(SS) = 0.986993 for BOTH final and best —
      the BK eigenvalue to six digits, i.e. the tangent anchor at weight
      1.0 makes the closed-loop spectrum the Blanchard-Kahn one. Zero
      buffer zombies, ergodic k_max 30.9, GH3 ergodic loss 2.0e-5,
      training metric ≈ ergodic truth. n=1: census still required before
      changing the canonical config default.
- [ ] Multi-seed stability census: fraction of runs with ρ<1 under each
      cure (the lottery framing demands seeds, not single runs).
- [ ] Re-measure the accuracy gap vs the TF reference using *final* (or
      post-cure best) checkpoints — all prior comparisons used
      zombie-selected artifacts.
- [ ] p>0 / ZLB variants: re-run estimator + stability diagnostics there.
