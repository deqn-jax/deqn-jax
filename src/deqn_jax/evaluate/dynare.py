"""Dynare cross-checks: moments / ghx / IRFs comparison + reporting."""

from typing import Any, Dict, List

import equinox as eqx
import jax
import jax.numpy as jnp

from deqn_jax.evaluate.diagnostics import (
    simulated_moments,
)

# ---------------------------------------------------------------------------
# Comparison to Dynare reference (perturbation solution)
# ---------------------------------------------------------------------------


def compare_to_dynare_moments(
    policy_net: eqx.Module,
    model,
    dynare_dir: str,
    n_periods: int = 10_000,
    seed: int = 123,
) -> Dict[str, Any]:
    """Diff the network's ergodic policy moments against Dynare's.

    Loads ``dynare_moments.csv`` and runs a long simulation under the
    network. For each DEQN policy whose name (after aliasing) appears in
    the Dynare output, reports both means and stds and their relative
    differences. The headline number is ``median_abs_std_diff_pct`` —
    "are policies moving the right amount" is the cleanest single check
    for non-degeneracy near the ergodic distribution.
    """
    from deqn_jax.dynare_io import deqn_policy_to_dynare, load_dynare_moments

    dyn_moments = load_dynare_moments(dynare_dir)
    net_moments = simulated_moments(policy_net, model, n_periods=n_periods, seed=seed)

    per_var: Dict[str, Dict[str, Any]] = {}
    mean_abs_diffs: List[float] = []
    std_abs_diffs: List[float] = []

    for pname in model.policy_names:
        dvar = deqn_policy_to_dynare(pname)
        if dvar not in dyn_moments or pname not in net_moments:
            continue
        d = dyn_moments[dvar]
        n = net_moments[pname]
        # Relative diff in pct, guarded against division by zero (use abs of
        # the dynare value as the scale; if it's tiny, fall back to absolute).
        d_mean = float(d["mean"])
        d_std = float(d["std"])
        n_mean = float(n["mean"])
        n_std = float(n["std"])
        scale_mean = max(abs(d_mean), 1e-6)
        scale_std = max(abs(d_std), 1e-6)
        mean_diff_pct = (n_mean - d_mean) / scale_mean * 100.0
        std_diff_pct = (n_std - d_std) / scale_std * 100.0
        per_var[pname] = {
            "dynare_var": dvar,
            "dynare_mean": d_mean,
            "net_mean": n_mean,
            "mean_diff_pct": mean_diff_pct,
            "dynare_std": d_std,
            "net_std": n_std,
            "std_diff_pct": std_diff_pct,
        }
        mean_abs_diffs.append(abs(mean_diff_pct))
        std_abs_diffs.append(abs(std_diff_pct))

    if not per_var:
        return {"per_var": {}, "n_compared": 0}

    sorted_means = sorted(mean_abs_diffs)
    sorted_stds = sorted(std_abs_diffs)
    n = len(sorted_means)
    return {
        "per_var": per_var,
        "n_compared": n,
        "median_abs_mean_diff_pct": sorted_means[n // 2],
        "median_abs_std_diff_pct": sorted_stds[n // 2],
        "max_abs_mean_diff_pct": sorted_means[-1],
        "max_abs_std_diff_pct": sorted_stds[-1],
    }


def compare_to_dynare_ghx(
    policy_net: eqx.Module,
    model,
    dynare_dir: str,
    perturb_sigma: float = 1.0e-3,
    n_perturbs: int = 4,
    seed: int = 0,
) -> Dict[str, Any]:
    """Diff the network's policy Jacobian at SS against Dynare's perturbation.

    Computes ``J_net`` by averaging ``jacrev(policy_net)`` at small
    perturbations of SS, then loads Dynare's perturbation matrix via
    ``dynare_io.load_dynare_jacobian``. Outputs Frobenius-norm difference
    plus per-policy row L2 and max-abs deviation. Sharp local-correctness
    test: if the policy gradient at SS doesn't match the linearization,
    the network's behaviour for small shocks is also off — even if
    ergodic means agree by coincidence.

    Why perturb instead of evaluating at exact SS:
        Policies that bound outputs via sigmoid / softplus / soft-floor
        regularizers can sit *exactly* at a saturating boundary at SS
        for some entries. The forward is finite there but the gradient
        can hit a derivative discontinuity (e.g. soft_floor's softplus
        having near-zero derivative just below the floor). For a
        well-posed network the behaviour is continuous off-SS, so
        averaging jacrev at ``SS + ε·N(0, I)`` over a few seeds gives
        a numerically clean estimate of the local linearization without
        moving meaningfully off the steady state. ``perturb_sigma=1e-3``
        is small enough that nonlinearity is negligible (linear policy
        recovers exactly within float roundoff in tests).

    Args:
        perturb_sigma: Std of the Gaussian perturbation around SS. Set
            to 0 to evaluate at exact SS (legacy behaviour).
        n_perturbs: Number of perturbed evaluations to average. Ignored
            when ``perturb_sigma == 0``.
        seed: PRNG seed for the perturbations (deterministic for
            reproducibility across eval runs).
    """
    from deqn_jax.dynare_io import load_dynare_jacobian

    assert model.steady_state_fn is not None
    ss_state, _ = model.steady_state_fn(model.constants)

    # Wrap the policy so jacrev sees a 1D-state → 1D-policy function.
    def _policy_at_state(s):
        out = policy_net(s)  # pyright: ignore[reportCallIssue]  # ty: ignore[call-non-callable]
        if out.ndim == 2:
            out = out[0]
        return out

    if perturb_sigma > 0 and n_perturbs > 0:
        keys = jax.random.split(jax.random.PRNGKey(seed), n_perturbs)
        Js = jnp.stack(
            [
                jax.jacrev(_policy_at_state)(
                    ss_state + perturb_sigma * jax.random.normal(k, ss_state.shape)
                )
                for k in keys
            ]
        )
        J_net = jnp.mean(Js, axis=0)
    else:
        J_net = jax.jacrev(_policy_at_state)(ss_state)
    J_dyn = load_dynare_jacobian(model, dynare_dir)

    # Soft-floor / sigmoid-bound code paths in the policy can produce NaN
    # or Inf gradients at exact-SS inputs (zero argument to log, derivative
    # of softplus at large negative input, etc). Detect this so the report
    # can flag it rather than silently propagating NaN to the diff norms.
    n_bad = int(jnp.sum(~jnp.isfinite(J_net)))
    has_nan = n_bad > 0

    diff = J_net - J_dyn
    fro = float(jnp.linalg.norm(diff))
    fro_dyn = float(jnp.linalg.norm(J_dyn))
    per_policy: Dict[str, Dict[str, float]] = {}
    for i, pname in enumerate(model.policy_names):
        row = diff[i]
        per_policy[pname] = {
            "l2": float(jnp.linalg.norm(row)),
            "max_abs": float(jnp.max(jnp.abs(row))),
            "dynare_row_l2": float(jnp.linalg.norm(J_dyn[i])),
            "net_row_l2": float(jnp.linalg.norm(J_net[i])),
            "row_nonfinite": int(jnp.sum(~jnp.isfinite(J_net[i]))),
        }
    return {
        "frobenius": fro,
        "frobenius_dynare": fro_dyn,
        "frobenius_relative": fro / max(fro_dyn, 1e-12),
        "per_policy": per_policy,
        "j_net_nonfinite_entries": n_bad,
        "j_net_has_nonfinite": has_nan,
    }


def compare_to_dynare_irfs(
    policy_net: eqx.Module,
    model,
    dynare_dir: str,
    horizon: int = 40,
    use_girf: bool = False,
) -> Dict[str, Any]:
    """Diff network IRFs against Dynare's per-shock IRFs.

    For each shock with a matching ``irf_e_<shock>.csv`` in ``dynare_dir``,
    runs our ``run_irf`` (or ``run_girf`` if ``use_girf=True``) and computes
    per-variable max-abs and L2 deviation over the horizon. Variables that
    appear in both Dynare's CSV and our IRF output (= states ∪ policies ∪
    definitions) are compared; the rest are skipped.
    """
    from deqn_jax.dynare_io import deqn_policy_to_dynare, load_dynare_irf
    from deqn_jax.irf import run_girf, run_irf

    if not model.shock_names:
        return {"per_shock": {}, "shocks_skipped": []}

    # Reverse alias: Dynare var name → DEQN name (so we can look up our IRF
    # output dict by Dynare's column header).
    dynare_to_deqn: Dict[str, str] = {}
    for pname in model.policy_names:
        dynare_to_deqn[deqn_policy_to_dynare(pname)] = pname
    # States, definitions, equations all keep their DEQN names in run_irf
    # output, so identity is fine for non-policy variables.

    per_shock: Dict[str, Dict] = {}
    shocks_skipped: List[str] = []

    for shock in model.shock_names:
        try:
            dyn = load_dynare_irf(dynare_dir, shock)
        except FileNotFoundError:
            shocks_skipped.append(shock)
            continue
        if use_girf:
            net = run_girf(policy_net, model, shock_name=shock, horizon=horizon)
        else:
            net = run_irf(policy_net, model, shock_name=shock, horizon=horizon)

        per_var: Dict[str, Dict[str, Any]] = {}
        max_abs_overall = 0.0
        for dvar, dyn_series in dyn.items():
            deqn_name = dynare_to_deqn.get(dvar, dvar)
            if deqn_name not in net:
                continue
            net_series = net[deqn_name]
            H = min(len(dyn_series), len(net_series))
            if H == 0:
                continue
            d_arr = jnp.asarray(dyn_series[:H])
            n_arr = jnp.asarray(net_series[:H])
            diff = n_arr - d_arr
            # Network IRFs can NaN out for unstable policies; scrub for the
            # diff norms but report the nonfinite count so the user knows.
            n_bad = int(jnp.sum(~jnp.isfinite(diff)))
            diff_finite = jnp.where(jnp.isfinite(diff), diff, 0.0)
            n_finite = jnp.where(jnp.isfinite(n_arr), n_arr, 0.0)
            ma = float(jnp.max(jnp.abs(diff_finite)))
            l2 = float(jnp.linalg.norm(diff_finite))
            per_var[dvar] = {
                "deqn_name": deqn_name,
                "max_abs_diff": ma,
                "l2_diff": l2,
                "dynare_max_abs": float(jnp.max(jnp.abs(d_arr))),
                "net_max_abs": float(jnp.max(jnp.abs(n_finite))),
                "horizon": H,
                "nonfinite_count": n_bad,
            }
            max_abs_overall = max(max_abs_overall, ma)
        per_shock[shock] = {
            "per_var": per_var,
            "max_abs_overall": max_abs_overall,
            "n_vars": len(per_var),
        }
    return {"per_shock": per_shock, "shocks_skipped": shocks_skipped}


def print_dynare_comparison(
    moments_diff: Dict[str, Any],
    ghx_diff: Dict[str, Any],
    irf_diff: Dict[str, Any],
    label: str = "",
) -> None:
    """Pretty-print all three Dynare-comparison diffs."""
    print(f"\n{'=' * 76}")
    print(f"Dynare comparison{' — ' + label if label else ''}")
    print(f"{'=' * 76}")

    # Moments
    n_compared = moments_diff.get("n_compared", 0)
    print(f"\n[moments]  {n_compared} overlapping policies")
    if n_compared:
        med_mean = moments_diff["median_abs_mean_diff_pct"]
        med_std = moments_diff["median_abs_std_diff_pct"]
        print(f"  median |Δmean|={med_mean:.2f}%   median |Δstd|={med_std:.2f}%")
        per_var = moments_diff["per_var"]
        print(
            f"  {'policy':>10s}  {'dyn_mean':>10s}  {'net_mean':>10s}  "
            f"{'Δmean%':>8s}  {'dyn_std':>10s}  {'net_std':>10s}  {'Δstd%':>8s}"
        )
        for pname, row in per_var.items():
            print(
                f"  {pname:>10s}  {row['dynare_mean']:>10.4g}  "
                f"{row['net_mean']:>10.4g}  {row['mean_diff_pct']:>+8.2f}  "
                f"{row['dynare_std']:>10.4g}  {row['net_std']:>10.4g}  "
                f"{row['std_diff_pct']:>+8.2f}"
            )

    # ghx
    fro_rel = ghx_diff["frobenius_relative"]
    fro = ghx_diff["frobenius"]
    n_bad = ghx_diff.get("j_net_nonfinite_entries", 0)
    print(
        f"\n[linearization]  ||J_net - J_dyn||_F = {fro:.4f}  "
        f"(relative to ||J_dyn||_F: {fro_rel * 100:.1f}%)"
    )
    if n_bad:
        print(
            f"  WARNING: jacrev(policy)(SS) produced {n_bad} non-finite "
            "entries — the policy has a discontinuous gradient at SS "
            "(likely a soft-floor / sigmoid-bound boundary). Frobenius is "
            "NaN; per-row counts below show which policies."
        )
    print(
        f"  {'policy':>10s}  {'row_L2':>10s}  {'row_max':>10s}  "
        f"{'dyn_norm':>10s}  {'nonfinite':>9s}"
    )
    for pname, pp in ghx_diff["per_policy"].items():
        print(
            f"  {pname:>10s}  {pp['l2']:>10.4f}  {pp['max_abs']:>10.4f}  "
            f"{pp['dynare_row_l2']:>10.4f}  {pp.get('row_nonfinite', 0):>9d}"
        )

    # IRFs
    per_shock = irf_diff.get("per_shock", {})
    skipped = irf_diff.get("shocks_skipped", [])
    print(f"\n[IRFs]  {len(per_shock)} shocks compared, {len(skipped)} skipped")
    if skipped:
        print(f"  skipped (no Dynare CSV): {', '.join(skipped)}")
    for shock, payload in per_shock.items():
        print(
            f"  shock={shock:>6s}  vars={payload['n_vars']:>3d}  "
            f"max_abs_diff={payload['max_abs_overall']:.4e}"
        )
