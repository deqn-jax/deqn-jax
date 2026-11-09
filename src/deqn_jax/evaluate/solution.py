"""Is a trained policy a solution of its model? One fixed set of gates.

Equation errors are taken at the states of a long simulation, with next
period's expectation computed by quadrature
(``euler_equation_errors(conditional_expectation=True)``), so an exact
solution scores rounding error only. For residuals in ratio form (the
``residuals_from_lagrangian`` default) an error of 1e-3 is 0.1% of marginal
utility. The thresholds are fixed so that whoever trains the model cannot
loosen them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from deqn_jax.evaluate.diagnostics import euler_equation_errors

MEAN_ERROR_MAX = 1e-3  # mean |error| of each equation on the simulated path
WORST_ERROR_MAX = 1e-2  # 99th percentile of |error|
OUTSIDE_SHARE_MAX = 0.01  # share of simulated periods outside the training region
EXACT_GAP_MAX = 1e-2  # largest relative gap to a known exact policy
N_PERIODS = 10_000
SEED = 123


@dataclass
class Gate:
    name: str
    passed: Optional[bool]  # None: reported, not gated
    value: str
    meaning: str


@dataclass
class SolutionCheck:
    solved: bool
    gates: List[Gate]
    equation_names: List[str]
    state_names: List[str]
    policy_names: List[str]
    errors: np.ndarray = field(repr=False)  # |residual|, [T, n_equations]
    states: np.ndarray = field(repr=False)  # [T, n_states]
    policies: np.ndarray = field(
        repr=False
    )  # network policy on the path, [T, n_policies]
    exact: Optional[np.ndarray] = field(default=None, repr=False)
    region: Optional[tuple] = field(default=None, repr=False)  # (low, high) per state

    def summary(self) -> dict:
        return {"solved": self.solved, "gates": [asdict(g) for g in self.gates]}


def _pct(x: float) -> str:
    return f"{100 * x:.3g}%"


def training_region(model, n: int = 20_000):
    """Per-state (low, high) of the model's training draws, or None."""
    if model.init_state_fn is None:
        return None
    draws = np.asarray(model.init_state_fn(jax.random.PRNGKey(0), n, model.constants))
    return draws.min(axis=0), draws.max(axis=0)


def check_solution(
    policy_net: Callable,
    model,
    exact_policy: Optional[Callable] = None,
    n_quadrature_points: Optional[int] = None,
) -> SolutionCheck:
    """Run the gates on ``policy_net`` (batched state -> policy)."""
    diag = euler_equation_errors(
        policy_net,
        model,
        n_periods=N_PERIODS,
        seed=SEED,
        n_quadrature_points=n_quadrature_points,
        conditional_expectation=True,
    )
    errors = np.abs(np.asarray(diag["residuals"], dtype=float))
    states = np.asarray(diag["states"], dtype=float)
    policies = np.asarray(policy_net(jnp.asarray(states)), dtype=float)
    names = list(diag["equation_names"])
    finite = bool(np.isfinite(errors).all() and np.isfinite(states).all())
    gates = [
        Gate(
            "Simulation stays finite",
            finite,
            "yes" if finite else "no",
            f"{N_PERIODS:,} simulated periods produce no NaN or infinite values.",
        )
    ]
    region, exact = training_region(model), None
    if finite:
        mean, worst = errors.mean(axis=0), np.quantile(errors, 0.99, axis=0)
        for i, eq in enumerate(names):
            gates.append(
                Gate(
                    f"Average error, {eq}",
                    bool(mean[i] <= MEAN_ERROR_MAX),
                    f"{mean[i]:.2e}",
                    f"Mean |error|; at most {MEAN_ERROR_MAX:g}.",
                )
            )
            gates.append(
                Gate(
                    f"Worst 1% error, {eq}",
                    bool(worst[i] <= WORST_ERROR_MAX),
                    f"{worst[i]:.2e}",
                    f"99th percentile of |error|; at most {WORST_ERROR_MAX:g}.",
                )
            )
        if region is not None:
            low, high = region
            tol = 1e-6 * np.maximum(high - low, 1e-12)
            share = float(
                ((states < low - tol) | (states > high + tol)).any(axis=1).mean()
            )
            gates.append(
                Gate(
                    "Simulation stays where the network was trained",
                    share <= OUTSIDE_SHARE_MAX,
                    _pct(share),
                    f"Share of periods with a state outside the training draws; at most "
                    f"{_pct(OUTSIDE_SHARE_MAX)}. Outside, the network extrapolates.",
                )
            )
        if exact_policy is not None:
            exact = np.asarray(
                exact_policy(jnp.asarray(states), model.constants), dtype=float
            )
            gap = float(
                (np.abs(policies - exact) / np.maximum(np.abs(exact), 1e-8)).max()
            )
            gates.append(
                Gate(
                    "Matches the exact solution",
                    gap <= EXACT_GAP_MAX,
                    _pct(gap),
                    f"Largest relative gap to the known solution on the path; at most "
                    f"{_pct(EXACT_GAP_MAX)}.",
                )
            )
        if model.steady_state_fn is not None:
            ss_state, ss_policy = model.steady_state_fn(model.constants)
            at_ss = np.asarray(policy_net(jnp.asarray(ss_state)[None, :]), dtype=float)[
                0
            ]
            ss_policy = np.asarray(ss_policy, dtype=float)
            gap = float(
                np.max(np.abs(at_ss - ss_policy) / np.maximum(np.abs(ss_policy), 1e-8))
            )
            gates.append(
                Gate(
                    "Policy at the deterministic steady state",
                    None,
                    _pct(gap),
                    "Gap between the network at the steady state and the steady-state policy; "
                    "with shocks they may differ a little (precaution), so not gated.",
                )
            )
    return SolutionCheck(
        solved=all(g.passed for g in gates if g.passed is not None),
        gates=gates,
        equation_names=names,
        state_names=list(model.state_names),
        policy_names=list(model.policy_names),
        errors=errors,
        states=states,
        policies=policies,
        exact=exact,
        region=region,
    )


def print_solution_check(check: SolutionCheck) -> None:
    print(f"\n{'SOLVED' if check.solved else 'NOT SOLVED'}\n")
    for g in check.gates:
        mark = {True: "pass", False: "FAIL", None: "info"}[g.passed]
        print(f"  {mark:4}  {g.name}: {g.value}")
