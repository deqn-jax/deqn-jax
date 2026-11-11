"""Output-slot reparametrizations of the disaster policy network.

Each reparametrization repurposes one policy output slot of
``DisasterPolicyNet``: the network learns a better-conditioned quantity
there, and the policy value is recovered from it. Three exist, each
enabled by a ``NetworkConfig`` flag:

- ``InvestmentBracket`` (``reparam_q_as_m``, §3.3 of the shape-priors doc):
  the ``q`` slot carries ``M = q · 𝓑(x)``, where
  ``𝓑(x) = 1 − S(x) − x·S'(x)`` is the investment-Euler bracket and
  ``x = µ_z·i/i_lag``; ``q = M / 𝓑(x)`` is recovered before the clip.
- ``CalvoPrice`` (``reparam_pi_as_kp_inner``, §3.1): the ``pi`` slot carries
  ``K_p_inner``, bounded to the image of the model's ``pi`` bounds; ``pi``
  and ``K_p`` are recovered after the clip so eq 2a holds up to the
  soft-floor wedge ``definitions()`` applies to the inner term.
- ``CalvoWage`` (``reparam_wtilda_as_kw_inner``, §3.1'): the ``w_tilda``
  slot carries ``K_w_inner``; ``w_tilda`` and ``K_w`` are recovered after
  the clip (and after ``CalvoPrice``) so eq 4a holds likewise.

Each is a frozen dataclass holding the indices and calibration constants it
reads, every one declared once with the model name it resolves from
(``_policy`` / ``_state`` / ``_constant``); ``from_model`` resolves them all
through ``index_of`` / ``constants_of``. The objects are hashable, so the
network keeps them as static fields. Slots they repurpose must use the
linear output link: the recovery works in level space.

Experimental: no shipped config enables them.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp
from jax import Array


def index_of(names: list, name: str, where: str, flag: str) -> int:
    """Position of ``name`` in the model's ``where`` list; raises naming it."""
    if name not in names:
        raise ValueError(f"{flag}=True requires {name!r} in {where}; got {names}")
    return names.index(name)


def constants_of(model, flag: str, *names: str) -> list:
    """Calibration constants a prior reads; raises naming every missing one."""
    missing = [n for n in names if n not in model.constants]
    if missing:
        raise ValueError(f"{flag}=True requires model.constants {missing}")
    return [float(model.constants[n]) for n in names]


def _policy(name: str):
    return dataclasses.field(metadata={"policy_names": name})


def _state(name: str):
    return dataclasses.field(metadata={"state_names": name})


def _constant(name: str):
    return dataclasses.field(metadata={"constants": name})


def _source(f: dataclasses.Field) -> tuple:
    ((where, name),) = f.metadata.items()
    return where, name


class _Reparam:
    """Shared resolution and validation; subclasses are frozen dataclasses."""

    flag: ClassVar[str]

    @classmethod
    def from_model(cls, model):
        names = {
            "policy_names": list(model.policy_names),
            "state_names": list(model.state_names or ()),
        }
        values = {}
        constant_fields = []
        for f in dataclasses.fields(cls):  # pyright: ignore[reportArgumentType]
            where, name = _source(f)
            if where == "constants":
                constant_fields.append((f.name, name))
            else:
                values[f.name] = index_of(names[where], name, where, cls.flag)
        constants = constants_of(model, cls.flag, *(n for _, n in constant_fields))
        values.update(zip((f for f, _ in constant_fields), constants))
        return cls(**values)

    def check_indices(self, n_states: int, n_policies: int) -> None:
        bounds = {"policy_names": n_policies, "state_names": n_states}
        for f in dataclasses.fields(self):  # pyright: ignore[reportArgumentType]
            where, _ = _source(f)
            if where == "constants":
                continue
            idx, bound = getattr(self, f.name), bounds[where]
            if not (0 <= idx < bound):
                raise ValueError(
                    f"{self.flag}=True requires valid {f.name}, got {idx} "
                    f"(bound {bound})"
                )

    @property
    def slot(self) -> int:
        """The policy output slot this reparametrization repurposes."""
        raise NotImplementedError

    def override_bounds(
        self, lower: tuple | None, upper: tuple | None, ss_state, n_policies: int
    ) -> tuple:
        """Clip bounds with this slot's domain substituted (default: none)."""
        return lower, upper

    def bk_target(self, bk_level: Array, state: Array) -> Array:
        """The BK policy, in this slot's reparametrized space.

        ``bk_level`` is ``ss_policy + P @ (state - ss_state)`` before any
        slot is substituted.
        """
        raise NotImplementedError


def _with_slot_bounds(lower, upper, n_policies, slot, lo, hi) -> tuple:
    new_lower = list(lower) if lower is not None else [-float("inf")] * n_policies
    new_upper = list(upper) if upper is not None else [float("inf")] * n_policies
    new_lower[slot] = lo
    new_upper[slot] = hi
    return tuple(new_lower), tuple(new_upper)


@dataclass(frozen=True)
class InvestmentBracket(_Reparam):
    """``reparam_q_as_m``: the q slot carries ``M = q · 𝓑(x)``.

    Aimed at the sign-flip pathology in eq 7; 𝓑(x) itself, and so the sign
    of µ_Υ q 𝓑(x), is not constrained.
    """

    flag: ClassVar[str] = "reparam_q_as_m"
    q_idx: int = _policy("q")
    i_idx: int = _policy("i")
    i_lag_idx: int = _state("i_lag")
    mu_z_idx: int = _state("mu_z")
    kappa: float = _constant("kappa")
    mu_z_ss: float = _constant("mu_z_ss")

    def __post_init__(self):
        if self.kappa <= 0.0:
            raise ValueError(f"{self.flag}=True requires kappa > 0, got {self.kappa}")

    @property
    def slot(self) -> int:
        return self.q_idx

    def _bracket(self, x):
        S_val = 0.5 * self.kappa * (x - self.mu_z_ss) ** 2
        S_prime = self.kappa * (x - self.mu_z_ss)
        return 1.0 - S_val - x * S_prime

    def bk_target(self, bk_level: Array, state: Array) -> Array:
        x_BK = (
            state[self.mu_z_idx]
            * bk_level[self.i_idx]
            / (state[self.i_lag_idx] + 1e-12)
        )
        return bk_level[self.q_idx] * self._bracket(x_BK)

    def recover(self, raw: Array, state: Array) -> Array:
        """q_t = M_t / 𝓑(x_t), with x_t from the POST-delta i_t. 𝓑 is floored
        at 1e-3 to keep the division well-conditioned near the sign flip."""
        x_t = state[self.mu_z_idx] * raw[self.i_idx] / (state[self.i_lag_idx] + 1e-12)
        B_t = self._bracket(x_t)
        B_safe = jnp.where(B_t > 1e-3, B_t, jnp.asarray(1e-3, dtype=B_t.dtype))
        return raw.at[self.q_idx].set(raw[self.q_idx] / B_safe)


@dataclass(frozen=True)
class CalvoPrice(_Reparam):
    """``reparam_pi_as_kp_inner``: the pi slot carries ``K_p_inner``.

    Encodes the Calvo asymptote in the parameterization so the MLP only has
    to learn a smooth K_p_inner.
    """

    flag: ClassVar[str] = "reparam_pi_as_kp_inner"
    pi_idx: int = _policy("pi")
    F_p_idx: int = _policy("F_p")
    K_p_idx: int = _policy("K_p")
    pi_lag_idx: int = _state("pi_lag")
    xi_p: float = _constant("xi_p")
    lambda_f: float = _constant("lambda_f")
    iota: float = _constant("iota")
    pi_ss: float = _constant("pi_ss")

    def __post_init__(self):
        if not (0.0 < self.xi_p < 1.0):
            raise ValueError(f"{self.flag}=True requires 0 < xi_p < 1, got {self.xi_p}")
        if self.lambda_f <= 1.0:
            raise ValueError(
                f"{self.flag}=True requires lambda_f > 1, got {self.lambda_f}"
            )

    @property
    def slot(self) -> int:
        return self.pi_idx

    def _pi_tilda(self, pi_lag):
        return (self.pi_ss**self.iota) * (pi_lag ** (1.0 - self.iota))

    def _kp_inner(self, pi_tilda, pi):
        """Forward Calvo map: pi → K_p_inner at the given indexation."""
        return (1.0 - self.xi_p * (pi_tilda / pi) ** (1.0 / (1.0 - self.lambda_f))) / (
            1.0 - self.xi_p
        )

    def override_bounds(self, lower, upper, ss_state, n_policies):
        """K_p_inner bounds derived FROM THE ORIGINAL pi bounds at the SS
        indexation, not the full Calvo domain (0, 1/(1−ξ_p)): high π maps to
        low K_p_inner, and the full domain would let the optimizer reach π
        outside the model's calibrated range. The lower π bound carries the
        Calvo-edge asymptote."""
        pi_tilda_ss = self._pi_tilda(float(ss_state[self.pi_lag_idx]))
        orig_pi_lower = float(lower[self.pi_idx]) if lower is not None else 0.5
        orig_pi_upper = float(upper[self.pi_idx]) if upper is not None else float("inf")
        kp_inner_lo = (
            self._kp_inner(pi_tilda_ss, orig_pi_upper)
            if orig_pi_upper < float("inf")
            else 0.01
        )
        kp_inner_hi = self._kp_inner(pi_tilda_ss, orig_pi_lower)
        kp_inner_lo = max(kp_inner_lo, 0.01)
        kp_inner_hi = min(kp_inner_hi, 1.0 / (1.0 - self.xi_p) - 0.01)
        return _with_slot_bounds(
            lower, upper, n_policies, self.pi_idx, kp_inner_lo, kp_inner_hi
        )

    def bk_target(self, bk_level, state):
        """K_p_inner_BK by the EXACT Calvo map of pi_BK. ss_policy[pi] is not
        overridden: with P[pi] still the pi-space slope, that would bias the
        BK value off the SS."""
        return self._kp_inner(
            self._pi_tilda(state[self.pi_lag_idx]), bk_level[self.pi_idx]
        )

    def recover(self, raw, state):
        """Post-clip: pi_t by the inverse map, and K_p = F_p · K_p_inner^{1−λ_f}
        so eq 2a holds up to the soft-floor wedge of definitions(). The
        (1−(1−ξ_p)·K_p_inner)/ξ_p factor is positive on the clipped domain."""
        kp_inner_t = raw[self.pi_idx]
        inner_term = (1.0 - (1.0 - self.xi_p) * kp_inner_t) / self.xi_p
        pi_t = self._pi_tilda(state[self.pi_lag_idx]) * inner_term ** (
            self.lambda_f - 1.0
        )
        K_p_t = raw[self.F_p_idx] * kp_inner_t ** (1.0 - self.lambda_f)
        raw = raw.at[self.pi_idx].set(pi_t)
        return raw.at[self.K_p_idx].set(K_p_t)


@dataclass(frozen=True)
class CalvoWage(_Reparam):
    """``reparam_wtilda_as_kw_inner``: the w_tilda slot carries ``K_w_inner``.

    Mirror of ``CalvoPrice`` on the wage side. Reads the pi slot (pi_BK for
    the wage inflation at BK) and pi_lag (wage indexation) whether or not
    ``CalvoPrice`` is on.
    """

    flag: ClassVar[str] = "reparam_wtilda_as_kw_inner"
    w_tilda_idx: int = _policy("w_tilda")
    F_w_idx: int = _policy("F_w")
    K_w_idx: int = _policy("K_w")
    pi_idx: int = _policy("pi")
    w_tilda_lag_idx: int = _state("w_tilda_lag")
    pi_lag_idx: int = _state("pi_lag")
    mu_z_idx: int = _state("mu_z")
    xi_w: float = _constant("xi_w")
    lambda_w: float = _constant("lambda_w")
    sigma_L: float = _constant("sigma_L")
    iota_w: float = _constant("iota_w")
    iota_mu: float = _constant("iota_mu")
    psi_L: float = _constant("psi_L")
    pi_ss: float = _constant("pi_ss")
    mu_z_ss: float = _constant("mu_z_ss")

    def __post_init__(self):
        if not (0.0 < self.xi_w < 1.0):
            raise ValueError(f"{self.flag}=True requires 0 < xi_w < 1, got {self.xi_w}")
        if self.lambda_w <= 1.0:
            raise ValueError(
                f"{self.flag}=True requires lambda_w > 1, got {self.lambda_w}"
            )

    @property
    def slot(self) -> int:
        return self.w_tilda_idx

    def _pi_w_tilda(self, pi_lag):
        return (self.pi_ss**self.iota_w) * (pi_lag ** (1.0 - self.iota_w))

    def override_bounds(self, lower, upper, ss_state, n_policies):
        """The K_w_inner domain (0.01, 1/(1−ξ_w) − 0.01): the original
        w_tilda bounds (lower ≈ 1.0, upper = ∞) have no meaningful image —
        they would confine wages to the deflationary side."""
        return _with_slot_bounds(
            lower,
            upper,
            n_policies,
            self.w_tilda_idx,
            0.01,
            1.0 / (1.0 - self.xi_w) - 0.01,
        )

    def _mu_w_tilda(self, mu_z):
        return (mu_z**self.iota_mu) * (self.mu_z_ss ** (1.0 - self.iota_mu))

    def bk_target(self, bk_level, state):
        mu_z = state[self.mu_z_idx]
        pi_w_BK = (
            bk_level[self.pi_idx]
            * mu_z
            * bk_level[self.w_tilda_idx]
            / (state[self.w_tilda_lag_idx] + 1e-12)
        )
        index_t = self._pi_w_tilda(state[self.pi_lag_idx]) * self._mu_w_tilda(mu_z)
        aw_exp = 1.0 / (1.0 - self.lambda_w)
        return (1.0 - self.xi_w * (index_t / pi_w_BK) ** aw_exp) / (1.0 - self.xi_w)

    def recover(self, raw, state):
        """Post-clip, after ``CalvoPrice.recover``: invert the Calvo formula
        for pi_w_t, so the K_w_inner that definitions() recomputes from
        (the wage index over pi_w) equals K_w_inner_t identically; derive
        w_tilda_t from pi_w_t = pi_t·µ_z,t·w_tilda_t/w_tilda_lag; then set
        K_w = (1/ψ_L)·K_w_inner^{1−λ_w(1+σ_L)}·w_tilda·F_w so eq 4a holds up to
        the soft-floor wedge. (Inverting eq 4a directly for w_tilda is wrong
        off the SS: definitions() would then recompute a different K_w_inner.)
        raw[pi] is the recovered pi_t when CalvoPrice is on, else the clipped
        network output."""
        w_tilda_lag = state[self.w_tilda_lag_idx]
        kw_inner_t = raw[self.w_tilda_idx]
        inner_term_w = (1.0 - (1.0 - self.xi_w) * kw_inner_t) / self.xi_w
        mu_z = state[self.mu_z_idx]
        pi_w_t = (
            self._pi_w_tilda(state[self.pi_lag_idx])
            * self._mu_w_tilda(mu_z)
            * inner_term_w ** (self.lambda_w - 1.0)
        )
        w_tilda_t = pi_w_t * w_tilda_lag / (raw[self.pi_idx] * mu_z + 1e-12)
        kw_exponent = 1.0 - self.lambda_w * (1.0 + self.sigma_L)
        K_w_t = (
            (1.0 / self.psi_L)
            * (kw_inner_t**kw_exponent)
            * w_tilda_t
            * raw[self.F_w_idx]
        )
        raw = raw.at[self.w_tilda_idx].set(w_tilda_t)
        return raw.at[self.K_w_idx].set(K_w_t)
