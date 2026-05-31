"""Diagnostic 1D plots of the troublesome equation slices.

Generates four PNGs in this directory:
  - calvo_bracket.png:       K_p_inner and K_p_inner^{1-λ_f} vs π/π̃
  - bgg_contract.png:        h(ω̄) participation function and Γ'-µG' denominator
  - investment_bracket.png:  1 - S(x) - x·S'(x) vs investment growth
  - elb_floor.png:           softplus ELB R(R_taylor)

These are 1D slices of multivariate equation residuals, intended to
illustrate where shape priors would help. See
``docs/dev/disaster_equation_shape_priors.md`` for the analysis.

Constants are pulled from src/deqn_jax/models/disaster/variables.py
defaults. Tune by editing the globals at the top.

Run from the repo root:
    uv run python docs/dev/figures/plot_diagnostic_curves.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Calibration globals (mirror disaster/variables.py)
XI_P, XI_W = 0.6, 0.6
LAMBDA_F, LAMBDA_W = 1.2, 1.2
SIGMA_L = 1.0
SIGMA_OMEGA = 0.26822
MU_MON = 0.22
KAPPA = 2.0
MU_Z_SS = 1.0041
R_LB = 1.0
R_LB_SHARPNESS = 500.0

OUT_DIR = Path(__file__).parent


def plot_calvo_bracket() -> None:
    """K_p_inner and its (1-λ_f) power vs π/π̃."""
    exp1 = 1.0 / (1.0 - LAMBDA_F)  # -5 for λ_f=1.2
    exp2 = 1.0 - LAMBDA_F  # -0.2 for λ_f=1.2
    edge = (1.0 / XI_P) ** (LAMBDA_F - 1)  # validity edge in π/π̃ direction

    r_grid = np.linspace(1.001, edge - 0.001, 400)
    inner = (1 - XI_P * (1.0 / r_grid) ** exp1) / (1 - XI_P)
    power = inner**exp2

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(r_grid, inner, lw=2, color="C0")
    axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].axvline(
        edge, color="red", ls="--", alpha=0.5, label=f"Calvo edge: π/π̃ = {edge:.3f}"
    )
    axes[0].set_xlabel("π/π̃")
    axes[0].set_ylabel("K_p_inner")
    axes[0].set_title(
        "K_p_inner(π/π̃) = (1 − ξ_p (π̃/π)^{1/(1−λ_f)})/(1−ξ_p)\nDrops to 0 at edge"
    )
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(r_grid, power, lw=2, color="C1")
    axes[1].axvline(edge, color="red", ls="--", alpha=0.5, label="Calvo edge")
    axes[1].set_xlabel("π/π̃")
    axes[1].set_ylabel("K_p_inner^{1−λ_f}  (=^{−0.2})")
    axes[1].set_title(
        "K_p_inner^{1−λ_f} → ∞ at edge\n(vertical asymptote in K_p definition)"
    )
    axes[1].set_yscale("log")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "calvo_bracket.png", dpi=120)
    plt.close()


def plot_bgg_contract() -> None:
    """Bank participation h(ω̄) and the eq 8 denominator Γ'(ω̄) − µG'(ω̄)."""
    omega = np.linspace(0.05, 2.5, 600)
    z_F = (np.log(omega) + 0.5 * SIGMA_OMEGA**2) / SIGMA_OMEGA
    z_G = (np.log(omega) - 0.5 * SIGMA_OMEGA**2) / SIGMA_OMEGA
    F = norm.cdf(z_F)
    G = norm.cdf(z_G)
    G_prime = norm.pdf(z_G) / (SIGMA_OMEGA * omega)
    Gamma = omega * (1 - F) + G
    h = Gamma - MU_MON * G
    contract_denom = (1 - F) - MU_MON * G_prime

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    peak = omega[np.argmax(h)]
    axes[0].plot(omega, h, lw=2, label="h(ω̄) = Γ − µG")
    axes[0].axvline(0.488, color="gray", ls=":", alpha=0.7, label="ω̄_ss = 0.488")
    axes[0].axvline(peak, color="red", ls="--", alpha=0.5, label=f"h peak ≈ {peak:.2f}")
    axes[0].set_xlabel("ω̄")
    axes[0].set_ylabel("h(ω̄)")
    axes[0].set_title("Bank participation function\n(only locally monotone)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(omega, contract_denom, lw=2, color="C2")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].axvline(0.488, color="gray", ls=":", alpha=0.7, label="ω̄_ss")
    axes[1].axvline(peak, color="red", ls="--", alpha=0.5, label="h peak ≈ singularity")
    axes[1].set_xlabel("ω̄")
    axes[1].set_ylabel("Γ'(ω̄) − µ G'(ω̄)")
    axes[1].set_title("Eq 8 denominator → 0 at h peak\n(endogenous singularity)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bgg_contract.png", dpi=120)
    plt.close()


def plot_investment_bracket() -> None:
    """Investment Euler bracket 1 − S(x) − x·S'(x)."""
    x_grid = np.linspace(0.6, 1.6, 400)
    S_val = 0.5 * KAPPA * (x_grid - MU_Z_SS) ** 2
    S_prime = KAPPA * (x_grid - MU_Z_SS)
    bracket = 1 - S_val - x_grid * S_prime

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_grid, bracket, lw=2)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(MU_Z_SS, color="gray", ls=":", alpha=0.7, label=f"µ_z_ss = {MU_Z_SS}")
    neg = bracket < 0
    if neg.any():
        ax.fill_between(
            x_grid,
            bracket,
            0,
            where=neg,
            alpha=0.2,
            color="red",
            label="bracket < 0 (sign flip)",
        )
    ax.set_xlabel("x = µ_z·i/i_lag")
    ax.set_ylabel("1 − S(x) − x·S'(x)")
    ax.set_title(
        "Investment Euler bracket\n(quadratic-in-deviation; sign flip far from SS)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "investment_bracket.png", dpi=120)
    plt.close()


def plot_elb_floor() -> None:
    """Softplus ELB R(R_taylor)."""
    R_taylor = np.linspace(0.985, 1.05, 600)
    R = R_LB + np.log(1 + np.exp(R_LB_SHARPNESS * (R_taylor - R_LB))) / R_LB_SHARPNESS

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(R_taylor, R, lw=2, label=f"Softplus floor (ξ={R_LB_SHARPNESS:.0f})")
    ax.plot(
        R_taylor, np.maximum(R_taylor, R_LB), "k--", alpha=0.5, label="Hard max (limit)"
    )
    ax.axvline(R_LB, color="red", ls=":", alpha=0.5, label=f"R_lb = {R_LB}")
    ax.set_xlabel("R_taylor")
    ax.set_ylabel("R")
    ax.set_title(
        f"ELB softplus\n(distortion at kink: {np.log(2) / R_LB_SHARPNESS:.4f} ≈ {np.log(2) / R_LB_SHARPNESS * 4 * 100:.1f} bp annualized)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "elb_floor.png", dpi=120)
    plt.close()


def main() -> None:
    plot_calvo_bracket()
    plot_bgg_contract()
    plot_investment_bracket()
    plot_elb_floor()
    print(f"Plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
