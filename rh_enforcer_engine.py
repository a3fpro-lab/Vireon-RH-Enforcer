#!/usr/bin/env python3
# =============================================================================
# RH ENFORCER ENGINE — Core Bayesian Evidence Logic (VIREON)
# =============================================================================
# Provides a reusable engine for:
#   - Loading zero / prime data
#   - Computing GUE spacing KL divergence
#   - Building ψ(x) via the explicit formula
#   - Defining an energy functional E[ψ]
#   - Calibrating λ to match the spacing KL in nats
#   - Evaluating log10 Bayes factors against off-line-zero scenarios
#
# Usage (example):
#
#   from rh_enforcer_engine import RHEngine
#
#   eng = RHEngine.from_files(
#       zeros_path="riemann_zeros_1e8.txt",
#       primes_path="primes_up_to_1e8.txt",
#       num_x=800,
#       x_min=1e3
#   )
#
#   bf = eng.log10_bayes_factor_against(k_spurious=1, beta=0.75)
#   print("log10 BF =", bf)
#
# Notes:
#   - In practice, START with small data (e.g. 1e4–1e5 zeros, primes ≤ 1e6).
#   - This is an EVIDENCE MODEL, not a formal proof of RH.
# =============================================================================

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from scipy.stats import entropy
from scipy.integrate import simpson


def wigner_surmise(s_vals: np.ndarray) -> np.ndarray:
    """
    GUE / β=2 Wigner surmise PDF on spacings.
    """
    s_vals = np.asarray(s_vals, dtype=np.float64)
    return (32.0 / np.pi**2) * s_vals**2 * np.exp(-4.0 * s_vals**2 / np.pi)


def compute_spacing_kl(
    primes: np.ndarray,
    bins: int = 400,
    s_max: float = 6.0,
) -> Tuple[float, float]:
    """
    Compute KL(p_emp || p_GUE) for unfolded prime gaps.

    Parameters
    ----------
    primes : np.ndarray
        Sorted array of primes (int64 or float).
    bins : int
        Number of histogram bins on [0, s_max].
    s_max : float
        Maximum spacing for histogram range [0, s_max].

    Returns
    -------
    KL_nats : float
        KL divergence in natural units (nats).
    KL_bits : float
        KL divergence in bits.
    """
    primes = np.asarray(primes, dtype=np.float64)
    gaps = np.diff(primes)
    mean_log = np.log(primes[:-1])
    s = gaps / mean_log  # unfolded spacings

    hist, edges = np.histogram(s, bins=bins, range=(0.0, s_max), density=True)
    s_centers = 0.5 * (edges[:-1] + edges[1:])

    p_emp = hist + 1e-18
    p_gue = wigner_surmise(s_centers)
    p_gue /= simpson(p_gue, s_centers)

    KL_nats = entropy(p_emp, p_gue)
    KL_bits = KL_nats / np.log(2.0)
    return KL_nats, KL_bits


def psi_explicit(
    xs: np.ndarray,
    gammas: np.ndarray,
    betas: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    ψ(x) ≈ x - ∑_{ρ} x^ρ/ρ (real part, doubled for conjugate pairs).

    Parameters
    ----------
    xs : np.ndarray
        Array of x values (real-positive).
    gammas : np.ndarray
        Imaginary parts γ_n of the zeros.
    betas : np.ndarray or None
        Real parts β_n; if None, all β = 1/2 (RH).

    Returns
    -------
    psi_vals : np.ndarray
        Approximate ψ(x) values (real).
    """
    xs_c = np.asarray(xs, dtype=np.complex128)
    gammas = np.asarray(gammas, dtype=np.float64)

    if betas is None:
        betas = np.full_like(gammas, 0.5, dtype=np.float64)
    else:
        betas = np.asarray(betas, dtype=np.float64)
        if betas.shape != gammas.shape:
            raise ValueError("betas and gammas must have the same shape")

    total = xs_c.copy()
    for beta, gamma in zip(betas, gammas):
        rho = beta + 1j * gamma
        total -= xs_c**rho / rho

    return 2.0 * total.real  # factor 2 for ±γ


def energy_functional(
    psi_vals: np.ndarray,
    xs: np.ndarray,
    log_xs: np.ndarray,
) -> float:
    """
    Energy-like functional measuring deviation of ψ(x) from x, rescaled and
    integrated over log x:

        E = ∫ ((ψ(x) - x)/sqrt(x))^2 * (1/x) d(log x)

    Parameters
    ----------
    psi_vals : np.ndarray
        ψ(x) values on xs.
    xs : np.ndarray
        x grid.
    log_xs : np.ndarray
        log(xs), used as integration variable.

    Returns
    -------
    E : float
        Energy value.
    """
    xs = np.asarray(xs, dtype=np.float64)
    psi_vals = np.asarray(psi_vals, dtype=np.float64)

    error = (psi_vals - xs) / np.sqrt(xs)
    weight = 1.0 / xs
    integrand = error**2 * weight

    return simpson(integrand, log_xs)


@dataclass
class RHEngine:
    """
    Core RH Enforcer engine: holds data, calibration, and evidence routines.
    """

    zeros: np.ndarray          # γ_n
    primes: np.ndarray         # p_n
    xs: np.ndarray             # x-grid
    log_xs: np.ndarray         # log(x-grid)
    KL_spacing_nats: float     # KL(p_emp || p_GUE) in nats
    KL_spacing_bits: float     # KL in bits
    E_RH: float                # baseline energy under RH
    lam: float                 # calibration factor mapping E → nats

    @classmethod
    def from_files(
        cls,
        zeros_path: str,
        primes_path: str,
        num_x: int = 800,
        x_min: float = 1e3,
    ) -> "RHEngine":
        """
        Construct an engine from disk data.

        Parameters
        ----------
        zeros_path : str
            Path to text file with imaginary parts γ_n, one per line.
        primes_path : str
            Path to text file with primes, one per line.
        num_x : int
            Number of x-grid points (log-spaced).
        x_min : float
            Minimum x on the grid.

        Returns
        -------
        RHEngine
        """
        zeros = np.loadtxt(zeros_path)
        primes = np.loadtxt(primes_path, dtype=np.int64)

        X_MAX = float(primes[-1])

        # 1. Spacing KL
        KL_nats, KL_bits = compute_spacing_kl(primes)

        # 2. x-grid + ψ(x) under RH (all β=1/2)
        xs = np.logspace(np.log10(x_min), np.log10(X_MAX), num_x)
        log_xs = np.log(xs)

        psi_rh = psi_explicit(xs, zeros)  # all β=1/2

        # 3. Baseline energy
        E_RH = energy_functional(psi_rh, xs, log_xs)

        # 4. Calibrate λ so that λ * E_RH ≈ KL_nats
        lam = KL_nats / max(E_RH, 1e-30)

        return cls(
            zeros=zeros,
            primes=primes,
            xs=xs,
            log_xs=log_xs,
            KL_spacing_nats=KL_nats,
            KL_spacing_bits=KL_bits,
            E_RH=E_RH,
            lam=lam,
        )

    @property
    def X_MAX(self) -> float:
        return float(self.primes[-1])

    @property
    def N_ZEROS(self) -> int:
        return int(self.zeros.shape[0])

    def misfit_increase(
        self,
        betas_fake: Sequence[float],
        gammas_fake: Optional[Sequence[float]] = None,
        height_strategy: str = "lowest",
    ) -> float:
        """
        Add k spurious zeros (possibly off the line) and return ΔM = M_fake - M_RH
        in nats, where M ≈ λ * E + KL_spacing_nats (KL term is fixed here).

        Parameters
        ----------
        betas_fake : sequence of float
            Real parts β of k counterfeit zeros.
        gammas_fake : sequence of float or None
            Imaginary parts γ for counterfeit zeros; if None and
            height_strategy='lowest', reuse the first k heights from self.zeros.
        height_strategy : str
            Currently only 'lowest' is implemented.

        Returns
        -------
        delta_M_nats : float
            Increase in misfit (nats).
        """
        betas_fake = np.asarray(betas_fake, dtype=np.float64)
        k = betas_fake.shape[0]

        if gammas_fake is None:
            if height_strategy != "lowest":
                raise NotImplementedError("Only 'lowest' height_strategy implemented when gammas_fake is None.")
            gammas_fake = self.zeros[:k]
        gammas_fake = np.asarray(gammas_fake, dtype=np.float64)

        # Full set of β, γ including RH zeros + fake ones
        betas_full = np.concatenate(
            [np.full(self.N_ZEROS, 0.5, dtype=np.float64), betas_fake]
        )
        gammas_full = np.concatenate(
            [self.zeros, gammas_fake]
        )

        psi_fake = psi_explicit(self.xs, gammas_full, betas_full)
        E_fake = energy_functional(psi_fake, self.xs, self.log_xs)
        delta_E = E_fake - self.E_RH

        # Misfit penalty in nats
        return self.lam * delta_E

    def log10_bayes_factor_against(
        self,
        k_spurious: int = 1,
        beta: float = 0.75,
        height_strategy: str = "lowest",
    ) -> float:
        """
        log10 Bayes factor in favor of RH against exactly k_spurious zeros
        at real part beta, placed according to height_strategy.

        Parameters
        ----------
        k_spurious : int
            Number of counterfeit zeros.
        beta : float
            Real part of each counterfeit zero.
        height_strategy : str
            Placement rule for imaginary parts. Only 'lowest' is implemented.

        Returns
        -------
        log10_BF : float
            log10 Bayes factor in favor of RH (data vs this off-line-zero scenario),
            under this model.
        """
        betas_fake = [beta] * k_spurious
        delta_M_nats = self.misfit_increase(
            betas_fake=betas_fake,
            gammas_fake=None,
            height_strategy=height_strategy,
        )
        # nats → log10: divide by ln(10)
        return delta_M_nats / np.log(10.0)

    # Convenience wrapper
    def rh_evidence(self, beta: float, k: int = 1) -> float:
        """
        Convenience alias for log10_bayes_factor_against.

        Parameters
        ----------
        beta : float
            Real part of zeros considered off-line.
        k : int
            Number of such zeros.

        Returns
        -------
        log10_BF : float
        """
        return self.log10_bayes_factor_against(k_spurious=k, beta=beta)


# Optional: simple self-test path when run directly
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python rh_enforcer_engine.py <zeros_path> <primes_path>")
        sys.exit(1)

    zeros_path = sys.argv[1]
    primes_path = sys.argv[2]

    print("Initializing RH engine from files...")
    eng = RHEngine.from_files(
        zeros_path=zeros_path,
        primes_path=primes_path,
        num_x=400,   # modest for quick sanity run
        x_min=1e3,
    )

    print("\n=== RH ENGINE SUMMARY ===")
    print(f"Zeros: {eng.N_ZEROS:,}")
    print(f"Max prime: {eng.X_MAX:,.0f}")
    print(f"KL(spacing) = {eng.KL_spacing_bits:.6f} bits")
    print(f"E_RH = {eng.E_RH:.3e}")
    print(f"λ = {eng.lam:.3e}")
    print("=========================\n")

    for beta in [0.60, 0.70, 0.80]:
        bf = eng.log10_bayes_factor_against(k_spurious=1, beta=beta)
        print(f"beta = {beta:.2f}, k=1 → log10 BF ≈ {bf:.2f}")
