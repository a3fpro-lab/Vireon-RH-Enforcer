#!/usr/bin/env python3
# =============================================================================
# TOY DEMO — RH ENFORCER ENGINE
# Uses fake primes + fake zeros to sanity-check the pipeline.
# =============================================================================

import numpy as np
from rh_enforcer_engine import RHEngine


def fake_primes_up_to(n: int) -> np.ndarray:
    """Very small sieve for demo only."""
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(n**0.5) + 1):
        if sieve[p]:
            sieve[p*p:n+1:p] = False
    return np.nonzero(sieve)[0].astype(np.int64)


def main():
    # 1. Fake primes up to ~1e5
    primes = fake_primes_up_to(100_000)

    # 2. Fake zeros: just γ_n ≈ n * π for n=1..1000 (completely synthetic)
    zeros = np.arange(1, 1001, dtype=float) * np.pi

    # 3. Build a small x-grid
    X_MAX = float(primes[-1])
    xs = np.logspace(3, np.log10(X_MAX), 200)
    log_xs = np.log(xs)

    # 4. Manually build the engine (bypassing from_files)
    from rh_enforcer_engine import (
        compute_spacing_kl,
        psi_explicit,
        energy_functional,
    )

    KL_nats, KL_bits = compute_spacing_kl(primes)
    psi_rh = psi_explicit(xs, zeros)
    E_RH = energy_functional(psi_rh, xs, log_xs)
    lam = KL_nats / max(E_RH, 1e-30)

    eng = RHEngine(
        zeros=zeros,
        primes=primes,
        xs=xs,
        log_xs=log_xs,
        KL_spacing_nats=KL_nats,
        KL_spacing_bits=KL_bits,
        E_RH=E_RH,
        lam=lam,
    )

    print("=== TOY DEMO (FAKE DATA) ===")
    print(f"Toy zeros: {eng.N_ZEROS}")
    print(f"Toy primes max: {eng.X_MAX:,.0f}")
    print(f"KL(spacing) ≈ {eng.KL_spacing_bits:.4f} bits")
    print(f"E_RH ≈ {eng.E_RH:.3e}")
    print(f"λ ≈ {eng.lam:.3e}")
    print()

    for beta in [0.6, 0.7, 0.8]:
        bf = eng.log10_bayes_factor_against(k_spurious=1, beta=beta)
        print(f"beta = {beta:.2f}, k=1 → log10 BF ≈ {bf:.2f} (toy data)")

    print("\nThis is *only* a plumbing sanity check with synthetic data.")


if __name__ == "__main__":
    main()
