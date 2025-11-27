#!/usr/bin/env python3
# =============================================================================
# RH ENFORCER v1.0 — Main Script
# Uses RHEngine to load data, calibrate, and print Bayes-factor table.
# =============================================================================

import sys
from rh_enforcer_engine import RHEngine


def main():
    if len(sys.argv) < 3:
        zeros_path = "riemann_zeros_1e8.txt"
        primes_path = "primes_up_to_1e8.txt"
        print(f"Using default paths:\n  zeros  = {zeros_path}\n  primes = {primes_path}\n")
    else:
        zeros_path = sys.argv[1]
        primes_path = sys.argv[2]

    print("Initializing RH Enforcer engine...")
    eng = RHEngine.from_files(
        zeros_path=zeros_path,
        primes_path=primes_path,
        num_x=800,   # can lower for speed (e.g. 200–400)
        x_min=1e3,
    )

    print("\n" + "=" * 80)
    print("           RIEMANN HYPOTHESIS BAYESIAN ENFORCER — NOVEMBER 2025")
    print("=" * 80)
    print(f"Data: {eng.N_ZEROS:,} zeros, primes ≤ {eng.X_MAX:,.0f}")
    print(f"KL(spacing) = {eng.KL_spacing_bits:.6f} bits")
    print(f"λ × E_RH term = {eng.lam * eng.E_RH:.6f} nats (calibrated to match KL)\n")

    print("log10(Bayes factor in favor of RH) against k counterfeit zeros at Re(ρ)=β:")
    print("     β \\ k-spurious →   1         2         5        10        50       100")
    print("-" * 80)

    k_values = [1, 2, 5, 10, 50, 100]
    beta_values = [0.51, 0.55, 0.60, 0.70, 0.80, 0.90, 0.99]

    for beta in beta_values:
        row = f"{beta:4.2f} | "
        for k in k_values:
            bf = eng.log10_bayes_factor_against(k_spurious=k, beta=beta)
            if bf > 9999:
                row += " >1e4    "
            else:
                row += f"{bf:8.1f}  "
        print(row)

    print("-" * 80)
    print("\nInterpretation (model-dependent):")
    print("• Under this Bayesian model, these data give enormous support to RH")
    print("  versus the specific off-line-zero scenarios in the table.")
    print("• This is evidence modeling, *not* a formal proof of RH.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
