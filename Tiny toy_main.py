#!/usr/bin/env python3
# Toy main that uses the small toy data files.

from rh_enforcer_engine import RHEngine

def main():
    zeros_path = "riemann_zeros_1e3_toy.txt"
    primes_path = "primes_up_to_1e5_toy.txt"

    print(f"Using toy data:\n  zeros  = {zeros_path}\n  primes = {primes_path}\n")

    eng = RHEngine.from_files(
        zeros_path=zeros_path,
        primes_path=primes_path,
        num_x=200,   # smaller grid for speed
        x_min=1e3,
    )

    print("\n=== RH ENFORCER — TOY RUN ===")
    print(f"Toy zeros: {eng.N_ZEROS}")
    print(f"Toy max prime: {eng.X_MAX:,.0f}")
    print(f"KL(spacing) ≈ {eng.KL_spacing_bits:.6f} bits")
    print(f"E_RH ≈ {eng.E_RH:.3e}")
    print(f"λ ≈ {eng.lam:.3e}\n")

    k_values = [1, 2, 5]
    beta_values = [0.6, 0.7, 0.8]

    print("log10 BF (toy data) vs k, β")
    print("     β \\ k →    1         2         5")
    print("-" * 60)

    for beta in beta_values:
        row = f"{beta:4.2f} | "
        for k in k_values:
            bf = eng.log10_bayes_factor_against(k_spurious=k, beta=beta)
            row += f"{bf:9.2f}  "
        print(row)

    print("\nNote: this is using synthetic zeros — just a plumbing demo.\n")

if __name__ == "__main__":
    main()
