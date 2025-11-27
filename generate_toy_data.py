#!/usr/bin/env python3
"""
Generate toy data files for RH Enforcer:

- primes_up_to_1e5_toy.txt   (real primes, small)
- riemann_zeros_1e3_toy.txt  (fake zeros, just n * pi)

These are ONLY for plumbing / demo. They are NOT real RH data.
"""

import numpy as np

def fake_primes_up_to(n: int):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    p = 2
    while p * p <= n:
        if sieve[p]:
            for k in range(p * p, n + 1, p):
                sieve[k] = False
        p += 1
    return [i for i, is_p in enumerate(sieve) if is_p]

def main():
    # 1. Real primes up to 1e5 (small but genuine)
    primes = fake_primes_up_to(100_000)
    with open("primes_up_to_1e5_toy.txt", "w") as f:
        for p in primes:
            f.write(f"{p}\n")
    print(f"Wrote {len(primes)} primes to primes_up_to_1e5_toy.txt")

    # 2. Fake zeros: gamma_n ≈ n * pi for n=1..1000 (purely synthetic)
    zeros = np.arange(1, 1001, dtype=float) * np.pi
    np.savetxt("riemann_zeros_1e3_toy.txt", zeros)
    print(f"Wrote {len(zeros)} fake zeros to riemann_zeros_1e3_toy.txt")

if __name__ == "__main__":
    main()
