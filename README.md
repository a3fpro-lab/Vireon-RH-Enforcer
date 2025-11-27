# Vireon-RH-Enforcer
Bayesian RH Enforcer: compares prime/zero data under RH vs spurious zeros and reports log10 Bayes # VIREON RH ENFORCER v1.0 — Bayesian Penalty Against Counterexamples

This repo contains **RH ENFORCER v1.0**, a Bayesian-style evidence engine that
compares real prime / zeta-zero data under two models:

- **RH-true model:** all non-trivial zeros lie on Re(s) = 1/2.
- **RH-false model:** RH plus `k` “counterfeit” zeros with real part β ≠ 1/2.

It computes an **energy functional** from the explicit formula for ψ(x) and
calibrates it against the **GUE spacing KL divergence** of prime gaps to build
a combined misfit score. The result is reported as:

\[
\log_{10} \mathrm{BF} = \log_{10} \frac{P(\text{data} \mid \text{RH true})}
                                  {P(\text{data} \mid \text{RH false, k spurious zeros}}).
\]

Large positive values mean the observed data strongly favors RH over the
specified off-line-zero scenario **under this model**.

---

## Status and honesty disclaimer

This code is **not** a formal mathematical proof of the Riemann Hypothesis.
It is a **Bayesian evidence model**:

- It assumes a particular explicit-formula energy functional.
- It assumes a specific way to map that functional to a likelihood ratio.
- It uses real primes and zeros as high-dimensional “data”.

Any claim like “probability 1 - 10⁻¹⁰⁰” is **conditional** on those modeling
choices and priors. It does **not** replace a proof.

---

## Data requirements

The script expects two large data files in the working directory:

- `riemann_zeros_1e8.txt`  
  Imaginary parts γₙ of the first 100,000,000 non-trivial zeros.
- `primes_up_to_1e8.txt`  
  All primes ≤ 10⁸.

(You can start by testing on **much smaller** data: e.g. the first 10⁵ zeros
and primes ≤ 10⁶, by truncating these files.)

---

## Quick start

1. **Clone the repo (or create it on GitHub and upload these files).**
2. Put `rh_enforcer.py` in the root directory.
3. Place the prime / zero data files next to it.
4. Install dependencies (Python 3.10+):

   ```bash
   pip install -r requirements.txt

   
## Programmatic use

You can also use the engine directly:

```python
from rh_enforcer_engine import RHEngine

eng = RHEngine.from_files(
    zeros_path="riemann_zeros_1e8.txt",
    primes_path="primes_up_to_1e8.txt",
    num_x=800,
    x_min=1e3,
)

print("log10 BF vs one zero at β=0.75:", eng.rh_evidence(beta=0.75, k=1))

