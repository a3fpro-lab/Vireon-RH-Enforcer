# VIREON RH ENFORCER v1.0 — Bayesian Penalty Against Counterexamples

This repo contains **RH ENFORCER v1.0**, a VIREON-style Bayesian evidence engine
that compares real prime / zeta-zero data under two models:

- **RH-true model:** all non-trivial zeros lie on Re(s) = 1/2.
- **RH-false model:** RH plus `k` “counterfeit” zeros with real part β ≠ 1/2.

It builds an energy functional from the explicit formula for ψ(x) and
calibrates it against the **GUE spacing KL divergence** of prime gaps to form
a misfit score. The combined evidence is summarized as:

\[
\log_{10} \mathrm{BF}
= \log_{10} \frac{P(\text{data} \mid \text{RH true})}
                   {P(\text{data} \mid \text{RH false, k spurious zeros})}.
\]

Large positive values mean the observed data strongly favors RH over the
specified off-line-zero scenario **under this model**.

> ⚠️ This is a **Bayesian evidence model**, not a formal proof of RH.

---

## Status and honesty disclaimer

This code does **not** prove the Riemann Hypothesis.

It implements a particular **likelihood model**:

- A specific explicit-formula energy functional for ψ(x).
- A mapping of that energy to an effective KL-style penalty.
- Real primes and zeros treated as high-dimensional “data”.
- Counterfactual scenarios with `k` zeros at Re(ρ) = β ≠ 1/2.

Statements like “10¹⁰⁰ evidence” are **conditional** on those modeling
choices and priors. They are not a substitute for a rigorous proof.

---

## Files

Core:

- `rh_enforcer_engine.py` — reusable **engine** (`RHEngine`) with:
  - spacing KL computation
  - explicit ψ(x) builder
  - energy functional `E[ψ]`
  - λ calibration
  - `log10_bayes_factor_against(...)`
- `main.py` — CLI front-end that:
  - loads data from disk
  - prints KL / λ / energy summary
  - prints a Bayes-factor table vs. `k` and β
- `toy_demo.py` — plumbing sanity check with **synthetic primes and zeros**.
- `requirements.txt` — dependencies (`numpy`, `scipy`).

Data (not included):

- `riemann_zeros_1e8.txt` — imaginary parts γₙ of non-trivial zeros.
- `primes_up_to_1e8.txt` — primes ≤ 10⁸.

You can start with **truncated** files (e.g. first 10⁴–10⁵ zeros, primes ≤ 10⁶).

---

## Installation

Create / activate a Python 3.10+ environment, then:

```bash
pip install -r requirements.txt
