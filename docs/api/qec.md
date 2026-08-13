# QEC (erasure-aware stabilizer decoding)

Generic, code-agnostic stabilizer-code utilities: Pauli-string commutation
(`pauli_commutes`), syndrome computation from any list of stabilizer
generators (`compute_syndrome`), and an erasure-aware decoder
(`erasure_aware_decode`) that exploits known error *locations* (e.g. a
heralded lost photon in a dual-rail photonic qubit) rather than only the
syndrome.

Erasure-aware decoding rests on a real, foundational result: Grassl, Beth
& Pellizzari, "Codes for the quantum erasure channel," Phys. Rev. A 56, 33
(1997) — a distance-*d* stabilizer code can correct up to *d*-1 erasures
(known-location errors), versus only floor((*d*-1)/2) arbitrary
(unlocated) errors. `erasure_aware_decode` doesn't hard-code that bound;
it emerges from the brute-force search itself (more heralded qubits than
the code can resolve typically yields zero or multiple syndrome-matching
assignments, so the function returns `None` — never a guess).

::: dense_evolution.qec

---

**See also**: promoted from Dense-Evolution-Discovery's Steane [[7,1,3]]
code investigation
([Block 6](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/steane_qec/) —
heralded-erasure conversion), where a Steane-specific version of this
decoder was first built and validated against STIM's native
`HERALDED_ERASE` noise channel: 0 decoding failures across every
double-erasure shot tested (>60,000 shots total, 40,000 trials × 10
physical error rates), versus a real ~25% failure rate for a standard
syndrome-only decoder blind to the erasure locations. Also grounded in
Gu, Vaknin, Retzker & Kubica, "Optimizing quantum error correction
protocols with erasure qubits," PRX Quantum 6, 040354 (2025),
arXiv:2408.00829.
