# QEC (stabilizer decoding: erasure-aware, MWPM, and blind brute-force)

Generic, code-agnostic stabilizer-code utilities: Pauli-string commutation
(`pauli_commutes`), syndrome computation from any list of stabilizer
generators (`compute_syndrome`), and three decoders covering three
different real-world settings:

- `erasure_aware_decode` — exploits known error *locations* (e.g. a
  heralded lost photon in a dual-rail photonic qubit) rather than only
  the syndrome.
- `pymatching_decode` — blind (no known locations) minimum-weight-perfect-matching
  decoding via the [`pymatching`](https://pypi.org/project/PyMatching/)
  package (optional dependency: `pip install dense-evolution[pymatching]`).
  Only works for "graph-like" codes where every qubit is checked by at
  most 2 stabilizers — true for the surface code and other topological
  codes, **not** true for every stabilizer code (Steane [[7,1,3]]'s
  weight-4 stabilizers check some qubits 3 times; `pymatching_decode`
  raises a clear `ValueError` rather than a confusing library traceback).
- `blind_minimum_weight_decode` — blind decoding for codes `pymatching_decode`
  structurally can't handle, like Steane. Pure Python, no new dependency:
  brute-forces every possible error in increasing weight order and
  returns the minimum-weight match. The obvious shortcut — calling
  `erasure_aware_decode` with every qubit passed as "heralded" — does
  **not** work (verified directly): with no qubit assumed error-free,
  many stabilizer-equivalent errors share the same syndrome, so
  `erasure_aware_decode`'s "exactly one match" criterion is essentially
  always violated. Minimum-weight selection is what makes blind decoding
  well-posed at all.

Erasure-aware decoding rests on a real, foundational result: Grassl, Beth
& Pellizzari, "Codes for the quantum erasure channel," Phys. Rev. A 56, 33
(1997) — a distance-*d* stabilizer code can correct up to *d*-1 erasures
(known-location errors), versus only floor((*d*-1)/2) arbitrary
(unlocated) errors. `erasure_aware_decode` doesn't hard-code that bound;
it emerges from the brute-force search itself (more heralded qubits than
the code can resolve typically yields zero or multiple syndrome-matching
assignments, so the function returns `None` — never a guess). The same
"return `None`, never guess" discipline applies to `pymatching_decode`
and `blind_minimum_weight_decode` whenever a syndrome is ambiguous.

::: dense_evolution.physics.qec

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
