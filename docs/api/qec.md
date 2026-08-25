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

`decode_with_erasure_fallback` composes the two-decoder split above into the real-world
decoding POLICY, not just another raw decoder: use `erasure_aware_decode` when there are
heralded qubits and it resolves the syndrome uniquely, otherwise fall back to
`blind_minimum_weight_decode`. Never worse than always calling blind decoding directly --
promoted from Dense-Evolution-Discovery's
[cosmic-ray-burst-as-erasure experiment](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/cosmic_ray_burst_validation/),
where this exact fallback logic was first written inline in a Monte Carlo loop testing
whether knowing WHICH qubits a real cosmic-ray burst hit (arXiv:2104.05219) lets a Steane
[[7,1,3]] code recover better than blind decoding alone.

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

`counts_in_intervals_dimension` answers a question upstream of decoding
strategy altogether: is a given stream of error/erasure timestamps
actually temporally clustered (bursty), or Poissonian? It generalizes
the "counts-in-spheres" fractal-dimension estimator used to measure the
transition to large-scale cosmic homogeneity —
[Scrimgeour et al., "The WiggleZ Dark Energy Survey: the transition to
large-scale cosmic homogeneity," MNRAS 425, 116 (2012), arXiv:1205.6812](https://arxiv.org/abs/1205.6812) —
from 3-D space down to 1-D time: for a homogeneous (Poisson) process the
mean count of other events within radius *r* of a reference event scales
as *r*¹ exactly; real burst-like noise (e.g. cosmic-ray-correlated error
bursts, arXiv:2104.05219 — the same physical source `cosmic_ray_burst_profile`
in `dense_evolution.mitigation.zne` models) depresses that exponent below 1.

The function always returns the log-log fit's R² alongside the estimated
dimension, never the number alone — a narrow or poorly-covered range of
window sizes can make a fitted slope meaningless. This is not a
hypothetical caveat: an unrelated spatial box-counting fit through only
4 points spanning a narrow range of scales once produced a fractal
dimension of **142** — physically impossible in 3 dimensions — purely
from fit noise, not from any real structure in the underlying data.
Always inspect R² (rule of thumb: below ~0.98 means don't trust the
dimension) before acting on the result.

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
