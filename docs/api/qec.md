# QEC (stabilizer decoding: erasure-aware, MWPM, and blind brute-force)

A stabilizer code protects information by encoding it across several physical qubits
such that certain "stabilizer" measurements always return the same value on an
error-free state -- a *syndrome*. When an error flips one of those measurements, the
syndrome pattern reveals (up to the code's own limits) which error happened, without
ever measuring the encoded data itself. This module is generic and code-agnostic:
syndrome computation and three decoders work for any stabilizer code you hand them,
not just one built in.

## Step 1. Do two Pauli operators commute?

```python
from dense_evolution.physics.qec import pauli_commutes

pauli_commutes('XII', 'ZZI')
```

```
False
```

`pauli_commutes(p1, p2)` checks whether two full-length Pauli strings commute --
`False` here because `X` and `Z` overlap (anti-commute) on qubit 0. This is the
building block a stabilizer measurement's outcome is built from: an error anticommuting
with a stabilizer flips that stabilizer's measured value, which is exactly the
syndrome bit `compute_syndrome` (Step 2) reads off.

## Step 2. The syndrome of a real error

```python
from dense_evolution.physics.qec import compute_syndrome

x_stabilizers = ['IIIXXXX', 'IXXIIXX', 'XIXIXIX']
z_stabilizers = ['IIIZZZZ', 'IZZIIZZ', 'ZIZIZIZ']
stabilizers = x_stabilizers + z_stabilizers

error = 'IIIZIII'
compute_syndrome(error, stabilizers)
```

```
(1, 0, 0, 0, 0, 0)
```

`stabilizers` here are the Steane [[7,1,3]] code's real 6 generators (3 checking `X`
errors, 3 checking `Z` errors) -- the running example for the rest of this page.
`error` is a `Z` flip on qubit 3 alone. `compute_syndrome` returns one bit per
stabilizer: `1` where that stabilizer anticommutes with the error. Only the first
`Z`-type stabilizer (`IIIZZZZ`, which includes qubit 3) flips -- the other five commute
with a lone `Z` error.

## Step 3. Recovering the error, blind

```python
from dense_evolution.physics.qec import blind_minimum_weight_decode

syndrome = compute_syndrome(error, stabilizers)
decoded = blind_minimum_weight_decode(syndrome, n_qubits=7, stabilizers=stabilizers)
decoded == error
```

```
True
```

`blind_minimum_weight_decode` never sees `error` -- only `syndrome` -- and searches
every possible Pauli error in increasing weight order, returning the minimum-weight
one that reproduces it (or `None` if more than one error at that weight matches,
never a guess). Brute force, `O(3**w)` at weight `w`, so it's for small codes and low
weights -- but it works on any stabilizer code, including ones (like Steane) the
matching-graph decoder below structurally cannot handle.

## Step 4. Recovering the error, with a known location

```python
from dense_evolution.physics.qec import erasure_aware_decode

erasure_aware_decode(syndrome, heralded_qubits=[3], n_qubits=7, stabilizers=stabilizers)
```

```
'IIIZIII'
```

If something upstream already knows *where* an error might have happened (a heralded
lost photon in a dual-rail photonic qubit, say) -- not just the syndrome --
`erasure_aware_decode` uses that location directly instead of searching blind. A
distance-3 code like Steane can resolve up to 2 heralded erasures, versus only 1
unlocated error blind -- knowing the location is strictly more powerful, the real
result behind quantum erasure-channel codes (Grassl, Beth & Pellizzari, Phys. Rev. A
56, 33 (1997)).

## Step 5. `pymatching`'s real limitation

```python
from dense_evolution.physics.qec import pymatching_decode

error_x = 'IIIXIII'
syndrome_z = compute_syndrome(error_x, z_stabilizers)
pymatching_decode(syndrome_z, z_stabilizers, n_qubits=7, error_type='X')
```

```
ValueError: pymatching's matching-graph decoder needs every qubit checked by AT MOST 2 stabilizers (a graph edge connects at most 2 detector nodes) -- qubit(s) [6] are each checked by [3] stabilizers here. ...
```

`pymatching_decode` (an optional dependency, `pip install dense-evolution[pymatching]`)
is much faster than blind brute-force decoding -- but only for "graph-like" codes,
where every qubit sits on at most 2 stabilizers (true for the surface code). Steane's
weight-4 stabilizers check some qubits 3 times, so `pymatching_decode` raises this
`ValueError` rather than silently returning a wrong answer -- `blind_minimum_weight_decode`
(Step 3) or `erasure_aware_decode` (Step 4) are the ones that work here instead.

## Step 6. Is a noise process bursty, or Poissonian?

```python
import numpy as np
from dense_evolution.physics.qec import counts_in_intervals_dimension

rng = np.random.default_rng(0)
window_sizes = [1, 2, 4, 8, 16, 32]

poisson_times = np.sort(rng.uniform(0, 100, 200))
dim, r2, _ = counts_in_intervals_dimension(poisson_times, window_sizes)
dim, r2
```

```
(1.011649262707353, 0.9998031727847672)
```

A separate question from decoding itself: is a stream of error/erasure timestamps
temporally clustered (bursty, e.g. a cosmic-ray impact) or Poissonian (uniformly
random)? For a homogeneous Poisson process, the mean count of nearby events within
radius `r` scales as `r^1` exactly -- `dim` above lands almost exactly on `1.0`, with
`r2` close to `1` confirming the fit itself is trustworthy. A genuinely bursty stream
(150 events packed into `[0, 10]`, 50 more into `[50, 52]`, otherwise identical in
count) gives a real, depressed exponent instead:

```python
burst_times = np.sort(np.concatenate([rng.uniform(0, 10, 150), rng.uniform(50, 52, 50)]))
counts_in_intervals_dimension(burst_times, window_sizes)[:2]
```

```
(0.7347741100131827, 0.9721804667576291)
```

`0.73` instead of `1.0` -- exactly the signature real burst-like error sources (cosmic
rays, correlated hardware glitches) produce. Always check `r2` before trusting the
dimension itself (rule of thumb: below `~0.98` means don't) -- a narrow or
poorly-covered range of window sizes can fit a meaningless slope.

---

## Details

**`decode_with_erasure_fallback`** composes Steps 3-4 into the real-world decoding
*policy*, not another raw decoder: use `erasure_aware_decode` when there are heralded
qubits and it resolves the syndrome uniquely, otherwise fall back to
`blind_minimum_weight_decode`. Never worse than always calling blind decoding directly
-- promoted from Dense-Evolution-Discovery's
[cosmic-ray-burst-as-erasure experiment](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/cosmic_ray_burst_validation/),
where this exact fallback logic was first written inline in a Monte Carlo loop testing
whether knowing *which* qubits a real cosmic-ray burst hit (arXiv:2104.05219) lets a
Steane code recover better than blind decoding alone.

**The naive shortcut that doesn't work**: calling `erasure_aware_decode` with *every*
qubit passed as heralded, instead of using `blind_minimum_weight_decode`, fails
in practice -- with no qubit assumed error-free, many stabilizer-equivalent errors
share the same syndrome, so `erasure_aware_decode`'s "exactly one match" criterion is
essentially always violated. Minimum-weight selection (Step 3) is what makes blind
decoding well-posed at all.

**Real validation, not just unit tests**: a Steane-specific version of this decoder was
checked against STIM's native `HERALDED_ERASE` noise channel -- 0 decoding failures
across every double-erasure shot tested (>60,000 shots total, 40,000 trials x 10
physical error rates), versus a real ~25% failure rate for a standard syndrome-only
decoder blind to the erasure locations. See Dense-Evolution-Discovery's Steane
[[7,1,3]] investigation and Gu, Vaknin, Retzker & Kubica, "Optimizing quantum error
correction protocols with erasure qubits," PRX Quantum 6, 040354 (2025),
arXiv:2408.00829.

**Never guesses**: every decoder here returns `None` on an ambiguous or unresolvable
syndrome rather than a best-effort guess -- more heralded qubits than the code can
actually resolve, or a blind search with more than one minimum-weight match, both
yield `None`.

::: dense_evolution.physics.qec
