# UCCSD (native excitation circuits)

> Circuits, not chemistry: this module builds gate sequences for the
> single/double fermionic excitations a UCCSD ansatz needs. Pairing this
> with an actual molecular Hamiltonian is [VQE's own job](dashboard_core_vqe.md).

A UCCSD ansatz applies one unitary per excitation out of a Hartree-Fock
reference: `exp(theta * (a_p^dagger a_q - a_q^dagger a_p))` for a single
excitation, and the four-index equivalent for a double. This module builds
those unitaries as real, exact gate sequences directly against the
Jordan-Wigner mapping — no PennyLane dependency, no Trotter error.

## Step 1. Which excitations exist for a given molecule

```python
from dense_evolution.circuits.uccsd import find_excitations

singles, doubles = find_excitations(electrons=2, n_qubits=4)
singles, doubles
```

```
([[0, 2], [1, 3]], [[0, 1, 2, 3]])
```

`find_excitations` is pure combinatorics, no quantum computation: with 2
electrons filling qubits `0` and `1` (occupied) and qubits `2`, `3` empty
(virtual), a single excitation moves one electron from an occupied to a
virtual orbital of the same spin; a double moves two at once. This matches
`qml.qchem.excitations(electrons, n_qubits)` exactly, without needing
PennyLane installed just to find out which excitations exist.

## Step 2. A single excitation, exact for any orbital distance

```python
import dense_evolution as de
from dense_evolution.circuits.uccsd import single_excitation_ops

p, q, theta = 0, 2, 0.4173
ops = single_excitation_ops(p, q, theta)

sim = de.DenseSVSimulator(4)
sim.sv[0] = 0
sim.sv[int('0011', 2)] = 1.0
sim.run_circuit(ops)
sim.get_statevector()
```

```
array with nonzero amplitudes at |0011> (0.9142) and |0110> (0.4053)
```

`single_excitation_ops(p, q, theta)` builds `exp(theta * (a_p^dagger a_q -
a_q^dagger a_p))` as `CNOT(p,q)`, a chain of `CNOT`s folding every
in-between qubit's Jordan-Wigner parity onto `p`, one `CRY(2*theta)`, then
undoing the fold — exact for any `p < q`, not just adjacent orbitals.
Starting from the Hartree-Fock reference `|0011>` (qubits 0,1 occupied), it
rotates into a superposition with `|0110>` — the `p=0 -> q=2` excited
configuration — matching `scipy.linalg.expm` of the exact fermionic
generator to machine precision (`~1e-16`), not a Trotter approximation.

## Step 3. A double excitation, exact when both pairs are adjacent

```python
from dense_evolution.circuits.uccsd import double_excitation_ops

p, q, r, s, theta = 0, 1, 2, 3, 0.4173
ops = double_excitation_ops(p, q, r, s, theta, ancilla1=4, ancilla2=5)

sim = de.DenseSVSimulator(6)
sim.sv[0] = 0
sim.sv[int('110000', 2)] = 1.0
sim.run_circuit(ops)
sim.get_statevector()
```

```
array with nonzero amplitudes at |110000> (0.9142) and |001100> (0.4053)
```

`double_excitation_ops` needs two spare ancilla qubits (`ancilla1`,
`ancilla2`, must be `|0>` on entry) whenever the occupied pair (`p, q`)
*and* the virtual pair (`r, s`) are each adjacent — exactly the case
`find_excitations` produces for a minimal active space like this one.
The ancillas come back to `|0>` on exit (verified above: no leakage
outside the logical `|110000>`/`|001100>` pair), so the same two ancillas
can be reused across every double excitation in an ansatz.

## Details

### Where this came from

UCCSD's excitation generators were previously only reachable through
PennyLane's own `FermionicSingleExcitation`/`FermionicDoubleExcitation`
decomposition. This module is a direct derivation against
`dense_evolution.physics.fermions.majorana_pauli_terms` (the package's own
already-verified Jordan-Wigner mapping) instead — every circuit identity
was checked against `scipy.linalg.expm` of the exact generator matrix,
independently reconstructed from raw ladder operators, before being
written here (see `tests/unit/test_uccsd.py`).

### When the double-excitation closed form doesn't apply

For an active space where the occupied pair or the virtual pair is
itself non-adjacent (3+ occupied or 3+ virtual orbitals, e.g. LiH/BeH2),
the closed-form Z-string folding used for single excitations does not
carry over: the naive fold leaves genuine leakage, not just an
unsimplified circuit. `double_excitation_ops` falls back automatically to
per-term exponentiation of the 8-term Pauli decomposition (via
`dense_evolution.circuits.trotter.pauli_rotation_ops`) whenever no
ancillas are given, or the pairs aren't both adjacent.

That fallback is exact too, not a Trotter approximation: verified
exhaustively across every computational basis state and several
non-adjacent `p<q<r<s` choices, to floating-point precision (`~1e-15`) at
every `theta` tested — the 8 terms don't commute as general operators, but
the generator only ever couples two basis states for any fixed setting of
the untouched qubits, and within that 2-dimensional subspace the per-term
exponentials compose without error. Only the gate count differs from the
closed form, never correctness.

### Gate count

The single-excitation closed form costs `2*(q-p) + 2` CX gates (plus the
`CRY`'s own 2 CX). The double-excitation closed form uses two Toffolis
(each itself decomposed into CX + single-qubit gates via
`QuantumTranspiler.decompose_toffoli`) around one `CRY`; the per-term
fallback costs one Pauli-rotation gadget per surviving term (up to 8) of
the double-excitation generator, via `pauli_rotation_ops`.

::: dense_evolution.circuits.uccsd
