"""
Native (PennyLane-free) UCCSD excitation circuits.

UCCSD's single/double fermionic excitation generators
(a_p^dagger a_q - h.c., a_p^dagger a_q^dagger a_r a_s - h.c.) were
previously only reachable through PennyLane's own FermionicSingleExcitation/
FermionicDoubleExcitation decomposition (see dashboard/core/vqe.py's
_uccsd_tape_to_qasm) -- not because a native circuit is theoretically
impossible, but because deriving the exact (non-Trotterized) circuit
identity requires real derivation, not just wiring PennyLane through.

This module IS that derivation, done directly against
dense_evolution.physics.fermions.majorana_pauli_terms (already-verified
Jordan-Wigner mapping) rather than against PennyLane's internals, and
every circuit identity below was verified against scipy.linalg.expm of
the exact generator matrix (independently reconstructed from raw
Jordan-Wigner ladder operators, not from PennyLane) before being written
here -- see tests/unit/test_uccsd.py.

Single excitation (any orbital distance): exact closed form, always.
G_pq = a_p^dagger a_q - a_q^dagger a_p decomposes (via
a_p^dagger=(chi_X(p)-i*chi_Y(p))/2) into exactly 2 Pauli strings that
share the Jordan-Wigner Z-string between p and q and differ only by an
X<->Y swap at p and q. This is the standard "Givens rotation" generator
(G^2 = 2I): exact circuit is CNOT(p,q), fold each Z-string qubit's
parity onto p via CNOT(k,p), CRY(2*theta, control=q, target=p), then
undo. Verified for adjacent and long-Z-string pairs alike (any p<q).

Double excitation: exact closed form when the occupied pair (p,q) is
adjacent (q=p+1) AND the virtual pair (r,s) is adjacent (s=r+1) -- the
gap between q and r (HOMO-LUMO gap) can be any length, no restriction.
This is the case dashboard/core/vqe.py's UCCSD ansatz always produces
for a 2-occupied/2-virtual active space (e.g. H2 in a minimal basis),
and generally whenever qml.qchem.excitations pairs consecutive occupied
orbitals with consecutive virtual orbitals. G_pqrs (8 Pauli terms)
reduces to a clean 2-level coupling between |p=0,q=1,r=1,s=1,...> and
its bit-flip, realized via a CNOT fan (mirroring the p,q,r,s pattern
onto q) plus a triple-controlled RY (Toffoli-mediated, 2 ancilla
qubits, always returned to |0>).

For a double excitation where the occupied pair or the virtual pair is
itself non-adjacent (occurs for active spaces with 3+ occupied or 3+
virtual orbitals, e.g. LiH/BeH2), the same closed-form Z-string folding
used for single excitations does NOT carry over cleanly (verified
directly: the naive fold leaves genuine leakage, not just an
unsimplified-but-correct circuit) -- deriving a comparably compact
closed form for that case is unresolved, so double_excitation_ops falls
back to per-term exponentiation of the 8 Pauli terms via
dense_evolution.circuits.trotter.pauli_rotation_ops for that specific
case, applied sequentially in a fixed term order.

That fallback turned out to be exact too, not a Trotter approximation:
verified exhaustively (every computational basis state, several p<q<r<s
choices including non-adjacent pairs, theta up to pi) against
scipy.linalg.expm of the exact generator matrix, to floating-point
precision (~1e-15) at every theta tested, not just small theta the way
a genuine first-order Trotter error would show. The 8 terms don't
commute as operators on the full Hilbert space, but the generator only
ever couples two basis states for any fixed setting of the qubits it
doesn't act on (same structure as the closed-form case), and within
that 2-dimensional invariant subspace the per-term exponentials
apparently compose without error -- not derived from first principles
here, just confirmed by the exhaustive check above. Only the GATE COUNT
differs from the closed form, not correctness.
"""
from typing import List, Tuple

from .trotter import pauli_rotation_ops

__all__ = ['find_excitations', 'single_excitation_ops', 'double_excitation_ops']


def find_excitations(electrons: int, n_qubits: int):
    """
    Enumerate spin-conserving single/double excitation index tuples for
    a Hartree-Fock reference with the given electron count -- pure
    classical combinatorics (occupied = orbitals [0, electrons), virtual
    = the rest, even index = spin-up, odd = spin-down under the standard
    Jordan-Wigner interleaved spin-orbital ordering), no quantum
    computation involved. Verified to reproduce
    qml.qchem.excitations(electrons, n_qubits) exactly across several
    (electrons, n_qubits) pairs -- see tests/unit/test_uccsd.py -- kept
    here so finding *which* excitations exist doesn't need PennyLane
    installed any more than building their circuits does.

    Returns (singles, doubles): singles is a list of [p, q] (p occupied,
    q virtual, same spin); doubles is a list of [p, q, r, s] with
    p<q both occupied, r<s both virtual, total spin conserved.
    """
    occupied = list(range(electrons))
    virtual = list(range(electrons, n_qubits))
    singles = [[p, q] for p in occupied for q in virtual if p % 2 == q % 2]
    doubles = []
    for i, p in enumerate(occupied):
        for q in occupied[i + 1:]:
            for j, r in enumerate(virtual):
                for s in virtual[j + 1:]:
                    if (p % 2 + q % 2) == (r % 2 + s % 2):
                        doubles.append([p, q, r, s])
    return singles, doubles


# ── Pauli-string algebra (needed only for the per-term fallback path's
# generator decomposition -- the closed-form circuits below are pure
# combinatorial gate templates and never call this at all) ──────────────

_MUL = {
    ('I', 'I'): ('I', 1), ('I', 'X'): ('X', 1), ('I', 'Y'): ('Y', 1), ('I', 'Z'): ('Z', 1),
    ('X', 'I'): ('X', 1), ('Y', 'I'): ('Y', 1), ('Z', 'I'): ('Z', 1),
    ('X', 'X'): ('I', 1), ('Y', 'Y'): ('I', 1), ('Z', 'Z'): ('I', 1),
    ('X', 'Y'): ('Z', 1j), ('Y', 'X'): ('Z', -1j),
    ('Y', 'Z'): ('X', 1j), ('Z', 'Y'): ('X', -1j),
    ('Z', 'X'): ('Y', 1j), ('X', 'Z'): ('Y', -1j),
}


def _pauli_mul_term(term_a, term_b):
    ca, da = term_a
    cb, db = term_b
    phase = ca * cb
    out = {}
    for q in set(da) | set(db):
        la, lb = da.get(q, 'I'), db.get(q, 'I')
        letter, ph = _MUL[(la, lb)]
        phase *= ph
        if letter != 'I':
            out[q] = letter
    return (phase, out)


def _psum_mul(terms_a, terms_b):
    return [_pauli_mul_term(ta, tb) for ta in terms_a for tb in terms_b]


def _psum_scale(terms, s):
    return [(c * s, d) for c, d in terms]


def _psum_combine(terms):
    acc = {}
    for c, d in terms:
        key = tuple(sorted(d.items()))
        acc[key] = acc.get(key, 0) + c
    return [(c, dict(key)) for key, c in acc.items() if abs(c) > 1e-12]


def _chi(mode_index, n_qubits):
    from ..physics.fermions import majorana_pauli_terms
    return majorana_pauli_terms(mode_index, n_qubits)[1]


def _a_dagger(p, n_qubits):
    return [(0.5, _chi(2 * p + 1, n_qubits)), (-0.5j, _chi(2 * p + 2, n_qubits))]


def _a(p, n_qubits):
    return [(0.5, _chi(2 * p + 1, n_qubits)), (0.5j, _chi(2 * p + 2, n_qubits))]


def _double_excitation_generator(p, q, r, s, n_qubits):
    """8 Pauli terms of a_p^dagger a_q^dagger a_r a_s - h.c. (JW-mapped).
    Used only by the per-term fallback path -- verified against direct
    fermionic ladder operators in tests/unit/test_uccsd.py."""
    term1 = _psum_mul(_psum_mul(_a_dagger(p, n_qubits), _a_dagger(q, n_qubits)),
                       _psum_mul(_a(r, n_qubits), _a(s, n_qubits)))
    term2 = _psum_mul(_psum_mul(_a_dagger(s, n_qubits), _a_dagger(r, n_qubits)),
                       _psum_mul(_a(q, n_qubits), _a(p, n_qubits)))
    return _psum_combine(term1 + _psum_scale(term2, -1))


# ── Single excitation: exact closed form, any p<q ───────────────────────

def single_excitation_ops(p: int, q: int, theta: float) -> List[Tuple]:
    """
    Exact circuit for exp(theta * (a_p^dagger a_q - a_q^dagger a_p)),
    the UCCS single-excitation unitary, for ANY p < q (any Jordan-Wigner
    Z-string length in between) -- no Trotter error, ever.

    Gate count: 2*(1 + (q-p-1)) CX + 1 CRY (itself 2 CX + 2 RY), so
    2*(q-p) + 2 CX total.
    """
    if not p < q:
        raise ValueError(f"single_excitation_ops requires p < q, got p={p}, q={q}")
    inter = list(range(p + 1, q))
    ops: List[Tuple] = []
    ops.append(('cx', p, q))
    for k in inter:
        ops.append(('cx', k, p))
    ops.extend(_cry_ops(2 * theta, control=q, target=p))
    for k in reversed(inter):
        ops.append(('cx', k, p))
    ops.append(('cx', p, q))
    return ops


def _cry_ops(theta, control, target):
    """CRY(theta) from native RY+CX: RY(theta/2)_t ; CX(c,t) ; RY(-theta/2)_t ; CX(c,t)."""
    return [
        ('ry', target, theta / 2),
        ('cx', control, target),
        ('ry', target, -theta / 2),
        ('cx', control, target),
    ]


# ── Double excitation ────────────────────────────────────────────────────

def double_excitation_ops(p: int, q: int, r: int, s: int, theta: float,
                           ancilla1: int = None, ancilla2: int = None) -> List[Tuple]:
    """
    Circuit for exp(theta * (a_p^dagger a_q^dagger a_r a_s - h.c.)), the
    UCCD double-excitation unitary, for p < q < r < s.

    Exact closed form (no Trotter error) when ancilla1/ancilla2 are both
    given AND the occupied pair is adjacent (q == p+1) AND the virtual
    pair is adjacent (s == r+1) -- the gap between q and r can be any
    length. This is what qml.qchem.excitations always produces for a
    2-occupied/2-virtual active space, and sometimes for larger ones.

    Falls back to per-term exponentiation of the 8-term Pauli
    decomposition whenever no ancillas are given, or the pairs aren't
    adjacent -- verified exact (not an approximation), just more gates
    than the closed form; see the module docstring for why the closed
    form doesn't generalize to a non-adjacent occupied or virtual pair
    yet.

    ancilla1, ancilla2 : two qubits that MUST be |0> on entry and are
    guaranteed returned to |0> on exit -- only touched by the
    closed-form path (never referenced by the per-term fallback), so
    the same two ancillas can be reused across every double excitation
    in an ansatz. Omit both (leave as None) to force the ancilla-free
    per-term path unconditionally -- e.g. when the caller's qubit
    register has no spare qubits to offer.
    """
    if not p < q < r < s:
        raise ValueError(f"double_excitation_ops requires p<q<r<s, got {p},{q},{r},{s}")

    have_ancillas = ancilla1 is not None and ancilla2 is not None
    if have_ancillas and q == p + 1 and s == r + 1:
        return _double_excitation_ops_closed_form(p, q, r, s, theta, ancilla1, ancilla2)
    return _double_excitation_ops_per_term(p, q, r, s, theta)


def _double_excitation_ops_closed_form(p, q, r, s, theta, a1, a2) -> List[Tuple]:
    ops: List[Tuple] = []
    ops.append(('cx', q, s))
    ops.append(('cx', q, r))
    ops.append(('cx', q, p))
    ops.append(('x', p))
    ops.extend(_toffoli_ops(p, r, a1))
    ops.extend(_toffoli_ops(a1, s, a2))
    ops.extend(_cry_ops(-2 * theta, control=a2, target=q))
    ops.extend(_toffoli_ops(a1, s, a2))
    ops.extend(_toffoli_ops(p, r, a1))
    ops.append(('x', p))
    ops.append(('cx', q, p))
    ops.append(('cx', q, r))
    ops.append(('cx', q, s))
    return ops


def _toffoli_ops(c1, c2, t):
    from .compiler import QuantumTranspiler
    return QuantumTranspiler.decompose_toffoli(c1, c2, t)


def _double_excitation_ops_per_term(p, q, r, s, theta) -> List[Tuple]:
    n_qubits = s + 1
    terms = _double_excitation_generator(p, q, r, s, n_qubits)
    ops: List[Tuple] = []
    for coeff, pauli_dict in terms:
        real_part = (coeff / 1j).real
        angle = -theta * real_part
        ops.extend(pauli_rotation_ops(pauli_dict, angle))
    return ops
