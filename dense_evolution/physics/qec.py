"""Stabilizer-code quantum error correction utilities: generic Pauli-string
commutation/syndrome primitives, an erasure-aware decoder, and a
minimum-weight-perfect-matching (MWPM) decoder.

Promoted from Dense-Evolution-Discovery's Steane [[7,1,3]] code
investigation (scripts/steane_code_block6_erasure_conversion.py), where a
Steane-specific version of `erasure_aware_decode` was first built and
verified: on shots with exactly 2 simultaneous heralded erasures, it
achieved exactly 0 decoding failures across 94-12,469 such shots at every
tested physical error rate (0 failures out of >60,000 double-erasure
shots total, 40,000 trials x 10 p-values), versus ~25% failure for a
standard syndrome-only decoder blind to the erasure locations -- a clean
confirmation of the real erasure-correction bound below. This version is
code-agnostic (works from any stabilizer generator list, not a
hand-built Steane-specific table), so it moved here instead of staying
Discovery-repo-specific research code.

Erasure-aware decoding exploits a real, foundational fact: Grassl, Beth &
Pellizzari, "Codes for the quantum erasure channel", Phys. Rev. A 56, 33
(1997) -- a distance-d stabilizer code can correct up to (d-1) ERASURES
(known-location errors, e.g. a heralded lost photon in a dual-rail
photonic qubit), versus only floor((d-1)/2) arbitrary (unlocated)
errors. Erasure location information is worth roughly twice as much as
an ordinary syndrome bit, because knowing WHERE the error is removes
exactly the ambiguity a blind syndrome-only decoder has to guess at.

`pymatching_decode` (prog.txt Sezione 4.3) fills the complementary,
far more common case: a real decoder for when NO erasure locations are
known at all (the standard setting for e.g. a surface code under generic
physical noise) -- `erasure_aware_decode`'s brute force (4**k over
heralded qubits) has no answer when k=0 heralded qubits, by design
(returns None). Backed by `pymatching` (Apache-2.0, oscarhiggott/PyMatching,
the standard minimum-weight-perfect-matching decoder for stabilizer
codes), an optional dependency (`pip install dense-evolution[pymatching]`),
not a required one -- most callers of this module (erasure-aware
decoding, raw syndrome computation) never need it.
"""
import itertools
from typing import Optional, Sequence

import numpy as np

try:
    import pymatching
    HAS_PYMATCHING = True
except ImportError:
    HAS_PYMATCHING = False


def _require_pymatching():
    if not HAS_PYMATCHING:
        raise ImportError(
            "MWPM decoding requires the 'pymatching' package. "
            "Install it with: pip install dense-evolution[pymatching]")

_ANTICOMMUTING_PAIRS = {
    frozenset(('X', 'Z')), frozenset(('X', 'Y')), frozenset(('Y', 'Z')),
}
PAULIS = ('I', 'X', 'Y', 'Z')


def pauli_commutes(p1: str, p2: str) -> bool:
    """Whether two equal-length Pauli strings (each character in IXYZ, no
    global phase) commute -- the standard symplectic rule: they commute
    iff the number of qubit positions where the local single-qubit
    Paulis anticommute (X/Z, X/Y, or Y/Z, in either order; I commutes
    with everything) is EVEN.

    >>> pauli_commutes('XX', 'ZZ')   # X,Z anticommute at both qubits -> 2 (even) -> commute
    True
    >>> pauli_commutes('XI', 'ZI')   # X,Z anticommute at 1 qubit -> 1 (odd) -> anticommute
    False
    """
    if len(p1) != len(p2):
        raise ValueError(f"Pauli strings must be equal length: {len(p1)} != {len(p2)}")
    n_anticommuting = sum(
        1 for a, b in zip(p1, p2)
        if a != 'I' and b != 'I' and a != b and frozenset((a, b)) in _ANTICOMMUTING_PAIRS
    )
    return n_anticommuting % 2 == 0


def compute_syndrome(pauli_error: str, stabilizers: Sequence[str]) -> tuple:
    """The syndrome (one bit per stabilizer generator, 1 = anticommutes /
    detected, 0 = commutes / undetected) a given Pauli error string would
    produce against `stabilizers` (a list of equal-length Pauli strings,
    the code's stabilizer generators -- X-type, Z-type, or mixed; this
    function doesn't assume a CSS structure)."""
    return tuple(0 if pauli_commutes(pauli_error, g) else 1 for g in stabilizers)


def _pauli_string(n_qubits: int, assignment: dict) -> str:
    chars = ['I'] * n_qubits
    for q, p in assignment.items():
        chars[q] = p
    return ''.join(chars)


def erasure_aware_decode(
    observed_syndrome: tuple,
    heralded_qubits: Sequence[int],
    n_qubits: int,
    stabilizers: Sequence[str],
) -> Optional[str]:
    """Erasure-aware decoder for any stabilizer code. Given the observed
    syndrome and a list of qubits KNOWN to have been erased (e.g. a
    heralded photon-loss event on a dual-rail-encoded qubit), brute-forces
    every Pauli assignment (I/X/Y/Z, 4**len(heralded_qubits) combinations)
    on just the heralded qubits and returns the unique full-length Pauli
    string reproducing `observed_syndrome` exactly.

    Returns `None` -- not a guess -- when there are zero heralded qubits,
    when the observed syndrome is not explained by any assignment on the
    heralded qubits alone, or when more than one assignment explains it
    (ambiguous). Both `None` cases mean: fall back to a standard
    syndrome-only decoder, or treat as a detected-but-uncorrectable
    event -- this function will not silently return a wrong-but-plausible
    correction. The number of heralded qubits this can actually resolve
    unambiguously is bounded by the code's real distance (Grassl, Beth &
    Pellizzari 1997: up to d-1 erasures) -- that bound emerges naturally
    from the brute-force search itself (more heralded qubits than the
    code can resolve typically yields zero or multiple matches), it is
    not hard-coded here.

    Cost is 4**len(heralded_qubits) syndrome computations, each O(n_qubits
    * len(stabilizers)) -- fine for the small numbers of simultaneous
    erasures a real per-shot noise rate produces (verified up to 2 in the
    original Steane investigation; tractable up to 4-5 for most small
    codes before it's worth switching to a smarter search).
    """
    if not heralded_qubits:
        return None

    matches = []
    for assignment_paulis in itertools.product(PAULIS, repeat=len(heralded_qubits)):
        assignment = dict(zip(heralded_qubits, assignment_paulis))
        candidate = _pauli_string(n_qubits, assignment)
        if compute_syndrome(candidate, stabilizers) == tuple(observed_syndrome):
            matches.append(candidate)

    if len(matches) != 1:
        return None
    return matches[0]


def pymatching_decode(
    observed_syndrome: Sequence[int],
    stabilizers: Sequence[str],
    n_qubits: int,
    error_type: str = 'X',
    weights: Optional[Sequence[float]] = None,
) -> str:
    """MWPM syndrome decoder (via `pymatching`) for a single physical error
    type, no erasure/heralding information needed -- the standard decoding
    setting `erasure_aware_decode` deliberately doesn't cover (it always
    returns None with zero heralded qubits).

    Restricted to same-purpose stabilizers detecting ONE error type at a
    time (the standard CSS setup: e.g. Z-type stabilizers decoding X
    errors, or vice versa) -- NOT the fully general mixed-Pauli case
    `compute_syndrome`/`erasure_aware_decode` accept. A code with both X-
    and Z-type errors to correct needs two separate calls, one per type
    (X-type stabilizers -> decode Z errors, Z-type stabilizers -> decode
    X errors), each contributing its own half of the full correction --
    the same two-pass structure any real CSS decoder uses, not a
    limitation specific to this wrapper.

    FURTHER REAL RESTRICTION (verified directly against pymatching, not
    assumed from its docs): a matching-graph decoder needs every qubit
    checked by AT MOST 2 stabilizers -- pymatching represents each
    potential error as a graph EDGE between (at most) 2 detector nodes, a
    structural fact about topological/surface codes (each qubit sits on
    an edge between exactly 2 checks), not a general property of
    stabilizer codes. Steane [[7,1,3]]'s weight-4 stabilizers are a real
    counterexample -- qubit 6 is checked by all 3 X-stabilizers at once
    (3 > 2), so `pymatching_decode` cannot be used for it at all
    (confirmed: raises ValueError below, not just slow or approximate) --
    use `erasure_aware_decode` for codes like that instead. Repetition
    codes and the surface/toric code family satisfy the <=2 constraint by
    construction; small non-topological codes generally do not.

    The check matrix pymatching needs is built from `stabilizers` via this
    module's own `pauli_commutes` (column q of row i is 1 iff stabilizer i
    anticommutes with a lone `error_type` error on qubit q) rather than a
    naive "non-identity entry" heuristic -- those disagree whenever a
    stabilizer has the SAME letter as `error_type` at some qubit (e.g. an
    'X' entry in a stabilizer being used to decode X errors: it commutes
    with an X error there and must NOT count as detecting it, but is very
    much non-identity).

    Parameters
    ----------
    observed_syndrome : sequence of int
        One bit per stabilizer, same convention as `compute_syndrome`'s
        return value (1 = that stabilizer detected/anticommuted).
    stabilizers : sequence of str
        The stabilizer generators used for this error type (e.g. every
        Z-type generator, to decode X errors), each a length-`n_qubits`
        Pauli string over IXYZ. Mixing generator types that detect
        different error types in the same call silently produces a wrong
        check matrix -- this function has no way to tell that apart from
        a correctly-scoped single-type list, so it isn't validated here.
    n_qubits : int
        Number of physical qubits (columns of the check matrix / length
        of the returned Pauli string).
    error_type : str, optional
        Which single-qubit Pauli error this decodes for -- one of 'X',
        'Y', 'Z'. Defaults to 'X'.
    weights : sequence of float, optional
        Per-qubit edge weight for MWPM (e.g. -log(p_q) for a known
        per-qubit physical error rate p_q) -- forwarded to
        `pymatching.Matching.from_check_matrix`. Defaults to pymatching's
        own default (uniform weight 1.0 for every qubit, i.e. no prior
        assumption about which qubits are more error-prone).

    Returns
    -------
    str
        Length-`n_qubits` Pauli string (only 'I' and `error_type`) giving
        the minimum-weight correction pymatching found.

    Raises
    ------
    ImportError
        If `pymatching` isn't installed (`pip install dense-evolution[pymatching]`).
    ValueError
        If `error_type` isn't one of 'X'/'Y'/'Z', if `observed_syndrome`
        doesn't have one entry per stabilizer, if a stabilizer's length
        isn't `n_qubits`, or if every stabilizer commutes with every
        possible `error_type` error (the check matrix would be all-zero --
        almost always means `stabilizers` is the wrong generator type for
        the requested `error_type`, not a real all-zero code).

    Examples
    --------
    >>> # 3-qubit repetition code, Z-type stabilizers, decoding X errors
    >>> stabilizers = ['ZZI', 'IZZ']
    >>> syndrome = compute_syndrome('IXI', stabilizers)  # X error on qubit 1
    >>> pymatching_decode(syndrome, stabilizers, n_qubits=3, error_type='X')
    'IXI'
    """
    _require_pymatching()

    if error_type not in ('X', 'Y', 'Z'):
        raise ValueError(f"error_type must be one of 'X', 'Y', 'Z', got {error_type!r}")
    if len(observed_syndrome) != len(stabilizers):
        raise ValueError(
            f"observed_syndrome has {len(observed_syndrome)} entries but there are "
            f"{len(stabilizers)} stabilizers -- these must match one-to-one"
        )
    for i, s in enumerate(stabilizers):
        if len(s) != n_qubits:
            raise ValueError(f"stabilizers[{i}] has length {len(s)}, expected n_qubits={n_qubits}")

    check_matrix = np.array(
        [[0 if pauli_commutes(s[q], error_type) else 1 for q in range(n_qubits)] for s in stabilizers],
        dtype=np.uint8,
    )
    if not check_matrix.any():
        raise ValueError(
            f"every stabilizer commutes with every possible {error_type} error -- "
            f"stabilizers is very likely the wrong generator type to decode {error_type} "
            f"errors (e.g. passing X-type stabilizers to decode X errors, which they "
            f"cannot detect by construction)"
        )
    checks_per_qubit = check_matrix.sum(axis=0)
    if (checks_per_qubit > 2).any():
        bad_qubits = np.where(checks_per_qubit > 2)[0].tolist()
        raise ValueError(
            f"pymatching's matching-graph decoder needs every qubit checked by AT MOST 2 "
            f"stabilizers (a graph edge connects at most 2 detector nodes) -- qubit(s) "
            f"{bad_qubits} are each checked by {checks_per_qubit[bad_qubits].tolist()} "
            f"stabilizers here. This is a real structural requirement of matching-graph "
            f"decoding (true for the surface code and other topological codes, where each "
            f"qubit sits on an edge between exactly 2 checks), NOT satisfied by every "
            f"stabilizer code -- e.g. Steane [[7,1,3]]'s weight-4 stabilizers check some "
            f"qubits 3 times, so pymatching_decode cannot be used for it; use "
            f"erasure_aware_decode instead for codes like that."
        )

    matching = pymatching.Matching.from_check_matrix(check_matrix, weights=weights)
    correction = matching.decode(np.asarray(observed_syndrome, dtype=np.uint8))

    return ''.join(error_type if bit else 'I' for bit in correction)
