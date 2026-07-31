"""
Pauli-string expectation values, computed directly from a statevector via
O(dim) bit manipulation -- the 2**n_qubits Hamiltonian matrix is never
built. The same technique (XOR a flip-mask into the basis-state indices,
track a per-qubit phase from the bit values) shows up hand-duplicated
across dozens of VQE/observable scripts built on this package, each with
its own slightly different bit-twiddling for whichever two or three Pauli
operators that script happened to need. This module factors it into one
tested, general implementation for an arbitrary Pauli string on any subset
of qubits.

Indexing convention: this package's DenseSVSimulator stores qubit 0 as the
*most* significant bit of the basis-state index (empirically: `('x', 0)`
on a 2-qubit register lands on index 2 = '10', not index 1) -- so qubit q
is bit (n_qubits - 1 - q) of the index, and every bit-position computed
here is translated through that offset rather than assuming qubit q is
bit q directly. The string form of a Pauli term reads left-to-right as
qubit 0 upward (`pauli_terms[q]` is the operator on qubit q), independent
of this internal bit-position detail.
"""
import numpy as np

__all__ = ['pauli_expectation', 'pauli_sum_expectation']


def _normalize_terms(pauli_terms, n_qubits=None):
    """Accepts a string ('IXYZ...', pauli_terms[q] = qubit q), a dict
    {qubit: 'X'|'Y'|'Z'|'I'}, or an iterable of (qubit, pauli) pairs.
    Returns a plain dict {qubit: 'X'|'Y'|'Z'} with identity terms dropped
    and every Pauli letter validated."""
    if isinstance(pauli_terms, str):
        if n_qubits is not None and len(pauli_terms) != n_qubits:
            raise ValueError(
                f"pauli_terms string has length {len(pauli_terms)}, "
                f"but n_qubits={n_qubits}")
        terms = {q: p.upper() for q, p in enumerate(pauli_terms) if p.upper() != 'I'}
    elif isinstance(pauli_terms, dict):
        terms = {int(q): str(p).upper() for q, p in pauli_terms.items() if str(p).upper() != 'I'}
    else:
        terms = {int(q): str(p).upper() for q, p in pauli_terms if str(p).upper() != 'I'}

    for q, p in terms.items():
        if p not in ('X', 'Y', 'Z'):
            raise ValueError(
                f"unknown Pauli operator {p!r} for qubit {q}, expected one of X, Y, Z, I")
        if q < 0:
            raise ValueError(f"qubit index {q} must be >= 0")
    return terms


def pauli_expectation(statevector, pauli_terms, n_qubits=None):
    """
    Exact expectation value <psi|P|psi> of a single Pauli string P on a
    pure statevector, computed in O(dim) without ever building the
    2**n_qubits matrix for P.

    Parameters
    ----------
    statevector : array-like, shape (2**n_qubits,)
        A normalized statevector (as returned by
        DenseSVSimulator.get_statevector()).
    pauli_terms : str | dict | iterable of (int, str)
        The Pauli string, in any of three equivalent forms:
          - a string, e.g. ``'XIZ'`` -- pauli_terms[q] is the operator on
            qubit q (qubit 0 first, left-to-right; see module docstring)
          - a dict ``{qubit: 'X'|'Y'|'Z'}`` -- omitted qubits are
            identity, convenient when only a few qubits are non-identity
          - an iterable of ``(qubit, pauli)`` pairs
        Any qubit not mentioned (or given 'I') is identity.
    n_qubits : int, optional
        Only used to validate a string-form pauli_terms' length up front;
        ignored for the dict/iterable forms.

    Returns
    -------
    float
        Real by construction: every Pauli string is Hermitian, so its
        expectation value on any state is real.

    Examples
    --------
    >>> import dense_evolution as de
    >>> sim = de.DenseSVSimulator(2)
    >>> sim.run_circuit([('h', 0), ('cx', 0, 1)])
    >>> de.pauli_expectation(sim.get_statevector(), 'ZZ')
    1.0
    >>> de.pauli_expectation(sim.get_statevector(), {0: 'X', 1: 'X'})
    1.0
    """
    statevector = np.asarray(statevector)
    dim = statevector.shape[0]
    inferred_n_qubits = dim.bit_length() - 1
    if 1 << inferred_n_qubits != dim:
        raise ValueError(f"statevector length {dim} is not a power of 2")

    terms = _normalize_terms(pauli_terms, n_qubits)
    if terms and max(terms) >= inferred_n_qubits:
        raise ValueError(
            f"pauli_terms references qubit {max(terms)}, but the statevector "
            f"only spans {inferred_n_qubits} qubits")

    if not terms:
        # Identity on every qubit: <psi|I|psi> = <psi|psi>, i.e. the norm
        # (1.0 for a properly normalized state, computed rather than
        # assumed so a mis-normalized input surfaces as a wrong answer,
        # not a silently hidden bug).
        return float(np.real(np.vdot(statevector, statevector)))

    def bit_pos(q):
        # qubit q -> bit position in the index (qubit 0 = MSB, see module
        # docstring).
        return inferred_n_qubits - 1 - q

    flip_mask = 0
    for q, p in terms.items():
        if p in ('X', 'Y'):
            flip_mask |= (1 << bit_pos(q))

    indices = np.arange(dim)
    source_idx = indices ^ flip_mask  # source_idx[out] = out ^ flip_mask

    coeff = np.ones(dim, dtype=np.complex128)
    for q, p in terms.items():
        bit = (source_idx >> bit_pos(q)) & 1
        if p == 'Y':
            coeff = coeff * np.where(bit == 0, 1j, -1j)
        elif p == 'Z':
            coeff = coeff * np.where(bit == 0, 1.0, -1.0)
        # 'X' flips the bit (already folded into flip_mask) and
        # contributes a coefficient of 1 -- no further action needed.

    p_psi = statevector[source_idx] * coeff
    return float(np.real(np.vdot(statevector, p_psi)))


def pauli_sum_expectation(statevector, terms, n_qubits=None):
    """
    Expectation value of a weighted sum of Pauli strings, i.e. a
    Hamiltonian given directly in Pauli form:
    ``sum_i coeff_i * <psi|P_i|psi>``.

    Unlike ``circuit_to_energy_fn``'s ``h_matrix @ statevector`` approach,
    this never builds the 2**n_qubits Hamiltonian matrix -- useful once
    the system is too large for a dense Hamiltonian to be practical, or
    simply when the Hamiltonian is more naturally expressed as a Pauli
    sum than as an explicit matrix.

    Parameters
    ----------
    statevector : array-like, shape (2**n_qubits,)
    terms : iterable of (coeff, pauli_terms)
        coeff : float or complex weight for that term.
        pauli_terms : in any form ``pauli_expectation`` accepts (string,
          dict, or pair-iterable).
    n_qubits : int, optional
        Forwarded to ``pauli_expectation`` for string-form term validation.

    Returns
    -------
    float

    Examples
    --------
    >>> # H = 1.0 * Z0 Z1 + 0.5 * X0  (a 2-site transverse-field-Ising term)
    >>> pauli_sum_expectation(sv, [(1.0, 'ZZ'), (0.5, {0: 'X'})])
    """
    total = 0.0
    for coeff, pauli_terms in terms:
        total += coeff * pauli_expectation(statevector, pauli_terms, n_qubits=n_qubits)
    return total
