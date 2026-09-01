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
import jax.numpy as jnp

__all__ = [
    'pauli_expectation', 'pauli_sum_expectation', 'pauli_hamiltonian_to_matrix',
    'pauli_sum_matvec', 'multiply_pauli_terms',
    'pauli_sum_matvec_jax', 'pauli_sum_expectation_jax', 'PauliSumOperator',
]

_PAULI_MATRICES = {
    'I': np.eye(2, dtype=np.complex128),
    'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


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


def _apply_pauli_term(statevector, terms, inferred_n_qubits):
    """P|psi> for a single already-normalized Pauli term (terms: plain
    dict {qubit: 'X'|'Y'|'Z'}, identity qubits already dropped), computed
    in O(dim) via the same flip-mask/phase technique pauli_expectation
    uses -- factored out here so both pauli_expectation (which reduces
    this to a scalar via vdot) and pauli_sum_matvec (which needs the
    vector itself, not a scalar) share one tested implementation instead
    of two copies of the same bit-twiddling."""
    if not terms:
        return statevector

    def bit_pos(q):
        return inferred_n_qubits - 1 - q

    flip_mask = 0
    for q, p in terms.items():
        if p in ('X', 'Y'):
            flip_mask |= (1 << bit_pos(q))

    dim = statevector.shape[0]
    indices = np.arange(dim)
    source_idx = indices ^ flip_mask

    coeff = np.ones(dim, dtype=np.complex128)
    for q, p in terms.items():
        bit = (source_idx >> bit_pos(q)) & 1
        if p == 'Y':
            coeff = coeff * np.where(bit == 0, 1j, -1j)
        elif p == 'Z':
            coeff = coeff * np.where(bit == 0, 1.0, -1.0)

    return statevector[source_idx] * coeff


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

    p_psi = _apply_pauli_term(statevector, terms, inferred_n_qubits)
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


def pauli_hamiltonian_to_matrix(terms, n_qubits):
    """
    Builds the real, explicit dense Hermitian Hamiltonian matrix for a
    weighted sum of Pauli strings, H = sum_i coeff_i * P_i -- the
    (2**n_qubits, 2**n_qubits) matrix pauli_sum_expectation deliberately
    avoids building. Use this when something downstream genuinely needs
    the matrix itself (exact diagonalization for a ground-state energy,
    a VQE cost function computed as ``<psi| H @ psi>`` instead of a
    Pauli-by-Pauli sum, ...), not just an expectation value.

    Same qubit-0-is-MSB convention as the rest of this module (see the
    module docstring): each term's matrix is the Kronecker product of
    per-qubit 2x2 Pauli matrices in qubit order 0..n_qubits-1, so this
    matrix's basis-state index lines up exactly with the one
    pauli_expectation/pauli_sum_expectation use -- H @ statevector and
    pauli_sum_expectation(statevector, terms) agree to floating-point
    precision for the same terms.

    Parameters
    ----------
    terms : iterable of (coeff, pauli_terms)
        Same format pauli_sum_expectation accepts: coeff is a real or
        complex weight, pauli_terms is a string/dict/pair-iterable in any
        form _normalize_terms accepts.
    n_qubits : int
        Total number of qubits the matrix spans (every term's qubits must
        be < n_qubits).

    Returns
    -------
    numpy.ndarray, shape (2**n_qubits, 2**n_qubits), dtype complex128
        Hermitian by construction (a real-weighted sum of Hermitian
        Pauli-string matrices, each a Kronecker product of Hermitian 2x2
        Pauli matrices -- Hermiticity is closed under both operations).

    Examples
    --------
    >>> H = pauli_hamiltonian_to_matrix([(1.0, 'ZZ'), (0.5, {0: 'X'})], n_qubits=2)
    >>> H.shape
    (4, 4)
    """
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
    dim = 1 << n_qubits
    H = np.zeros((dim, dim), dtype=np.complex128)

    for coeff, pauli_terms in terms:
        normalized = _normalize_terms(pauli_terms, n_qubits)
        if normalized and max(normalized) >= n_qubits:
            raise ValueError(
                f"term references qubit {max(normalized)}, but n_qubits={n_qubits}"
            )
        term_matrix = np.array([[1.0]], dtype=np.complex128)
        for q in range(n_qubits):
            letter = normalized.get(q, 'I')
            term_matrix = np.kron(term_matrix, _PAULI_MATRICES[letter])
        H += coeff * term_matrix

    return H


def pauli_sum_matvec(vector, terms, n_qubits=None):
    """
    H @ vector for a Hamiltonian given as a weighted sum of Pauli strings,
    computed in O(dim * n_terms) WITHOUT ever building the (2**n_qubits,
    2**n_qubits) matrix pauli_hamiltonian_to_matrix materializes -- the
    matrix-free counterpart needed for an iterative/sparse eigensolver
    (e.g. scipy.sparse.linalg.eigsh via a LinearOperator wrapping this
    function), where pauli_hamiltonian_to_matrix's O(dim**2) memory is
    exactly the thing being avoided.

    Same qubit-0-is-MSB convention as the rest of this module: this
    agrees with pauli_hamiltonian_to_matrix(terms, n_qubits) @ vector to
    floating-point precision for the same terms -- same underlying
    per-term application as pauli_expectation, just returning the vector
    P|psi> instead of reducing it to <psi|P|psi>.

    Parameters
    ----------
    vector : array-like, shape (2**n_qubits,)
        Not required to be normalized (this is a linear map, not an
        expectation value) -- e.g. an intermediate Lanczos vector, not
        necessarily a physical statevector.
    terms : iterable of (coeff, pauli_terms)
        Same format pauli_sum_expectation/pauli_hamiltonian_to_matrix
        accept.
    n_qubits : int, optional
        Only used to validate string-form terms' length; inferred from
        vector's length otherwise (same convention as pauli_expectation).

    Returns
    -------
    numpy.ndarray, shape (2**n_qubits,), dtype complex128

    Examples
    --------
    >>> import numpy as np
    >>> terms = [(1.0, 'ZZ'), (0.5, {0: 'X'})]
    >>> v = np.array([1, 0, 0, 0], dtype=complex)
    >>> pauli_sum_matvec(v, terms, n_qubits=2)
    array([1. +0.j, 0. +0.j, 0. +0.j, 0.5+0.j])
    >>> pauli_hamiltonian_to_matrix(terms, n_qubits=2) @ v
    array([1. +0.j, 0. +0.j, 0. +0.j, 0.5+0.j])
    """
    vector = np.asarray(vector, dtype=np.complex128)
    dim = vector.shape[0]
    inferred_n_qubits = dim.bit_length() - 1
    if 1 << inferred_n_qubits != dim:
        raise ValueError(f"vector length {dim} is not a power of 2")

    result = np.zeros(dim, dtype=np.complex128)
    for coeff, pauli_terms in terms:
        normalized = _normalize_terms(pauli_terms, n_qubits)
        if normalized and max(normalized) >= inferred_n_qubits:
            raise ValueError(
                f"term references qubit {max(normalized)}, but vector only "
                f"spans {inferred_n_qubits} qubits")
        result += coeff * _apply_pauli_term(vector, normalized, inferred_n_qubits)

    return result


def _apply_pauli_term_jax(statevector, terms, inferred_n_qubits):
    """JAX-native counterpart of _apply_pauli_term -- pure jnp ops, no
    np.asarray/float() cast on `statevector`, so this stays valid under
    jax.grad/jax.jit tracing (same split as _jsd_vectors/_jsd_vectors_jax
    in dense_evolution.backends.mps). `terms` (already-normalized dict)
    and `inferred_n_qubits` are always plain Python objects, never traced
    -- only `statevector` is ever a tracer here."""
    if not terms:
        return statevector

    def bit_pos(q):
        return inferred_n_qubits - 1 - q

    flip_mask = 0
    for q, p in terms.items():
        if p in ('X', 'Y'):
            flip_mask |= (1 << bit_pos(q))

    dim = statevector.shape[0]
    indices = jnp.arange(dim)
    source_idx = indices ^ flip_mask

    coeff = jnp.ones(dim, dtype=jnp.complex128)
    for q, p in terms.items():
        bit = (source_idx >> bit_pos(q)) & 1
        if p == 'Y':
            coeff = coeff * jnp.where(bit == 0, 1j, -1j)
        elif p == 'Z':
            coeff = coeff * jnp.where(bit == 0, 1.0, -1.0)

    return statevector[source_idx] * coeff


def pauli_sum_matvec_jax(vector, terms, n_qubits=None):
    """JAX-native counterpart of pauli_sum_matvec: H @ vector for a
    Hamiltonian given as a weighted sum of Pauli strings, never building
    the 2**n_qubits matrix -- but pure jnp internally (no np.asarray/
    float() on `vector`), so unlike pauli_sum_matvec itself this stays
    valid under jax.grad/jax.jit tracing. Verified to agree with
    pauli_sum_matvec to floating-point precision, and to give correct
    gradients (matching a dense pauli_hamiltonian_to_matrix reference)
    -- see test_pauli_sum_jax_matches_numpy_and_is_differentiable.

    `terms`/`n_qubits` must stay plain Python objects (never traced) --
    only `vector` may be a JAX tracer.

    PRECISION GOTCHA: this function never calls dense_evolution.config's
    ensure_x64() itself (it's a pure math function, no opinion on global
    JAX state) -- if nothing else in the process has constructed a
    DenseSVSimulator/QuantumHardwareRegistry/circuit_to_energy_fn yet
    (the only things that call ensure_x64() lazily), JAX is still at its
    float32 default, and this silently runs at complex64 precision with
    no error, just ~1e-7 relative accuracy instead of ~1e-16 -- verified
    directly (a standalone correctness selftest run before constructing
    anything else failed at max_diff=7.10e-07, exactly float32 relative
    precision, until dense_evolution.set_precision(True) was called
    first). Call dense_evolution.set_precision(True) yourself up front
    if you're using this standalone, before anything else has a chance
    to enable x64 for you.

    This is what lets a differentiable VQE loop reach 20+ qubits at all.
    circuit_to_energy_fn's own h_matrix @ statevector path needs a dense
    (2**n_qubits, 2**n_qubits) matrix -- physically impossible to hold
    much past ~14 qubits (2**28 complex128 entries = 4GB, x4 per extra
    qubit) -- while the statevector itself stays linear in dim (2**20
    complex128 = 16MB at 20 qubits, no problem at all). Drop this in as
    circuit_to_energy_fn's `h_matrix` argument via PauliSumOperator
    (below), whose only job is wrapping this behind `__matmul__` since
    that's the only operation energy_fn performs on h_matrix:

        from dense_evolution import circuit_to_energy_fn, PauliSumOperator
        energy_fn, n_params = circuit_to_energy_fn(circuit, n_qubits=20)
        h_op = PauliSumOperator(terms, n_qubits=20)
        energy, sv = energy_fn(theta, h_op)
        grad = jax.grad(lambda th: energy_fn(th, h_op)[0])(theta)
    """
    dim = vector.shape[0]
    inferred_n_qubits = dim.bit_length() - 1
    if 1 << inferred_n_qubits != dim:
        raise ValueError(f"vector length {dim} is not a power of 2")

    vector = jnp.asarray(vector, dtype=jnp.complex128)
    result = jnp.zeros(dim, dtype=jnp.complex128)
    for coeff, pauli_terms in terms:
        normalized = _normalize_terms(pauli_terms, n_qubits)
        if normalized and max(normalized) >= inferred_n_qubits:
            raise ValueError(
                f"term references qubit {max(normalized)}, but vector only "
                f"spans {inferred_n_qubits} qubits")
        result = result + coeff * _apply_pauli_term_jax(vector, normalized, inferred_n_qubits)

    return result


def pauli_sum_expectation_jax(statevector, terms, n_qubits=None):
    """JAX-native counterpart of pauli_sum_expectation -- same
    sum_i coeff_i * <psi|P_i|psi>, pure jnp internally so it stays valid
    under jax.grad/jax.jit (unlike pauli_sum_expectation, whose
    np.asarray(statevector) forces concretization). Returns a jnp float
    scalar, not a Python float -- call float(...) yourself outside a
    traced context if you need one. `terms`/`n_qubits` must stay plain
    Python objects; only `statevector` may be a tracer."""
    dim = statevector.shape[0]
    inferred_n_qubits = dim.bit_length() - 1
    if 1 << inferred_n_qubits != dim:
        raise ValueError(f"statevector length {dim} is not a power of 2")

    statevector = jnp.asarray(statevector, dtype=jnp.complex128)
    total = jnp.zeros((), dtype=jnp.float64)
    for coeff, pauli_terms in terms:
        normalized = _normalize_terms(pauli_terms, n_qubits)
        if normalized and max(normalized) >= inferred_n_qubits:
            raise ValueError(
                f"term references qubit {max(normalized)}, but statevector only "
                f"spans {inferred_n_qubits} qubits")
        p_psi = _apply_pauli_term_jax(statevector, normalized, inferred_n_qubits)
        total = total + coeff * jnp.real(jnp.vdot(statevector, p_psi))

    return total


class PauliSumOperator:
    """Matrix-free Hamiltonian wrapper: presents a Pauli-sum Hamiltonian
    (the same `terms` format pauli_sum_expectation/pauli_sum_matvec_jax
    accept) as an object supporting `@`, so it can be dropped in wherever
    a dense h_matrix is expected without ever materializing one.

    Written specifically for circuit_to_energy_fn(circuit, n_qubits)'s
    energy_fn(theta, h_matrix, ...), whose only use of h_matrix is
    `h_matrix @ statevector` -- see pauli_sum_matvec_jax's docstring for
    the full worked example. `terms`/`n_qubits` are fixed at construction
    (plain Python objects, never traced); only the vector passed to
    `@` may be a JAX tracer, keeping the whole thing jax.grad-safe."""

    def __init__(self, terms, n_qubits):
        self.terms = terms
        self.n_qubits = n_qubits

    def __matmul__(self, vector):
        return pauli_sum_matvec_jax(vector, self.terms, n_qubits=self.n_qubits)


_SAME_QUBIT_PAULI_PRODUCT = {
    ('X', 'X'): (1, None), ('Y', 'Y'): (1, None), ('Z', 'Z'): (1, None),
    ('X', 'Y'): (1j, 'Z'), ('Y', 'X'): (-1j, 'Z'),
    ('Y', 'Z'): (1j, 'X'), ('Z', 'Y'): (-1j, 'X'),
    ('Z', 'X'): (1j, 'Y'), ('X', 'Z'): (-1j, 'Y'),
}


def multiply_pauli_terms(factors):
    """
    Multiplies several Pauli-string OPERATORS together into one combined
    term, tracking the i^k phase picked up whenever two factors act on the
    same qubit (X*Y=iZ, Y*X=-iZ, etc.) -- the exact symbolic algebra
    needed to compose Pauli strings by hand, e.g. building the "Klein
    factor" total-parity operator for a set of Jordan-Wigner-mapped
    Majorana modes (see dense_evolution.physics.fermions.total_parity_operator),
    or any other manual Pauli-operator product.

    This multiplies OPERATORS (order matters -- Pauli matrices don't
    commute), unlike pauli_hamiltonian_to_matrix/pauli_sum_expectation,
    which take a SUM of independent terms (order doesn't matter there).

    Promoted from Dense-Evolution-Discovery's dashboard_core.wormhole
    module (`_multiply_pauli_dicts`), where it was originally written to
    combine independently-Jordan-Wigner-mapped Majorana operators across
    the two sides of a wormhole-teleportation simulation -- a generic
    Pauli-algebra operation with no dependency on that use case, so it
    belongs here instead of duplicated wherever it's next needed.

    Parameters
    ----------
    factors : iterable of (coeff, pauli_terms)
        Same (coeff, pauli_terms) pair format pauli_hamiltonian_to_matrix's
        `terms` accepts -- no bare-term shorthand (a Python int coefficient
        would be indistinguishable from a bare (qubit, pauli) pair, e.g.
        (1, 'X'), so this deliberately doesn't try to support one; pass
        (1.0, pauli_terms) explicitly instead). Applied left to right --
        the first factor is the leftmost operator in the product.

    Returns
    -------
    (complex, dict)
        combined_coeff : the product of every factor's own coefficient,
        times the i^k phase accumulated from same-qubit collisions.
        combined_pauli_dict : {qubit: 'X'|'Y'|'Z'}, identity qubits
        dropped -- ready for pauli_hamiltonian_to_matrix / pauli_expectation
        / another multiply_pauli_terms call.

    Examples
    --------
    >>> multiply_pauli_terms([(1.0, 'X'), (1.0, 'Y')])  # X0 * Y0 = i*Z0
    (1j, {0: 'Z'})
    >>> multiply_pauli_terms([(2.0, 'X'), (3.0, {1: 'Z'})])  # disjoint qubits, no collision
    (6.0, {0: 'X', 1: 'Z'})
    """
    merged = {}
    total_coeff = 1.0 + 0j
    for coeff, pauli_terms in factors:
        total_coeff *= coeff
        normalized = _normalize_terms(pauli_terms)
        for q, p in normalized.items():
            if q not in merged:
                merged[q] = p
            else:
                phase, new_p = _SAME_QUBIT_PAULI_PRODUCT[(merged[q], p)]
                total_coeff *= phase
                if new_p is None:
                    del merged[q]
                else:
                    merged[q] = new_p
    return total_coeff, merged
