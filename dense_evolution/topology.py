"""
Entangling-layer topology helpers.

Every variational circuit (VQE, QAOA, hardware-efficient ansätze) needs an
entangling layer, and hand-writing it as a `for` loop of two-qubit gates is
one of the most repeated patterns across quantum-circuit code. Other
libraries give it a name and a single call instead (Qiskit's
`TwoLocal(entanglement=...)`, PennyLane's `qml.broadcast(pattern=...)`).
`entangling_layer` is the Dense-Evolution equivalent: it returns a plain
list of gate tuples in the circuit format `run_circuit` already accepts, so
it drops straight into any existing circuit list via concatenation.
"""

VALID_PATTERNS = ('linear', 'circular', 'full', 'star', 'brick')


def entangling_layer(n_qubits, pattern='linear', gate='cx', reverse=False, hub=0):
    """
    Build a list of two-qubit gate tuples connecting n_qubits according to
    a named topology.

    Patterns
    --------
    'linear'   -- chain: (0,1), (1,2), ..., (n-2,n-1)
    'circular' -- linear + one wraparound edge (n-1,0) ("ring"); identical
                  to 'linear' when n_qubits == 2, since there is only one
                  possible edge between two qubits
    'full'     -- every pair (i,j) with i<j ("complete"/all-to-all)
    'star'     -- a single hub qubit connected to every other qubit
    'brick'    -- alternating even/odd layers: (0,1)(2,3).. then (1,2)(3,4)..
                  ("brickwork"/staircase, the pattern behind most Trotterized
                  and hardware-efficient ansätze)

    Parameters
    ----------
    n_qubits : int
        Number of qubits involved, must be >= 2.
    pattern : str
        One of VALID_PATTERNS.
    gate : str
        Two-qubit gate name applied to every edge (e.g. 'cx', 'cz', 'cy').
        Not validated against dense_evolution.gates.GATES here, so a custom
        gate name registered elsewhere still works.
    reverse : bool
        Swap (control, target) -> (target, control) for every edge. Some
        ansätze alternate direction layer to layer for symmetry.
    hub : int
        Hub qubit index, only used by pattern='star'.

    Returns
    -------
    list[tuple[str, int, int]]
    """
    if n_qubits < 2:
        raise ValueError(f"entangling_layer needs at least 2 qubits, got {n_qubits}")
    if pattern not in VALID_PATTERNS:
        raise ValueError(f"unknown pattern {pattern!r}, expected one of {VALID_PATTERNS}")

    if pattern == 'linear':
        edges = [(q, q + 1) for q in range(n_qubits - 1)]
    elif pattern == 'circular':
        edges = [(q, q + 1) for q in range(n_qubits - 1)]
        if n_qubits > 2:
            edges.append((n_qubits - 1, 0))
    elif pattern == 'full':
        edges = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    elif pattern == 'star':
        if not (0 <= hub < n_qubits):
            raise ValueError(f"hub={hub} out of range for n_qubits={n_qubits}")
        edges = [(hub, q) for q in range(n_qubits) if q != hub]
    elif pattern == 'brick':
        even = [(q, q + 1) for q in range(0, n_qubits - 1, 2)]
        odd = [(q, q + 1) for q in range(1, n_qubits - 1, 2)]
        edges = even + odd

    if reverse:
        edges = [(b, a) for a, b in edges]

    return [(gate, a, b) for a, b in edges]
