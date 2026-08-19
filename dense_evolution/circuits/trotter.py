"""
Real-time Hamiltonian evolution as an actual gate circuit (Trotterization)
-- did not exist anywhere in this package before. Every existing piece of
"evolution" machinery here is either gate-based-and-fixed (a hand-written
or VQE-optimized circuit template) or exact-and-not-a-circuit
(dashboard_core.hamiltonians.ground_state_energy's dense diagonalization).
Nothing composed exp(-i*H*t) for an arbitrary Hamiltonian into gates a
real quantum computer could run.

Originated in research/wormhole_syk.py, where it closed an explicit,
previously-open follow-on: reproducing a traversable-wormhole-teleportation
signal (arXiv:2604.10090) first via exact matrix exponentiation (cheap,
but not what real hardware executes), then via this module's Trotterized
gate circuit -- verified the signal wasn't an artifact of the exact-
evolution shortcut, it survives with real gates too. Neither function
here is specific to that experiment or to SYK physics; both drop
straight into any future feature needing exp(-i*H*t) as gates (a
Trotterized VQE-adjacent ansatz, quench dynamics, etc.).

pauli_rotation_ops is exact for a single Pauli-string term (fidelity
1.0 against scipy.linalg.expm, verified in tests/unit/test_trotter.py for
1-4 qubit mixed X/Y/Z strings, not just Z-strings); trotter_evolve_ops
composes many such terms via the first-order product formula by default,
which is an *approximation* whose error shrinks as n_steps grows (also
verified: infidelity drops roughly 4x per doubling of steps against a
real, non-trivial multi-qubit Hamiltonian, consistent with the expected
quadratic convergence of first-order Trotter error in state overlap).

order=2 selects the second-order (Strang/symmetric) product formula
instead -- each step applies the terms forward at half the angle, then
backward (reversed order) at half the angle again:
[prod_k exp(-i*c_k*P_k*dt/2)] * [prod_k(reversed) exp(-i*c_k*P_k*dt/2)],
which cancels the first-order formula's leading error term (verified in
tests/unit/test_trotter.py: infidelity drops roughly 16x per doubling of
steps, consistent with the expected quartic convergence of second-order
Trotter error in state overlap, vs. order=1's ~4x). Costs 2x the gates
of order=1 for the same n_steps -- the standard second-order tradeoff,
worth it when n_steps would otherwise need to be large for accuracy
(e.g. the noise-robustness experiments in wormhole_syk_teleportation.py,
where gate count directly limits how much depolarizing noise the
circuit accumulates).
"""

__all__ = ['pauli_rotation_ops', 'trotter_evolve_ops']


def pauli_rotation_ops(pauli_dict, angle):
    """Gate-tuple circuit for exp(-i*angle*P), P a Pauli string given as
    {qubit: 'X'/'Y'/'Z'} -- basis-change + CNOT-staircase + RZ + inverse,
    the same identity already used elsewhere in this codebase
    (dashboard_core.vqe's UCCSD/QAOA-style ZZ interactions) generalized
    here to arbitrary mixed X/Y/Z strings, not just Z-strings.

    This package's rz(theta) = exp(-i*theta/2*Z) (checked directly
    against scipy.linalg.expm when this was written, not assumed from
    convention) -- rz(2*angle) on the accumulator qubit therefore gives
    exactly exp(-i*angle*Z) on the accumulated parity.

    Parameters
    ----------
    pauli_dict : dict
        {qubit: 'X'|'Y'|'Z'}. An empty dict (identity term) returns [].
    angle : float

    Returns
    -------
    list[tuple]
        Gate tuples ready for DenseSVSimulator.run_circuit /
        QASMParser-compatible circuits.
    """
    qubits = sorted(pauli_dict.keys())
    if not qubits:
        return []
    ops = []
    for q in qubits:
        letter = pauli_dict[q]
        if letter == 'X':
            ops.append(('h', q))
        elif letter == 'Y':
            ops.append(('sdg', q))
            ops.append(('h', q))
    for i in range(len(qubits) - 1):
        ops.append(('cx', qubits[i], qubits[i + 1]))
    ops.append(('rz', qubits[-1], 2 * angle))
    for i in reversed(range(len(qubits) - 1)):
        ops.append(('cx', qubits[i], qubits[i + 1]))
    for q in qubits:
        letter = pauli_dict[q]
        if letter == 'X':
            ops.append(('h', q))
        elif letter == 'Y':
            ops.append(('h', q))
            ops.append(('s', q))
    return ops


def trotter_evolve_ops(terms, t, n_steps, order=1):
    """Trotter product formula for exp(-i*H*t), H = sum_k c_k*P_k.

    order=1 (default): [prod_k exp(-i*c_k*P_k*(t/n_steps))]^n_steps.
    Term order within one step follows `terms`' own order, identical
    every repetition (not re-randomized per step).

    order=2: Strang/symmetric splitting -- each step is a forward half-
    angle pass through `terms` followed by a backward half-angle pass
    through `terms` reversed, [prod_k exp(-i*c_k*P_k*dt/2)] *
    [prod_k(reversed) exp(-i*c_k*P_k*dt/2)], repeated n_steps times.
    Quadratically more accurate than order=1 for the same n_steps (see
    module docstring), at 2x the gate count per step.

    Parameters
    ----------
    terms : list[(float, dict)]
        (coefficient, pauli_dict) pairs, e.g. from
        dense_evolution.pauli_hamiltonian_to_matrix's own term format,
        or dense_evolution.majorana_pauli_terms products.
    t : float
        Total evolution time.
    n_steps : int
        Number of Trotter steps -- higher is more accurate and more
        gates, the standard Trotter accuracy/cost tradeoff.
    order : int
        1 (default) or 2 -- see above.

    Returns
    -------
    list[tuple]
        Gate tuples for the whole Trotterized evolution.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")
    dt = t / n_steps
    if order == 1:
        step_ops = []
        for c, pdict in terms:
            step_ops.extend(pauli_rotation_ops(pdict, c * dt))
    else:
        half_dt = dt / 2
        step_ops = []
        for c, pdict in terms:
            step_ops.extend(pauli_rotation_ops(pdict, c * half_dt))
        for c, pdict in reversed(terms):
            step_ops.extend(pauli_rotation_ops(pdict, c * half_dt))
    return step_ops * n_steps
